#!/usr/bin/env python3
import os
import pprint
import random
import sys
sys.path.append("FSRL")
from fsrl.utils.net.common import ActorCritic
# Set this before everything
os. environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataclasses import asdict, dataclass, field
import ast
import warnings # FIXME: Fix this warning eventually
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import highway_env
import bullet_safety_gym
import gymnasium as gym
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.discrete import Discrete # Must be gymnasium, not gym for type checking
import numpy as np
import torch.nn as nn
import safety_gymnasium

import torch
from torch.distributions import Independent, Normal
import pyrallis

from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.common import Net, get_dict_state_decorator, DataParallelNet
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv, RayVectorEnv
from tianshou.data import VectorReplayBuffer, ReplayBuffer

# To render the environemnt and agent
import matplotlib.pyplot as plt
sys.path.append("FSRL")
from fsrl.config.cpo_cfg import TrainCfg
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.continuous import DoubleCritic, Critic
from fsrl.data import FastCollector
from fsrl.policy import CPO
from fsrl.trainer import OnpolicyTrainer
from utils.utils import load_environment

from typing import Union, List

import argparse

# CPO arguments
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('--task', type=str, default="parking-v0", help='Task for training')
parser.add_argument('--project', type=str, default="2-constraints-absolute", help='Project name')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
parser.add_argument('--target_kl', type=float, default=0.01, help='Target KL divergence')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--step_per_epoch', type=int, default=20000, help='Steps per epoch')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma value for reinforcement learning')
parser.add_argument('--cost_limit', type=float, nargs='+', default=[3.0, 3.0], help='Cost limit values as a list', metavar='FLOAT')
parser.add_argument('--render', type=float, default=None, help='Render interval (if applicable)')
parser.add_argument('--render_mode', type=str, default=None, help='Mode for rendering')
parser.add_argument('--thread', type=int, default=320, help='Number of threads')

# Environment argumnets
parser.add_argument('--constrained_rl', type=bool, default=True, help='Identifier for constrained RL')
parser.add_argument('--cost_delta_distance', type=float, default=4.0, help='The maximum distance to line points until costs incur')
parser.add_argument('--quantized_line_points', type=int, default=20, help='Number of quantized points for each parking line')
parser.add_argument('--absolute_cost_distance', type=bool, default=True, help='Indicates whether absolute cost function is used instead of gradual')
parser.add_argument('--cost_speed_limit', type=float, default=4.0, help='The maximum speed until costs incur')
parser.add_argument('--absolute_cost_speed', type=bool, default=True, help='Indicates whether absolute cost function is used instead of gradual')


args = parser.parse_args()

@dataclass
class MyCfg(TrainCfg):
    task: str = args.task
    epoch: int = args.epoch # Get epoch from command-line arguments
    lr: float = args.lr
    render: float = args.render # The rate at which it renders (e.g., .001)
    render_mode: str = args.render_mode # "rgb_array" or "human" or None
    thread: int = args.thread # If use CPU to train
    step_per_epoch: int = args.step_per_epoch
    target_kl: float = args.target_kl
    project: str = args.project
    gamma: float = args.gamma
    # cost_limit: Union[List, float] = field(default_factory=lambda: args.cost_limit)
    cost_limit: Union[List, float] = field(default_factory=lambda: [3.0, 3.0])
    
    worker: str = "ShmemVectorEnv"
    # Decide which device to use based on availability
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
    # Points are around the parking lot and in the middle
    random_starting_locations = [[0,0], [40, 40], [-40,-40], [40, -40], [-40, 40], [0, 40], [-40, 0]]

with open(MyCfg.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
ENV_CONFIG.update({
    # Costs
    "constrained_rl": args.constrained_rl,
    # Cost-distance
    "cost_delta_distance": args.cost_delta_distance,
    "quantized_line_points": args.quantized_line_points,
    "absolute_cost_distance": args.absolute_cost_distance,
    # Cost-speed
    "cost_speed_limit": args.cost_speed_limit,
    "absolute_cost_speed": args.absolute_cost_speed
    })

@pyrallis.wrap()
def train(args: MyCfg):
    # set seed and computing
    seed_all(args.seed)
    torch.set_num_threads(args.thread)

    task = args.task
    # default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else TrainCfg()
    default_cfg = TrainCfg()
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        default_cfg.task = args.task
        default_cfg.seed = args.seed
        default_cfg.device = args.device
        default_cfg.logdir = args.logdir
        default_cfg.project = args.project
        default_cfg.group = args.group
        default_cfg.suffix = args.suffix
        args = default_cfg

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        if isinstance(args.cost_limit, list): # Since you can have more than one cost limit
            args.group = args.task 
            for index, cost_limit in enumerate(args.cost_limit):
                args.group += f"-cost{index}-" + str(int(cost_limit))
        else: 
            args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    if MyCfg.random_starting_locations:
        # Make a list of initialized environments with different starting positions
        env_training_list, env_testing_list = [], []
        for _ in range(training_num):
            ENV_CONFIG.update({"start_location": random.choice(MyCfg.random_starting_locations)})
            env_training_list.append(ENV_CONFIG)
        for _ in range(args.testing_num):
            ENV_CONFIG.update({"start_location": random.choice(MyCfg.random_starting_locations)})
            env_testing_list.append(ENV_CONFIG)

        train_envs = worker([lambda: load_environment(env_training_list[i]) for i in range(training_num)])
        test_envs = worker([lambda: load_environment(env_testing_list[i]) for i in range(args.testing_num)])
    else:
        train_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(training_num)])
        test_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(args.testing_num)])

    # This env is used strictly to evaluate the observation and action shapes for CPO
    env = load_environment(ENV_CONFIG)

    # set seed and computing
    seed_all(args.seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(args.thread)

    # Model
    # Get the shapes of the states and actions to be transfered to a tensor
    if isinstance(env.observation_space, Dict):
        # TODO: This is hardcoded please fix
        dict_state_shape = {
            "achieved_goal": (6,),
            "observation": (6,),
            "desired_goal": (6,)
        }
        decorator_fn, state_shape = get_dict_state_decorator(dict_state_shape, list(dict_state_shape.keys()))
        global Net, ActorProb, Critic, DataParallelNet # Fixes UnboundLocalError
        # Apply decorator to overwrite the forward pass in the Tensorflow module to allow for dict object
        Net = decorator_fn(Net)
        ActorProb = decorator_fn(ActorProb)
        Critic = decorator_fn(Critic)
        DataParallelNet = decorator_fn(DataParallelNet)
    else: 
        state_shape = env.observation_space.shape or env.observation_space.n

    action_shape = env.action_space.shape or env.action_space.n

    if isinstance(env.action_space, Discrete):
        max_action = env.action_space.n
    else:
        # max_action = env.action_space.n
        max_action = env.action_space.high[0]

    # W/ DataParallelNet For cuda Parallelization
    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
                ActorProb(
                    net,
                    action_shape,
                    max_action=max_action,
                    unbounded=args.unbounded,
                    device=None
                ).to(args.device)
        )
        critic = [DataParallelNet(
                        Critic(
                            Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
                            device=None
                        ).to(args.device)
                    ) for _ in range(2)
        ]
    else:
        actor = ActorProb(
            net,
            action_shape,
            max_action=max_action,
            unbounded=args.unbounded,
            device=args.device
        ).to(args.device)
        critic = [
            Critic(
                Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
                device=args.device
            ).to(args.device) for _ in range(2)
        ]
    if not torch.cuda.is_available():
        torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    if args.last_layer_scale:
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = CPO(
        actor,
        critic,
        optim,
        dist,
        logger=logger,
        # CPO specific arguments
        device=args.device,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        damping_coeff=args.damping_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        l2_reg=args.l2_reg,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        # Base Policy Common arguments
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_scheduler=None
    )

    # Collectors
    if isinstance(train_envs, gym.Env):
        buffer = ReplayBuffer(args.buffer_size)
    else:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = FastCollector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True,
    )
    test_collector = FastCollector(
        policy, test_envs
    ) if test_envs is not None else None

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

    def checkpoint_fn():
        return {"model": policy.state_dict()}

    if args.save_ckpt:
        logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )

    for epoch, epoch_stat, info in trainer:
        logger.store(tab="train", cost_limit_distance=args.cost_limit[0], cost_limit_speed=args.cost_limit[1])
        print(f"Epoch: {epoch}")
        print(info)

    if __name__ == "__main__":
        pprint.pprint(info)
        # Let's watch its performance!
        # Update the starting location
        if MyCfg.random_starting_locations:
            ENV_CONFIG.update({"starting_location": random.choice(MyCfg.random_starting_locations)})
        env = load_environment(ENV_CONFIG)
        policy.eval()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        policy.train()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
