import copy
import os
import argparse
os. environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pprint
import random
import sys
from dataclasses import asdict
sys.path.append("FSRL")
from fsrl.utils.net.common import ActorCritic
from dataclasses import asdict, dataclass, field

import ast
# import warnings # FIXME: Fix this warning eventually
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
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
from fsrl.config.cvpo_cfg import TrainCfg
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.continuous import SingleCritic, DoubleCritic, Critic
from fsrl.data import FastCollector
from fsrl.policy import CVPO
from fsrl.trainer import OffpolicyTrainer
from utils import load_environment

from typing import Tuple, Union, List

# CPO arguments
# TODO None of these actually work --> there exist predefined arguments somewhere I can't find
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('-f', default='cvpo', help=argparse.SUPPRESS)
parser.add_argument('--task', type=str, default="parking-v0", help='Task for training')
parser.add_argument('--project', type=str, default="CVPO", help='Project name')
parser.add_argument('--epoch', type=int, default=400, help='Number of epochs')
parser.add_argument('--step_per_epoch', type=int, default=20000, help='Steps per epoch')
parser.add_argument('--gamma', type=float, default=0.97, help='Gamma value for reinforcement learning')
parser.add_argument('--cost_limit', type=float, nargs='+', default=[5.0, 5.0], help='Cost limit values as a list', metavar='FLOAT')
parser.add_argument('--render', type=float, default=None, help='Render interval (if applicable)')
parser.add_argument('--render_mode', type=str, default=None, help='Mode for rendering')
parser.add_argument('--thread', type=int, default=320, help='Number of threads')

# Environment argumnets
# parser.add_argument('--constrained_rl', type=bool, default=True, help='Identifier for constrained RL')
parser.add_argument('--constraint_type', type=str, nargs='*', default=["distance", "speed"], help='List of constraint types to use')
parser.add_argument('--cost_speed_limit', type=float, default=4.0, help='The maximum speed until costs incur')
parser.add_argument('--absolute_cost_speed', type=bool, default=True, help='Indicates whether absolute cost function is used instead of gradual')
args = parser.parse_args()

@dataclass
class MyCfg(TrainCfg):
    task: str = args.task
    epoch: int = args.epoch # Get epoch from command-line arguments
    render: float = args.render # The rate at which it renders (e.g., .001)
    render_mode: str = args.render_mode # "rgb_array" or "human" or None
    thread: int = args.thread # If use CPU to train
    step_per_epoch: int = args.step_per_epoch
    project: str = args.project
    gamma: float = args.gamma
    step_per_epoch: int = 2000 # Shorten since parking env has small episodes
    double_critic: bool = True # Increases stability and promotes exploration since it is pessimistic on the Q-values it predicts
    # cost_limit: Union[List, float] = field(default_factory=lambda: args.cost_limit)
    cost_limit: Union[List, float] = field(default_factory=lambda: [2.0, 2.0])
    constraint_type: list[str] = field(default_factory=lambda: ["distance", "speed"])
    hidden_sizes: Tuple[int, ...] = (128, 128)
    
    worker: str = "ShmemVectorEnv"
    # Decide which device to use based on availability
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
    # Points are around the parking lot and in the middle
    random_starting_locations = [[0,32]]

with open(MyCfg.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
ENV_CONFIG.update({
    "add_walls": True,
    # Costs
    # "constrained_rl": args.constrained_rl,
    "constraint_type": args.constraint_type,
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
    default_cfg = MyCfg()
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
    logger.save_config(cfg, verbose=args.verbose)

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    if MyCfg.random_starting_locations:
        def generate_env_config(num):
            return [{"start_location": random.choice(MyCfg.random_starting_locations)} for _ in range(num)]
        def get_updated_config(i, env_list):
            updated_config = copy.deepcopy(ENV_CONFIG)
            updated_config.update(env_list[i])
            return updated_config

        # Make a list of initialized environments with different starting positions
        env_training_list = generate_env_config(training_num)
        env_testing_list = generate_env_config(args.testing_num)

        train_envs = worker([lambda i=i: load_environment(get_updated_config(i, env_training_list)) for i in range(training_num)])
        test_envs = worker([lambda i=i: load_environment(get_updated_config(i, env_testing_list)) for i in range(args.testing_num)])
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
        global Net, ActorProb, DoubleCritic, SingleCritic, DataParallelNet # Fixes UnboundLocalError
        # Apply decorator to overwrite the forward pass in the Tensorflow module to allow for dict object
        Net = decorator_fn(Net)
        ActorProb = decorator_fn(ActorProb)
        DoubleCritic = decorator_fn(DoubleCritic)
        SingleCritic = decorator_fn(SingleCritic)
        DataParallelNet = decorator_fn(DataParallelNet)
    else: 
        state_shape = env.observation_space.shape or env.observation_space.n

    action_shape = env.action_space.shape or env.action_space.n

    if isinstance(env.action_space, Discrete):
        max_action = env.action_space.n
    else:
        # max_action = env.action_space.n
        max_action = env.action_space.high[0]

    assert hasattr(
        env.spec, "max_episode_steps"
    ), "Please use an env wrapper to provide 'max_episode_steps' for CVPO"

    use_cuda = torch.cuda.is_available()
    # Create Actor
    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor_constructor = ActorProb(net, action_shape, max_action=max_action, unbounded=args.unbounded, conditioned_sigma=args.conditioned_sigma, device=None if use_cuda else args.device)
    actor = DataParallelNet(actor_constructor).to(args.device) if use_cuda else actor_constructor.to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # Create Critics
    if args.double_critic:
        net1 = Net(
            state_shape,
            action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device
        )
        net2 = Net(
            state_shape,
            action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device
        )
        critic_constructor = lambda: DoubleCritic(net1, net2, device=None if use_cuda else args.device).to(args.device)
    else:
        net_c = Net(
            state_shape,
            action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device
        )
        critic_constructor = lambda: SingleCritic(net_c, device=None if use_cuda else args.device).to(args.device)

    critics = [DataParallelNet(critic_constructor()) for _ in range(2)] if use_cuda else [critic_constructor() for _ in range(2)]
    critic_optim = torch.optim.Adam(
        nn.ModuleList(critics).parameters(), lr=args.critic_lr
    )

    if not args.conditioned_sigma and not use_cuda:
        torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critics)
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

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = CVPO(
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        logger=logger,
        action_space=env.action_space,
        dist_fn=dist,
        # max_episode_steps=env.spec.max_episode_steps,
        max_episode_steps=args.step_per_epoch, # TODO Check that this is somewhat right
        cost_limit=args.cost_limit,
        tau=args.tau,
        gamma=args.gamma,
        n_step=args.n_step,
        # E-step
        estep_iter_num=args.estep_iter_num,
        estep_kl=args.estep_kl,
        estep_dual_max=args.estep_dual_max,
        estep_dual_lr=args.estep_dual_lr,
        sample_act_num=args.sample_act_num,  # for continous action space
        # M-step
        mstep_iter_num=args.mstep_iter_num,
        mstep_kl_mu=args.mstep_kl_mu,
        mstep_kl_std=args.mstep_kl_std,
        mstep_dual_max=args.mstep_dual_max,
        mstep_dual_lr=args.mstep_dual_lr,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        lr_scheduler=None
    )

    # collector
    if isinstance(train_envs, gym.Env):
        buffer = ReplayBuffer(args.buffer_size)
    else:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = FastCollector(
        policy,
        train_envs,
        buffer,
        exploration_noise=False,
        constraint_type=args.constraint_type,
    )
    test_collector = FastCollector(policy, test_envs, constraint_type=args.constraint_type)

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

    def checkpoint_fn():
        return {"model": policy.state_dict()}

    if args.save_ckpt:
        logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        constraint_type=args.constraint_type,
        step_per_epoch=args.step_per_epoch,
        update_per_step=args.update_per_step,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )

    for epoch, epoch_stat, info in trainer:
        logger.store(tab="train" ,cost_limit_distance=args.cost_limit[0], cost_limit_speed=args.cost_limit[1])
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
