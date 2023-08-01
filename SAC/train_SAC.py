#!/usr/bin/env python3
import os
import pprint
import sys
sys.path.append("FSRL")
from fsrl.policy.sac_lag import SACLagrangian
from fsrl.trainer.offpolicy import OffpolicyTrainer
from fsrl.utils.net.common import ActorCritic
# Set this before everything
os. environ['WANDB_DISABLED'] = 'True'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import asdict, dataclass
import pickle
import ast
import warnings # FIXME: Fix this warning eventually
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import highway_env
import bullet_safety_gym
import gymnasium as gym
import numpy as np
import torch.nn as nn
try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")

import torch
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv, RayVectorEnv
from tianshou.data import VectorReplayBuffer

# To render the environemnt and agent
import matplotlib.pyplot as plt
sys.path.append("FSRL")
from fsrl.agent import SACLagAgent
from fsrl.config.sacl_cfg import TrainCfg
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name
from fsrl.utils.net.continuous import DoubleCritic
from tianshou.utils.net.common import Net, get_dict_state_decorator, DataParallelNet
from tianshou.utils.net.continuous import ActorProb
from utils.utils import load_environment
from fsrl.utils.exp_util import auto_name, seed_all
from gymnasium.spaces.dict import Dict
from fsrl.data import FastCollector

TASK_TO_CFG = {
    # HighwayEnv tasks
    "roundabout-v0": TrainCfg, # TODO: Change the configs for HighEnv tasks
}

# Make my own config params
@dataclass
class MyCfg(TrainCfg):
    task: str = "parking-v0"
    epoch: int = 250
    lr: float = 0.001
    render: float = None # The rate at which it renders (e.g., .001)
    render_mode: str = None # "rgb_array" or "human" or None
    thread: int = 320 # If use CPU to train
    step_per_epoch: int = 20000
    target_kl: float = 0.01
    project: str = "fast-safe-rl"
    worker: str = "ShmemVectorEnv"
    # worker: str = "RayVectorEnv"
    # Decide which device to use based on availability
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    gamma: float = .99
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoal.txt'

with open(MyCfg.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)


@pyrallis.wrap()
def train(args: MyCfg):
    # set seed and computing
    seed_all(args.seed)
    torch.set_num_threads(args.thread)

    task = args.task
    default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else TrainCfg()
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
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(training_num)])
    test_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(args.testing_num)])


    # model
    env = load_environment(ENV_CONFIG)
    # Get the shapes of the states and actions to be transfered to a tensor
    if isinstance(env.observation_space, Dict):
        # TODO: This is hardcoded for the HighwayEnv-parking environment | FIXME
        dict_state_shape = {
            "achieved_goal": (6,),
            "observation": (6,),
            "desired_goal": (6,)
        }
        decorator_fn, state_shape = get_dict_state_decorator(dict_state_shape, list(dict_state_shape.keys()))
        global Net, ActorProb, DoubleCritic, DataParallelNet # Fixes UnboundLocalError
        # Apply decorator to overwrite the forward pass in the Tensorflow module to allow for dict object
        Net = decorator_fn(Net)
        ActorProb = decorator_fn(ActorProb)
        DoubleCritic = decorator_fn(DoubleCritic)
        DataParallelNet = decorator_fn(DataParallelNet)
    else: 
        state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # W/ DataParallelNet For cuda Parallelization
    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
            ActorProb(
                net,
                action_shape,
                max_action=max_action,
                device=None, # Set device to none with DataParallelNet Wrapper
                conditioned_sigma=args.conditioned_sigma,
                unbounded=args.unbounded
            ).to(args.device)
        )
        critics = []
        for _ in range(2):
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
            critics.append(DataParallelNet(DoubleCritic(net1, net2, device=None).to(args.device)))
    else:
        actor = ActorProb(
                    net,
                    action_shape,
                    max_action=max_action,
                    device=args.device,
                    conditioned_sigma=args.conditioned_sigma,
                    unbounded=args.unbounded
                ).to(args.device)
        critics = []
        for _ in range(2):
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
            critics.append(DoubleCritic(net1, net2, device=args.device).to(args.device))
    
    # Optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(
        nn.ModuleList(critics).parameters(), lr=args.critic_lr
    )

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

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)
    
    policy = SACLagrangian(
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        logger=logger,
        alpha=args.alpha,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=None,
        n_step=args.n_step,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        reward_normalization=False,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_scheduler=None
    )

    # collector
    train_collector = FastCollector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = FastCollector(policy, test_envs)

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
        logger.store(tab="train", cost_limit=args.cost_limit)
        print(f"Epoch: {epoch}")
        print(info)
    
    if __name__ == "__main__":
        pprint.pprint(info)
        # Let's watch its performance!
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