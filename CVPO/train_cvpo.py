

import copy
import os
# os. environ['WANDB_DISABLED'] = 'True'
# os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pprint
import random
import sys
sys.path.append("FSRL")
from fsrl.utils.net.common import ActorCritic
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

import wandb
wandb.init(project="CVPO-sweep")

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

# To render the environemnt and agent
from fsrl.config.cvpo_cfg import TrainCfg
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.data import FastCollector
from fsrl.agent import CVPOAgent
from utils import load_environment

from typing import Tuple, Union, List

import argparse

# CPO arguments
# TODO None of these actually work --> there exist predefined arguments somewhere I can't find
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('--task', type=str, default="parking-v0", help='Task for training')
# parser.add_argument('--project', type=str, default="2-constraints-absolute", help='Project name')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
parser.add_argument('--target_kl', type=float, default=0.01, help='Target KL divergence')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--step_per_epoch', type=int, default=20000, help='Steps per epoch')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma value for reinforcement learning')
parser.add_argument('--cost_limit', type=float, nargs='+', default=[3.0, 3.0], help='Cost limit values as a list', metavar='FLOAT')
parser.add_argument('--render', type=float, default=None, help='Render interval (if applicable)')
parser.add_argument('--render_mode', type=str, default=None, help='Mode for rendering')
parser.add_argument('--thread', type=int, default=320, help='Number of threads')
parser.add_argument('--normalize_obs', type=bool, default=True, help='normalization of observation')
parser.add_argument('--actor_lr', type=float, default=.001, help='actor learning rate')
parser.add_argument('--critic_lr', type=float, default=.001, help='critic learning rate')

# Environment argumnets
parser.add_argument('--constraint_type', type=str, nargs='+', default=["lines", "speed"], help='List of constraint types to use')
parser.add_argument('--speed_limit', type=float, default=4.0, help='The maximum speed until costs incur')
parser.add_argument('--absolute_cost_speed', type=bool, default=True, help='Indicates whether absolute cost function is used instead of gradual')

# args = wandb.config

args = parser.parse_args()


@dataclass
class MyCfg(TrainCfg):
    task: str = args.task
    project: str = "Line-constraint"
    epoch: int = 700 # Get epoch from command-line arguments
    step_per_epoch: int = 1000
    cost_limit: Union[List, float] = field(default_factory=lambda: [5.0])
    constraint_type: list[str] = field(default_factory=lambda: ["lines"])
    worker: str = "ShmemVectorEnv"
    # Decide which device to use based on availability
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
    hidden_sizes: Tuple[int, ...] = (128, 128)
    random_starting_locations = [[0,0]] # Support of starting position
    render_mode: str = "rgb_array"
    save_interval: int = 4 # The frequency of saving model per number of epochs
    verbose: bool = False

    # # Wandb params
    # optim_critic_iters: int = wandb.config.optim_critic_iters
    # last_layer_scale =b.config.last_layer_scale
    # max_backtracks: int = wandb.config.max_backtracks
    # gae_lambda: float = wandb.config.gae_lambda
    # target_kl: float = wandb.config.target_kl
    # l2_reg: float = wandb.config.l2.reg
    # gamma: float = wandb.config.gamma
    actor_lr: float = wandb.config.actor_lr
    critic_lr: float = wandb.config.critic_lr
    normalize_obs: bool = wandb.config.normalize_obs

with open(MyCfg.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
ENV_CONFIG.update({
    "observation": {        
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": MyCfg.normalize_obs
    },
    "start_angle": -np.math.pi/2, # This is radians
    # Costs
    "constraint_type": args.constraint_type,
    # Cost-speed
    "speed_limit": args.speed_limit
})

@pyrallis.wrap()
def train(args: MyCfg):
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
        args.group = args.task + "-cost-" + str(args.cost_limit)
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)
    
    # This env is used strictly to evaluate the observation and action shapes for CPO
    try:
        demo_env = load_environment(ENV_CONFIG)
    except:
        demo_env = gym.make(args.task, render_mode=args.render_mode)
    
    agent = CVPOAgent(
        env=demo_env,
        logger=logger,
        cost_limit=args.cost_limit,
        constraint_type=args.constraint_type,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        # CVPO arguments,
        estep_iter_num=args.estep_iter_num,
        estep_kl=args.estep_kl,
        estep_dual_max=args.estep_dual_max,
        estep_dual_lr=args.estep_dual_lr,
        sample_act_num=args.sample_act_num,
        mstep_iter_num=args.mstep_iter_num,
        mstep_kl_mu=args.mstep_kl_mu,
        mstep_kl_std=args.mstep_kl_std,
        mstep_dual_max=args.mstep_dual_max,
        mstep_dual_lr=args.mstep_dual_lr,
        # other algorithm params,
        actor_lr = args.actor_lr,
        critic_lr = args.critic_lr,
        gamma = args.gamma,
        n_step = args.n_step,
        tau = args.tau,
        hidden_sizes=args.hidden_sizes,
        double_critic=args.double_critic,
        conditioned_sigma=args.conditioned_sigma,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    try:
        # Start your vehicle at a random starting position 
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
    except:
        train_envs = worker([lambda: gym.make(args.task, render_mode=args.render_mode) for _ in range(training_num)])
        test_envs = worker([lambda: gym.make(args.task, render_mode=args.render_mode) for _ in range(args.testing_num)])

    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        # repeat_per_collect=args.repeat_per_collect,
        buffer_size=args.buffer_size,
        testing_num=args.testing_num,
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,
        save_interval=args.save_interval,
        resume=args.resume,
        save_ckpt=args.save_ckpt,
        verbose=args.verbose,
    )

    if __name__ == "__main__":
        # Let's watch its performance!
        env = load_environment(ENV_CONFIG)
    
        agent.policy.eval()
        collector = FastCollector(agent.policy, env, constraint_type=args.constraint_type)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["avg_total_cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        agent.policy.train()
        collector = FastCollector(agent.policy, env, constraint_type=args.constraint_type)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["avg_total_cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()