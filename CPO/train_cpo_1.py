#!/usr/bin/env python3
import os
# Set this before everything
os. environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import asdict, dataclass
import pickle
import ast
import sys
import warnings # FIXME: Fix this warning eventually
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import gymnasium as gym
import bullet_safety_gym
import highway_env
try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")

import torch
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv, RayVectorEnv
import numpy as np

# To render the environemnt and agent
import matplotlib.pyplot as plt
sys.path.append("FSRL")
from fsrl.agent import CPOAgent
from fsrl.config.cpo_cfg import (
    TrainCfg,
)
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name
from utils.utils import load_environment

TASK_TO_CFG = {
    # HighwayEnv tasks
    "parking-v0": TrainCfg,
    "roundabout-v0": TrainCfg,
}

# Make my own config params
@dataclass
class MyCfg(TrainCfg):
    task: str = "parking-v0"
    cost_limit: float = 30 # The distance when surpassing the threshold 
    epoch: int = 300
    lr: float = 1e-3
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
    batch_size: int = 50000 # As seen in CPO paper
    l2_reg: float = 0.01
    max_backtracks: int = 200
    gae_lambda: float = 0.92
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoal.txt'

with open(MyCfg.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Update the steering_range since np can't be paresed in .txt file
ENV_CONFIG.update({"steering_range": np.deg2rad(50)}) # it is typical to be between 30-50 irl

@pyrallis.wrap()
def train(args: MyCfg):
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

    demo_env = load_environment(ENV_CONFIG)

    agent = CPOAgent(
        env=demo_env,
        logger=logger,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        damping_coeff=args.damping_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        cost_limit=args.cost_limit,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(training_num)])
    test_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(args.testing_num)])

    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
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
        from fsrl.data import FastCollector
        env = load_environment(ENV_CONFIG, render_mode='human')
        agent.policy.eval()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        agent.policy.train()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()