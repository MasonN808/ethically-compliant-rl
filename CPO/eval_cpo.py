#!/usr/bin/env python3
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import bullet_safety_gym
import warnings # FIXME: Fix this warning eventually
warnings.filterwarnings("ignore", category=DeprecationWarning) 
try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import DataParallelNet
sys.path.append("FSRL")
from fsrl.agent import CPOAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all
import ast
from utils.utils import load_environment
import numpy as np


@dataclass
class EvalConfig:
    # Need to get relative path of the experiment that you'd like to evaluate
    path: str = "logs/fast-safe-rl/parking-v0-cost-30/cpo_batch_size50000_cost30_l2_reg0.01_step_per_epoch20000-6860"
    best: bool = True
    eval_episodes: int = 3
    parallel_eval: bool = False
    # This was originally a bool; must be changed to float
    render: float = .01
    train_mode: bool = False
    render_mode: str = "rgb_array"
    device = "cpu"
    env_config_file: str = 'configs/ParkingEnv/env-image.txt'

with open(EvalConfig.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Update the steering_range since np can't be paresed in .txt file
ENV_CONFIG.update({"steering_range": np.deg2rad(50)}) # it is typical to be between 30-50 irl


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)

    task = cfg["task"]
    demo_env = load_environment(ENV_CONFIG, render_mode=args.render_mode)

    agent = CPOAgent(
        env=demo_env,
        logger=BaseLogger(),
        device=args.device,
        thread=cfg["thread"],
        seed=cfg["seed"],
        cost_limit=cfg["cost_limit"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    if args.parallel_eval:
        test_envs = ShmemVectorEnv(
            [lambda: load_environment(ENV_CONFIG, render_mode=args.render_mode) for _ in range(args.eval_episodes)]
        )
    else:
        test_envs = load_environment(ENV_CONFIG, render_mode=args.render_mode)

    rews, lens, cost = agent.evaluate(
        test_envs=test_envs,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        train_mode=args.train_mode
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()