#!/usr/bin/env python3
import copy
import os
import random
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
import re

@dataclass
class EvalConfig:
    # Need to get relative path of the experiment that you'd like to evaluate
    path: str = "logs/2-constraints-absolute/parking-v0-cost0-5-cost1-5/cpo_lr0.0005_step_per_epoch20000-8198"
    best: bool = True
    eval_episodes: int = 2
    parallel_eval: bool = False
    # This was originally a bool; must be changed to float
    render: float = .01
    convert_to_gif: bool = True
    train_mode: bool = False
    render_mode: str = "rgb_array"
    # render_mode: str = "human"
    device = "cpu"
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
    # Points are around the parking lot and in the middle
    random_starting_locations = [[0, 32]]

if EvalConfig.env_config_file:
    with open(EvalConfig.env_config_file) as f:
        data = f.read()
    # reconstructing the data as a dictionary
    ENV_CONFIG = ast.literal_eval(data)
    # Update the steering_range since np can't be paresed in .txt file
    ENV_CONFIG.update({"steering_range": np.deg2rad(50)}) # it is typical to be between 30-50 irl

    # Get the unique 4 char id of the file at the end of the file name
    match = re.search(r'-([\w]+)$', EvalConfig.path)
    EvalConfig.experiment_id = "----"
    if match:
        EvalConfig.experiment_id = match.group(1)
    else:
        print("Pattern not found")

    # Get the algorithm used
    match = re.search(r'/(\w+?)_', EvalConfig.path)
    EvalConfig.constraints = True
    if match:
        EvalConfig.algorithm = match.group(1)
        if EvalConfig.algorithm == "ppol":
            EvalConfig.constraints = False
    else:
        print("Pattern not found")

@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)

    try:
        demo_env = load_environment(ENV_CONFIG)
    except:
        demo_env = gym.make(cfg["task"], render_mode=args.render_mode)
    
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
        constraint_type = cfg["constraint_type"]
    )

    rews, lens, cost = agent.evaluate(
        env_config = ENV_CONFIG,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        render_mode = args.render_mode,
        train_mode=args.train_mode,
        experiment_id=args.experiment_id,
        random_starting_locations = args.random_starting_locations,
        algorithm = args.algorithm,
        convert_to_gif = args.convert_to_gif
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()