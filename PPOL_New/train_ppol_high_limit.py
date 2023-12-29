#!/usr/bin/env python3
import argparse
import ast
import os
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'True'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
import numpy as np
import wandb
import sys
sys.path.append("stable_baselines3")
from stable_baselines3 import PPOL
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils import load_environment
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import TrainCfg
from dataclasses import dataclass, field
import pyrallis


class Cfg(TrainCfg):
    speed_limit: float = 100
    wandb_project_name: str = "New-PPOL-SpeedLimit=" + str(speed_limit)
    env_config: str = "configs/ParkingEnv/env-default.txt"
    epochs: int = 50
    total_timesteps: int = 100000
    batch_size: int = 512

    # Lagrangian Parameters
    constraint_type: list[str] = field(default_factory=lambda: ["speed"])
    cost_threshold: list[float] = field(default_factory=lambda: [8])
    lagrange_multiplier: bool = True
    K_P: float = 1
    K_I: float = 1
    K_D: float = 2

@pyrallis.wrap()
def train(args: Cfg):
    run = wandb.init(name="ppol-highway-parking", project=args.wandb_project_name, sync_tensorboard=True)

    with open(args.env_config) as f:
        config = f.read()
    # Reconstructing the data as a dictionary
    env_config = ast.literal_eval(config)
    # Overriding certain keys in the environment config
    env_config.update({
        "start_angle": -np.math.pi/2, # This is radians
        "constraint_type": args.constraint_type,
        "speed_limit": args.speed_limit
    })
    # Load the Highway env from the config file
    env = FlattenObservation(load_environment(env_config))

    # Vectorize env for stablebaselines
    env = DummyVecEnv([lambda: env])

    # Initialize the PPO agent with an MLP policy
    agent = PPOL("MlpPolicy",
                 env,
                 n_costs=len(args.constraint_type), 
                 cost_threshold=args.cost_threshold, 
                 lagrange_multiplier=args.lagrange_multiplier,
                 K_P=args.K_P,
                 K_I=args.K_I,
                 K_D=args.K_D,
                 batch_size=args.batch_size,
                 verbose=1)

    # Train the agent with the callback
    for i in range(args.epochs):
        agent.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False)
        if i % 5 == 0:
            path = f"PPOL_New/models/{args.wandb_project_name}/{run.id}/model_epoch({i})"
            # Check if the directory already exists
            if not os.path.exists(path):
                # If it doesn't exist, create it
                os.makedirs(path)
                print(f"Directory created: {path}")
            else:
                print(f"Directory already exists: {path}")

            agent.save(path)

    # Test the trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f"Final reward: {rewards.mean()}")
        env.render()
    env.close()

if __name__ == "__main__":
    train()
