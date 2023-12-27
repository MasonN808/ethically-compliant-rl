#!/usr/bin/env python3
import argparse
import ast
import os
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
import numpy as np
import torch
import wandb
import sys
sys.path.append("stable_baselines3")
from stable_baselines3 import PPOL
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils import load_environment, check_build_path
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import TrainCfg
from dataclasses import dataclass, field
import pyrallis


class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the rewards
        reward = self.locals['rewards']
        self.logger.record('reward', reward)
        # Outputs all the values from the logger as a dictionary
        logs = self.logger.name_to_value.copy()
        wandb.log(logs)
        # Continue training
        return True


parser = argparse.ArgumentParser(description='PPO_Lagrange')
parser.add_argument('--speed_limit', type=int, default=2, help='Speed limit')
args = parser.parse_args()

@dataclass
class Cfg(TrainCfg):
    speed_limit: float = args.speed_limit
    wandb_project_name: str = "New-PPOL-SpeedLimit=" + str(speed_limit)
    env_config: str = "configs/ParkingEnv/env-default.txt"
    epochs: int = 400
    total_timesteps: int = 100000

    # Lagrangian Parameters
    constraint_type: list[str] = field(default_factory=lambda: ["speed"])
    cost_threshold: list[float] = field(default_factory=lambda: [2])
    K_P: float = 0.05
    K_I: float = 0.0005
    K_D: float = 0.1

@pyrallis.wrap()
def train(args: Cfg):
    wandb.init(name="ppol-highway-parking", project=args.wandb_project_name, sync_tensorboard=True)

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

    # Create WandbLoggingCallback
    callback = WandbLoggingCallback()

    # Initialize the PPO agent with an MLP policy
    agent = PPOL("MlpPolicy",
                 env,
                 n_costs=len(args.constraint_type), 
                 cost_threshold=args.cost_threshold, 
                 K_P=args.K_P,
                 K_I=args.K_I,
                 K_D=args.K_D,
                 verbose=1)

    # Train the agent with the callback
    for i in range(args.epochs):
        agent.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)
        path = f"PPOL_New/models/{args.wandb_project_name}/model_epoch({i})_timesteps({args.total_timesteps})"
        # Check if path exists and build if if it does not
        check_build_path(path)
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
