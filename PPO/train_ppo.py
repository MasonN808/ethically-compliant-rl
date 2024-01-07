#!/usr/bin/env python3
import ast
import numpy as np
import torch
import os
import wandb
import sys
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'True'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
sys.path.append("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils import load_environment
from gymnasium.wrappers import FlattenObservation
import pyrallis
from ppo_cfg import TrainCfg
from dataclasses import dataclass
from gymnasium.wrappers import RecordEpisodeStatistics

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the rewards
        reward = self.locals['rewards']
        is_success = int(self.locals['infos'][0].get('is_success') == True)
        self.logger.record('reward', reward)
        self.logger.record('is_success', is_success)
        # Outputs all the values from the logger as a dictionary
        logs = self.logger.name_to_value.copy()
        wandb.log(logs)
        # Continue training
        return True

@dataclass
class Cfg(TrainCfg):
    wandb_project_name: str = "PPO"
    env_name: str = "HighwayEnv" # Following are permissible: HighwayEnv, ParkingEnv
    env_config: str = f"configs/{env_name}/default.txt"
    epochs: int = 150
    total_timesteps: int = 100000
    batch_size: int = 256
    num_envs: int = 1

@pyrallis.wrap()
def train(args: Cfg):
    # Initialize wandb
    run = wandb.init(project=args.wandb_project_name, sync_tensorboard=True)
    run.name = run.id + str(args.env_name)

    with open(args.env_config) as f:
        data = f.read()
    # Reconstructing the data as a dictionary
    env_config = ast.literal_eval(data)
    # Overriding certain keys in the environment config
    # env_config.update({
    #     "start_angle": -np.math.pi/2, # This is radians
    # })
    env_config.update({
        "simulation_frequency": 1,
        "lanes_count": 4,
        "vehicles_count": 40,
    })

    def make_env(env_config):
        def _init():
            # Load the Highway env from the config file
            env = FlattenObservation(load_environment(env_config))
            # Add Wrapper to record stats in env
            env = RecordEpisodeStatistics(env)
            return env
        return _init
    
    envs = [make_env(env_config) for _ in range(args.num_envs)]
    env = DummyVecEnv(envs) 

    # Initialize the PPO agent with an MLP policy
    agent = PPO("MlpPolicy", # TODO: Double check that this needs to be a string
                 env,
                 batch_size=args.batch_size,
                 verbose=1)
    
    # Create WandbLoggingCallback
    callback = WandbLoggingCallback(env)
    
    # Train the agent with the callback
    for i in range(args.epochs):
        agent.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)
        if i % 5 == 0:
            path = f"PPO/models/{args.wandb_project_name}/{run.id}/model_epoch({i})"
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
