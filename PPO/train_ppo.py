#!/usr/bin/env python3
import ast
import numpy as np
import os
import wandb
import sys
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
sys.path.append("stable_baselines3") # Since outside of current directory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from utils import load_environment
from gymnasium.wrappers import FlattenObservation
from ppo_cfg import TrainCfg
from dataclasses import dataclass
import pyrallis
from gymnasium.wrappers import RecordEpisodeStatistics

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the rewards
        if isinstance(self.locals['rewards'], np.ndarray):
            reward = np.mean(self.locals['rewards'])
        else:
            reward = self.locals['rewards']
        is_success = int(self.locals['infos'][0].get('is_success') == True)
        avg_speed = self.locals['infos'][0].get('avg_speed')
        max_speed = self.locals['infos'][0].get('max_speed')
        self.logger.record('reward', reward)
        self.logger.record('is_success', is_success)
        self.logger.record('avg_speed', avg_speed)
        self.logger.record('max_speed', max_speed)
        # Outputs all the values from the logger as a dictionary
        logs = self.logger.name_to_value.copy()
        wandb.log(logs)
        # Continue training
        return True

@dataclass
class Cfg(TrainCfg):
    # wandb_project_name: str = "PPO+PPOL"
    wandb_project_name: str = "QUALITATIVE-TEST"
    run_dscrip: str = "PPO"
    env_name: str = "ParkingEnv" # Following are permissible: HighwayEnv, ParkingEnv
    env_config: str = f"configs/{env_name}/default.txt"
    epochs: int = 300
    total_timesteps: int = 10000
    batch_size: int = 512
    num_envs: int = 1
    model_save_interval: int = 5
    seed: int = None
    # env_logger_path: str = "PPO/env_logger.txt"
    env_logger_path: str = None

@pyrallis.wrap()
def train(args: Cfg):
    # set_random_seed(args.seed)
    # Initialize wandb
    run = wandb.init(project=args.wandb_project_name, sync_tensorboard=True)
    run.name = run.id + "-" + str(args.env_name) + "-" + args.run_dscrip

    with open(args.env_config) as f:
        data = f.read()
    # Reconstructing the data as a dictionary
    env_config = ast.literal_eval(data)
    # Overriding certain keys in the environment config
    env_config.update({
        "start_angle": -np.math.pi/2, # This is radians
    })
    # env_config.update({
    #     "simulation_frequency": 1,
    #     "lanes_count": 4,
    #     "vehicles_count": 40,
    # })

    def make_env(env_config):
        def _init():
            # Load the Highway env from the config file
            env = FlattenObservation(load_environment(env_config, env_logger_path=args.env_logger_path))
            # Add Wrapper to record stats in env
            env = RecordEpisodeStatistics(env)
            return env
        return _init
    
    envs = [make_env(env_config) for _ in range(args.num_envs)]
    env = DummyVecEnv(envs) 

    # Initialize the PPO agent with an MLP policy
    agent = PPO("MlpPolicy",
                 env,
                 batch_size=args.batch_size,
                 verbose=1,
                 seed=args.seed)
    
    # Create WandbLoggingCallback
    callback = WandbLoggingCallback(env)
    
    # Train the agent with the callback
    for i in range(args.epochs):
        agent.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)
        if i % args.model_save_interval == 0:
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
