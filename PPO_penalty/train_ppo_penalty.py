#!/usr/bin/env python3
import ast
import argparse
import numpy as np
import torch
import os
import wandb
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
import sys
sys.path.append("stable_baselines3")
from stable_baselines3 import PPO_Penalty
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils import load_environment
from gymnasium.wrappers import FlattenObservation

# seed = 10

# Set the numpy seed
# np.random.seed(seed)

# Set the pytorch seed
# Set the seed for CPU
# torch.manual_seed(seed)

# If you're using CUDA:
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

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
    
def float_or_string(value):
    try:
        return float(value)
    except ValueError:
        return value

# Create an argument parser
parser = argparse.ArgumentParser(description='PPO_Penalty')

# Add an argument for the beta value
parser.add_argument('--beta', type=float_or_string, default=1, help='Value of KL penalty coefficient')

# Parse the command line arguments
args = parser.parse_args()

# Initialize wandb
wandb.init(name=f"ppo-KLpenalty-beta({args.beta})-parking", project="PPO-Penalty-Report5", sync_tensorboard=True)

with open('configs/ParkingEnv/env-default.txt') as f:
    data = f.read()

# Reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Overriding certain keys in the environment config
ENV_CONFIG.update({
    "start_angle": -np.math.pi/2, # This is radians
})

# Load the Highway env from the config file
env = FlattenObservation(load_environment(ENV_CONFIG))

# Stable baselines usually works with vectorized environments, 
# so even though CartPole is a single environment, we wrap it in a DummyVecEnv
env = DummyVecEnv([lambda: env])

# Create WandbLoggingCallback
callback = WandbLoggingCallback()

dynamic_beta = False
beta = args.beta
if args.beta == "dynamic":
   dynamic_beta = True
   beta = 5.0
# Initialize the PPO agent with an MLP policy
agent = PPO_Penalty(MlpPolicy, env, learning_rate=.001, beta=beta, dynamic_beta=dynamic_beta, target_kl=.01, verbose=1)

# Train the agent with the callback
time_steps = 100000
epochs = 40
for i in range(epochs):
  agent.learn(total_timesteps=time_steps, callback=callback, reset_num_timesteps=False)
  if i % 5 == 0:
    agent.save(f"PPO_penalty/models/{args.beta}/model_epoch({i})_timesteps({time_steps})")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = agent.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f"Final reward: {rewards.mean()}")
    env.render()
env.close()
