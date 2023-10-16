import ast
import gym
import numpy as np
import torch
from utils import load_environment, evaluate_policy_and_capture_frames, save_frames_as_gif
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import FlattenObservation

seed = 10

# Set the numpy seed
np.random.seed(seed)

# Set the pytorch seed
# Set the seed for CPU
torch.manual_seed(seed)

# If you're using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

with open('configs/ParkingEnv/env-default.txt') as f:
    data = f.read()

# Reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Overriding certain keys in the environment config
ENV_CONFIG.update({
    "start_angle": -np.math.pi/2, # This is radians
})

# Load the Highway env from the config file
env = FlattenObservation(load_environment(ENV_CONFIG, render_mode="rgb_array"))

# Stable baselines usually works with vectorized environments, 
# so even though CartPole is a single environment, we wrap it in a DummyVecEnv
env = DummyVecEnv([lambda: env])

# Load the trained agent
agent = PPO.load("PPO/models/model_epoch(7)_timesteps(100000).zip", env=env)

# A modified version of evaluate_policy() from stable_baslelines3
mean_reward, std_reward, frames= evaluate_policy_and_capture_frames(agent, env, n_eval_episodes=7)

# Create the gif from the frames
save_frames_as_gif(frames, path=f'PPO/gifs/car_evaluation.gif')


print(mean_reward, std_reward)
