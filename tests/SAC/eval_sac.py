import ast
import os
import sys
import gym
import numpy as np
import torch
import pyrallis
from utils import load_environment, evaluate_policy_and_capture_frames, save_frames_as_gif
sys.path.append("stable_baselines3")
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium.wrappers import FlattenObservation
from sac_cfg import TrainCfg

@pyrallis.wrap()
def evaluate(args: TrainCfg):
    for i in range(0, 1000, 5):
        # Path to your saved model
        model_path = f"SAC/models/SAC/o2x30h2d/model_epoch({i}).zip"
        # Parsing path for gif path
        parsed_gif_file = model_path.split("/models/")[-1][:-4]
        
        # Parse the directory
        # Splitting the string by '/'
        parts = model_path.split('/')
        parsed_gif_dir = parts[2] + '/' + parts[3]


        gif_dir = f"SAC/gifs/{parsed_gif_dir}"
        gif_path = f"SAC/gifs/{parsed_gif_file}.gif"
        # Create the path if it does not exist
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)


        with open('configs/ParkingEnv/default.txt') as f:
            data = f.read()

        # Reconstructing the data as a dictionary
        ENV_CONFIG = ast.literal_eval(data)
        # Overriding certain keys in the environment config
        ENV_CONFIG.update({
            "start_angle": -np.math.pi/2, # This is radians
            "duration": 60,
        })

        # Load the Highway env from the config file
        env = FlattenObservation(load_environment(ENV_CONFIG, render_mode="rgb_array"))

        # Stable baselines usually works with vectorized environments, 
        # so even though CartPole is a single environment, we wrap it in a DummyVecEnv
        env = DummyVecEnv([lambda: env])

        # Load the saved data
        data, params, _ = load_from_zip_file(model_path)

        # Load the trained agent
        agent = SAC(
                    policy=data["policy_class"], 
                    env=env, 
                    device='auto',
                )
        
        # Load the model state
        agent.set_parameters(params)

        # A modified version of evaluate_policy() from stable_baslelines3
        mean_reward, std_reward, frames= evaluate_policy_and_capture_frames(agent, env, n_eval_episodes=1)

        # Create the gif from the frames
        save_frames_as_gif(frames, path=gif_path)

        print(mean_reward, std_reward)

if __name__=="__main__":
    evaluate()
