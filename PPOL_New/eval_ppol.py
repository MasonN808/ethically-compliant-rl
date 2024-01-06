import ast
import os
import sys
import gym
import numpy as np
import pyrallis
import torch as th
from utils import load_environment, evaluate_policy_and_capture_frames, save_frames_as_gif
sys.path.append("stable_baselines3")
from stable_baselines3 import PPOL
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import TrainCfg



@pyrallis.wrap()
def evaluate(args: TrainCfg):
    for i in range(0, 1000, 10):
        # Path to your saved model
        model_path = f"PPOL_New/models/New-PPOL-NoMultipiler-SpeedLimit=2/1ugan1k1/model_epoch({i}).zip"
        # Parsing path for gif path
        parsed_gif_file = model_path.split("/models/")[-1][:-4]
        
        # Parse the directory
        # Splitting the string by '/'
        parts = model_path.split('/')
        parsed_gif_dir = parts[2] + '/' + parts[3]


        gif_dir = f"PPOL_New/gifs/{parsed_gif_dir}"
        gif_path = f"PPOL_New/gifs/{parsed_gif_file}.gif"
        # Create the path if it does not exist
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)


        with open('configs/ParkingEnv/env-default.txt') as f:
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
        agent = PPOL(
                    policy=data["policy_class"], 
                    env=env, 
                    device='auto',
                    n_costs=len(args.constraint_type),
                    cost_threshold=args.cost_threshold,
                    K_P=args.K_P,
                    K_I=args.K_I,
                    K_D=args.K_D,
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
