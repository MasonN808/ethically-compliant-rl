import ast
import os
import sys
import gym
import numpy as np
import pyrallis
import torch as th
from utils import load_environment, evaluate_policy_and_capture_frames, save_frames_as_gif, verify_and_solve_path, verify_path
sys.path.append("stable_baselines3")
from stable_baselines3 import PPOL
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import EvalCfg
from dataclasses import dataclass, field

@dataclass
class Cfg(EvalCfg):
    n_eval_episodes: int = 6
    seed: int = 7 # Use seed 7 for all evaluations
    model_directory: str = "tests/PPOL_New/models/ent-coefficient-ppol/797ofrdf"

    model_epoch: int = 8
    model_save_interval: int = 5
    loop_over_epochs: bool = False

    # PID Lagrangian Params
    constraint_type: list[str] = field(default_factory=lambda: ["speed"])
    cost_threshold: list[float] = field(default_factory=lambda: [8])
    K_P: float = 1
    K_I: float = 1
    K_D: float = 2
 
@pyrallis.wrap()
def evaluate(args: Cfg):
    model_epoch = args.model_epoch
    #TODO: extract config from zip files to avoid re createing the params in the eval config
    model_zip_file = args.model_directory + f"/model_epoch({model_epoch}).zip"
    while verify_path(model_zip_file, is_directory=False):
        # Parsing path for gif path
        parsed_gif_file = model_zip_file.split("/models/")[-1][:-4]
        
        # Parse the directory
        # Splitting the string by '/'
        parts = model_zip_file.split('/')
        # parts[3] -> the project name
        # parts[4] -> run id
        parsed_gif_dir = parts[3] + '/' + parts[4]

        gif_dir = f"tests/PPOL_New/gifs/{parsed_gif_dir}"
        gif_path = f"tests/PPOL_New/gifs/{parsed_gif_file}"
        # Create the path if it does not exist
        verify_and_solve_path(gif_dir)

        # Load ParkingEnv configuration
        with open('configs/ParkingEnv/default.txt') as f:
            data = f.read()

        # Reconstructing the data as a dictionary
        ENV_CONFIG = ast.literal_eval(data)
        # Overriding certain keys in the environment config
        ENV_CONFIG.update({
            "start_angle": -np.math.pi/2, # This is radians
            "duration": 60,
            "simulation_frequency": 30,
            "policy_frequency": 30,
        })

        # Load the Highway env from the config file
        env = FlattenObservation(load_environment(ENV_CONFIG, render_mode="rgb_array"))

        # Stable baselines usually works with vectorized environments, 
        # so even though CartPole is a single environment, we wrap it in a DummyVecEnv
        env = DummyVecEnv([lambda: env])

        # Load the saved data
        data, params, _ = load_from_zip_file(model_zip_file)

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
                    seed=args.seed
                )
        
        # Load the model state
        agent.set_parameters(params)

        # A modified version of evaluate_policy() from stable_baslelines3
        mean_reward, std_reward, frames = evaluate_policy_and_capture_frames(agent, env, n_eval_episodes=args.n_eval_episodes)
        for i in range(args.n_eval_episodes):
            popped_frames = frames.pop()
            modified_gif_path = gif_path + f"_{i}.gif"
            # Create the gif from the frames
            # duration = the number of miliseconds shown per frame
            save_frames_as_gif(path=modified_gif_path, frames=popped_frames, duration=20)

        print(mean_reward, std_reward)

        if not args.loop_over_epochs:
            break
        
        model_epoch += args.model_save_interval
        model_zip_file = args.model_directory + f"/model_epoch({model_epoch}).zip"

if __name__=="__main__":
    evaluate()
