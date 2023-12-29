import ast
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
    seed = 10

    # Set the numpy seed
    np.random.seed(seed)

    # Set the pyth seed
    # Set the seed for CPU
    th.manual_seed(seed)

    # If you're using CUDA:
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

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

    # Path to your saved model
    path_to_zip_file = "PPOL_New/models/New-PPOL-SpeedLimit=8/model_epoch(20)_timesteps(100000).zip"

    # Load the saved data
    data, params, _ = load_from_zip_file(path_to_zip_file)

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
    mean_reward, std_reward, frames= evaluate_policy_and_capture_frames(agent, env, n_eval_episodes=3)

    # Create the gif from the frames
    save_frames_as_gif(frames, path=f'PPOL_New/gifs/car_evaluation.gif')

    print(mean_reward, std_reward)

if __name__=="__main__":
    evaluate()
