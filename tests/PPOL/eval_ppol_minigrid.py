import ast
import numpy as np
import pyrallis
from utils import load_environment, evaluate_policy_and_capture_frames, save_frames_as_gif, verify_and_solve_path, verify_path

from stable_baselines3 import PPOL
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv
from ppol_cfg import EvalCfg
from dataclasses import dataclass, field
import gymnasium as gym


@dataclass
class Cfg(EvalCfg):
    n_eval_episodes: int = 5
    seed: int = 7 # Use seed 7 for all evaluations
    model_directory: str = "tests/PPOL/models/mini-grid/qfgkmstx"

    model_epoch: int = 10
    model_save_interval: int = 2
    loop_over_epochs: bool = False

    # PID Lagrangian Params
    # constraint_type: list[str] = field(default_factory=lambda: ["lines"])
    # cost_threshold: list[float] = field(default_factory=lambda: [4])
    constraint_type: list[str] = field(default_factory=lambda: [])
    cost_threshold: list[float] = field(default_factory=lambda: [])
    K_P: float = 2
    K_I: float = 1
    K_D: float = 1
    lagrange_multiplier: bool = False

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

        gif_dir = f"tests/PPOL/gifs/{parsed_gif_dir}"
        gif_path = f"tests/PPOL/gifs/{parsed_gif_file}"
        # Create the path if it does not exist
        verify_and_solve_path(gif_dir)

        env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
        env = DummyVecEnv([lambda: env])

        # Load the saved data
        data, params, _ = load_from_zip_file(model_zip_file)

        # Load the trained agent
        agent = PPOL(
                    policy=data["policy_class"], 
                    policy_kwargs=data["policy_kwargs"], 
                    env=env, 
                    device='cpu',
                    n_costs=len(args.constraint_type),
                    cost_threshold=args.cost_threshold,
                    K_P=args.K_P,
                    K_I=args.K_I,
                    K_D=args.K_D,
                    lagrange_multiplier=args.lagrange_multiplier,
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
