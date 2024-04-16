#!/usr/bin/env python3
import ast
import os
from typing import Optional, Dict, List
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
import numpy as np
import wandb


from stable_baselines3 import PPOL
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor


from utils import load_environment, verify_and_solve_path
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import TrainCfg
from dataclasses import dataclass, field, asdict
import pyrallis
from gymnasium.wrappers import RecordEpisodeStatistics
import gymnasium as gym
# import Minigrid
np.seterr(divide='ignore', invalid='ignore') # Useful for lidar observation


class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the rewards
        if isinstance(self.locals['rewards'], np.ndarray):
            reward = np.mean(self.locals['rewards'])
        else:
            reward = self.locals['rewards']

        if self.locals['infos'][0].get('cost'):
            cost = self.locals['infos'][0].get('cost')[0] # TODO: For more one constraint
            # cost = self.locals['infos'][0].get('cost')
            self.logger.record('cost', cost)

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
    wandb_project_name: str = "mini-grid"
    env_name: str = "MiniGrid-Empty-16x16-v1"
    epochs: int = 5
    total_timesteps: int = 100000
    batch_size: int = 2048
    num_envs: int = 1
    model_save_interval: int = 2
    policy_kwargs: Dict[str, List[int]] = field(default_factory=lambda: {'net_arch': [64, 64], 'features_extractor_class': CombinedExtractor})
    seed: int = 1
    ent_coef: float = .002
    env_logger_path: str = None
    run_dscrip: str = "Hazards"
    device: str = "cuda"

    # Lagrangian Parameters
    constraint_type: list[str] = field(default_factory=lambda: ["hazards"])
    cost_threshold: list[float] = field(default_factory=lambda: [0])
    # constraint_type: list[str] = field(default_factory=lambda: [])
    # cost_threshold: list[float] = field(default_factory=lambda: [])
    lagrange_multiplier: bool = True
    K_P: float = 1
    K_I: float = 1
    K_D: float = 2

    notes: str = "Changed cost_values-d to cost_values in Loss function since this may be causing issues with the cost prediction."

@pyrallis.wrap()
def train(args: Cfg):
    run = wandb.init(project=args.wandb_project_name, notes=args.notes, sync_tensorboard=True)
    run.name = run.id + "-" + str(args.env_name) + "-" + args.run_dscrip
    
    # Log all the config params to wandb
    params_dict = asdict(args)
    wandb.config.update(params_dict)

    def make_env(env_name):
        def _init():
            # Load the Highway env from the config file
            # env = FlattenObservation()
            env = gym.make(env_name)
            # Add Wrapper to record stats in env
            env = RecordEpisodeStatistics(env)
            return env
        return _init
    
    envs = [make_env(args.env_name) for _ in range(args.num_envs)]
    env = DummyVecEnv(envs) 

    # Initialize the PPO agent with an MLP policy
    agent = PPOL("MlpPolicy",
                 env,
                 n_costs=len(args.constraint_type),
                 cost_threshold=args.cost_threshold,
                 lagrange_multiplier=args.lagrange_multiplier,
                 K_P=args.K_P,
                 K_I=args.K_I,
                 K_D=args.K_D,
                 batch_size=args.batch_size,
                 verbose=0,
                 ent_coef=args.ent_coef,
                 policy_kwargs=args.policy_kwargs,
                 seed=args.seed,
                 device=args.device)

    # Create WandbLoggingCallback
    callback = WandbLoggingCallback(env)
    
    # Train the agent with the callback
    for i in range(args.epochs):
        agent.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)
        if i % args.model_save_interval == 0:
            path = f"tests/PPOL/models/{args.wandb_project_name}/{run.id}/model_epoch({i})"
            verify_and_solve_path(f"tests/PPOL/models/{args.wandb_project_name}/{run.id}")
            agent.save(path)

if __name__ == "__main__":
    train()
