#!/usr/bin/env python3
import ast
import os
from typing import Optional, Dict, List
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'True'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
import numpy as np
import wandb
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorManyCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, LiDARCNN

from stable_baselines3 import PPOL
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from utils import load_environment, verify_and_solve_path
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import TrainCfg
from dataclasses import dataclass, field, asdict
import pyrallis
from gymnasium.wrappers import RecordEpisodeStatistics
np.seterr(divide='ignore', invalid='ignore') # Useful for lidar observation


# class CustomCNN(nn.Module):
#     def __init__(self, observation_space: gym.spaces.Dict, n_additional_features, features_dim=512):
#         super(CustomCNN, self).__init__()
#         n_input_channels = observation_space['image'].shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass of a sample image
#         with th.no_grad():
#             n_flatten = self.cnn(th.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]

#         self.additional_features_layer = nn.Linear(n_additional_features, 64)
#         self.features_dim = features_dim
#         self.post_cnn = nn.Sequential(
#             nn.Linear(n_flatten + 64, self.features_dim),
#             nn.ReLU(),
#         )

#     def forward(self, observations):
#         image_obs = observations['image']
#         additional_features = observations['additional_features']
#         cnn_out = self.cnn(image_obs)
#         additional_out = self.additional_features_layer(additional_features)
#         combined_out = th.cat((cnn_out, additional_out), dim=1)
#         return self.post_cnn(combined_out)

# class CustomFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512):
#         super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)
#         self.extractor = CustomCNN(observation_space=observation_space.spaces, n_additional_features=18, features_dim=features_dim)

#     def forward(self, observations):
#         return self.extractor(observations)

# class CustomPolicy(ActorManyCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         kwargs['features_extractor_class'] = CustomFeatureExtractor
#         kwargs['features_extractor_kwargs'] = {'features_dim': 512}
#         super(CustomPolicy, self).__init__(*args, **kwargs)


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
            cost = self.locals['infos'][0].get('cost')[0]
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
    speed_limit: Optional[float] = None
    wandb_project_name: str = "ppol-LIDAR-CNN"
    env_name: str = "ParkingEnv" # Following are permissible: HighwayEnv, ParkingEnv
    env_config: str = f"configs/{env_name}/default.txt"
    epochs: int = 40
    total_timesteps: int = 100000
    batch_size: int = 512
    num_envs: int = 1
    model_save_interval: int = 2
    policy_kwargs: Dict[str, List[int]] = field(default_factory=lambda: {'net_arch': [140, 140], 'features_extractor_class': LiDARCNN})
    seed: int = 1
    ent_coef: float = .001
    env_logger_path: str = None
    run_dscrip: str = f"Lines-Seed={seed}"
    start_location: list = field(default_factory=lambda: [0, 0])
    extra_lines: bool = True # Adds additional horizonatal lines in the parking environment

    # Lagrangian Parameters
    constraint_type: list[str] = field(default_factory=lambda: ["lines"])
    cost_threshold: list[float] = field(default_factory=lambda: [3])
    lagrange_multiplier: bool = True
    K_P: float = 1
    K_I: float = 1
    K_D: float = 2

@pyrallis.wrap()
def train(args: Cfg):
    run = wandb.init(project=args.wandb_project_name, sync_tensorboard=True)
    run.name = run.id + "-" + str(args.env_name) + "-" + args.run_dscrip
    
    # Log all the config params to wandb
    params_dict = asdict(args)
    wandb.config.update(params_dict)

    with open(args.env_config) as f:
        config = f.read()
    # Reconstructing the data as a dictionary
    env_config = ast.literal_eval(config)
    # Overriding certain keys in the environment config
    env_config.update({
        "observation": {
            "type": "KinematicsLidarObservation",
            "cells": 100,
            "maximum_range": 60,
            "normalize": True,
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
        },
        "start_angle": -np.math.pi/2, # This is radians
        "constraint_type": args.constraint_type,
        "speed_limit": args.speed_limit,
        "start_location": args.start_location,
        "extra_lines": args.extra_lines,
    })

    # def make_env(env_config):
    #     def _init():
    #         # Load the Highway env from the config file
    #         env = FlattenObservation(load_environment(env_config, env_logger_path=args.env_logger_path))
    #         # Add Wrapper to record stats in env
    #         env = RecordEpisodeStatistics(env)
    #         return env
    #     return _init
    def make_env(env_config):
        def _init():
            # Add Wrapper to record stats in env
            env = RecordEpisodeStatistics(load_environment(env_config, env_logger_path=args.env_logger_path))
            return env
        return _init
    
    envs = [make_env(env_config) for _ in range(args.num_envs)]
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
                 seed=args.seed)

    # Create WandbLoggingCallback
    callback = WandbLoggingCallback(env)
    
    # Train the agent with the callback
    for i in range(args.epochs):
        agent.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)
        if i % args.model_save_interval == 0:
            path = f"tests/PPOL_New/models/{args.wandb_project_name}/{run.id}/model_epoch({i})"
            verify_and_solve_path(f"tests/PPOL_New/models/{args.wandb_project_name}/{run.id}")
            agent.save(path)

if __name__ == "__main__":
    train()
