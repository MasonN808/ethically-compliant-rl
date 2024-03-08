#!/usr/bin/env python3
import ast
import os
from typing import Optional
# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
import numpy as np
import wandb
import torch as th
import torch.nn as nn

from stable_baselines3 import PPOL
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorManyCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from utils import load_environment, verify_and_solve_path
from gymnasium.wrappers import FlattenObservation
from ppol_cfg import TrainCfg
from dataclasses import dataclass, field, asdict
import pyrallis
from gymnasium.wrappers import RecordEpisodeStatistics
import gymnasium as gym


class CustomCNN(nn.Module):
    def __init__(self, observation_space: gym.Space, n_additional_features, out_features_dim):
        super(CustomCNN, self).__init__()
        # Define the CNN part
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # Define the part that processes additional features
        self.additional_feature_layers = nn.Linear(n_additional_features, 64)

        self.linear = nn.Sequential(nn.Linear(n_flatten, out_features_dim), nn.ReLU())

        # Define the final fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 + , 512),
            nn.ReLU(),
            # More layers as needed
            nn.Linear(512, n_actions) # Assuming a discrete action space
        )

    def forward(self, image_obs, additional_features):
        # Process the image observation through CNN
        cnn_out = self.cnn_layers(image_obs)
        cnn_out = cnn_out.view(cnn_out.size(0), -1) # Flatten the output
        
        # Process additional features
        additional_features_out = self.additional_feature_layers(additional_features)
        
        # Concatenate the outputs
        combined_out = th.cat((cnn_out, additional_features_out), dim=1)
        
        # Pass the combined output through the final layers
        return self.fc_layers(combined_out)

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.extractor = CustomCNN(n_additional_features=18)

    def forward(self, observations):
        image_obs = observations["image"]
        additional_features = observations["additional_features"]
        return self.extractor(image_obs, additional_features)

class CustomPolicy(ActorManyCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           features_extractor_class=CustomFeatureExtractor,
                                           features_extractor_kwargs={"features_dim": 512},
                                           net_arch=[],
                                           )

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
    wandb_project_name: str = "ppol-CNN-lines"
    env_name: str = "ParkingEnv" # Following are permissible: HighwayEnv, ParkingEnv
    env_config: str = f"configs/{env_name}/default.txt"
    epochs: int = 40
    total_timesteps: int = 100000
    batch_size: int = 512
    num_envs: int = 1
    model_save_interval: int = 2
    seed: int = 1
    ent_coef: float = .001
    env_logger_path: str = None
    run_dscrip: str = f"Lines-Seed={seed}"
    start_location: list = field(default_factory=lambda: [0, 0])
    extra_lines: bool = True # Adds additional horizonatal lines in the parking environment 
    use_closest_line_distance_in_obs: bool = False # Adds the minimum distance from the car to all lines in the observation

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
    # Instance of config
    # Convert the dataclass instance to a dictionary
    params_dict = asdict(args)
    # Log the parameters to wandb
    wandb.config.update(params_dict)

    with open(args.env_config) as f:
        config = f.read()
    # Reconstructing the data as a dictionary
    env_config = ast.literal_eval(config)
    # Overriding certain keys in the environment config
    env_config.update({
        "start_angle": -np.math.pi/2, # This is radians
        "constraint_type": args.constraint_type,
        "speed_limit": args.speed_limit,
        "start_location": args.start_location,
        "extra_lines": args.extra_lines,
        "use_closest_line_distance_in_obs": args.use_closest_line_distance_in_obs,
    })

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

    policy_kwargs = dict(net_arch=[128, 128, 128, 128, 128, 128, 128])

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
                 policy_kwargs=policy_kwargs,
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
