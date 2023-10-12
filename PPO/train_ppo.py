import ast
import gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils import load_environment

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log all the values from the logger
        logs = self.logger.get_log_dict()
        wandb.log(logs)
        # Continue training
        return True

# Initialize wandb
wandb.init(name="ppo-highway-parking", project="PPO", sync_tensorboard=True)

with open('configs/ParkingEnv/env-default.txt') as f:
    data = f.read()

# Reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Overriding certain keys in the environment config
ENV_CONFIG.update({
    "start_angle": -np.math.pi/2, # This is radians
})

# Load the Highway env from the config file
env = load_environment(ENV_CONFIG)

# Stable baselines usually works with vectorized environments, 
# so even though CartPole is a single environment, we wrap it in a DummyVecEnv
env = DummyVecEnv([lambda: env])

# Create WandbLoggingCallback
callback = WandbLoggingCallback()

# Initialize the PPO agent with an MLP policy
agent = PPO(MlpPolicy, env, verbose=1)

# Train the agent with the callback
agent.learn(total_timesteps=100000, callback=callback)
# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = agent.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
