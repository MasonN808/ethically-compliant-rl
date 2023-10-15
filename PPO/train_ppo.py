import ast
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils import load_environment
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

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Outputs all the values from the logger as a dictionary
        logs = self.logger.name_to_value.copy()
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
env = FlattenObservation(load_environment(ENV_CONFIG))

# Stable baselines usually works with vectorized environments, 
# so even though CartPole is a single environment, we wrap it in a DummyVecEnv
env = DummyVecEnv([lambda: env])

# Create WandbLoggingCallback
callback = WandbLoggingCallback()

# Initialize the PPO agent with an MLP policy
agent = PPO(MlpPolicy, env, verbose=1, seed=seed)

# Train the agent with the callback
# agent.learn(total_timesteps=100000, callback=callback, progress_bar=True)
time_steps = 150000
epochs = 200
for i in range(epochs):
  agent.learn(total_timesteps=time_steps, callback=callback, reset_num_timesteps=False)
  agent.save(f"PPO/models/model_epoch({i})_timesteps({time_steps})")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = agent.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f"Final reward: {rewards.mean()}")
    env.render()
env.close()
