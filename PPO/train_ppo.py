import ast
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.policies import MlpPolicy
from PPO.utils import load_environment

with open('configs/ParkingEnv/env-default.txt') as f:
    data = f.read()
# Reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)
# Overriding certain keys in the environment config
ENV_CONFIG.update({
    "start_angle": -np.math.pi/2, # This is radians
})

# load the Highway env from the config file
demo_env = load_environment(ENV_CONFIG)

# Create the environment
env = gym.make('CartPole-v1')

# Stable baselines usually works with vectorized environments, 
# so even though CartPole is a single environment, we wrap it in a DummyVecEnv
env = DummyVecEnv([lambda: env])

# Initialize the PPO agent with an MLP policy
agent = PPO(MlpPolicy, env, verbose=1)

# Train the agent
agent.learn(total_timesteps=100000)

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = agent.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
