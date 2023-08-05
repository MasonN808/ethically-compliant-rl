import gymnasium as gym
import highway_env
import numpy as np

env = gym.make("parking-v0", render_mode="human")
env.configure({
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1], 
        "normalize": False
    },
    "manual_control": True,
    "real_time_rendering": True,
    'cost_delta_distance': 2,
    'add_walls': False,
    "steering_range": np.deg2rad(50),
})
env.reset()
done = False
while not done:
    _, _, done, _, _ = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    env.render()