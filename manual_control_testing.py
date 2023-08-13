import gymnasium as gym
import highway_env
import numpy as np

env = gym.make("parking-v0", render_mode="human")
env.configure({
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False # TODO THIS DOES NOTHING
    },
    "action": {
        "type": "ContinuousAction"
    },
    # This determines the weights to the difference between the desired_goal and achieved_goal
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "show_trajectories": False,
    "success_goal_reward": 0.12,
    "collision_reward": -5,
    "simulation_frequency": 100,
    "policy_frequency": 5,
    "duration": 10,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0, 0],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": False,
    "start_location": [0, 0],
    "manual_control": True,

    # Costs
    "constrained_rl": True,
    # Cost-distance
    "cost_delta_distance": 4,
    "quantized_line_points": 20,
    # Cost-speed
    "cost_speed_limit": 2,
})
env.reset()
done = False
while not done:
    obs, rew, done, _, info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    # achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    # print(f'achieved_goal: {achieved_goal}')
    print(f'desired_goal: {desired_goal}')
    cost = info['cost']
    print(f'cost: {cost}')

    env.render()