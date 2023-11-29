import gymnasium as gym
import highway_env
import numpy as np

env = gym.make("parking-v0", render_mode="human")
env.configure({
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": True
    },
    "action": {
        "type": "ContinuousAction"
    },
    # This determines the weights to the difference between the desired_goal and achieved_goal
    "reward_weights": [1, .3, .02, .02, 0.02, 0],
    "show_trajectories": False,
    "success_goal_reward": -0.12, # set to negative if using alternative reward function
    "collision_reward": -5,
    "simulation_frequency": 100,
    "policy_frequency": 30,
    "duration": 70,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": False,
    "start_location": [0, 32],
    "start_angle": -np.math.pi/2, # This is radians

    "manual_control": True,

    # Cost-speed
    "constraint_type": ["speed"],
    "speed_limit": 2,
})

env.reset()
done = False
while not done:
    obs, rew, done, _, info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    achieved_goal = obs['achieved_goal'][5]
    desired_goal = obs['desired_goal'][5]
    # print(f'achieved_goal: {achieved_goal}')
    # print(f'desired_goal: {desired_goal}')
    # print(info)
    # cost = info['cost']
    # print(f'cost: {cost}')
    print(f'reward: {rew}')
    # print(rew)
    # print(len(obs))

    env.render()