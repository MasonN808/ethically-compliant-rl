import gymnasium as gym

env = gym.make("parking-v0", render_mode="human")
env.configure({
    "id": "parking-v0",
    "import_module": "highway_env",
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
        "additional_features": [], # To add line coordinates
    },
    "action": {
        "type": "ContinuousAction"
    },
    # This determines the weights to the difference between the desired_goal and achieved_goal
    "reward_weights": [1.2, 0.3, 0.06, 0.06, 0.02, 0],
    "show_trajectories": False,
    "success_goal_reward": 0.12, # TODO: Change this to positive when using constant negative reward
    "collision_reward": -5,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 20, # seconds
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": False,
    "start_location": [40, 30],
    "manual_control": True,
    "extra_lines": True,

    # Costs
    "constraint_type":["lines"],
    # "speed_limit": 3,
})

env.reset()
done = False
i=0
while not done:
    i+=1
    print(i)
    obs, rew, done, _, info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    achieved_goal = obs['achieved_goal'][5]
    desired_goal = obs['desired_goal'][5]
    # print(f'achieved_goal: {achieved_goal}')
    # print(f'desired_goal: {desired_goal}')
    # print(info)
    cost = info['cost']
    print(f'cost: {cost}')
    # print(f'reward: {rew}')
    # print(rew)
    # print(len(obs))

    env.render()