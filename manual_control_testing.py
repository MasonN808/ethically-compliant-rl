import gymnasium as gym
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Useful for lidar observation

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="human")
env.reset()
done = False
i=0
while not done:
    i+=1
    # print(i)
    obs, rew, done, _, info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    # achieved_goal = obs['achieved_goal'][5]
    # desired_goal = obs['desired_goal'][5]
    # print(f'achieved_goal: {achieved_goal}')
    # print(f'desired_goal: {desired_goal}')
    # print(info)
    # cost = info['cost']
    # print(f'cost: {cost}')
    # print(f'reward: {rew}')
    # print(rew)
    print(obs)
    print(len(obs))

    env.render()