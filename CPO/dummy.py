import time
import gym

# Testing rendering
if __name__=='__main__':
    env = gym.make('CartPole-v0', render_mode="human")
    env.reset()

    for i in range(1000):
        env.step(env.action_space.sample())
        env.render()
        time.sleep(0.01)