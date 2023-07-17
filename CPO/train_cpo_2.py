import bullet_safety_gym
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger


if __name__=='__main__':
    task = "SafetyCarCircle-v0"
    # init logger
    logger = TensorboardLogger("logs", log_txt=True, name=task)
    # init the PPO Lag agent with default parameters
    agent = PPOLagAgent(gym.make(task), logger)
    # init the envs
    training_num, testing_num = 10, 1
    train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(testing_num)])

    agent.learn(train_envs, test_envs, epoch=100)