import json
import imageio
import gymnasium as gym
import logging
import numpy as np
logger = logging.getLogger(__name__)

def load_environment(env_config, render_mode=None):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :return: the environment
    """
    # Load the environment config from file
    if not isinstance(env_config, dict):
        with open(env_config) as f:
            env_config = json.loads(f.read())

    # Make the environment
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])
    try:
        env = gym.make(env_config['id'], render_mode=render_mode)
        # Save env module in order to be able to import it again
        env.import_module = env_config.get("import_module", None)
    except KeyError:
        raise ValueError("The gym register-id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        print("import_module", env_config["import_module"])
        raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
                                        'the "import_module" key of the environment configuration'.format(
                                            env_config['id']))

    # Configure the environment, if supported
    try:
        env.unwrapped.configure(env_config)
        # Reset the environment to ensure configuration is applied
        env.reset()
    except AttributeError as e:
        logger.info("This environment does not support configuration. {}".format(e))
    return env


# Replaces evaluate_policy() from stable_baselines3 with a version that outputs frames to be transformed into a gif
def evaluate_policy_and_capture_frames(model, env, n_eval_episodes=10):
    episode_rewards = []
    frames = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Capture frame
            frame = env.render(mode='rgb_array')
            frames.append(frame)

        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward, frames

# Duration is same as fpss
def save_frames_as_gif(frames, path='./gym_animation.gif', duration=30):
    imageio.mimsave(path, frames, duration=duration)
