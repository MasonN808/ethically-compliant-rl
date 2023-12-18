import copy
import json
import random
import gymnasium as gym
import logging
import numpy as np
logger = logging.getLogger(__name__)

# From https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/common/factory.py
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

def seed(self, seed):
    np.random.seed(seed)

def generate_env_config(num: int, starting_locations):
    return [{"start_location": random.choice(starting_locations)} for _ in range(num)]

def get_updated_config(index: int, env_list: list, env_config: dict):
    updated_config = copy.deepcopy(env_config)
    updated_config.update(env_list[index])
    return updated_config
