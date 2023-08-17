import ast
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import sys

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
import torch
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv
from tianshou.utils.net.common import Net, get_dict_state_decorator, DataParallelNet
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

sys.path.append("FSRL")
from fsrl.data import FastCollector
from fsrl.policy import CVPO
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model, mp4_to_gif, seed_all
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic
from utils import load_environment
from gymnasium.spaces.dict import Dict

# For video monitoring the environment
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import re

@dataclass
class EvalConfig:
    # Relative path to experiment
    path: str = "logs/CVPO/parking-v0-cost0-2-cost1-2/cvpo-a6cf"
    # Get the unique 4 char id of the file at the end of the file name
    match = re.search(r'-([\w]+)$', path)
    experiment_id = "----"
    if match:
        experiment_id = match.group(1)
    else:
        print("Pattern not found")

    # Get the algorithm used
    match = re.search(r'/([a-zA-Z]+)[_-][^/]+$', path)
    algorithm = "----"
    if match:
        algorithm = match.group(1)
    else:
        print("Pattern not found")

    best: bool = True
    eval_episodes: int = 2
    convert_to_gif: bool = True
    parallel_eval: bool = False
    # This was originally a bool; must be changed to float
    render: float = .005
    train_mode: bool = False
    render_mode: str = "rgb_array"
    device = "cpu"
    worker: BaseVectorEnv = ShmemVectorEnv
    env_config_file: str = 'configs/ParkingEnv/env-evaluation.txt'
    # env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
    monitor_mode: bool = True
    video_recorder: VideoRecorder = None # Keep this None
    # random_starting_locations = [[0,0], [40, 40], [-40,-40], [40, -40], [-40, 40], [0, 40], [-40, 0]]
    random_starting_locations = [[0,32]]


with open(EvalConfig.env_config_file) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)

@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)
    # seed
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["thread"])

    logger = BaseLogger()

    # model
    env = load_environment(ENV_CONFIG, render_mode=args.render_mode)
    # Get the shapes of the states and actions to be transfered to a tensor
    if isinstance(env.observation_space, Dict):
        # TODO: This is hardcoded please fix
        dict_state_shape = {
            "achieved_goal": (6,),
            "observation": (6,),
            "desired_goal": (6,)
        }
        decorator_fn, state_shape = get_dict_state_decorator(dict_state_shape, list(dict_state_shape.keys()))
        global Net, ActorProb, Critic, DataParallelNet # Fixes UnboundLocalError
        # Apply decorator to overwrite the forward pass in the Tensorflow module to allow for dict object
        Net = decorator_fn(Net)
        ActorProb = decorator_fn(ActorProb)
        Critic = decorator_fn(Critic)
        DataParallelNet = decorator_fn(DataParallelNet)
    else: 
        state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    use_cuda = torch.cuda.is_available()
    # Create Actor
    net = Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device)
    actor_constructor = ActorProb(net, action_shape, max_action=max_action, unbounded=cfg["unbounded"], conditioned_sigma=cfg["conditioned_sigma"], device=None if use_cuda else args.device)
    actor = DataParallelNet(actor_constructor).to(args.device) if use_cuda else actor_constructor.to(args.device)

    # Create Critics
    if cfg["double_critic"]:
        net1 = Net(
            state_shape,
            action_shape,
            hidden_sizes=cfg["hidden_sizes"],
            concat=True,
            device=args.device
        )
        net2 = Net(
            state_shape,
            action_shape,
            hidden_sizes=cfg["hidden_sizes"],
            concat=True,
            device=args.device
        )
        critic_constructor = lambda: DoubleCritic(net1, net2, device=None if use_cuda else args.device).to(args.device)
    else:
        net_c = Net(
            state_shape,
            action_shape,
            hidden_sizes=cfg["hidden_sizes"],
            concat=True,
            device=args.device
        )
        critic_constructor = lambda: SingleCritic(net_c, device=None if use_cuda else args.device).to(args.device)

    critics = [DataParallelNet(critic_constructor()) for _ in range(2)] if use_cuda else [critic_constructor() for _ in range(2)]

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = CVPO(
        actor=actor,
        critics=critics,
        actor_optim=None,
        critic_optim=None,
        logger=logger,
        dist_fn=dist,
        cost_limit=cfg["cost_limit"],
        action_space=env.action_space,
        max_episode_steps=cfg["step_per_epoch"] #TODO Check if this is rigth
    )
    policy.load_state_dict(model["model"])
    policy.eval()
    
    # (1) For each epsisode, make a video and store it
    # (2) Convert the video into a gif and store it
    video_index = 0
    for _ in range(0, args.eval_episodes):
        ENV_CONFIG.update({"start_location": random.choice(args.random_starting_locations)})
        test_env = load_environment(ENV_CONFIG, render_mode=args.render_mode)
        # Check if the file name exists
        # If not, loop through the indices until you reach an available index
        name = f"./videos/{args.algorithm}/mp4s/{args.algorithm}-{args.experiment_id}-{video_index}.mp4"
        filename = Path(name)
        while filename.exists():
            video_index += 1
            name = f"./videos/{args.algorithm}/mp4s/{args.algorithm}-{args.experiment_id}-{video_index}.mp4"
            filename = Path(name)
        video_recorder = VideoRecorder(test_env, name)
        # Collector
        eval_collector = FastCollector(policy, test_env)
        result = eval_collector.collect(n_episode=1, render=args.render, video_recorder=video_recorder)

        # Optionally turn the mp4 into a gif immediately
        if args.convert_to_gif:
            mp4_to_gif(mp4_path=f"./videos/{args.algorithm}/mp4s/{args.algorithm}-{args.experiment_id}-{video_index}.mp4",
                        gif_path=f"./videos/{args.algorithm}/gifs/{args.algorithm}-{args.experiment_id}-{video_index}.gif")

        rews, lens, cost = result["rew"], result["len"], result["avg_total_cost"]
        print(f'rews: {rews}', f'lens: {lens}', f'cost: {cost}')

if __name__ == "__main__":
    eval()
