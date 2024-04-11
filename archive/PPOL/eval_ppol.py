import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import sys

import bullet_safety_gym
import numpy as np

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
from fsrl.policy import PPOLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model, mp4_to_gif, seed_all
from utils.utils import load_environment, parse_between_slashes
from gymnasium.spaces.dict import Dict

# For video monitoring the environment
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import re

@dataclass
class EvalConfig:
    # Relative path to experiment
    path: str = "logs/PPOL-SpeedConstraint-200sDuration-Speed=16/parking-v0-cost0-2/ppol_cost2_gamma0.999_lr0.0003_step_per_epoch3000_target_kl0.01-31c0"

    # Get the unique 4 char id of the file at the end of the file name
    match = re.search(r'-([\w]+)$', path)
    experiment_id = "----"
    if match:
        experiment_id = match.group(1)
    else:
        print("Pattern not found")

    # Get the algorithm used
    match = re.search(r'/(\w+?)_', path)
    if match:
        algorithm = match.group(1)
    else:
        print("Pattern not found")

    epoch_model_number: int = 525
    best: bool = True
    eval_episodes: int = 3 
    convert_to_gif: bool = True
    parallel_eval: bool = False
    constraint_type: list[str] = field(default_factory=lambda: ["speed"])
    # This was originally a bool; must be changed to float
    render: float = .005
    train_mode: bool = False
    render_mode: str = "rgb_array"
    device = "cpu"
    worker: BaseVectorEnv = ShmemVectorEnv
    env_config_file: str = 'configs/ParkingEnv/env-evaluation.txt'
    monitor_mode: bool = True
    video_recorder: VideoRecorder = None # Keep this None
    # random_starting_locations = [[0,0], [40, 40], [-40,-40], [40, -40], [-40, 40], [0, 40], [-40, 0]]
    random_starting_locations = [[0,0]]


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best, epoch_model_number=args.epoch_model_number)
    # seed
    seed_all(111)
    torch.set_num_threads(cfg["thread"])

    with open(EvalConfig.env_config_file) as f:
        data = f.read()
    # reconstructing the data as a dictionary
    ENV_CONFIG = ast.literal_eval(data)

    # Overriding certain keys in the environment config
    ENV_CONFIG.update({
        "start_angle": -np.math.pi/2, # This is radians
        # Costs
        "constraint_type": args.constraint_type,
        "speed_limit": 2, # TODO make constraint type and limit a dictionary for easier implementation
})

    logger = BaseLogger()

    # model
    env = load_environment(ENV_CONFIG, render_mode=args.render_mode)
    # Get the shapes of the states and actions to be transfered to a tensor
    if isinstance(env.observation_space, Dict):
        # TODO: This is hardcoded please fix
        dict_state_shape = {
            "observation": (6,),
            "achieved_goal": (6,),
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

    # model
    # W/ DataParallelNet For cuda Parallelization
    net = Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
                ActorProb(
                    net,
                    action_shape,
                    max_action=max_action,
                    unbounded=cfg["unbounded"],
                    device=None
                ).to(args.device)
        )
        critic = [DataParallelNet(
                        Critic(
                            Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device),
                            device=None
                        ).to(args.device)
                    ) for _ in range(1 + len(cfg["constraint_type"]))
        ]
    else:
        actor = ActorProb(
            net,
            action_shape,
            max_action=max_action,
            unbounded=cfg["unbounded"],
            device=args.device
        ).to(args.device)
        critic = [
            Critic(
                Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device),
                device=args.device
            ).to(args.device) for _ in range(1 + len(cfg["constraint_type"]))
        ]

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOLagrangian(
        actor,
        critic,
        None,
        dist,
        logger=logger,
        use_lagrangian=cfg["use_lagrangian"],
        action_space=env.action_space,
        max_batchsize=20000,
    )
    policy.load_state_dict(model["model"])
    policy.eval()

    video_index = 0
    for _ in range(0, EvalConfig.eval_episodes):
        # Get the project description from path
        description = parse_between_slashes(args.path)
        ENV_CONFIG.update({"start_location": random.choice(EvalConfig.random_starting_locations)})
        test_env = load_environment(ENV_CONFIG, render_mode=EvalConfig.render_mode)
        # Check if the file name exists
        # If not, loop through the indices until you reach an available index
        mp4_path = f"./videos/{EvalConfig.algorithm}/{description}/mp4s/{EvalConfig.algorithm}-{EvalConfig.experiment_id}-{video_index}.mp4"
        filename = Path(mp4_path)

        # Ensure the directory exists
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        while filename.exists():
            video_index += 1
            mp4_path = f"./videos/{EvalConfig.algorithm}/{description}/mp4s/{EvalConfig.algorithm}-{EvalConfig.experiment_id}-{video_index}.mp4"
            filename = Path(mp4_path)
        video_recorder = VideoRecorder(test_env, mp4_path)
        # Collector
        # eval_collector = FastCollector(policy, test_env, constraint_type=cfg["constraint_type"])
        eval_collector = FastCollector(policy, test_env, constraint_type=["speed"])
        result = eval_collector.collect(n_episode=1, render=EvalConfig.render, video_recorder=video_recorder)


        # Optionally turn the mp4 into a gif immediately
        if EvalConfig.convert_to_gif:
            gif_path = f"./videos/{EvalConfig.algorithm}/{description}/gifs/{EvalConfig.algorithm}-{EvalConfig.experiment_id}-{video_index}.gif"
            filename = Path(gif_path)
            # Ensure the directory exists
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True, exist_ok=True)
            mp4_to_gif(mp4_path=mp4_path, gif_path=gif_path)

        rews, lens, cost = result["rew"], result["len"], result["avg_total_cost"]
        print(f'rews: {rews}', f'lens: {lens}', f'cost: {cost}')


if __name__ == "__main__":
    eval()
