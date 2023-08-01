import ast
from dataclasses import asdict, dataclass
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
from fsrl.policy import PPOLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model, seed_all
from utils.utils import load_environment
from gymnasium.spaces.dict import Dict

@dataclass
class EvalConfig:
    # Relative path to experiment
    path: str = "logs/fast-safe-rl/parking-v0-cost-10/ppol_lr0.001_step_per_epoch20000_target_kl0.01-7d4f"
    best: bool = True
    eval_episodes: int = 3
    parallel_eval: bool = False
    # This was originally a bool; must be changed to float
    render: float = .001
    train_mode: bool = False
    render_mode: str = "rgb_array"
    device = "cpu"
    worker: BaseVectorEnv = ShmemVectorEnv
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoal.txt'

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
    env = load_environment(ENV_CONFIG, render_mode="rbg_array")
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
                    ) for _ in range(2)
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
            ).to(args.device) for _ in range(2)
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

    # collector
    test_envs = args.worker(
        [lambda: load_environment(ENV_CONFIG) for _ in range(args.eval_episodes)]
    )
    eval_collector = FastCollector(policy, test_envs)
    result = eval_collector.collect(n_episode=args.eval_episodes, render=args.render)
    rews, lens, cost = result["rew"], result["len"], result["cost"]
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()
