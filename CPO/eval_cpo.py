import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import CPOAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all


@dataclass
class EvalConfig:
    # Need to get relative path of the experiment that you'd like to evaluate
    path: str = "logs/fast-safe-rl-slurm/SafetyCarGoal1Gymnasium-v0-cost-10/cpo_cost10_step_per_epoch10000-426f"
    best: bool = True
    eval_episodes: int = 2
    parallel_eval: bool = False
    device: str = "cpu"
    # This was originally a bool; must be changed to float
    render: float = .001
    train_mode: bool = False
    render_mode: str = "human"


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)

    task = cfg["task"]
    demo_env = gym.make(task, render_mode=args.render_mode)

    agent = CPOAgent(
        env=demo_env,
        logger=BaseLogger(),
        device=args.device,
        thread=cfg["thread"],
        seed=cfg["seed"],
        cost_limit=cfg["cost_limit"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    if args.parallel_eval:
        test_envs = ShmemVectorEnv(
            [lambda: gym.make(task, render_mode=args.render_mode) for _ in range(args.eval_episodes)]
        )
    else:
        test_envs = gym.make(task, render_mode=args.render_mode)

    rews, lens, cost = agent.evaluate(
        test_envs=test_envs,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        train_mode=args.train_mode
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()