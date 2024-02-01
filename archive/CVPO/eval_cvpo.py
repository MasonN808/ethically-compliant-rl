import ast
from dataclasses import dataclass
import sys

import bullet_safety_gym
import safety_gymnasium
import gymnasium as gym
import re
import numpy as np
import pyrallis

sys.path.append("FSRL")
from fsrl.agent import CVPOAgent
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model
from utils import load_environment

@dataclass
class EvalConfig:
    # Relative path to experiment
    path: str = "logs/CVPO-sweep-700-epochs-high-limit-lines/parking-v0-cost-[100000]/cvpo_actor_lr0.001_constraint_typelines_cost100000_critic_lr0.0005_gamma0.099_step_per_epoch1000-0baa"
    best: bool = False
    # TODO Create a most recent checkpoint model
    epoch_model_number: int = 100 # For a specific checkpoint model 
    eval_episodes: int = 2
    parallel_eval: bool = False
    # This was originally a bool; must be changed to float
    render: float = .01
    convert_to_gif: bool = True
    train_mode: bool = False
    render_mode: str = "rgb_array"
    # render_mode: str = "human"
    device = "cpu"
    env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
    # Points are around the parking lot and in the middle
    random_starting_locations = [[0, 0]]

# Get the unique 4 char id of the file at the end of the file name
match = re.search(r'-([\w]+)$', EvalConfig.path)
EvalConfig.experiment_id = "----"
if match:
    EvalConfig.experiment_id = match.group(1)
else:
    print("Pattern not found")

# Get the algorithm used
match = re.search(r'/(\w+?)_', EvalConfig.path)
EvalConfig.constraints = True
if match:
    EvalConfig.algorithm = match.group(1)
    if EvalConfig.algorithm == "ppol":
        EvalConfig.constraints = False
else:
    print("Pattern not found")

@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best, epoch_model_number=args.epoch_model_number)

    with open(EvalConfig.env_config_file) as f:
        data = f.read()
    # reconstructing the data as a dictionary
    ENV_CONFIG = ast.literal_eval(data)
    ENV_CONFIG.update({
        "steering_range": np.deg2rad(50),  # it is typical to be between 30-50 irl
        "start_angle": -np.math.pi/2, # This is radians
        # Costs
        "constraint_type": cfg["constraint_type"],
    })

    demo_env = load_environment(ENV_CONFIG)

    agent = CVPOAgent(
        env=demo_env,
        logger=BaseLogger(),
        device=args.device,
        cost_limit=cfg["cost_limit"],
        constraint_type=cfg["constraint_type"],
        thread=cfg["thread"],
        seed=cfg["seed"],
        # CVPO arguments
        estep_iter_num=cfg["estep_iter_num"],
        estep_kl=cfg["estep_kl"],
        estep_dual_max=cfg["estep_dual_max"],
        estep_dual_lr=cfg["estep_dual_lr"],
        sample_act_num=cfg["sample_act_num"],
        mstep_iter_num=cfg["mstep_iter_num"],
        mstep_kl_mu=cfg["mstep_kl_mu"],
        mstep_kl_std=cfg["mstep_kl_std"],
        mstep_dual_max=cfg["mstep_dual_max"],
        mstep_dual_lr=cfg["mstep_dual_lr"],
        # other algorithm params
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        gamma=cfg["gamma"],
        n_step=cfg["n_step"],
        tau=cfg["tau"],
        hidden_sizes=cfg["hidden_sizes"],
        double_critic=cfg["double_critic"],
        conditioned_sigma=cfg["conditioned_sigma"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
        deterministic_eval=cfg["deterministic_eval"],
        action_scaling=cfg["action_scaling"],
        action_bound_method=cfg["action_bound_method"],
    )

    rews, lens, cost = agent.evaluate(
        env_config = ENV_CONFIG,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        render_mode = args.render_mode,
        train_mode=args.train_mode,
        experiment_id=args.experiment_id,
        random_starting_locations = args.random_starting_locations,
        algorithm = args.algorithm,
        convert_to_gif = args.convert_to_gif
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()