#!/usr/bin/env python3
import os
from dataclasses import asdict, dataclass
import pickle
import ast
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import highway_env
import bullet_safety_gym
import gymnasium as gym
try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")

import torch
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

# To render the environemnt and agent
import matplotlib.pyplot as plt
sys.path.append("FSRL")
from fsrl.agent import CPOAgent
from fsrl.config.cpo_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg,
)
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name

# from rl_agents.agents.common.factory import load_agent, load_environment
from utils.utils import load_environment

TASK_TO_CFG = {
    # bullet safety gym tasks
    "SafetyCarRun-v0": Bullet1MCfg,
    "SafetyBallRun-v0": Bullet1MCfg,
    "SafetyBallCircle-v0": Bullet1MCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyDroneRun-v0": TrainCfg,
    "SafetyAntRun-v0": TrainCfg,
    "SafetyDroneCircle-v0": Bullet5MCfg,
    "SafetyAntCircle-v0": Bullet10MCfg,
    # safety gymnasium tasks
    "SafetyPointCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyHalfCheetahVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyHopperVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetySwimmerVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyWalker2dVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyAntVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyHumanoidVelocityGymnasium-v1": Mujoco20MCfg,
    # HighwayEnv tasks
    "roundabout-v0": TrainCfg, # TODO: Change the configs for HighEnv tasks
}

# HIGHWAY_ENV_TO_CFG = {
#     "roundabout-v0": HighwayEnvCfg,
#     "parking-v0": HighwayEnvCfg,
# }

import os
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# Make my own config params
@dataclass
class MyCfg(TrainCfg):
    # task: str = "SafetyPointCircle1Gymnasium-v0"
    task: str = "parking-v0"
    epoch: int = 2
    lr: float = 0.01
    # render: float = .001
    render: float = None # The rate at which it renders
    render_mode: str = "rgb_array"
    # render_mode: str = None # If you don't want renders after training
    thread: int = 160 # If use CPU to train
    step_per_epoch = 100
    project: str = "fast-safe-rl"
    slurm: bool = False
    # Decide which device to use based on availability
    # device = (
    #         "cuda"
    #         if torch.cuda.is_available()
    #         else "mps"
    #         if torch.backends.mps.is_available()
    #         else "cpu"
    #     )
    device = "cpu"

ENV_CONFIG_FILE = 'configs/ParkingEnv/env-image.txt'
with open(ENV_CONFIG_FILE) as f:
    data = f.read()
# reconstructing the data as a dictionary
ENV_CONFIG = ast.literal_eval(data)


@pyrallis.wrap()
def train(args: MyCfg):
    task = args.task
    default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else TrainCfg()
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        default_cfg.task = args.task
        default_cfg.seed = args.seed
        default_cfg.device = args.device
        default_cfg.logdir = args.logdir
        default_cfg.project = args.project
        default_cfg.group = args.group
        default_cfg.suffix = args.suffix
        args = default_cfg

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)
    # logger = BaseLogger()

    # demo_env = gym.make(args.task, render_mode=args.render_mode)
    demo_env = load_environment(ENV_CONFIG)

    # Some config testing
    print("Observation Space: {}".format(demo_env.observation_space))
    print("Action Space: {}".format(demo_env.action_space))
    print("Render Mode: {}".format(demo_env.render_mode))

    agent = CPOAgent(
        env=demo_env,
        logger=logger,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        lr=args.lr,
        hidden_sizes=args.hidden_sizes,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        damping_coeff=args.damping_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        cost_limit=args.cost_limit,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        slurm=args.slurm,
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(training_num)])
    test_envs = worker([lambda: load_environment(ENV_CONFIG) for _ in range(args.testing_num)])
    #     train_envs = worker([lambda: gym.make(args.task) for _ in range(training_num)])
    #     test_envs = worker([lambda: gym.make(args.task) for _ in range(args.testing_num)])

    # start training
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=args.epoch,
        episode_per_collect=args.episode_per_collect,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        buffer_size=args.buffer_size,
        testing_num=args.testing_num,
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,
        save_interval=args.save_interval,
        resume=args.resume,
        save_ckpt=args.save_ckpt,
        verbose=args.verbose,
    )
    if __name__ == "__main__":
        # Let's watch its performance!
        from fsrl.data import FastCollector
        env = load_environment(ENV_CONFIG, render_mode='human')
        agent.policy.eval()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        agent.policy.train()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()