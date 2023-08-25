import ast
from dataclasses import dataclass
import sys

import bullet_safety_gym
import safety_gymnasium
import gymnasium as gym

import pyrallis
sys.path.append("FSRL")
from fsrl.agent import CVPOAgent
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model
from utils import load_environment

# For video monitoring the environment
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import re
import numpy as np

@dataclass
class EvalConfig:
    # Relative path to experiment
    path: str = "logs/CVPO-sweep-700epochs/parking-v0-cost-[5.0]/cvpo_actor_lr0.001_constraint_typelines_cost5.0_critic_lr0.0005_gamma0.099_step_per_epoch1000-a31a"
    best: bool = False
    # TODO Create a most recent checkpoint model
    epoch_model_number: int = 588 # For a specific checkpoint model 
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

#     # Get the unique 4 char id of the file at the end of the file name
#     match = re.search(r'-([\w]+)$', path)
#     experiment_id = "----"
#     if match:
#         experiment_id = match.group(1)
#     else:
#         print("Pattern not found")

#     # Get the algorithm used
#     match = re.search(r'/([a-zA-Z]+)[_-][^/]+$', path)
#     algorithm = "----"
#     if match:
#         algorithm = match.group(1)
#     else:
#         print("Pattern not found")

#     best: bool = True
#     eval_episodes: int = 2
#     convert_to_gif: bool = True
#     parallel_eval: bool = False
#     # This was originally a bool; must be changed to float
#     render: float = .005
#     train_mode: bool = False
#     render_mode: str = "rgb_array"
#     device = "cpu"
#     worker: BaseVectorEnv = ShmemVectorEnv
#     env_config_file: str = 'configs/ParkingEnv/env-evaluation.txt'
#     # env_config_file: str = 'configs/ParkingEnv/env-kinematicsGoalConstraints.txt'
#     monitor_mode: bool = True
#     video_recorder: VideoRecorder = None # Keep this None
#     # random_starting_locations = [[0,0], [40, 40], [-40,-40], [40, -40], [-40, 40], [0, 40], [-40, 0]]
#     random_starting_locations = [[0,32]]


# with open(EvalConfig.env_config_file) as f:
#     data = f.read()
# # reconstructing the data as a dictionary
# ENV_CONFIG = ast.literal_eval(data)

# @pyrallis.wrap()
# def eval(args: EvalConfig):
#     cfg, model = load_config_and_model(args.path, args.best)
#     # seed
#     seed_all(cfg["seed"])
#     torch.set_num_threads(cfg["thread"])

#     logger = BaseLogger()

#     # model
#     env = load_environment(ENV_CONFIG, render_mode=args.render_mode)
#     # Get the shapes of the states and actions to be transfered to a tensor
#     if isinstance(env.observation_space, Dict):
#         # TODO: This is hardcoded please fix
#         dict_state_shape = {
#             "observation": (6,),
#             "achieved_goal": (6,),
#             "desired_goal": (6,)
#         }
#         decorator_fn, state_shape = get_dict_state_decorator(dict_state_shape, list(dict_state_shape.keys()))
#         global Net, ActorProb, Critic, DataParallelNet # Fixes UnboundLocalError
#         # Apply decorator to overwrite the forward pass in the Tensorflow module to allow for dict object
#         Net = decorator_fn(Net)
#         ActorProb = decorator_fn(ActorProb)
#         Critic = decorator_fn(Critic)
#         DataParallelNet = decorator_fn(DataParallelNet)
#     else: 
#         state_shape = env.observation_space.shape or env.observation_space.n
#     action_shape = env.action_space.shape or env.action_space.n
#     max_action = env.action_space.high[0]

#     use_cuda = torch.cuda.is_available()
#     # Create Actor
#     net = Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device)
#     actor_constructor = ActorProb(net, action_shape, max_action=max_action, unbounded=cfg["unbounded"], conditioned_sigma=cfg["conditioned_sigma"], device=None if use_cuda else args.device)
#     actor = DataParallelNet(actor_constructor).to(args.device) if use_cuda else actor_constructor.to(args.device)

#     # Create Critics
#     if cfg["double_critic"]:
#         net1 = Net(
#             state_shape,
#             action_shape,
#             hidden_sizes=cfg["hidden_sizes"],
#             concat=True,
#             device=args.device
#         )
#         net2 = Net(
#             state_shape,
#             action_shape,
#             hidden_sizes=cfg["hidden_sizes"],
#             concat=True,
#             device=args.device
#         )
#         critic_constructor = lambda: DoubleCritic(net1, net2, device=None if use_cuda else args.device).to(args.device)
#     else:
#         net_c = Net(
#             state_shape,
#             action_shape,
#             hidden_sizes=cfg["hidden_sizes"],
#             concat=True,
#             device=args.device
#         )
#         critic_constructor = lambda: SingleCritic(net_c, device=None if use_cuda else args.device).to(args.device)

#     critics = [DataParallelNet(critic_constructor()) for _ in range(2)] if use_cuda else [critic_constructor() for _ in range(2)]

#     def dist(*logits):
#         return Independent(Normal(*logits), 1)

#     policy = CVPO(
#         actor=actor,
#         critics=critics,
#         actor_optim=None,
#         critic_optim=None,
#         logger=logger,
#         dist_fn=dist,
#         cost_limit=cfg["cost_limit"],
#         action_space=env.action_space,
#         max_episode_steps=cfg["step_per_epoch"] #TODO Check if this is rigth
#     )
#     policy.load_state_dict(model["model"])
#     policy.eval()
    
#     # (1) For each epsisode, make a video and store it
#     # (2) Convert the video into a gif and store it
#     video_index = 0
#     for _ in range(0, args.eval_episodes):
#         ENV_CONFIG.update({"start_location": random.choice(args.random_starting_locations)})
#         test_env = load_environment(ENV_CONFIG, render_mode=args.render_mode)
#         # Check if the file name exists
#         # If not, loop through the indices until you reach an available index
#         name = f"./videos/{args.algorithm}/mp4s/{args.algorithm}-{args.experiment_id}-{video_index}.mp4"
#         filename = Path(name)
#         while filename.exists():
#             video_index += 1
#             name = f"./videos/{args.algorithm}/mp4s/{args.algorithm}-{args.experiment_id}-{video_index}.mp4"
#             filename = Path(name)
#         video_recorder = VideoRecorder(test_env, name)
#         # Collector
#         eval_collector = FastCollector(policy, test_env)
#         result = eval_collector.collect(n_episode=1, render=args.render, video_recorder=video_recorder)

#         # Optionally turn the mp4 into a gif immediately
#         if args.convert_to_gif:
#             mp4_to_gif(mp4_path=f"./videos/{args.algorithm}/mp4s/{args.algorithm}-{args.experiment_id}-{video_index}.mp4",
#                         gif_path=f"./videos/{args.algorithm}/gifs/{args.algorithm}-{args.experiment_id}-{video_index}.gif")

#         rews, lens, cost = result["rew"], result["len"], result["avg_total_cost"]
#         print(f'rews: {rews}', f'lens: {lens}', f'cost: {cost}')

# if __name__ == "__main__":
#     eval()

