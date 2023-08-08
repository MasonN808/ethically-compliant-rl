#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --gpus-per-task=A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --qos medium


srun "/nas/ucb/mason/ethically-compliant-rl/PPO/train_ppo.py"