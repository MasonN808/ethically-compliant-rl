#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=50gb
#SBATCH --time=50:00:00
#SBATCH -c 30
#SBATCH --qos medium


srun --gpus=0 -c30 --mem=50gb "/nas/ucb/mason/ethically-compliant-rl/SAC/train_SAC.py"
# srun -n1 -N1 --gpus=1 --mem=50gb "/nas/ucb/mason/ethically-compliant-rl/PPO/train_ppo.py"
# wait
