#!/bin/sh
#SBATCH --job-name=ppo
#SBATCH --cpus-per-task=10
#SBATCH --mem=20gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --qos scavenger


srun -N1 -n1 "/nas/ucb/mason/ethically-compliant-rl/PPO/train_ppo.py"