#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=40gb
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --gpus=1
#SBATCH --qos medium


srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py"