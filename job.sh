#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --gpus=0
#SBATCH --time=04:00:00

srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo.py"