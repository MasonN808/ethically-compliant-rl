#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00

srun --mem=80gb --qos=medium --gpus=1 -c 10 --time=20:00:00 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py"
srun --mem=80gb --qos=medium --gpus=1 -c 10 --time=20:00:00 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py"