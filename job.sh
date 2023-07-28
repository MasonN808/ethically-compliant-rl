#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00

srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait