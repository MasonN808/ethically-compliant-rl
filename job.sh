#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=100gb
#SBATCH --time=20:00:00
#SBATCH -n 2
#SBATCH -c 10
#SBATCH --qos medium


srun --exclusive -n 1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun --exclusive -n 1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait