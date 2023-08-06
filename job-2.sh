#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=100gb
#SBATCH --time=20:00:00
#SBATCH -c 8
#SBATCH -n 2
#SBATCH --gpus=2
#SBATCH --qos=high
#SBATCH --gpus-per-task=A6000:1

srun -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait