#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=100gb
#SBATCH --gpus=2
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 10
#SBATCH --qos medium


srun -n 1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -n 1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait