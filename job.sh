#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=100gb
#SBATCH --time=20:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 5
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=2
#SBATCH --qos medium


srun --exclusive -n 1 -N 1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun --exclusive -n 1 -N 1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait