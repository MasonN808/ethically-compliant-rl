#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=100gb
#SBATCH --time=30:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 5
#SBATCH --gpus=2
#SBATCH --qos medium


srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -N1 -n1 --gpus=1"/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
wait
