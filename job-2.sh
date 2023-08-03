#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=120gb
#SBATCH --time=40:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --qos medium


srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py"
