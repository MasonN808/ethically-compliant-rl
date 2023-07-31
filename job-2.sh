#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=80gb
#SBATCH --time=30:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --gpus=1
#SBATCH --qos medium


srun -n1 -N1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
wait
