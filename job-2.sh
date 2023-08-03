#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=120gb
#SBATCH --time=40:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 8
#SBATCH --gpus=2
#SBATCH --qos medium


srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait
