#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=100gb
#SBATCH --time=50:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 20
#SBATCH --gpus=2
#SBATCH --qos medium


srun -n1 -N1 --gpus=1 --mem=50gb "/nas/ucb/mason/ethically-compliant-rl/SAC/train_sac.py" &
srun -n1 -N1 --gpus=1 --mem=50gb "/nas/ucb/mason/ethically-compliant-rl/CPO/train_ppo.py" &
wait
