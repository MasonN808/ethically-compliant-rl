#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=80gb
#SBATCH --time=20:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 5
#SBATCH --gpus=2
#SBATCH --qos medium


srun -n1 -N1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -n1 -N1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait
