#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60gb
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=30:00:00
#SBATCH --qos default


srun -N1 -n1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py" &
srun -N1 -n1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom_2.py" &
wait
