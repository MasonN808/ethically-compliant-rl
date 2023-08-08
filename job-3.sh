#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --gpus-per-task=A6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=90gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --qos medium


srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py"
