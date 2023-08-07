#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --cpus-per-task=30
#SBATCH --mem=80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --qos medium


srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py"