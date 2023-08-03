#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=40gb
#SBATCH --time=15:00:00
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --qos=high


srun "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py"
