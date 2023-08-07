#!/bin/sh
#SBATCH --gpus-per-task=A6000:1
#SBATCH --job-name=safe-rl
#SBATCH --mem=40gb
#SBATCH --time=20:00:00
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --qos=high



