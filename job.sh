#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --gpus-per-task=A6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80gb
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=30:00:00
#SBATCH --qos medium


srun -n1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -n1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
wait
