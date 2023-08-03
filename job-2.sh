#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --mem=40gb
#SBATCH --time=40:00:00
#SBATCH -c 8
#SBATCH --qos medium


srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_1.py" &
srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_2.py" &
srun -N1 -n1 --gpus=1 "/nas/ucb/mason/ethically-compliant-rl/PPO/train_ppo.py" &
wait
