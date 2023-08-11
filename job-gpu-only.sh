#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30gb
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --time=30:00:00
#SBATCH --qos scavenger


SCRIPTS=(
"/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py" 
"/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py" --cost_limit 10.0 10.0
"/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py" --target_kl 0.015
)

for SCRIPT in ${SCRIPTS[@]}; do
    srun -N1 -n1 python3 $SCRIPT &
done
wait