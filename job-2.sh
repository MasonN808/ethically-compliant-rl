#!/bin/bash
#SBATCH --job-name=safe-rl
#SBATCH --cpus-per-task=9
#SBATCH --mem=40gb
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=30:00:00
#SBATCH --qos scavenger

SCRIPTS=(
"/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py"
"/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom_2.py"
)

for SCRIPT in \${SCRIPTS[@]}; do
    srun -N1 -n1 python3 \$SCRIPT &
done
wait