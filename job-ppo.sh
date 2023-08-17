#!/bin/bash
#SBATCH --job-name=ppo
#SBATCH --cpus-per-task=7
#SBATCH --mem=40gb
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=30:00:00
#SBATCH --qos scavenger


# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPO/train_ppo.py"

SCRIPTS=(
"$BASE_SCRIPT --lr .001 --epoch 300 --target_kl .012 $ARGS"
"$BASE_SCRIPT --lr .0005 --epoch 300 --target_kl .012 $ARGS" 
"$BASE_SCRIPT --lr .001 --epoch 300 --target_kl .01 $ARGS" 
"$BASE_SCRIPT --lr .0005 --epoch 300 --target_kl .01 $ARGS"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait