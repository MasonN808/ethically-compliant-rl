#!/bin/sh
#SBATCH --job-name=safe-rl
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30gb
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --time=30:00:00
#SBATCH --qos scavenger


# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo_custom.py"

SCRIPTS=(
"$BASE_SCRIPT --lr .0005 --epoch 450 $ARGS" 
"$BASE_SCRIPT --lr .0005 --epoch 450 $ARGS"
"$BASE_SCRIPT --lr .0005 --epoch 450 $ARGS"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait