#!/bin/bash
#SBATCH --job-name=CVPO-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --time=30:00:00
#SBATCH --qos scavenger


# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/CVPO/train_cvpo_copy.py"

SCRIPTS=(
"$BASE_SCRIPT --epoch 400 $ARGS"
"$BASE_SCRIPT --epoch 400 $ARGS"
"$BASE_SCRIPT --epoch 300 $ARGS"
"$BASE_SCRIPT --epoch 300 $ARGS"
"$BASE_SCRIPT --epoch 200 $ARGS"
"$BASE_SCRIPT --epoch 200 $ARGS"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait