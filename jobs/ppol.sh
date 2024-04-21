#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=minigrid          # Name of the job
#SBATCH --gres=gpu:1             # Request one GPU
#SBATCH --mem=5gb                # Memory allocated
#SBATCH --nodes=3                 # Number of nodes
#SBATCH --ntasks=6                # Number of tasks
#SBATCH --time=1-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/tests/PPOL/train_ppol_safety_grid.py"

SEEDS=("1")
PIDs=("0.0" "0.001" "0.01" "0.1" "1" "10")

# Run the script as many times as the number of nodes in parallel
for SEED in "${SEEDS[@]}"; do
    for PID in "${PIDs[@]}"; do
        srun -N1 -n1 python3 $BASE_SCRIPT --seed $SEED --K_P $PID --K_I $PID --K_D $PID &
    done
done

wait