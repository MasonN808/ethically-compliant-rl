#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=cpo+           # Name of the job
#SBATCH --cpus-per-task=5         # Number of CPUs per task
#SBATCH --mem=15gb                # Memory allocated
#SBATCH --nodes=5                 # Number of nodes
#SBATCH --ntasks=12                # Number of tasks
#SBATCH --time=3-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/CPO/train_cpo.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS"
)

SPEED_VALUES=("2" "4" "8" "16")  # Add the beta values you want to test

# Run the script as many times as the number of nodes in parallel
for i in {1..3}; do
    for SCRIPT in "${SCRIPTS[@]}"; do
        for SPEED in "${SPEED_VALUES[@]}"; do
            echo "Running script with speed limit: $SPEED"
            srun -N1 -n1 python3 $SCRIPT --speed_limit $SPEED &
        done
    done
done

wait