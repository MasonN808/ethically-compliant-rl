#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppol           # Name of the job
#SBATCH --cpus-per-task=7         # Number of CPUs per task
#SBATCH --mem=40gb                # Memory allocated
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks=8                # Number of tasks
#SBATCH --time=3-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPOL/train_ppol.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS"
)

COST_VALUES=("2" "4" "8" "16" "32")  # Add the beta values you want to test

# Run the script as many times as the number of nodes in parallel
for i in {1..3}; do
    for SCRIPT in "${SCRIPTS[@]}"; do
        for COST in "${COST_VALUES[@]}"; do
            srun -N1 -n1 python3 $SCRIPT --cost_limit $COST &
        done
    done
done

wait