#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=sac-parking         # Name of the job
#SBATCH --cpus-per-task=3         # Number of CPUs per task
#SBATCH --ntasks=5  # Specify the number of CPU cores
#SBATCH --mem=25gb                # Memory allocated
#SBATCH --nodes=5                 # Number of nodes
#SBATCH --time=2-00:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/SAC/train_sac.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS"
)

for i in {1..5}; do
    for SCRIPT in "${SCRIPTS[@]}"; do
        srun -N1 -n1 python3 $SCRIPT &
    done
done

wait