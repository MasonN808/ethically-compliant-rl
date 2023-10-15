#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppo-highway         # Name of the job
#SBATCH --cpus-per-task=6         # Number of CPUs per task
#SBATCH --mem=20gb                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=10:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ecrl/PPO/train_ppo.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS" 
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait
