#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppo-penalty         # Name of the job
#SBATCH --cpus-per-task=5         # Number of CPUs per task
#SBATCH --mem=15gb                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=1-00:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPO_penalty/train_ppo_penalty.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS" 
)

BETA_VALUES=("dynamic" "0" ".1" ".5" "5" "50" "200" "1000")  # Add the beta values you want to test

for SCRIPT in "${SCRIPTS[@]}"; do
    for BETA in "${BETA_VALUES[@]}"; do
        srun -N1 -n1 python3 $SCRIPT --beta $BETA &
    done
done

wait
