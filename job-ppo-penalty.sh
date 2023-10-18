#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppo-penalty         # Name of the job
#SBATCH --cpus-per-task=5         # Number of CPUs per task
#SBATCH --mem=45gb                # Memory allocated
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks=8                # Number of tasks
#SBATCH --time=24:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPO_penalty/train_ppo_penalty.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS" 
)

BETA_VALUES=("0.0" ".01" "0.05" "0.1" "0.3" "1.0" "3" "10" "40")  # Add the beta values you want to test

for SCRIPT in "${SCRIPTS[@]}"; do
    for BETA in "${BETA_VALUES[@]}"; do
        srun -N1 -n1 python3 $SCRIPT --beta $BETA &
    done
done

wait
