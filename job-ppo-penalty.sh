#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppo-penalty         # Name of the job
#SBATCH --gpus-per-task=A6000:1
#SBATCH --cpus-per-task=8         # Number of CPUs per task
#SBATCH --mem=20gb                # Memory allocated
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --time=23:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPO_penalty/train_ppo_penalty.py"

SCRIPTS=(
    "$BASE_SCRIPT $ARGS" 
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait
