#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppo-parking         # Name of the job
#SBATCH --cpus-per-task=3         # Number of CPUs per task
#SBATCH --mem=10gb                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1              # Specify the number of CPU cores
#SBATCH --time=3-00:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPO/train_ppo.py"

for i in {1..1}; do
    srun -N1 -n1 python3 $BASE_SCRIPT $ARGS &
done

wait