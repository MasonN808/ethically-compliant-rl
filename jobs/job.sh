#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=test           # Name of the job
#SBATCH --cpus-per-task=1         # Number of CPUs per task
#SBATCH --mem=1gb                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=01:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPOL_New/train_ppol.py"

srun -N1 -n1 python3 $BASE_SCRIPT --speed_limit 5.0

wait
