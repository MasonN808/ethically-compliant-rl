#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppol           # Name of the job
#SBATCH --cpus-per-task=7         # Number of CPUs per task
#SBATCH --mem=30gb                # Memory allocated
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks=8                # Number of tasks
#SBATCH --time=2-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/PPOL/train_ppol.py"

# Extract number of nodes from SLURM settings
NUM_NODES=$(grep "^#SBATCH --nodes=" $0 | cut -d'=' -f2)

# Run the script as many times as the number of nodes in parallel
for i in $(seq 1 $NUM_NODES); do
    srun -N1 -n1 python3 $BASE_SCRIPT $ARGS &
done

wait