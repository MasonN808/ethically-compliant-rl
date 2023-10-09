#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=sweeps         # Name of the job
#SBATCH --cpus-per-task=6         # Number of CPUs per task
#SBATCH --mem=40gb                # Memory allocated
#SBATCH --nodes=6                 # Number of nodes
#SBATCH --ntasks=6                # Number of tasks
#SBATCH --time=3-00:00:00         # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
# source activate your_environment_name

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ecrl/CVPO/train_cvpo.py"

SCRIPTS=(
    "$BASE_SCRIPT --epoch 200 $ARGS" 
    "$BASE_SCRIPT --epoch 200 $ARGS"
    "$BASE_SCRIPT --epoch 200 $ARGS" 
    "$BASE_SCRIPT --epoch 200 $ARGS"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait
