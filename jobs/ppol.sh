#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=ppol-lines-lidar          # Name of the job
#SBATCH --cpus-per-task=3         # Number of CPUs per task
#SBATCH --mem=3gb                # Memory allocated
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --time=3-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/tests/PPOL_New/train_ppol_lidar.py"

# srun -N1 -n1 python3 $BASE_SCRIPT $ARGS

SEEDS=("1")  # Vary seeds
ENT_COEFS=(".001" ".002" ".004" ".008")  # Vary entropy coefficents

# Run the script as many times as the number of nodes in parallel
for SEED in "${SEEDS[@]}"; do
    for ENT_COEF in "${ENT_COEFS[@]}"; do
        srun -N1 -n1 python3 $BASE_SCRIPT --seed $SEED --ent_coef $ENT_COEF &
    done
done

wait