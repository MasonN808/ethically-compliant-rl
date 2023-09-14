#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=sweeps         # Name of the job
#SBATCH --cpus-per-task=6           # Number of CPUs per task
#SBATCH --mem=40gb                  # Memory allocated
#SBATCH --nodes=6                   # Number of nodes
#SBATCH --ntasks=6                  # Number of tasks
#SBATCH --time=3-00:00:00           # Maximum run time of the job (set to 2 days)
#SBATCH --qos=scavenger             # Quality of Service of the job

# Set the wandb API key
export WANDB_API_KEY='9762ecfe45a25eda27bb421e664afe503bb42297'

# An array of paths to the sweep configuration files
SWEEP_FILES=(
    "/nas/ucb/mason/ethically-compliant-rl/CPO/sweep-cpo.yaml"
    "/nas/ucb/mason/ethically-compliant-rl/CVPO/sweep-cvpo.yaml"
)

# Template for the command that will be run for each sweep
BASE_SCRIPT_TEMPLATE="wandb agent %s"

# Number of parallel jobs for each sweep
NUM_JOBS=3

# Function to execute a single sweep
run_sweep() {
    # Assign the first argument of the function to SWEEP_FILE
    SWEEP_FILE=$1

    # Extract the name of the sweep from the file name
    SWEEP_NAME=$(basename "$SWEEP_FILE" | cut -d'-' -f2 | cut -d'.' -f1)

    # Use the extracted name for the sweep
    SWEEP_LINE=$(wandb sweep --name="$SWEEP_NAME-700-epochs-no-walls" $SWEEP_FILE 2>&1 | grep "wandb agent")

    # This command extracts the SWEEP_ID from the captured line using awk
    SWEEP_ID=$(echo $SWEEP_LINE | awk '{print $NF}')
  
    # Replace %s in the template with the SWEEP_ID
    BASE_SCRIPT=$(printf "$BASE_SCRIPT_TEMPLATE" "$SWEEP_ID")

    # Start NUM_JOBS parallel jobs for the current sweep
    for ((i=1; i<=NUM_JOBS; i++)); do
        srun -N1 -n1 $BASE_SCRIPT &  # Execute the BASE_SCRIPT command in the background
    done

    wait  # Wait for all background jobs of this sweep to complete
}

# Start sweeps for all files in parallel
for SWEEP_FILE in "${SWEEP_FILES[@]}"; do
    run_sweep $SWEEP_FILE &  # Call run_sweep function in the background for parallel execution
done

wait  # Wait for all sweeps to complete
