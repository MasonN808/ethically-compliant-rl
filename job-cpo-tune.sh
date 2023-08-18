Copy code
#!/bin/bash
#SBATCH --job-name=CPO-tune
#SBATCH --cpus-per-task=7
#SBATCH --mem=40gb
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=30:00:00
#SBATCH --qos scavenger

# Initialize the sweep to get the SWEEP_ID
SWEEP_ID=$(wandb sweep /nas/ucb/mason/ethically-compliant-rl/CPO/sweep.yaml)

BASE_SCRIPT="wandb agent $SWEEP_ID"

# Number of jobs you want to run in parallel.
NUM_JOBS=2

for ((i=1; i<=NUM_JOBS; i++)); do
    srun -N1 -n1 $BASE_SCRIPT &
done

wait