#!/bin/bash
#SBATCH --job-name=CVPO-1-const
#SBATCH --cpus-per-task=7
#SBATCH --mem=40gb
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=30:00:00
#SBATCH --qos scavenger


# Get all arguments passed to the script
ARGS="$@"

BASE_SCRIPT="/nas/ucb/mason/ethically-compliant-rl/CVPO/train_cvpo.py"
# BASE_SCRIPT2="/nas/ucb/mason/ethically-compliant-rl/CVPO/train_cvpo_copy.py"
# BASE_SCRIPT3="/nas/ucb/mason/ethically-compliant-rl/CVPO/train_cvpo_copy_copy.py"


SCRIPTS=(
"$BASE_SCRIPT $ARGS" 
"$BASE_SCRIPT $ARGS"
# "$BASE_SCRIPT2 $ARGS" 
# "$BASE_SCRIPT2 $ARGS"
# "$BASE_SCRIPT3 $ARGS"
# "$BASE_SCRIPT3 $ARGS"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    srun -N1 -n1 python3 $SCRIPT &
done

wait