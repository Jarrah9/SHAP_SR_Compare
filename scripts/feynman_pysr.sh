#!/bin/bash
#SBATCH --job-name=feynman_pysr
#SBATCH --account=pawsey0411
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/joatham/SR-XAI-Comparison/scripts/feynman_pysr-%j.log

PROJECT_ROOT="/scratch/pawsey0411/joatham/SR-XAI-Comparison"
export JULIA_DEPOT_PATH="/scratch/pawsey0411/joatham/.julia_depot"
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_ROOT"
if [ -e "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e

echo "Starting job at $(date)"
set +e
python -u feynmanPySR.py
STATUS=$?
set -e
echo "Finished job at $(date) with status $STATUS"
exit "$STATUS"
