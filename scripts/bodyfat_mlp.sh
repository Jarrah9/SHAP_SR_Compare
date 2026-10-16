#!/bin/bash
#SBATCH --job-name=bodyfat_mlp
#SBATCH --account=pawsey0411
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/pawsey0411/joatham/SR-XAI-Comparison/scripts/bodyfat_mlp-%j.log

set -euo pipefail
PROJECT_ROOT="/scratch/pawsey0411/joatham/SR-XAI-Comparison"
cd "$PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found at $PROJECT_ROOT/.venv" >&2
    exit 1
fi
source "$PROJECT_ROOT/.venv/bin/activate"

# Independent CV fits share the allocation. Each fit uses one BLAS/OpenMP
# thread, so 32 workers cannot accidentally launch 32 threads each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting body-fat MLP at $(date), job $SLURM_JOB_ID, CPUs $SLURM_CPUS_PER_TASK"
trap 'status=$?; echo "Finished body-fat MLP at $(date), status $status"' EXIT
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    python -u bodyfatMLP.py "$@"
