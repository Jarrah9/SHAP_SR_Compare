#!/bin/bash
#SBATCH --job-name=bodyfat_prepare
#SBATCH --account=pawsey0411
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/pawsey0411/joatham/SR-XAI-Comparison/scripts/bodyfat_prepare-%j.log

PROJECT_ROOT="/scratch/pawsey0411/joatham/SR-XAI-Comparison"

cd "$PROJECT_ROOT"
if [ -e "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

DATA_FILE="$PROJECT_ROOT/data/bodyfat/bodyfat.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: body-fat dataset not found at $DATA_FILE"
    echo "Download bodyfat.csv locally from:"
    echo "https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset"
    echo "Then copy it to $DATA_FILE before resubmitting this job."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -e

echo "Starting job at $(date)"
set +e
python -u bodyfat_prepare_data.py
STATUS=$?
set -e
echo "Finished job at $(date) with status $STATUS"
exit "$STATUS"
