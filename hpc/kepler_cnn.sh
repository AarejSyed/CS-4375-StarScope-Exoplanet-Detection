#!/bin/bash

#SBATCH --job-name=kepler_cnn
#SBATCH --output=logs/kepler_cnn_%j.out
#SBATCH --error=logs/kepler_cnn_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-preempt
#SBATCH --mem=16G
#SBATCH --time=12:00:00

#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aarej.syed@utdallas.edu

set -e

# -----------------------------

# Variables

# -----------------------------

SCRIPT_NAME="starscope_hpc.py"
PROJECT_DIR="$HOME/kepler_cnn"
SCRATCH_DIR="$HOME/scratch/kepler_cnn_$SLURM_JOB_ID"
RESULTS_DEST_DIR="$PROJECT_DIR/$SLURM_JOB_ID"

ENV_PATH="/groups/yousefi/syed/wave_height_prediction_3_11"

LOG_PREFIX="[${SLURM_JOB_ID}]"

echo "$LOG_PREFIX Restart count: ${SLURM_RESTART_COUNT:-0}"

# -----------------------------

# Setup scratch

# -----------------------------

rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

cp -r "$PROJECT_DIR/$SCRIPT_NAME" "$SCRATCH_DIR"
cd $SCRATCH_DIR

# -----------------------------

# Activate environment

# -----------------------------

module load miniconda
source ~/.bashrc
conda activate "$ENV_PATH"

# -----------------------------

# Run experiment

# -----------------------------

echo "$LOG_PREFIX Running experiment"
python $SCRIPT_NAME

# -----------------------------

# Copy results back

# -----------------------------

echo "$LOG_PREFIX Syncing results"
mkdir -p "$RESULTS_DEST_DIR"
cp -v experiments.txt confusion_matrix.png roc_curve.png "$RESULTS_DEST_DIR"

# -----------------------------

# Cleanup

# -----------------------------

echo "$LOG_PREFIX Cleaning scratch"
cd ..
rm -rf "$SCRATCH_DIR"

echo "$LOG_PREFIX Job complete"

