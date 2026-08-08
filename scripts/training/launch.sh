#!/bin/bash
# Launch a training/calibration job (from repo root):
#   bash scripts/training/launch.sh <script-name-without-.jl> [ARRAY]
#   bash scripts/training/launch.sh 1_constLW 0-7
#
SCRIPT=${1:?"Usage: ./scripts/training/launch.sh <name-without-.jl> [ARRAY]"}

ARRAY=${2:-0}
mkdir -p slurm_logs

sbatch --job-name="$SCRIPT"  \
       --array="$ARRAY"  \
       --output="slurm_logs/${SCRIPT}_%A_%a.out" \
       --error="slurm_logs/${SCRIPT}_%A_%a.err" \
       scripts/training/submit.sh "$SCRIPT"