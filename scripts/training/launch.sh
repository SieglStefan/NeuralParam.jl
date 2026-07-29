#!/bin/bash
# Launch a training/calibration job (from repo root):
#   ./scripts/training/launch.sh <script-name-without-.jl> [ARRAY]
#   ./scripts/training/launch.sh 1_calibrate_const_linear 0-7
#
SCRIPT=${1:?"Usage: ./scripts/training/launch.sh <name-without-.jl> [ARRAY]"}
ARRAY=${2:-0}
mkdir -p slurm_logs
sbatch --array="$ARRAY" scripts/training/submit.sh "$SCRIPT"