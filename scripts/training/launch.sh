#!/bin/bash
# Launch a training/calibration job (from repo root):
#   ./scripts/train/launch.sh <script-name-without-.jl> [ARRAY]
#   ./scripts/train/launch.sh 1_calibrate_const_linear 0-7
#
SCRIPT=${1:?"Usage: ./scripts/train/launch.sh <name-without-.jl> [ARRAY]"}
ARRAY=${2:-0}
mkdir -p logs
sbatch --array="$ARRAY" scripts/train/submit.sh "$SCRIPT"