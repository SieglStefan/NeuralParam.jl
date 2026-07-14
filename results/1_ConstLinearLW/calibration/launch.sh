#!/bin/bash
#
# Launcher for calibration jobs (from repo root):
#   ./results/1_ConstLinearLW/calibration/launch.sh NAME [ARRAY] [N_IC]
#
# NAME  = run name          (folder \calibration\calib_NAME)
# ARRAY = which variants    (see calibration.jl), e.g. 0 or 0-7
# N_IC  = default n_ic      (optional, 3rd arg), default: 5
#
# Examples:
#   ./results/1_ConstLinearLW/calibration/launch.sh default 0         - creates \calib_default with only task 0
#   ./results/1_ConstLinearLW/calibration/launch.sh initValue 0-7     - creates \calib_initValue with tasks 0 to 7


NAME=$1
ARRAY=${2:-0}
NIC=${3:-5}

[ -z "$NAME" ] && { echo "ERROR: NAME fehlt. Usage: launch.sh NAME [ARRAY] [N_IC]" >&2; exit 1; }

RUN="calib_${NAME}"
RUNPATH="results/1_ConstLinearLW/calibration/$RUN"
mkdir -p "$RUNPATH"

sbatch --job-name="CLLW_calib_${NAME:-run}" \
       --array="$ARRAY" \
       --output="$RUNPATH/slurm-%A_%a.out" \
       --error="$RUNPATH/slurm-%A_%a.err" \
       --export=ALL,RUN="$RUN",N_IC="$NIC" \
       results/1_ConstLinearLW/calibration/submit.sh
echo "-> $RUN  (n_ic default = $NIC, tasks $ARRAY)"
