#!/bin/bash
#
# Launch rollout generation (from repo root):
#   ./scripts/rollout/launch.sh [ARRAY]
#
# ARRAY = which variants (see generate_rollout.jl), e.g. "0" or "0-2"

ARRAY=${1:-0}
mkdir -p slurm_logs

sbatch --job-name="rollout" \
       --array="$ARRAY" \
       --output="slurm_logs/rollout_%A_%a.out" \
       --error="slurm_logs/rollout_%A_%a.err" \
       scripts/rollout/submit.sh