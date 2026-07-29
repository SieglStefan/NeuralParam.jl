#!/bin/bash
#
# Launch reference generation (from repo root):
#   ./scripts/reference/launch.sh [ARRAY]
#
# ARRAY = which variants (see generate_reference.jl), e.g. "0" or "0-2" (default: "0")

ARRAY=${1:-0}
mkdir -p slurm_logs

sbatch --job-name="ref" \
       --array="$ARRAY" \
       --output="slurm_logs/ref_%A_%a.out" \
       --error="slurm_logs/ref_%A_%a.err" \
       scripts/reference/submit.sh