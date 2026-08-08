#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00

set -euo pipefail

scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID:-0}" || true

module purge
module load julia/1.10.10

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1

echo "Host $(hostname) | Task ${SLURM_ARRAY_TASK_ID:-0}"
julia --project=. scripts/rollout/generate_rollout.jl