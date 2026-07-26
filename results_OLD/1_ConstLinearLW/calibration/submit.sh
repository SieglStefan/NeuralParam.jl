#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail
module purge
module load julia/1.10.10

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1
export RUN="${RUN:-run_${SLURM_ARRAY_JOB_ID}}"

RUNPATH="results/1_ConstLinearLW/calibration/$RUN"

# Move slurm .out and .err files into the respective folders
move_logs() {
    local dir="$RUNPATH/task_${SLURM_ARRAY_TASK_ID}"
    local base="$RUNPATH/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    [ -d "$dir" ] || return 0
    mv "$base.out" "$dir/" 2>/dev/null || true
    mv "$base.err" "$dir/" 2>/dev/null || true
}
trap move_logs EXIT

echo "Host $(hostname) | ArrayJob $SLURM_ARRAY_JOB_ID | Task $SLURM_ARRAY_TASK_ID | RUN=$RUN"
julia --project=. results/1_ConstLinearLW/calibration/calibration.jl
