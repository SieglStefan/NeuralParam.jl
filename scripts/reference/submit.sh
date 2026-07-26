#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail
module purge
module load julia/1.10.10          # <- match your cluster's Julia

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1

echo "Host $(hostname) | Task ${SLURM_ARRAY_TASK_ID:-0}"
julia --project=. scripts/reference/generate_reference.jl