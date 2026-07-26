#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=train
#SBATCH --output=slurm_logs/train-%A_%a.out
#SBATCH --error=slurm_logs/train-%A_%a.err
set -euo pipefail

SCRIPT=${1:?"Usage: sbatch --array=... scripts/train/submit.sh <name-without-.jl>"}

module purge
module load julia/1.10.10
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1

echo "Host $(hostname) | Task ${SLURM_ARRAY_TASK_ID:-0} | scripts/train/$SCRIPT.jl"
julia --project=. "scripts/train/$SCRIPT.jl"