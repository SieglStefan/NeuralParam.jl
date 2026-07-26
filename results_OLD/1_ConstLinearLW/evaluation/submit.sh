#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail
module purge
module load julia/1.10.10

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1              


SCRIPTS=(evaluation.jl evaluation_ab.jl)
SCRIPT=${SCRIPTS[${EVA_SCRIPT:-0}]}

export TASK=${SLURM_ARRAY_TASK_ID:-0}     # welcher Teil

echo "Host $(hostname) | Script $SCRIPT | Part $TASK | EVA_NAME=${EVA_NAME:-}"
julia --project=. "results/1_ConstLinearLW/evaluation/$SCRIPT"
