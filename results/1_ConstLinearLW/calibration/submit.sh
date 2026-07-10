#!/bin/bash
#SBATCH --job-name=calib_constlinearlw
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0                                                  # Default: nur task 0 (single)
#SBATCH --output=results/1_ConstLinearLW/calibration/%x-%A_%a.out
#SBATCH --error=results/1_ConstLinearLW/calibration/%x-%A_%a.err

set -euo pipefail
module purge
module load julia/1.10.10

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1
export RUN="${RUN:-run_${SLURM_ARRAY_JOB_ID}}"    # ✅ nutzt RUN vom Launcher; sonst Job-ID

echo "Host $(hostname) | ArrayJob $SLURM_ARRAY_JOB_ID | Task $SLURM_ARRAY_TASK_ID"
julia --project=. results/1_ConstLinearLW/calibration/calibration.jl