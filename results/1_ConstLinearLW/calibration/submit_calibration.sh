#!/bin/bash
#SBATCH --job-name=calib_constlinearlw
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-4                                            
#SBATCH --output=results/1_ConstLinearLW/calibration/%x-%A_%a.out  
#SBATCH --error=results/1_ConstLinearLW/calibration/%x-%A_%a.err

set -euo pipefail
module purge
module load julia/1.10.10

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1

echo "Host $(hostname) | Job $SLURM_JOB_ID | CPUs $SLURM_CPUS_PER_TASK"
julia --project=. results/1_ConstLinearLW/calibration/calibration.jl