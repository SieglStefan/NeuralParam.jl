#!/bin/bash
#SBATCH --job-name=training_neurallinearlw
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=results/2_NeuralLinearLW/training/%x-%j.out
#SBATCH --error=results/2_NeuralLinearLW/training/%x-%j.err

set -euo pipefail
module purge
module load julia/1.10.11

export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=1

echo "Host $(hostname) | Job $SLURM_JOB_ID | CPUs $SLURM_CPUS_PER_TASK"
julia --project=. results/2_NeuralLinearLW/training/training.jl