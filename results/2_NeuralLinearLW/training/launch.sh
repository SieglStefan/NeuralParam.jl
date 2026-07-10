#!/bin/bash

NAME=$1
ARRAY=${2:-0}
NIC=${3:-5}                                           # Default n_ic (3. Argument)
RUN="run_${NAME:+${NAME}_}$(date +%Y-%m-%d_%H-%M-%S)"
RUNPATH="results/2_NeuralLinearLW/training/$RUN"
mkdir -p "$RUNPATH"

sbatch --job-name="$NLLW_train_${NAME:-run}" --array="$ARRAY" \
       --output="$RUNPATH/slurm-%A_%a.out" \
       --error="$RUNPATH/slurm-%A_%a.err" \
       --export=ALL,RUN="$RUN",N_IC="$NIC" \
       results/2_NeuralLinearLW/training/submit.sh
echo "-> $RUN  (n_ic default = $NIC)"