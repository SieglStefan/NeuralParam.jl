#!/bin/bash
#
# Launcher for evaluation jobs (from repo root):
#   ./results/1_ConstLinearLW/evaluation/launch.sh NAME [ARRAY]
#
# NAME  = evaluation name          (folder \evaluation\eval_NAME)
# ARRAY = which evaluation         (0 = evaluation.jl, 1 = evaluation_initialValue.jl,...)
# SUBARRAY = task                  (evaluation.jl: 1 = skill, 2 = stability, 3 = benchmark, 4 = ab, 5 = profile)
#
# Examples:
#   XXX/launch.sh test1 0              - all tasks from evaluation.jl
#   XXX/launch.sh test1 0 1            - task 1 from evaluation.jl
#   XXX/launch.sh test1 0 1,2          - task 1 and 2 from evaluation.jl
#   XXX/launch.sh test1 1 1            - task 1 from evaluation_initialValue.jl
#   (XXX = ./results/1_ConstLinearLW/evaluation)


NAME=$1
SCRIPT=${2:-0}
TASKS=${3:-1-4}

[ -z "$NAME" ] && { echo "ERROR: NAME is missing. Usage: launch.sh NAME SCRIPT [TASKS]" >&2; exit 1; }

RUNPATH="results/1_ConstLinearLW/evaluation/$NAME"
mkdir -p "$RUNPATH"

sbatch --job-name="CLLW_eval_${NAME}" \
       --array="$TASKS" \
       --output="$RUNPATH/slurm-%A_%a.out" \
       --error="$RUNPATH/slurm-%A_%a.err" \
       --export=ALL,EVA_NAME="$NAME",EVA_SCRIPT="$SCRIPT" \
       results/1_ConstLinearLW/evaluation/submit.sh
echo "-> evaluation/$NAME  (script $SCRIPT, parts $TASKS)"
