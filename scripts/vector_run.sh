#! /usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --output=exp/slurm_logs/%j.out

set -e

source scripts/vector_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

./run.sh "$@" -s -p "$P" -w "$W" -x "--no-progress-bar"
