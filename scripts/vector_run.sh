#! /usr/bin/env bash

#SBATCH --wait
#SBATCH --partition=t4v2,rtx6000
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp/slurm_logs/%j.out

set -e

source scripts/vector_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

source ./run.sh "$@" -s -p "$P" -w "$W" -x "--no-progress-bar"
