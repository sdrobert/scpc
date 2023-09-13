#! /usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --output=exp/slurm_logs/%j.out
#SBATCH --exclude=gpu081,gpu0[90-99]

set -e

source scripts/vector_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
./run.sh -zs -p "$P" -w "$W" -x "--quiet" "$@"
