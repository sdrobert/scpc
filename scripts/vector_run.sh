#! /usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --output=exp/slurm_logs/%j.out
#SBATCH --exclude=gpu081,gpu0[90-99]

# Vector institute run script wrapper for sbatch. Uses preemption to handle
# dynamic job lengths. Sets run flags according to environment variables
#
# e.g.
# sbatch --gres=gpu:t4:1 ./scripts/cc_run.sh -m cpc.small -v 1
# sbatch ./scripts/cc_run.sh ./scripts/zrc_run.sh -m cpc.small -v 1

set -e

source scripts/vector_env.sh

P="${SLURM_NTASKS:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

script="./run.sh"
if [ -f "$1" ]; then
  script="$1"
  shift
fi

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
"$script" -zs -p "$P" -w "$W" -x "--quiet" "$@"
