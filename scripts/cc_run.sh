#! /usr/bin/env bash
#SBATCH --wait
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=exp/slurm_logs/slurm-%j.out

set -e

source scripts/cc_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

source ./run.sh "$@" -s -p "$P" -w "$W" -x "--no-progress-bar"
