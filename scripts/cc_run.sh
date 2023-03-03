#! /usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --signal=B:SIGUSR1@90
#SBATCH --output=exp/slurm_logs/slurm-%j.out

set -e
export NCCL_DEBUG=INFO

# we don't rely on pytorch-lightning to requeue b/c CC doesn't allow requeuing
# via scontrol. Following https://docs.alliancecan.ca/wiki/Running_jobs instead
do_requeue() {
    echo -e "SIGUSR1 passed. Requeueing job"
    sbatch scripts/cc_run.sh "${CC_RUN_OLD_ARGS[@]}"
    echo -e "Exiting"
    exit 1
}

source scripts/cc_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"
CC_RUN_OLD_ARGS=( "$@" )
trap do_requeue SIGUSR1

source ./run.sh "$@" -s -p "$P" -w "$W" # -x "--no-progress-bar"
