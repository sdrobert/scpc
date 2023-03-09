#! /usr/bin/env bash
#SBATCH --wait
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-100%1
#SBATCH --output=exp/slurm_logs/slurm-%A.out

# we don't rely on pytorch-lightning to requeue b/c CC doesn't allow requeuing
# via scontrol. Following https://docs.alliancecan.ca/wiki/Running_jobs, we
# create an array of jobs instead. If we can complete the run.sh call, we
# cancel all remaining elements in the array.

export NCCL_DEBUG=INFO

source scripts/cc_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

./run.sh "$@" -s -p "$P" -w "$W" -x "--no-progress-bar"
r=$?
sleep 5
scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
sleep 10
exit $r
