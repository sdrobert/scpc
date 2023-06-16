#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-30%1
#SBATCH --output=exp/slurm_logs/slurm-%A.out
#SBATCH --open-mode=append

source scripts/cc_env.sh

P="${SLURM_TASKS_PER_NODE:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

./scripts/zrc_run.sh "$@" -s -p "$P" -w "$W"
r=$?
sleep 5
scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
sleep 10
exit $r
