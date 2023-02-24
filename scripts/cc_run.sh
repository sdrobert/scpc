#! /usr/bin/env bash
#SBATCH --wait
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=exp/slurm_logs/slurm-%j.out

source scripts/cc_env.sh

env
srun bash ./run.sh "$@" -x "--enable_progress_bar=False --devices=${SLURM_TASKS_PER_NODE:-1} --num_nodes=${SLURM_NNODES:-1} --num-workers=${SLURM_CPUS_PER_TASK:-4}"