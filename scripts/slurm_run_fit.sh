#! /usr/bin/env bash

#SBATCH --job-name=scpc-run-fit
#SBATCH --export=ALL
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=exp/slurm_logs/%j.out

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun bash ./run.sh -m "${1:-cpc.deft}" -o -x "--devices=${SBATCH_GPUS_PER_NODE} --enable_progress_bar=False --num_nodes=${SLURM_JOB_NUM_NODES} --num-workers=4"
