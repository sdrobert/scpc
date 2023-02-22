#! /usr/bin/env bash

#SBATCH --job-name=scpc-run-fit
#SBATCH --export=ALL
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=exp/slurm_logs/%j.out
#SBATCH --requeue

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun bash ./run.sh -m "${1:-cpc.deft}" -o -x "--devices=4 --enable_progress_bar=False --num_nodes=1 --num-workers=4"
