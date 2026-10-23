#!/usr/bin/env bash
# Batch version of your salloc line, plus requeue on preemption and enough CPUs for the numpy env.
#SBATCH --job-name=m4
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
# sbatch --export=ALL,DATASET=coco_search18 scripts/job.sbatch
bash scripts/run_all.sh          # finished caches/runs are skipped, so a requeued job resumes
