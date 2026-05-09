#!/bin/bash
#SBATCH --job-name=hubble_agg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_DIR="/gpfs/radev/project/q_chen/zs437/hubble_memorization_pipeline"

cd "${PROJECT_DIR}"

source /gpfs/radev/project/q_chen/zs437/miniconda3/etc/profile.d/conda.sh
conda activate hubble

python scripts/aggregate_results.py \
  --raw_dir results/raw \
  --out_dir results/summary
