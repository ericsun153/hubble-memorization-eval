#!/bin/bash
#SBATCH --job-name=hubble_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="/gpfs/radev/project/q_chen/zs437/hubble_memorization_pipeline"
SCRATCH_DIR="/gpfs/radev/scratch/q_chen/zs437"

cd "${PROJECT_DIR}"

source /gpfs/radev/project/q_chen/zs437/miniconda3/etc/profile.d/conda.sh
conda activate hubble

export HF_HOME="${SCRATCH_DIR}/hf_cache"
export HF_DATASETS_CACHE="${SCRATCH_DIR}/hf_datasets_cache"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${SCRATCH_DIR}/hubble_raw" data logs

python scripts/discover_hubble_datasets.py \
  --out_dir "${SCRATCH_DIR}/hubble_raw" \
  --max_rows_per_dataset 2000

python scripts/build_eval_questions.py \
  --raw_dir "${SCRATCH_DIR}/hubble_raw" \
  --out_path data/eval_questions.jsonl \
  --max_items 1000 \
  --prefix_words 50 \
  --target_words 50
