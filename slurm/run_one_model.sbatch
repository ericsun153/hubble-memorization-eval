#!/bin/bash
#SBATCH --job-name=hubble_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

set -euo pipefail

PROJECT_DIR="/gpfs/radev/project/q_chen/zs437/hubble_memorization_pipeline"
SCRATCH_DIR="/gpfs/radev/scratch/q_chen/zs437"

cd "${PROJECT_DIR}"

source /gpfs/radev/project/q_chen/zs437/miniconda3/etc/profile.d/conda.sh
conda activate hubble

export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
export HF_DATASETS_CACHE="${SCRATCH_DIR}/hf_datasets_cache"
export TORCH_HOME="${SCRATCH_DIR}/torch_cache"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" results/raw logs

MODEL_JSON=$(python - <<PY
import json
models = json.load(open("configs/models_core.json"))
idx = int("${SLURM_ARRAY_TASK_ID}")
print(json.dumps(models[idx]))
PY
)

MODEL_NAME=$(python - <<PY
import json
m = json.loads('''${MODEL_JSON}''')
print(m["name"])
PY
)

MODEL_ID=$(python - <<PY
import json
m = json.loads('''${MODEL_JSON}''')
print(m["model_id"])
PY
)

REVISION=$(python - <<PY
import json
m = json.loads('''${MODEL_JSON}''')
print(m["revision"])
PY
)

echo "Running ${MODEL_NAME}"
echo "Model ID: ${MODEL_ID}"
echo "Revision: ${REVISION}"

python scripts/run_generation_eval.py \
  --model_name "${MODEL_NAME}" \
  --model_id "${MODEL_ID}" \
  --revision "${REVISION}" \
  --eval_path data/eval_questions.jsonl \
  --out_dir results/raw \
  --max_new_tokens 80 \
  --temperature 0.0
