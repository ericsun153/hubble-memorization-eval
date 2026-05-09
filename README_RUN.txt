Pipeline location:
  /gpfs/radev/project/q_chen/zs437/hubble_memorization_pipeline

Scratch/cache:
  /gpfs/radev/scratch/q_chen/zs437

First install/use env:
  source /gpfs/radev/project/q_chen/zs437/miniconda3/etc/profile.d/conda.sh
  conda activate hubble
  pip install -r requirements.txt

Run data preparation:
  sbatch slurm/prepare_data.sbatch

Check jobs:
  squeue -u zs437

Start with 1B models only:
  sbatch slurm/run_1b_first.sbatch

Run all 8 core Hubble models:
  sbatch slurm/run_one_model.sbatch

Aggregate results:
  sbatch slurm/aggregate.sbatch

Important outputs:
  data/eval_questions.jsonl
  results/raw/*.jsonl
  results/summary/model_task_summary.csv
  results/summary/standard_vs_perturbed.csv
  results/summary/all_results.parquet

If partition name is wrong:
  edit #SBATCH --partition=gpu or #SBATCH --partition=standard in slurm/*.sbatch

If HF access/token is needed:
  huggingface-cli login

Smoke test without dataset:
  python scripts/run_local_smoke_test.py

Then test one small model manually:
  python scripts/run_generation_eval.py \
    --model_name hubble_1b_100b_standard \
    --model_id allegrolab/hubble-1b-100b_toks-standard-hf \
    --revision 48000 \
    --eval_path data/eval_questions_smoke.jsonl \
    --out_dir results/raw \
    --max_examples 2
