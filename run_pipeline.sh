#!/bin/bash
set -euo pipefail

cd /gpfs/radev/project/q_chen/zs437/hubble_memorization_pipeline

source /gpfs/radev/project/q_chen/zs437/miniconda3/etc/profile.d/conda.sh
conda activate hubble

echo "Step 1: prepare data"
DATA_JOB=$(sbatch --parsable slurm/prepare_data.sbatch)
echo "Data job id: ${DATA_JOB}"

echo "Step 2: run 1B after data finishes"
ONEB_JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} slurm/run_1b_first.sbatch)
echo "1B job id: ${ONEB_JOB}"

echo "Step 3: run 8B after data finishes"
EIGHTB_JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} slurm/run_8b_multi_gpu.sbatch)
echo "8B job id: ${EIGHTB_JOB}"

echo "Step 4: aggregate after 1B and 8B finish"
AGG_JOB=$(sbatch --parsable --dependency=afterok:${ONEB_JOB}:${EIGHTB_JOB} slurm/aggregate.sbatch)
echo "Aggregate job id: ${AGG_JOB}"

echo "Submitted all jobs."
echo "Check status with:"
echo "squeue -u zs437"