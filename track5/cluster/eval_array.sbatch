#!/bin/bash
# TC2 guide section 11 - 15-condition robustness matrix.
#
# THE QoS ALLOWS TWO JOBS: MaxJobsPU=2 and MaxSubmitPU=2 (docs/cluster_profile.md).
# Array tasks count against those limits on most Slurm configurations, so this
# defaults to THREE condition groups with --array=0-2%1: one task runs at a
# time, and the whole array occupies one submission slot if arrays are counted
# as a single job, or is rejected outright if they are counted per task. Confirm
# the array policy before the first submission.
#
#   mkdir -p logs
#   sbatch cluster/eval_array.sbatch                 # 3 groups, 1 at a time
#   sbatch --array=0-14%1 cluster/eval_array.sbatch  # per-condition tasks
#   ARRAY_MODE=conditions sbatch --array=0-14%1 cluster/eval_array.sbatch
#
# If arrays are disallowed entirely, submit a single job that loops over every
# condition - src.evaluate --condition all --resume is resumable per condition,
# so a job that dies partway through costs only the unfinished rows:
#
#   sbatch --array=0 --export=ALL,ARRAY_MODE=all cluster/eval_array.sbatch
#
# Never submit 15 independent jobs.
#
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=26G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --job-name=aigc_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
umask 077

PROJECT_ROOT=/persistent/path/aigc-track5/repo
DATA_ROOT=/persistent/path/aigc-track5/data
HF_CACHE_ROOT=/persistent/path/aigc-track5/huggingface
RUN_DIR=/persistent/path/aigc-track5/runs/frozen_model
CHECKPOINT="${RUN_DIR}/best.pt"
MANIFEST="${DATA_ROOT}/manifests/dev.parquet"

GROUPS=(clean_jpeg blur_resize noise_color_crop)
CONDITIONS=(
  clean jpeg_90 jpeg_70 jpeg_50 jpeg_30
  blur_05 blur_10 blur_20
  resize_050 resize_025
  noise_002 noise_005 noise_010
  jitter_pm20 crop_80
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${ARRAY_MODE:-groups}" in
  all)        CONDITION="all" ;;
  conditions) CONDITION="${CONDITIONS[${TASK_ID}]}" ;;
  *)          CONDITION="group:${GROUPS[${TASK_ID}]}" ;;
esac

module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate aigc-track5

export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}/metrics"
cd "${PROJECT_ROOT}"

date --iso-8601=seconds
hostname
nvidia-smi
echo "condition selector: ${CONDITION}"

# --resume skips any condition whose completion marker still matches this
# checkpoint/config/manifest, so a requeued or re-submitted task only does the
# rows that are actually outstanding. Metrics are written per condition, not at
# process end.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python -u -m src.evaluate \
  --config configs/dinov3l512.yaml \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --condition "${CONDITION}" \
  --data-root "${DATA_ROOT}/raw" \
  --cache-dir "${DATA_ROOT}/cache/eval" \
  --denylist-dir "${DATA_ROOT}/denylist" \
  --resume \
  --output "${RUN_DIR}/metrics/"

# The protected COCO-val2017 x DALL-E-Advanced benchmark is NOT run here. It
# needs a frozen model, a CPU manifest audit first, and an explicit
# --protected-run, and it happens exactly once.
