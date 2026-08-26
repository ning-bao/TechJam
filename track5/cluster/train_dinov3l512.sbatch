#!/bin/bash
# TC2 guide section 9 - production training segment.
#
# Sized to the measured QoS (docs/cluster_profile.md): 1 GPU, 10 CPU, 30 GB and
# MaxWall 6h per user, so this asks for 8 CPU / 26 GB / 05:50:00 and leaves
# headroom. --max-wall-minutes 330 (5 h 30) is the primary safety mechanism:
# the application stops itself, checkpoints and exits 0 with ~20 min of
# allocation left. --signal=USR1@300 is the second line of defence, and only
# works because python runs as an srun step.
#
# MaxJobsPU=2 / MaxSubmitPU=2: at most two of these may be queued or running at
# once, so chain them two at a time, never pre-submit a long chain.
#
#   mkdir -p logs
#   FIRST=$(sbatch --parsable cluster/train_dinov3l512.sbatch)
#   sbatch --dependency=afterok:${FIRST} cluster/train_dinov3l512.sbatch
#
# Inspect the first segment and test resume before chaining further. Do not add
# --requeue until TC2 support confirms requeueing is enabled and tested.
#
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=26G
#SBATCH --time=05:50:00
#SBATCH --signal=USR1@300
#SBATCH --job-name=dinov3l512
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=/persistent/path/aigc-track5/repo
DATA_ROOT=/persistent/path/aigc-track5/data
HF_CACHE_ROOT=/persistent/path/aigc-track5/huggingface
RUN_ROOT=/persistent/path/aigc-track5/runs
RUN_ID=dinov3l512_seed0
RUN_DIR="${RUN_ROOT}/${RUN_ID}"

module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate aigc-track5

export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}/telemetry" "${RUN_DIR}/provenance"
cd "${PROJECT_ROOT}"

date --iso-8601=seconds
hostname
nvidia-smi
module list
git rev-parse HEAD || true
git status --short || true

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu \
  --format=csv \
  --loop=60 > "${RUN_DIR}/telemetry/gpu_${SLURM_JOB_ID}.csv" &
SMI_PID=$!
cleanup() { kill "${SMI_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# src.train refuses to start if the protected-set denylist is missing or
# incomplete, or if the train manifest intersects it (constraint C2).
# --resume auto also refuses a changed run definition unless overridden.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python -u -m src.train \
  --config configs/dinov3l512.yaml \
  --train-manifest "${DATA_ROOT}/manifests/train.parquet" \
  --dev-manifest "${DATA_ROOT}/manifests/dev.parquet" \
  --data-root "${DATA_ROOT}/raw" \
  --cache-dir "${DATA_ROOT}/cache/eval" \
  --run-dir "${RUN_DIR}" \
  --resume auto \
  --max-wall-minutes 330

# On a clean stop the segment exits 0 and writes segment_complete.json with
# stop_reason=wall_budget|signal:USR1|max_steps; "complete": true means the
# whole run finished and no successor is needed.
