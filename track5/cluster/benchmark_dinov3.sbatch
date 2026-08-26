#!/bin/bash
# TC2 guide section 8 - DINOv3-L/512 qualification benchmark on the L40S.
#
# Measured profile this job is written for (see docs/cluster_profile.md):
#   NVIDIA L40S, 46,068 MiB usable, compute capability 8.9 -> BF16 is native,
#   CUDA driver 12.8, QoS caps: 1 GPU, 10 CPU, 30 GB, MaxWall 6h,
#   MaxJobsPU=2 / MaxSubmitPU=2.
#
# Sweep: 512 px, BF16, micro-batches 8/4/2/1 (largest first so an OOM is
# followed by smaller batches), activation checkpointing OFF first and ON only
# for a configuration that OOMs or exceeds 90% peak allocated VRAM, inference
# and full forward/backward/optimizer, 20 warm-up + 100 measured steps, no
# torch.compile. Expected wall clock is well under the 30 min requested.
#
# Replace the /persistent/path placeholders and confirm partition/QoS/module
# with the section 2 discovery commands. `mkdir -p logs` before sbatch.
#
#   mkdir -p logs
#   sbatch cluster/benchmark_dinov3.sbatch
#
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=26G
#SBATCH --time=00:30:00
#SBATCH --job-name=aigc_bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=/persistent/path/aigc-track5/repo
DATA_ROOT=/persistent/path/aigc-track5/data
HF_CACHE_ROOT=/persistent/path/aigc-track5/huggingface
RESULT_DIR=/persistent/path/aigc-track5/runs/benchmark

module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate aigc-track5

export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

mkdir -p "${RESULT_DIR}/telemetry"
cd "${PROJECT_ROOT}"

date --iso-8601=seconds
hostname
nvidia-smi
module list

# Coarse job-level telemetry (section 13). The benchmark also samples
# utilization at 250 ms into the JSON, attributed by GPU UUID, so it stays
# correct under CUDA_VISIBLE_DEVICES remapping.
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu \
  --format=csv \
  --loop=30 > "${RESULT_DIR}/telemetry/gpu_${SLURM_JOB_ID}.csv" &
SMI_PID=$!
cleanup() { kill "${SMI_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# End-to-end rows use the real corpus when the manifest exists, otherwise
# deterministic synthetic images; either way the JSON records which.
MANIFEST_ARGS=()
if [[ -f "${DATA_ROOT}/manifests/train.parquet" ]]; then
  MANIFEST_ARGS=(--manifest "${DATA_ROOT}/manifests/train.parquet"
                 --data-root "${DATA_ROOT}/raw")
fi

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python -u scripts/benchmark_gpu.py \
  --model facebook/dinov3-vitl16-pretrain-lvd1689m \
  --fallback facebook/dinov2-with-registers-large \
  --resolutions 512 \
  --precisions bf16 \
  --micro-batches 8,4,2,1 \
  --modes train,inference \
  --data compute,dataloader \
  --checkpointing auto \
  --vram-headroom-pct 90 \
  --workers 8 \
  --warmup-steps 20 \
  --steps 100 \
  "${MANIFEST_ARGS[@]}" \
  --out "${RESULT_DIR}/benchmark_${SLURM_JOB_ID}.json"

# Then apply the section 8 go/no-go gates to the JSON:
#   peak_allocated_pct <= 90 and >= 8 train img/s   -> keep DINOv3-L/512
#   < 5 img/s after tuning, or repeated OOM         -> DINOv2-with-registers lane
#   inference >= 20 img/s -> the 15-condition matrix fits one resumable job
# Only if 512 fails the gates, rerun with --resolutions 448,384.
