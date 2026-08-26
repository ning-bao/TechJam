#!/bin/bash
# TC2 guide §7 — first GPU job: hardware/software probe.
#
# Before submitting, replace every /persistent/path placeholder and confirm the
# partition, QoS and module name with the §2 discovery commands. `mkdir -p logs`
# on the head node first: Slurm does not create missing log directories.
#
#   mkdir -p logs
#   sbatch cluster/probe_gpu.sbatch
#
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=aigc_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
umask 077

PROJECT_ROOT=/persistent/path/aigc-track5/repo
HF_CACHE_ROOT=/persistent/path/aigc-track5/huggingface
REPORT_DIR=/persistent/path/aigc-track5/runs/probe

module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate aigc-track5

# Set before Python starts: the HF libraries read these at import time.
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

mkdir -p "${REPORT_DIR}"
cd "${PROJECT_ROOT}"

hostname
date --iso-8601=seconds
module list
nvidia-smi

python -u - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "PyTorch cannot see the allocated GPU"
props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": props.name,
    "vram_gib": round(props.total_memory / 2**30, 2),
    "compute_capability": f"{props.major}.{props.minor}",
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
}, indent=2))
PY

# Loads the gated DINOv3-L/16 from the offline cache (falls back to
# DINOv2-with-registers-L and says so), then runs a 512px forward+backward.
# Exits non-zero if CUDA, the weights, or the pass fails.
python -u -m src.cluster_probe \
  --config configs/dinov3l512.yaml \
  --resolution 512 \
  --output "${REPORT_DIR}/probe_${SLURM_JOB_ID}.json"
