# TC2 GPU Cluster Guide for the Track 5 AIGC Detector

**Purpose:** run DINOv3-L/512 training, robustness evaluation, calibration, and final inference safely on the university's CCDS GPU Cluster TC2.

**Project assumptions:** single-image authentic-vs-AIGC classification; `facebook/dinov3-vitl16-pretrain-lvd1689m` is the primary backbone; the complete submission remains below 2B parameters; COCO val2017 and the WildFake DALL·E Advanced subset are excluded from training; and every run must be reproducible and resumable.

**Source status:** the local TC2 user guide was updated 13 August 2025. Cluster policy and hardware can change. Commands marked “discover” below override the examples in this document.

## 1. Decisions at a glance

| Decision | Requirement |
|---|---|
| Where work runs | Never compile, preprocess, train, or evaluate on the head node. Use Slurm batch jobs. The head node is for editing, lightweight file management, and `sbatch`/status commands. |
| Default allocation | Design for **one GPU, one node, at most 10 CPU cores, 30 GB host RAM, and six hours** because those are the `normal` QoS examples in the TC2 guide. Query the real limits before submission. |
| Primary model | Qualify DINOv3-L/16 at **512×512** with a 15-minute benchmark job. Do not estimate its runtime from consumer-GPU anecdotes. |
| Precision | BF16 when the assigned GPU supports it; otherwise FP16 with `GradScaler`. Do not silently fall back to FP32. |
| Memory strategy | Start with micro-batch 1 or 2, activation checkpointing, native PyTorch scaled-dot-product attention, and gradient accumulation. Peak allocated VRAM must stay below 90% in the qualification run. |
| Job length | Request 5 h 50 min under a six-hour QoS and make the program stop, checkpoint, and exit by 5 h 30 min. Long training is a sequence of resumable jobs, not one fragile job. |
| Storage | Raw data is immutable. Checkpoints and manifests live on persistent shared storage. Node-local storage is disposable and only used for staged shards when the site provides it. |
| Model/cache access | Pre-cache gated DINOv3 weights. Compute jobs should work with `HF_HUB_OFFLINE=1`; no Hugging Face token may appear in scripts or logs. |
| Protected benchmark | Run the COCO-val2017-vs-DALL·E-Advanced benchmark only after model, temperature, and threshold are frozen. Assert zero train/calibration overlap first. |
| Multi-GPU | Do not request more than one GPU unless `mytcinfo`/`sacctmgr` explicitly grants it. A node containing 4–8 cards does not imply that the user may allocate them. |

**Recommendation:** treat the measured qualification job as the only valid answer to “how long will DINOv3-L/512 take?” The cluster may make it substantially faster than a 24 GB consumer GPU, but the unknown GPU model, VRAM, queue policy, and storage bandwidth dominate the answer.

## 2. Mandatory discovery before any GPU work

Run these read-only commands after SSH login. Replace `<username>` and confirm the partition name rather than assuming it.

```bash
whoami
mytcinfo
sacctmgr show user <username> withassoc format=user,account,partition,qos
sacctmgr show qos <your-qos> withassoc \
  format=name%20,MaxTRESPU%50,MaxJobsPU%12,MaxSubmitPU%14,MaxWall%12
scontrol show partition MGPU-TC2
sinfo -N -p MGPU-TC2 -o "%N %t %G %m %c %f"
module avail
module spider anaconda
module spider cuda
quota -s
df -h .
```

Record the answers in `docs/cluster_profile.md`:

```text
Date checked:
Cluster / partition:
Account and QoS:
Max GPUs per user:
Max concurrent jobs / submitted jobs:
Max CPU cores and host RAM across the user's jobs:
Max wall time:
GPU models and VRAM that can be allocated:
Persistent project/data path and quota:
Scratch or node-local path, quota, and purge policy:
Internet access from compute nodes:
Job arrays allowed:
Dependencies allowed:
Requeue and pre-timeout signals allowed:
Approved Anaconda/Python module:
Driver version / supported CUDA runtime:
```

The following require an administrator answer if the commands do not reveal them:

1. Is there a CPU partition for manifest construction, archive extraction, hashing, and augmentation-cache generation?
2. May package environments and gated model weights be downloaded from the head node, or must that happen in a provisioning job?
3. Which storage is persistent and backed up, and which storage is deleted at job end or purged periodically?
4. Are `--signal`, `--requeue`, job dependencies, and arrays enabled for the assigned QoS?
5. Can the team obtain a temporary higher-wall-time or multi-GPU QoS for the hackathon?

**Recommendation:** do not submit a long job until the completed profile proves the request is within the *aggregate per-user* QoS. Two individually valid jobs can collectively exceed the user's CPU, RAM, or GPU cap.

## 3. Split the project into cluster jobs

| Workload | Resource type | Suggested limit | Output that must persist |
|---|---:|---:|---|
| Manifest, hashes, denylist, metadata audit | CPU batch partition | 2–4 CPU, 8–16 GB | Immutable CSV/Parquet manifests and hashes |
| Archive extraction or sharding | CPU batch partition | 4–8 CPU, 16–24 GB | Tar/WebDataset shards plus index |
| VAE reconstruction data, if retained in the plan | One GPU | Measured; segment if needed | Completed shards and a progress ledger |
| Environment/GPU probe | One GPU | 15 min, 4 CPU, 16 GB | Hardware/software report |
| DINOv3-L/512 benchmark | One GPU | 15–30 min, 8 CPU, 24–26 GB | Images/s, step time, peak VRAM, GPU trace |
| Frozen-feature baseline | One GPU | 1–3 h | Cached features, head, metrics |
| End-to-end DINOv3 training | One GPU | Repeated 5 h 50 min jobs | Atomic checkpoints, metrics, run metadata |
| Dev robustness matrix | One GPU | One job if projected <5.5 h; otherwise split | One result row per completed condition |
| Temperature scaling/reporting | CPU or short GPU job | <1 h | Frozen temperature, threshold, plots |
| Final protected inference | One GPU | Based on measured throughput + 20% margin | Predictions JSON, matrix CSV, run receipt |

Do not generate data, launch notebooks, or run “quick” Python tests on the head node. If TC2 has no CPU-only partition, ask the administrator which batch resource to use rather than silently doing CPU work on the login host.

**Recommendation:** create independent, idempotent entry points for `prepare_manifest`, `benchmark`, `train`, `evaluate`, and `predict`. Every entry point must resume from existing outputs and refuse to overwrite a completed protected run.

## 4. Environment and dependency requirements

### 4.1 Create one pinned environment

TC2 uses environment modules and does not permit `sudo` or installation into system directories. Discover the current module name first; `anaconda/25.5.1` is only the version shown in the 2025 guide.

```bash
module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda create --name aigc-track5 python=3.11 -y
conda activate aigc-track5
```

Install binary packages with Conda where practical and use `pip` only inside this environment. Pin every direct dependency. Do not modify the base environment. Do not add FlashAttention or xFormers unless the native PyTorch implementation is demonstrably too slow: locally compiled CUDA extensions add failure modes and can violate the head-node policy.

After installation, persist both human-readable and exact locks:

```bash
conda env export --from-history > environment.from-history.yml
conda list --explicit > environment.conda-lock.txt
python -m pip freeze > environment.pip-freeze.txt
```

If environment creation is not allowed on the head node, put these commands in an administrator-approved CPU provisioning job. Never spend a long GPU allocation solving an environment repeatedly.

### 4.2 Let PyTorch own the CUDA runtime by default

Most current PyTorch Conda packages/wheels include their CUDA user-space runtime; the cluster supplies the NVIDIA driver. Loading a second, incompatible CUDA module can create confusing library conflicts. Therefore:

1. Install a PyTorch build compatible with the cluster driver.
2. Do **not** load `cuda/12.8.0` merely because it appears in the old guide.
3. Load a CUDA toolkit module only when building an approved CUDA extension, and match its version to the PyTorch build.
4. Verify the actual combination in a compute job with `nvidia-smi` and the probe in §7.

### 4.3 Cache gated weights once

Choose a persistent cache path with enough quota. Do not use an unknown node-local path.

```bash
export HF_HOME=/persistent/path/aigc-track5/huggingface
export HF_HUB_CACHE=/persistent/path/aigc-track5/huggingface/hub
hf auth login
hf download facebook/dinov3-vitl16-pretrain-lvd1689m
```

Run the download only where the university permits network activity. Store the token through the Hugging Face credential mechanism or a protected environment/secret facility; never commit it, put it in `#SBATCH` directives, or echo it to a log. Once the cache is complete, compute jobs set:

```bash
export HF_HOME=/persistent/path/aigc-track5/huggingface
export HF_HUB_CACHE=/persistent/path/aigc-track5/huggingface/hub
export HF_HUB_OFFLINE=1
```

Hugging Face reads these variables when its libraries are imported, so set them before Python starts.

**Recommendation:** keep DINOv2-with-registers cached as an ungated fallback. A gate, licence, or compute-node network problem must not block the critical path.

## 5. Data and storage layout

Use explicit persistent paths agreed with the administrator. A recommended logical layout is:

```text
/persistent/path/aigc-track5/
├── data/
│   ├── raw/                 # read-only after verification
│   ├── manifests/           # paths, labels, source, generator, SHA-256
│   └── shards/              # optional 0.5–2 GB sequential-read shards
├── huggingface/
├── runs/
│   └── <run_id>/
│       ├── checkpoints/
│       ├── metrics/
│       ├── provenance/
│       └── telemetry/
└── protected_outputs/       # write-once final artifacts
```

Requirements:

- Keep raw downloads immutable and record archive checksums.
- Use manifests to define every split. Include `image_id`, normalized path, label, dataset, source, generator family, original dimensions, encoding, byte size, and content hash.
- Maintain a hash denylist for all 4,998 COCO val2017 real images and 8,843 WildFake DALL·E Advanced fakes. Training and calibration jobs must abort if their manifest intersects it.
- Avoid millions of small random file reads over shared storage. Prefer indexed tar/WebDataset shards when storage profiling shows metadata I/O is a bottleneck.
- If `${SLURM_TMPDIR}` exists, stage only the hot shards needed by the current job, verify the copy, and copy checkpoints/results back to persistent storage before exit. Treat `${SLURM_TMPDIR}` as disposable.
- Never place the only copy of a checkpoint on node-local storage.
- Budget storage before download: raw archives + extracted/sharded copy + caches + at least 15–25 GB per active full-fine-tune run for rotating checkpoints and optimizer state. Measure the real checkpoint size after the smoke test.

**Recommendation:** do not duplicate the full 150 GB corpus into node-local storage per job. Stage a bounded shard window or stream from shared storage; benchmark both for 100–200 steps and retain the faster stable option.

## 6. Required project interfaces

The repository should expose these non-interactive commands before long jobs begin:

```text
python -m src.cluster_probe
python -m src.benchmark_train --config <yaml> --steps <n> --warmup-steps <n>
python -m src.train --config <yaml> --run-dir <path> --resume auto --max-wall-minutes <n>
python -m src.evaluate --config <yaml> --condition <name> --resume --output <path>
python -m src.predict --checkpoint <path> --manifest <path> --output <json>
```

Each must:

- return a non-zero exit code on failure;
- print the resolved configuration and output path;
- use `python -u` or flush logs continuously;
- write partial output atomically (`.tmp` then rename);
- be safe to rerun;
- record the Git commit, manifest hash, configuration hash, seed, package lock hash, Slurm job ID, node, GPU, driver, and PyTorch/CUDA versions.

`src.predict` must emit exactly the organiser's contract, for example:

```json
{"image_path":"relative/or/required/path.jpg","pred":0.873214}
```

where `pred` is the calibrated probability of AIGC, not a raw logit.

**Recommendation:** make this interface a CI gate using a 16-image fixture. Cluster scripts then remain thin launchers instead of containing experiment logic.

## 7. First GPU job: hardware and software probe

Create the log directory on the head node before `sbatch`; Slurm does not create missing parent directories for log paths.

```bash
mkdir -p logs
sbatch cluster/probe_gpu.sbatch
```

Example `cluster/probe_gpu.sbatch`:

```bash
#!/bin/bash
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

module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate aigc-track5

export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

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
report = {
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": props.name,
    "vram_gib": round(props.total_memory / 2**30, 2),
    "compute_capability": f"{props.major}.{props.minor}",
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
}
print(json.dumps(report, indent=2))
PY

python -u -m src.cluster_probe
```

Replace the module version, QoS, and paths with the discovered values. Do not add `--nodelist`; letting Slurm choose the node reduces queue time and reveals the hardware mix that the project must tolerate.

After the job starts, identify its assigned card and inspect the result:

```bash
squeue -u <username>
scontrol -d show jobid <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
seff <jobid>
```

**Pass gate:** CUDA is visible; model weights load offline; the assigned GPU name/VRAM is recorded; BF16 or FP16 works; a forward and backward pass completes; and no token or private path is leaked in logs.

## 8. DINOv3-L/512 qualification benchmark

DINOv3-L uses 16×16 patches, so 512 is a valid exact size. Do not substitute 504: a patch-16 implementation may crop or reject it. Valid nearby sizes include 384, 448, 480, 496, and 512.

Benchmark the **real training path**: real decoder, the production stochastic augmentation pipeline, mixed precision, optimizer, gradient accumulation, and dataloader. A bare backbone forward pass is not a training estimate.

For each candidate setting, use 20 warm-up steps and at least 100 timed optimizer steps. Record:

- assigned GPU and VRAM;
- micro-batch and gradient-accumulation factor;
- effective batch size;
- input size;
- precision;
- activation checkpointing state;
- median and p95 data time and step time;
- processed images/s, counting both views if paired-view consistency is enabled;
- peak allocated and reserved CUDA memory;
- host MaxRSS and CPU efficiency;
- GPU and memory utilization from the telemetry log.

Test in this order:

1. 512 px, BF16, micro-batch 1, activation checkpointing on.
2. Increase micro-batch to 2 if peak allocated VRAM remains below 80%.
3. Compare dataloader workers 2, 4, and at most the allocated CPU count.
4. If too slow, compare 448 then 384 before changing the backbone.
5. Only after the best stable setting is found, test disabling activation checkpointing; it trades more memory for speed.

### Runtime conversion

Use measured end-to-end throughput:

```text
epoch_hours = number_of_processed_training_images / (images_per_second × 3600)
matrix_hours = 207,615 / (inference_images_per_second × 3600)
```

“Processed images” includes every view. A paired-view loss over 320k source images processes roughly 640k images unless views share computation.

| Measured training throughput | 120k images | 150k images | 320k images |
|---:|---:|---:|---:|
| 5 img/s | 6.7 h | 8.3 h | 17.8 h |
| 8 img/s | 4.2 h | 5.2 h | 11.1 h |
| 10 img/s | 3.3 h | 4.2 h | 8.9 h |
| 15 img/s | 2.2 h | 2.8 h | 5.9 h |

| Measured inference throughput | Full 15-condition, 13,841-image matrix |
|---:|---:|
| 10 img/s | 5.77 h before overhead |
| 15 img/s | 3.84 h before overhead |
| 20 img/s | 2.88 h before overhead |
| 25 img/s | 2.31 h before overhead |

Add at least 15–20% for startup, I/O, metric aggregation, and checkpoint/result writes.

### Go/no-go gates

| Observation | Decision |
|---|---|
| Peak allocated VRAM ≤90%, no OOM, ≥8 train img/s | Keep DINOv3-L/512. A 320k-image epoch is about 11.1 h or less, so segment it across two six-hour allocations. |
| Stable at 5–8 img/s | Still feasible for 120–150k images; a 320k epoch requires roughly 2–3 allocations. Start only if queue time and deadline permit. |
| <5 img/s but model quality is promising | Reduce to 448 or 384, remove paired-view consistency, or freeze early blocks. Rebenchmark. |
| <4 img/s after tuning, or repeated OOM | Switch to DINOv2-with-registers-B/L or a parameter-efficient/partial-unfreeze lane. Do not burn the deadline on a prestige backbone. |
| Inference ≥20 img/s | Full robustness matrix fits in about 2.9 h before overhead; one resumable job is reasonable. |
| Inference <12 img/s | Split evaluation by condition. Do not risk the full matrix in one six-hour job. |

**Recommendation:** on an A100/H100-class allocation the plan may be much faster than the consumer-GPU estimate; on an older or bandwidth-limited card it may be slower. Commit to DINOv3-L/512 only after this table is populated with TC2 measurements.

## 9. Production training job

This template deliberately stays under the example six-hour wall limit and uses fewer than the example 10 CPU/30 GB aggregate caps. Adjust only after checking the actual QoS.

Example `cluster/train_dinov3l512.sbatch`:

```bash
#!/bin/bash
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
git rev-parse HEAD
git status --short
module list

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu \
  --format=csv \
  --loop=60 > "${RUN_DIR}/telemetry/gpu_${SLURM_JOB_ID}.csv" &
SMI_PID=$!

cleanup() {
  kill "${SMI_PID}" 2>/dev/null || true
}

trap cleanup EXIT

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python -u -m src.train \
  --config configs/dinov3l512.yaml \
  --train-manifest "${DATA_ROOT}/manifests/train.parquet" \
  --run-dir "${RUN_DIR}" \
  --resume auto \
  --max-wall-minutes 330
```

The `--max-wall-minutes 330` application timer is the primary safety mechanism. `--signal=USR1@300` asks Slurm to signal job steps five minutes before the allocation ends; launching Python as an `srun` step ensures that its handler receives the signal. (`srun` inside an `sbatch` allocation is appropriate; the TC2 warning concerns interactive command-line `srun` tied to an SSH session.) The signal is a second line of defence and is valid only when `src.train` implements the contract below. Do not add `--requeue` until TC2 support confirms that requeueing is enabled and tested.

Submit one segment, inspect it, then chain the next only when resume has been tested:

```bash
FIRST_JOB=$(sbatch --parsable cluster/train_dinov3l512.sbatch)
sbatch --dependency=afterok:${FIRST_JOB} cluster/train_dinov3l512.sbatch
```

An `afterok` successor does not run if the first job fails. That is desirable: inspect the failure instead of automatically continuing from a corrupt or missing checkpoint. If the program exits cleanly after its planned time budget, it should use exit code 0 and the successor resumes.

**Recommendation:** use manual/dependent six-hour segments as the portable default. Add automatic requeue only after a two-minute sacrificial test proves checkpoint, signal, and requeue behavior end to end.

## 10. Checkpoint-and-resume contract

`src.train` must implement all of the following before a run longer than one allocation:

1. **Atomic write:** write `checkpoint_step_<n>.tmp`, flush and close it, then rename within the same filesystem to `checkpoint_step_<n>.pt`. Update `latest.json` atomically only after the checkpoint is complete.
2. **Complete state:** model, classifier, optimizer, scheduler, AMP scaler when used, EMA state when used, epoch, global optimizer step, micro-step/accumulation position, best metric, early-stopping state, sampler state, and resolved configuration.
3. **Random state:** Python, NumPy, CPU Torch, and all CUDA RNG states. Record deterministic settings and seeds; exact replay can still vary across PyTorch releases, platforms, or nondeterministic kernels.
4. **Data position:** save sampler epoch/order or resume at a documented epoch boundary. Never silently replay a random fraction of an epoch while reporting it as new data.
5. **Compatibility check:** `--resume auto` must reject a changed model, label map, manifest hash, augmentation recipe, effective batch size, or optimizer definition unless an explicit override is given and logged.
6. **Signals:** on `SIGUSR1` or `SIGTERM`, stop scheduling new batches, finish or discard the current accumulation safely, save a checkpoint, write `segment_complete.json`, and exit.
7. **Time budget:** poll monotonic elapsed time and start a graceful save at least 10–15 minutes before `--max-wall-minutes`.
8. **Retention:** keep `best.pt`, the newest two complete recovery checkpoints, and a small weights-only final artifact. Delete an older checkpoint only after the new one is verified loadable.
9. **Resume test:** after the smoke run, resume for at least 20 optimizer steps and verify step number, learning rate, loss scale, validation result, and sampler position.

A full 303M-parameter AdamW training checkpoint can be several gigabytes because optimizer and mixed-precision state exceed the inference weights. Checkpoint every 30–45 minutes unless measured write time makes that excessive; also checkpoint at each validation boundary and graceful exit.

**Recommendation:** make a checkpoint validator load the newest file on CPU and verify required keys, shapes, config hash, and manifest hash. A file that merely exists is not a recoverable checkpoint.

## 11. Robustness evaluation on Slurm

The required matrix contains 15 rows: clean plus 14 transformed conditions.

```text
clean
jpeg_q90, jpeg_q70, jpeg_q50, jpeg_q30
blur_0.5, blur_1.0, blur_2.0
resize_0.5, resize_0.25
noise_0.02, noise_0.05, noise_0.10
color_jitter_0.20
center_crop_0.80
```

Requirements:

- Generate transforms deterministically from image ID and condition seed.
- Apply exactly one named transform per row unless the benchmark specification says otherwise.
- Save scores and metrics after each condition, not only at process end.
- If a row already has a matching checkpoint/config/manifest hash and a completion marker, `--resume` skips it.
- Record decode failures rather than dropping them silently.
- Fit temperature and select any decision threshold on the calibration split only. Apply the frozen values unchanged to all transformed and protected rows.
- Report AUROC, AP, balanced accuracy at the fixed threshold, TPR/FPR at the specified operating points, Brier score, and ECE per condition and generator.

If measured inference plus 20% overhead is below 5 h 30 min, a single resumable job may loop over all conditions. Otherwise split it. An optional array template is:

```bash
#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%1
#SBATCH --job-name=aigc_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
umask 077

PROJECT_ROOT=/persistent/path/aigc-track5/repo
HF_CACHE_ROOT=/persistent/path/aigc-track5/huggingface
RESULT_ROOT=/persistent/path/aigc-track5/runs/frozen_model/metrics

CONDITIONS=(
  clean jpeg_q90 jpeg_q70 jpeg_q50 jpeg_q30
  blur_0.5 blur_1.0 blur_2.0
  resize_0.5 resize_0.25
  noise_0.02 noise_0.05 noise_0.10
  color_jitter_0.20 center_crop_0.80
)
CONDITION="${CONDITIONS[${SLURM_ARRAY_TASK_ID}]}"

module purge
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate aigc-track5

export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python -u -m src.evaluate \
  --config configs/frozen_submission.yaml \
  --condition "${CONDITION}" \
  --resume \
  --output "${RESULT_ROOT}/${CONDITION}.json"
```

`%1` ensures that only one array element runs at a time, matching the example one-GPU user cap. Arrays still count against submission/job limits; confirm `MaxSubmitPU` and array policy before use. If arrays are disallowed, create three group jobs: clean/JPEG, blur/resize, and noise/color/crop.

For the protected COCO-vs-DALL·E run, first execute a CPU manifest audit that asserts zero hash overlap with train/dev/calibration, then submit inference. Do not inspect examples and retrain afterward; that converts the demonstration benchmark into development data.

**Recommendation:** run the complete transform harness first on a 100-image fixture and then on the ordinary dev set. The protected matrix should be a single frozen-model event with an immutable run receipt.

## 12. One GPU versus multiple GPUs

The default is one GPU because the TC2 guide's `normal` QoS example grants `gres/gpu=1`, even though nodes contain 4–8 cards. Requesting an unavailable second GPU can leave the job pending indefinitely or violate the aggregate QoS.

If the administrator grants multiple GPUs on one node:

- use DistributedDataParallel, not `DataParallel`;
- launch with `torchrun --standalone --nnodes=1 --nproc-per-node=gpu` inside the batch allocation;
- use a distributed sampler and call `set_epoch` each epoch;
- scale effective batch deliberately; do not accidentally change the optimization recipe;
- have rank 0 alone write checkpoints and metrics;
- reduce validation predictions across ranks without duplicate examples;
- benchmark scaling efficiency before reserving scarce cards.

Do not use multi-node training for this hackathon. It adds rendezvous, filesystem, and failure complexity with little benefit for a 303M-parameter model.

**Recommendation:** prefer a faster single card and short queue over multi-GPU complexity. Use two or more GPUs only if the measured wall-clock saving is critical and the QoS explicitly supports the request.

## 13. Monitoring and resource tuning

Useful commands from the head node:

```bash
squeue -u <username>
squeue -la
scontrol show jobid <jobid>
scontrol -d show jobid <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,Timelimit,AllocTRES,MaxRSS,ExitCode
seff <jobid>
sinfo -N -l -p MGPU-TC2
scancel <jobid>
```

TC2 also publishes periodic GPU status files under `/tc2share/Cluster-Information` with names such as `MSAI-ActiveNode-*`. The per-job telemetry in §9 is preferable because it is tied to the run and sampled every minute.

Interpretation:

| Symptom | Likely cause | Action |
|---|---|---|
| GPU utilization repeatedly <50% while memory is allocated | Decoder, augmentation, small-file I/O, or too few workers | Compare data time vs compute time; tune workers; shard/stage data; use pinned memory and persistent workers. |
| GPU near 100%, dataloader queue full | Compute-bound healthy run | Improve precision/attention/kernel settings only if the deadline requires it. |
| VRAM OOM at startup | Model/batch/optimizer too large | Micro-batch 1, activation checkpointing, BF16/FP16; then lower resolution. Do not solve VRAM OOM by requesting host `--mem`. |
| Host OOM / `OUT_OF_MEMORY` | Too many workers, prefetch, cache, or large manifests | Lower workers/prefetch and host RAM use; then request measured `--mem`. |
| `QOSMaxGRESPerUser` | GPU total across user's jobs exceeds QoS | Wait/cancel another GPU job or request a different approved QoS. |
| `QOSMaxCpuPerUserLimit` or `QOSMaxMemoryPerUser` | Aggregate CPU/RAM request is too high | Reduce resources or avoid concurrent jobs. |
| `QOSMaxJobsPerUserLimit` | Too many running jobs | Use dependency chains or wait; do not spam submissions. |
| `Priority` or `Resources` pending | Normal scheduling wait | Keep the valid job queued; shorten/relax requests only when useful. |
| CUDA library/symbol error | PyTorch runtime and loaded CUDA module conflict | `module purge`; use the tested module set; avoid mixed toolkits. |
| `TIMEOUT` with no current checkpoint | Missing time/signal handling | Fix the training contract before resubmitting. |
| `NODE_FAIL`, `DRAIN`, or reset | Cluster node failure/maintenance | Validate the last atomic checkpoint and resume on a new allocation. |
| Disk quota exceeded | Duplicate data or accumulated checkpoints | Check quota, rotate only verified checkpoints, remove recomputable cache after confirmation. |

Use `seff` after every completed job. Lower requested CPU/RAM only when repeated measurements show excess; an under-provisioned dataloader can make expensive GPU time slower.

**Recommendation:** optimize for GPU-seconds to a valid result, not the highest instantaneous utilization. Sustained ≥70% GPU utilization is a useful target, but correctness, deterministic transforms, and recoverable checkpoints come first.

## 14. Reproducibility, security, and audit trail

Every run directory must contain:

```text
provenance/
├── command.txt
├── resolved_config.yaml
├── git_commit.txt
├── git_diff.patch           # only if the tree was dirty
├── environment.conda-lock.txt
├── environment.pip-freeze.txt
├── manifest.sha256
├── checkpoint.sha256
├── cluster.json             # job, node, GPU, driver, CUDA, modules
└── seeds.json
```

Additional rules:

- Use a unique `run_id`; never reuse a run directory for a changed configuration.
- Refuse a long run from a dirty Git tree unless the diff is archived automatically.
- Hash configurations and manifests after resolving defaults.
- Keep dataset licences and the DINOv3 licence with any redistributed derivative checkpoint. Do not relicense DINOv3-derived weights as MIT.
- Set `umask 077` in jobs when paths, tokens, or gated artefacts may be private.
- Never print environment variables wholesale: that can expose tokens.
- Do not commit credentials, private cluster hostnames, absolute personal paths, or protected predictions.
- Store the final model's calibration temperature and decision threshold inside the model card and inference config.

**Recommendation:** make `run_receipt.json` a required deliverable. It should let another teammate reproduce which exact data, code, weights, calibration, GPU environment, and Slurm job produced each result.

## 15. Recommended execution sequence

### Phase 0 — one-time setup

1. Complete `cluster_profile.md` and administrator questions.
2. Allocate persistent paths and confirm quotas/purge policy.
3. Build and lock the Conda environment using the permitted provisioning route.
4. Cache DINOv3 and DINOv2 weights and test offline loading.
5. Build immutable manifests and the protected-set denylist in a CPU batch job.

### Phase 1 — qualify the hardware

1. Submit the 15-minute probe.
2. Submit the real 512-pixel benchmark on the actual augmentation/training path.
3. Select precision, micro-batch, accumulation, worker count, and activation-checkpoint setting from measurements.
4. Estimate every epoch and full evaluation matrix with 20% overhead.
5. Apply the go/no-go gates in §8.

### Phase 2 — train safely

1. Run a 30-minute end-to-end smoke job that creates a checkpoint.
2. Resume it in a second job for at least 20 optimizer steps.
3. Validate the checkpoint on CPU and compare continuous-vs-resumed behavior.
4. Launch the first 5 h 50 min production segment.
5. Inspect logs, `seff`, GPU telemetry, data time, and validation metrics before chaining more segments.
6. Freeze the candidate only after the dev robustness gate is met.

### Phase 3 — evaluate and submit

1. Run all 15 conditions on an ordinary dev set and repair the harness, not the protected data.
2. Fit temperature and threshold on the calibration split; freeze them with a config hash.
3. Run error analysis and select the final single model/ensemble under the <2B full-pipeline limit.
4. Assert protected-set hash isolation.
5. Submit protected inference with measured runtime plus 20% margin.
6. Verify JSON schema, image count, path order/uniqueness, probability range, checkpoint/config hashes, and completion markers.
7. Copy final outputs to persistent storage and a second authorised location before the allocation ends.

**Recommendation:** schedule the environment/cache/data tasks first because they are the longest-lead blockers. The first GPU reservation should answer feasibility, not begin an unmeasured overnight experiment.

## 16. Submission checklists

### Before the first GPU job

- [ ] Actual QoS, account, partition, wall time, GPU, CPU, RAM, and job limits recorded.
- [ ] Persistent and scratch paths plus quotas/purge policy confirmed.
- [ ] Compute-node internet policy known; model loads with offline cache.
- [ ] Conda/module versions pinned; no `sudo`; no system/base environment changes.
- [ ] `logs/` exists before `sbatch`.
- [ ] Probe exits non-zero if CUDA/model access fails.

### Before full fine-tuning

- [ ] Real-path benchmark includes decode and augmentations.
- [ ] Peak allocated VRAM ≤90%; no rising-memory leak over ≥100 timed steps.
- [ ] Throughput and epoch estimate recorded with 20% margin.
- [ ] Checkpoint is atomic, complete, validated, and successfully resumed.
- [ ] `SIGUSR1`, `SIGTERM`, and application wall timer tested.
- [ ] Train/dev/calibration manifests pass source/group split and protected denylist gates.
- [ ] Run directory is unique and provenance receipt is complete.

### Before protected inference

- [ ] Model, ensemble membership, temperature, threshold, transforms, and TTA are frozen.
- [ ] Zero protected/train/dev/calibration content-hash overlap asserted.
- [ ] Full 15-row matrix has already completed on non-protected data.
- [ ] Projected runtime fits the request with ≥20% margin; evaluation resumes per condition.
- [ ] Output has exactly one unique row per expected image and `0 ≤ pred ≤ 1`.
- [ ] Final checkpoint, config, manifest, JSON, and CSV checksums recorded.
- [ ] No protected outputs will be used for retraining or retuning.

## 17. Sources

- **CCDS GPU Cluster TC2 User Guide**, local attachment supplied for this project, updated 13 August 2025. This is the authority for TC2-specific rules such as no compute on the head node, Slurm/QoS use, example `normal` limits, module/Conda availability, and monitoring commands. Re-query live cluster configuration because the document's values are examples.
- [Slurm `sbatch` documentation](https://slurm.schedmd.com/sbatch.html) — batch directives, dependencies, signals, resources, and submission behavior.
- [Slurm job arrays](https://slurm.schedmd.com/job_array.html) — array indexes and concurrency limits.
- [Slurm job-state codes](https://slurm.schedmd.com/job_state_codes.html) — pending, running, failure, timeout, and node-failure states.
- [PyTorch general checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) — saving model and optimizer state for resumed training.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) — RNG control and limits of exact reproducibility.
- [Hugging Face Hub environment variables](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables) — `HF_HOME`, `HF_HUB_CACHE`, token handling, and offline mode.
- [DINOv3 ViT-L/16 model card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) — gated model identity, architecture, parameter scale, and licence link.
