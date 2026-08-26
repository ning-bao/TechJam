# Cluster profile (TC2 guide section 2)

Rows marked **confirmed** came from the cluster's own discovery commands. Rows
marked **TO CONFIRM** still need an answer before a long job is submitted — two
individually valid jobs can still exceed the aggregate per-user cap.

## TC2 — confirmed

| Item | Value |
|---|---|
| Partition | `MGPU-TC2` |
| QoS | `normal` |
| GPU model | NVIDIA **L40S** |
| Usable VRAM | 46,068 MiB (~45.0 GiB) |
| Compute capability | 8.9 (Ada) — **BF16 is native**, no FP16/GradScaler fallback needed |
| NVIDIA driver | **570.148.08** |
| CUDA (driver-supported) | 12.8 |
| Max GPUs per user | 1 |
| Max CPU cores | 10 |
| Max host RAM | 30 GB |
| MaxJobsPU | 2 |
| MaxSubmitPU | 2 |
| MaxWall | 6 h |
| Production training request | 1 GPU, 8 CPU, 26 GB, `--time=05:50:00` |
| Application exit/checkpoint | `--max-wall-minutes 330` (5 h 30), ~20 min left for the final save |

Driver 570.148.08 supports the CUDA 12.8 runtime, so a `cu128` PyTorch build is
the right choice and no `cuda/*` module needs loading — let PyTorch own its
runtime (TC2 §4.2). Verify inside a compute job, never on the head node.

### What this profile already forces in the repo

| Consequence | Where |
|---|---|
| `precision: bf16` resolves to real BF16 on cc 8.9; `resolve_precision` only drops to FP16+GradScaler on a card without BF16, and never silently to FP32 | `configs/dinov3l512.yaml`, `track5/train/loop.py` |
| 46 GB is ~2.8x the local 5070 Ti, so micro-batch 8 at 512 px is worth testing here and is out of reach locally | `cluster/benchmark_dinov3.sbatch` sweeps 8/4/2/1 |
| **MaxSubmitPU=2** caps an `afterok` chain at two deep — chain segments two at a time, never pre-submit a long chain | `cluster/train_dinov3l512.sbatch` |
| **MaxJobsPU=2** rules out 15 parallel evaluation jobs; the matrix runs as 3 condition groups with `--array=0-2%1`, or as one resumable job | `cluster/eval_array.sbatch` |
| Every job requests 8 CPU / 26 GB, under the 10 / 30 cap, so a second queued job does not breach the aggregate limit | all four sbatch files, enforced by `tests/test_cluster_scripts.py` |

### Throughput budget (to be replaced with measurements)

`cluster/benchmark_dinov3.sbatch` is the only valid source for these. Until it
runs, the §8 conversions are:

```text
epoch_hours  = processed_training_images / (train_images_per_second x 3600)
matrix_hours = 207,615 / (inference_images_per_second x 3600)
```

Add 15–20 % for startup, I/O and result writes. A 15-condition matrix over a
13,841-image dev set fits one 5 h 30 job only if inference clears ~12 img/s.

## TC2 — TO CONFIRM

| Item | Command / owner |
|---|---|
| Account name | `sacctmgr show user <user> withassoc format=user,account,partition,qos` |
| Persistent project/data path and quota | `quota -s`, `df -h .` |
| Scratch / `${SLURM_TMPDIR}` path, quota, purge policy | administrator |
| Internet access from compute nodes | administrator |
| **Job arrays allowed, and how array tasks count against MaxSubmitPU=2** | administrator — blocks `eval_array.sbatch` |
| `--signal` and `--requeue` enabled for `normal` | administrator — `--signal=USR1@300` is the second line of defence in `train_dinov3l512.sbatch` |
| Approved Anaconda module (`anaconda/25.5.1` assumed) | `module spider anaconda` |
| CPU-only partition for manifests/hashing/extraction | administrator |

Until the CPU-partition question is answered, manifest building, hashing and the
denylist must not run on the head node.

## Local workstation — confirmed

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16,303 MiB (~15.9 GiB) |
| Architecture | Blackwell, sm_120 (compute capability 12.0) |
| Driver model | Windows WDDM — the desktop compositor holds VRAM the benchmark cannot use |
| PyTorch | 2.11.0+cu128 (see `environment.pip-freeze.txt`) |

### Measured: DINOv3-L/16 @ 512 px, real pretrained weights

| Setting | Value |
|---|---|
| Precision | BF16 |
| Micro-batch | 2 |
| Activation checkpointing | off |
| Throughput | **14.42 img/s** (train, fwd+bwd+optimizer) |
| Median step | 138.7 ms |
| p95 step | 146.6 ms |
| PyTorch peak allocated | 6.119 GiB |
| GPU utilization | 95 % |

**Read the memory figure carefully.** 6.119 GiB is 38.4 % of the 15.92 GiB the
card has, but ~5.2 GiB is already held by the Windows desktop, so the honest
figure is ~57 % of the ~10.7 GiB this process can actually get. The benchmark
now records `vram_usable_gib`, `device_used_by_others_gib` and
`peak_allocated_pct_of_available` from `torch.cuda.mem_get_info`, and the
activation-checkpointing escalation triggers on the available-based number — the
total-based one would have looked safe right up to the OOM.

At 14.42 img/s this configuration already clears the §8 "≥ 8 img/s → keep
DINOv3-L/512" gate **on consumer hardware**, and 95 % utilization says the run is
compute-bound rather than starved by the dataloader. The L40S should be faster
still; only its numbers feed the go/no-go decision.

**The remaining local sweep is deferred until desktop VRAM is cleared.** Under
WDDM the measurement is only meaningful with an idle desktop; a background 5 GB
compositor/browser allocation shifts the usable ceiling and can OOM a batch size
that would otherwise fit. When the desktop is clear:

```bash
F:/Hackathon/.venv/Scripts/python.exe -u scripts/benchmark_gpu.py \
  --resolutions 512,448,384 --precisions bf16 --micro-batches 2,1 \
  --modes train,inference --data compute,dataloader --checkpointing auto \
  --workers 4 --warmup-steps 20 --steps 100 --resume \
  --out reports/benchmark_local_5070ti.json
```

`--resume` keeps the rows already measured; only the outstanding configurations
run again.

## Environment

`environment.pip-freeze.txt` is the exact verified set (Windows, Python 3.13.0,
cu128 wheels); `pyproject.toml` pins `transformers>=5.15.1,<5.16` because
DINOv3ViT is the D1 primary backbone. On TC2, install a torch build matching
driver 570.148.08 first, then regenerate the conda locks there per §4.1.
