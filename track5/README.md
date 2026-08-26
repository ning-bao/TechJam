# track5 — robust AIGC image detection under transforms

End-to-end fine-tuned ViT with a distortion sampler. Backbone pair (PLAN D1):
**DINOv3-L/16 primary** (`facebook/dinov3-vitl16-pretrain-lvd1689m`, 303.1M,
gated) with **DINOv2-with-registers as the ungated fallback** — configs declare
both, and `build_model` resolves the fallback automatically when the gated
checkpoint cannot be loaded, so the critical path never blocks on gate approval.

Decisions and schedule: [../PLAN.md](../PLAN.md). Module contracts:
[../INTERFACES.md](../INTERFACES.md). Cluster rules:
[../TC2_GPU_CLUSTER_GUIDE.md](../TC2_GPU_CLUSTER_GUIDE.md).

MIT-licensed code. DINOv3 weights carry Meta's community licence (PLAN D11):
never relicense a DINOv3-derived checkpoint as MIT. Everything else on the
critical path is Apache-2.0/MIT.

## Entry points

Non-interactive, rerunnable, non-zero exit on failure, atomic output writes
(TC2 §6). Run from this directory.

```bash
python -u -m src.cluster_probe --config configs/dinov3l512.yaml --output probe.json
python -u scripts/benchmark_gpu.py --out reports/bench.json
python -u -m src.train --config configs/dinov3l512.yaml --run-dir runs/dinov3l512_seed0 --resume auto --max-wall-minutes 330
python -u -m src.predict --checkpoint runs/x/best.pt --input <dir> --output preds.jsonl
```

| Command | Purpose |
|---|---|
| `src.cluster_probe` | GPU/driver/precision probe + one 512px forward+backward |
| `scripts/benchmark_gpu.py` | qualification sweep: resolution x precision x micro-batch x activation checkpointing, inference and train, compute-only vs end-to-end dataloader |
| `src.train` | one resumable training segment (`--resume auto`, `--max-wall-minutes`, SIGUSR1/SIGTERM) |
| `src.evaluate` | one robustness condition (or a group, or all 15): calibrated probabilities, AUROC/AP/bAcc/Brier/ECE, per-generator breakdown, `--resume` per condition |
| `src.predict` | organiser format: `{"image_path": ..., "pred": 0.87}` per image |
| `scripts/predict.py` | repo-internal INTERFACES §7 format (meta/predictions/errors) |
| `scripts/build_manifests.py` | manifests; `--source wildfake_csv` indexes WildFake from its label CSVs. **Strict by default** — fails if a requested image is missing; `--allow-missing` is development mode |
| `scripts/build_denylist.py` | protected-set denylist (content hashes + protected paths) |
| `scripts/eval_matrix.py` | whole-matrix runner writing one CSV (single-process alternative to `src.evaluate`) |
| `scripts/probe_gate.py` | PLAN D4 shortcut-probe gate |

`src.predict` writes a **JSON array** by default — the organiser has only said
"JSON", and an array parses with any JSON reader. `--format jsonl` keeps the
one-record-per-line form. For the final submission pass `--protected-run`: it
implies `--strict` (a decode failure fails the job instead of emitting a
`pred=0.5` recovery score), refuses to overwrite a completed run, and writes a
receipt. The `pred=0.5` recovery mode stays available for exploratory runs.

Slurm launchers: [cluster/probe_gpu.sbatch](cluster/probe_gpu.sbatch),
[cluster/benchmark_dinov3.sbatch](cluster/benchmark_dinov3.sbatch),
[cluster/train_dinov3l512.sbatch](cluster/train_dinov3l512.sbatch),
[cluster/eval_array.sbatch](cluster/eval_array.sbatch). Replace the
`/persistent/path` placeholders and confirm partition/QoS/module with the TC2 §2
discovery commands first; `mkdir -p logs` before `sbatch`. `MaxJobsPU=2` rules
out 15 parallel evaluation jobs, so the matrix runs as three condition groups at
`--array=0-2%1`, or as one resumable `--condition all` job.

## Hardware

Measured profile and open discovery items:
[docs/cluster_profile.md](docs/cluster_profile.md).

* **TC2 — NVIDIA L40S**, 46,068 MiB, cc 8.9 (BF16 native), driver 12.8. QoS: 1
  GPU / 10 CPU / 30 GB / MaxWall 6 h, MaxJobsPU = MaxSubmitPU = 2. Production
  segments request 8 CPU / 26 GB / 05:50:00 and stop themselves at 330 min.
  Because `MaxSubmitPU=2`, chain segments two at a time.
* **Local — RTX 5070 Ti**, 16,303 MiB, sm_120, Windows WDDM. The full local
  training benchmark is deferred until desktop VRAM is cleared; WDDM
  compositor allocations otherwise invalidate the peak-VRAM figures.

The L40S benchmark sweeps 512 px / BF16 / micro-batch 8→4→2→1 with activation
checkpointing off first, adding the on row only for a configuration that OOMs or
exceeds 90 % peak allocated VRAM. No `torch.compile` in the baseline.

## Order of operations

```bash
# 1. protected set first — nothing may train until C2 is enforceable
python -u scripts/build_denylist.py --coco-val
# 2a. manifests, development build while downloads run (skips what is absent)
python -u scripts/build_manifests.py --source wildfake_csv --out data/manifests/wildfake.parquet --per-family-limit 20000 --allow-missing
# 2b. the final build must be complete: scope it and drop --allow-missing
python -u scripts/build_manifests.py --source wildfake_csv --out data/manifests/wildfake.parquet --csvs ddim.csv,ddpm.csv,adm.csv,vqdm.csv --per-family-limit 20000
# 3. qualify the hardware, then apply the TC2 §8 go/no-go gates
sbatch cluster/probe_gpu.sbatch && sbatch cluster/benchmark_dinov3.sbatch
# 4. training segments, chained with --dependency=afterok
```

`build_denylist.py` exits non-zero if the COCO val2017 archive is incomplete, and
`src.train` refuses to start against an incomplete denylist. Re-run it once the
downloads finish.

## Constraint C2 (protected set)

COCO val2017 and the **entire** WildFake DALL·E family — 55,638 Typical plus
8,843 Advanced = 64,481 images — are never training, selection, or calibration
data. Four independent gates, so a renamed path alone cannot bypass denial:

| Gate | Where it comes from |
|---|---|
| generator family (`dalle`) | CSV `Architecture`/`Category`, mapped family |
| source dataset (`coco_val2017`) | manifest `source` column |
| path key | `data/denylist/protected_paths.parquet`, built from the label CSVs so it covers images that are *not* downloaded |
| content hash / pHash ≤ 4 | `data/denylist/denylist.parquet`, built from the archives that exist |

`wildfake_csv` manifest building refuses protected rows outright (by CSV,
architecture, category, family *and* path); `src.train` aborts if the train
manifest trips any gate or if the denylist is known-incomplete; `src.evaluate`
demands an explicit `--protected-run` and refuses to repeat a completed one.

**Denylist ≠ benchmark.** The denylist covers *all* images in the canonical
val2017 archive; the demonstration benchmark is the 4,998-image subset listed by
WildFake's `real_coco.csv`. Archive images outside that subset are still denied
from training but are not evaluated — `--source coco_val_demo` builds the
benchmark manifest, and the denylist receipt lists the denied-only images.

## Tests

```bash
python -m pytest -q
```

GPU-only and weight-downloading tests are deselected by default
(`-m 'not gpu and not needs_weights'` in `pyproject.toml`).
