# INTERFACES.md — binding contracts between modules

Package root: `track5/src/track5` (installed as `track5`, src-layout).
All paths below relative to `track5/`. If a contract here conflicts with your
preference, the contract wins. If it conflicts with PLAN.md, report it.

## 1. Utils (already written by orchestrator — import, don't reimplement)

```python
from track5.utils.seed import seed_everything, item_seed
# seed_everything(seed: int) -> None            # python/numpy/torch(if present)
# item_seed(*parts: str|int) -> int             # stable blake2b-based 32-bit seed

from track5.utils.config import load_config, config_hash
# load_config(path) -> dict                     # plain YAML dict
# config_hash(cfg: dict) -> str                 # sha256[:12] of canonical dump

from track5.utils.hashing import file_sha256
# file_sha256(path) -> str                      # hex digest
```

## 2. Eval atoms — `src/track5/transforms/eval_atoms.py` (FROZEN after Day 0)

The 15 conditions, exact names (these strings appear in manifests, cache dirs,
CSVs — never change them):

```
clean
jpeg_90  jpeg_70  jpeg_50  jpeg_30          # PIL/libjpeg, quality=N
blur_05  blur_10  blur_20                   # Gaussian blur sigma 0.5 / 1.0 / 2.0
resize_050  resize_025                      # bicubic downscale to 0.50x / 0.25x, then bicubic
                                            #   upscale back to original size (spec: "then upscale")
noise_002  noise_005  noise_010             # additive Gaussian sigma 0.02/0.05/0.10 of 255, seeded
jitter_pm20                                 # brightness/contrast/saturation each U[0.8,1.2], seeded, NO hue
crop_80                                     # center crop, SIDE convention (0.8 * each side)
```

API:
```python
ATOMS: dict[str, dict]           # name -> params
def apply_atom(img: PIL.Image.Image, atom: str, seed: int) -> PIL.Image.Image
def encode_atom(img, atom) -> bytes   # condition's re-encode: JPEG(q=N) for jpeg_*, else PNG
def apply_and_encode(src_bytes: bytes, atom: str, seed: int) -> bytes
```
- `crop_80` must also support area convention via `ATOMS`-level variant
  `crop_80_area` (16th atom, excluded from the standard 15, used for the D9
  ambiguity check).
- Determinism: identical output **bytes** for identical (src_bytes, atom, seed)
  across runs and processes. Seed for an image = `item_seed(sha256, atom, global_seed)`.
- Both classes always go through the identical path — no class-conditional logic.

## 3. Train sampler — `src/track5/transforms/train_sampler.py`

PLAN D5 verbatim. `TrainDistortionSampler(cfg, generator)` with
`__call__(img: PIL.Image, rng: np.random.Generator) -> PIL.Image`.
- Draw: 30% clean / 55% one corruption / 15% two.
- Family weights within a corruption draw: jpeg .30, resize .20, blur .15,
  noise .15, jitter .10, crop .10.
- Ranges: JPEG Q25–100 with two encoder paths (PIL libjpeg + cv2), balanced
  single/double compression history; resize 0.20–1.00× with 4 kernels
  (nearest/bilinear/bicubic/lanczos), 50% of draws upscaled back to the
  original size with an independently drawn kernel, 50% stay rescaled;
  blur σ0–2.3; noise σ0–0.11 half applied
  before JPEG half after; jitter 0.75–1.25 (no hue); crop 0.75–1.00.
- Banned (do not implement): MixUp, CutMix, hue, solarize.

## 4. Manifest schema — parquet at `data/manifests/<name>.parquet`

Path grammar (`track5.data.resolve.resolve_image_bytes(data_root, path)`):
plain file `WildFake/Images/Real/x.jpg`; zip member (no extraction)
`COCO/train2017.zip#train2017/000000123.jpg`; parquet row (HF image.bytes)
`SID_Set/data/train-00007-of-00283.parquet#123`.

Columns (exact names/dtypes):
```
path            str    # relative to track5/data/raw (or data/derived for VAE recons)
sha256          str
phash           str    # 64-bit hex
label           int8   # 0=real 1=fake
source          str    # coco_train2017 | coco_val2017 | wildfake | sid_set | vae_recon
generator_family str  # "" for real; else sd | mj | adm | ddpm | vqdm | gan | glide | dalle | flux | vae_sd15 | vae_sdxl | other
width           int32
height          int32
format          str    # jpeg | png | webp | other
file_bytes      int64
jpeg_quality    int16  # estimated luma-table quality, -1 if not jpeg/unknown
n_recompress    int8   # known JPEG recompression count, -1 unknown
split           str    # unassigned | train | dev | calib | shadow | heldout_ood | denied
```
- Manifests are built with split="unassigned"; Day-1 split logic assigns the
  real splits, `apply_denylist` marks "denied".
- `split="denied"` for anything hitting the denylist (never dropped silently).
- Denylist file: `data/denylist/denylist.parquet` with columns
  (sha256, phash, reason), reasons: `coco_val2017`, `wildfake_dalle`.
- Denylist match = exact sha256 OR phash Hamming distance ≤ 4.

## 5. Model API — `src/track5/models/`

```python
from track5.models import build_model
model = build_model(cfg)     # cfg["model"]: {backbone: "facebook/dinov2-with-registers-base"|"...-large",
                             #   head: "linear", pool: "cls", pretrained: bool, stub: bool}
logits = model(pixels)       # pixels [B,3,H,W] float, ImageNet-normalized; logits [B] (no sigmoid)
```
- `stub: true` → tiny random Conv+pool stand-in (same API) so tests/CI never
  download weights.
- Preprocessing contract (`src/track5/models/preprocess.py`):
  `train_crop(img, rng, size=448)` random native-res crop, reflect-pad if small;
  `eval_crop(img, size=448)` center crop, reflect-pad if small. NEVER resize
  before cropping. `to_tensor(img)` → normalized float tensor.
- Ensemble: `AverageEnsemble([model,...])` averaging **probabilities** (post-
  per-model calibration), not logits.

## 6. Checkpoint format (torch.save dict)

```python
{"state_dict": ..., "config": <full cfg dict>, 
 "calibration": {"temperature": float|None, "alpha": float|None, "threshold": float|None},
 "meta": {"config_hash": str, "epoch": int, "step": int, "metrics": dict, "code_version": str}}
```

## 7. predict.py — STABLE CLI (submission contract, never break)

```
.venv/Scripts/python.exe scripts/predict.py \
  --input <dir | .txt filelist | .csv with path column> \
  --checkpoint <path.pt> --out preds.json [--batch-size 32] [--device auto|cuda|cpu] [--tta none|crop5]
```
Output JSON:
```json
{"meta": {"model_hash": "...", "config_hash": "...", "temperature": 1.0,
          "alpha": 0.0, "threshold": 0.5, "n_ok": 123, "n_err": 1},
 "predictions": [{"path": "...", "score": 0.9731, "label": 1}],
 "errors": [{"path": "...", "error": "UnidentifiedImageError: ..."}]}
```
- `score` = calibrated `sigmoid((z + alpha)/T)` = p(AIGC); `label` = score ≥ threshold.
- Decode/read failures go to `errors`, NEVER a silent 0.5 in `predictions`.

## 8. Eval — `src/track5/eval/`

```python
from track5.eval.metrics import auroc, average_precision, bacc, fpr_at_95tpr, ece, all_metrics
# all_metrics(y, scores, threshold) -> dict with keys:
#   auroc, ap, bacc, fpr_at_95tpr, ece, n_real, n_fake
# ece: 15 equal-width bins on score
from track5.eval.calibrate import fit_temperature_alpha   # (logits, y) -> (T, alpha), NLL-minimizing
from track5.eval.threshold import freeze_threshold        # max bAcc s.t. FPR<=0.05 on CLEAN dev; returns float
from track5.eval.bootstrap import bootstrap_ci            # (y, scores, metric_fn, n=1000, seed) -> (lo, hi)
```
- Matrix runner (`src/track5/eval/matrix.py` + `scripts/eval_matrix.py`):
  for each atom × manifest split: transform via `apply_and_encode` (cache to
  `data/cache/eval/<atom>/<sha256>.<ext>`, reuse if exists), score via model,
  emit `reports/matrix_<config_hash>.csv` with rows =
  (atom, n, auroc, ap, bacc, fpr_at_95tpr, ece, delta_clean_bacc, ci_lo, ci_hi,
  model_hash, config_hash, atoms_version).
- Model-selection scalar: `worst_case_bacc` = min bAcc over
  {clean, jpeg_30, blur_20, resize_025, noise_010}.

## 9. Config YAML shape (configs/*.yaml)

```yaml
seed: 17
data: {train_manifest: ..., dev_manifest: ..., crop: 448, batch_size: 32, workers: 8}
model: {backbone: facebook/dinov2-with-registers-base, head: linear, pool: cls, pretrained: true, stub: false}
train: {epochs: ..., lr_head: ..., lr_backbone: ..., weight_decay: ..., warmup_frac: ...,
        loss: bce|focal, focal_gamma: 2.0, focal_alpha: 0.5, ema_decay: 0.999, swa: false,
        precision: bf16, grad_accum: 1, freeze_backbone: false}
distortion: {enabled: true}    # train_sampler on/off (off for the frozen floor probe)
eval: {every_steps: ..., worst_case_atoms: [clean, jpeg_30, blur_20, resize_025, noise_010]}
```

## 10. Dataset — `src/track5/data/dataset.py`

```python
ManifestDataset(manifest_path, split, crop=448, distortion_sampler=None, seed=...)
# __getitem__ -> {"pixels": FloatTensor[3,448,448], "label": int, "idx": int}
```
Order: load bytes → decode RGB → (train only) distortion sampler on PIL →
crop policy (§5) → to_tensor. Skip-and-log unreadable files (partial downloads).
