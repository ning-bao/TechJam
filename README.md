# Robust Detection of AI-Generated Images Under Real-World Transforms

TikTok TechJam 2026, Track 5. A fine-tuned vision transformer that decides
whether an image is AI-generated, and keeps deciding correctly after the image
has been JPEG-compressed, blurred, resized, noised, colour-jittered or cropped.

**Headline, on a generator the model has never seen** (the organiser's
demonstration benchmark: COCO val2017 reals vs DALL·E 3 Advanced fakes, with the
entire DALL·E family — 64,482 images — denylisted from training):

| condition | bAcc | AUROC | FPR@95TPR | ECE |
|---|---|---|---|---|
| clean | **0.9891** | 0.9998 | 0.000 | 0.013 |
| JPEG q30 | 0.9643 | 0.9931 | 0.024 | 0.025 |
| blur σ2.0 | 0.9667 | 0.9967 | 0.012 | 0.035 |
| resize 0.25× | **0.9584** | 0.9909 | 0.032 | 0.029 |
| noise σ0.10 | 0.9614 | 0.9938 | 0.026 | 0.032 |

Mean bAcc 0.968 · worst case 0.9584 · **maximum degradation from clean 3.1
points**. n = 13,843 per condition.

The number we think matters most is not in that table. It is the gap between
in-distribution and out-of-distribution performance: **1.0–2.5 points** between
dev (which shares all five generator families with training) and the protected
benchmark (which shares none). A detector that had memorised "what FLUX and SD
decoders look like" would fall off a cliff there. This one does not.

## Quick start

```bash
# Python 3.13, MIT-licensed code
pip install -e track5
cd track5

# fetch the submission checkpoint (DINOv3 License, NOT MIT - see WEIGHTS_LICENSE.md)
mkdir -p runs/dinov3l448_d4
curl -L -o runs/dinov3l448_d4/epoch1_best_calibrated.pt \
    https://github.com/ning-bao/TechJam/releases/download/submission-epoch1-step7000/epoch1_best_calibrated.pt

# score a directory of images -> JSON array of {image_path, pred}
python -u -m src.predict \
    --checkpoint runs/dinov3l448_d4/epoch1_best_calibrated.pt \
    --input <image-dir> \
    --output preds.json
```

`pred` is a calibrated probability that the image is AI-generated:
`sigmoid((z + α) / T)` with T, α and the decision threshold τ frozen before
inference and carried inside the checkpoint. A decode failure is written to
`preds.json.errors.json` and announced on stderr — never silently scored as 0.5.
Full CLI reference: [track5/README.md](track5/README.md).

## What is in this repository

| Path | What |
|---|---|
| [track5/src/track5/](track5/src/track5/) | the library: data, transforms, model, training loop, evaluation, calibration |
| [track5/src/predict.py](track5/src/predict.py) | **the submission CLI** — image directory in, `{image_path, pred}` JSON out |
| [track5/scripts/](track5/scripts/) | manifest building, denylist construction, probes, matrix evaluation |
| [track5/tests/](track5/tests/) | 331 tests, run in CI on every push |
| [track5/reports/](track5/reports/) | measured results: training, evaluation, data preparation, verification |
| [track5/analysis/](track5/analysis/) | calibration and decision-curve analysis (R) |
| [PLAN.md](PLAN.md) | every design decision, numbered, with the evidence behind it |
| [INTERFACES.md](INTERFACES.md) | module contracts frozen on day 0 |

## How it works

**Backbone.** DINOv3-L/16 at 448 px, fully fine-tuned end to end (303.1M
parameters, 15% of the 2B budget). DINOv2-with-registers is wired as an ungated
fallback so a licence-gate delay could never block the critical path.

**Why end-to-end and not a linear probe.** Measured on our own data: a frozen
backbone with a trained head reaches 0.737 worst-case bAcc; full fine-tuning
reaches 0.9685. **+23.2 points.** The published figure we were working from
predicted +26.7, so this is close to expectation and settles the question
empirically rather than by argument.

**Crop, never resize.** Training reads random 448 px crops at native
resolution. Downscaling an image to fit a model destroys exactly the
high-frequency evidence a detector depends on.

**Distortion sampler.** 30% of training images clean, 55% one corruption, 15%
two, applied identically to both classes. JPEG, resize (four kernels), blur,
noise, colour jitter, crop — each drawn slightly past test severity. MixUp,
CutMix and hue jitter are banned by design: published evidence has them
degrading this task badly (78.6 vs 92.2 bAcc in one case).

**Model selection is worst-case, not average.** The checkpoint is chosen by the
*minimum* bAcc across five conditions, so a model that scores well on clean
images and collapses under noise cannot win.

## The trap we had to disarm first

An AIGC corpus assembled from public datasets is usually separable without
looking at the picture at all. Ours was: **a probe reading only the JPEG
quantization table scored 0.974 bAcc.** A model trained on that data would learn
"PNG means fake", report 97% on its own test split, and collapse on the
benchmark.

Four shortcut probes were run on post-crop, post-normalization metadata — what
the model actually receives — and training was gated on all four scoring below
0.60:

| probe | raw corpus | after normalization | after size matching |
|---|---|---|---|
| JPEG quality | 0.974 | 0.496 | **0.500** |
| dimensions | 0.732 | 0.500 | **0.500** |
| file size | 0.589 | 0.620 | **0.497** |

Container statistics are equalised by a seeded plan that never sees the label,
then file size is equalised by stratified selection. One residual is documented
rather than hidden: a real photograph arrives already JPEG-compressed and that
cannot be undone, so total compression history and final-encode quality cannot
both be equalised. We equalise the final encode, because that is what the probe
reads and what a detector most easily latches onto.

## Reproducing the results

```bash
cd track5

# 1. the protected set is denied before anything can train
python -u scripts/build_denylist.py --coco-val

# 2. manifests (strict: fails on a missing image rather than skipping it)
python -u scripts/build_manifests.py --source wildfake_csv \
    --out data/manifests/wildfake.parquet --per-family-limit 20000

# 2b. the train/dev/calibration splits. Manifests are gitignored regenerable
#     artifacts, so this step is what produces data/manifests/*.parquet
python -u scripts/build_training_set.py --crop 448

# 3. shortcut probes — training refuses to start until all four score < 0.60
python -u scripts/probe_gate.py --manifest data/manifests/train.parquet

# 4. train (resumable segments; ~1.5 s/step on an RTX 5070 Ti)
python -u -m src.train --config configs/dinov3l448_d4.yaml \
    --run-dir runs/dinov3l448_d4 --resume auto --max-wall-minutes 330

# 5. the 15-condition robustness matrix
python -u scripts/eval_matrix.py --checkpoint runs/dinov3l448_d4/epoch1_best.pt \
    --manifest data/manifests/dev_eval2k.parquet --atoms all
```

Every step is non-interactive, rerunnable, exits non-zero on failure, and writes
output atomically. Tests: `python -m pytest -q` (331 collected, GPU-only and
weight-downloading tests deselected by default).

The protected-benchmark numbers in the table above are **not** reproduced by
this sequence, deliberately — see the constraint below.

## Constraint we imposed on ourselves

COCO val2017 and the entire WildFake DALL·E family (64,482 images) are never
training, model-selection, or calibration data. Four independent gates enforce
it, so a renamed file alone cannot slip through: generator family, source
dataset, path key, and content hash / perceptual hash. Manifest building refuses
protected rows outright; training aborts if the manifest trips any gate or if the
denylist is incomplete; the evaluation CLI demands an explicit `--protected-run`
flag and refuses to repeat a completed one.

The rules do not require this. The benchmark does not count toward the score, and
nothing forbids reporting results on it. We imposed the constraint because a
detector's whole value is performance on generators it has not seen, and a number
obtained by tuning against the benchmark measures nothing.

**We overran it, and this is the disclosure.** The budget was one final inference
run. Two exploratory reads happened first (2026-08-27 19:30 and 2026-08-28 02:29,
both uncalibrated logits at a naive 0.5 threshold), recorded in
`track5/reports/matrix_protected_*.csv`. Neither participated in choosing what to
submit: that decision was made on dev and on held-out generator families, and is
documented in the limitations note. We are reporting the overrun rather than
deleting the artifacts before submission.

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Project description | Devpost |
| 2 | Public repository + scoring script | this repo, [track5/src/predict.py](track5/src/predict.py) |
| 3 | Demo video | linked from Devpost |
| 4 | Robustness evaluation summary | [track5/reports/robustness_summary.md](track5/reports/robustness_summary.md) |
| 5 | Error analysis note | [track5/reports/error_analysis.md](track5/reports/error_analysis.md) |

## Limitations and what we would do with more time

Written up in full in [track5/KNOWN_LIMITATIONS.md](track5/KNOWN_LIMITATIONS.md).
The short version:

- **The 448 px crop decision excludes four generator families** (adm, ddpm,
  vqdm, gan) from training, because their images are smaller than the crop and
  nothing is ever padded. This is the largest unratified trade-off in the
  project.
- **A second training epoch made the model worse on unseen generators** and we
  submitted the first. The cause is a real bug we found and are disclosing:
  the augmentation RNG had no epoch term, so epoch 2 replayed epoch 1's exact
  augmented bytes. We fixed it and reran the second epoch with fresh draws: that
  recovered 2–4 points of the held-out loss and still trailed epoch 1 by 6.6
  points worst-case — the bug was an amplifier, the second epoch itself the
  dominant cause. Epoch 1 stands.
- **Calibration does not transfer to unseen generators.** ECE rises from 0.013
  in-distribution to 0.10–0.19 on held-out families. This is the price of
  fitting temperature once on a deployment mixture and never refitting per
  transform — refitting per condition would be tuning on the evaluation
  conditions, which we consider indefensible.
- **Our hard-case evaluation sets are not mutually comparable at `clean`.** The
  100 human-curated real photographs are JPEG originals spanning 3.1–103.8 MP
  (median 19.9); the 100 generated images are PNG at 1–1.6 MP. A fixed 448 px
  centre crop therefore covers a median 1.0% of a real frame and 18.7% of a
  generated one. Only the JPEG conditions, where every image is re-encoded
  identically, are clean comparisons.
- **All 100 curated real photographs come from one platform** (Unsplash), so a
  false-positive rate measured on them cannot separate "this kind of photograph
  is hard" from "this platform's processing pipeline has a signature".
- **2026 consumer endpoints evade the detector.** On our own hard-case set the
  seen-family Flux control is caught 25/25 while gpt-image-2 and Gemini 3 are
  missed 70 of 75 at the frozen operating point — confidently, not marginally.
  The no-watermark control refutes watermark-reading, and DALL·E 3 (also
  unseen) holds 0.9891 — so it is the newest generation specifically
  (KNOWN_LIMITATIONS item 11).

## Team contributions

| | |
|---|---|
| **Bao Ning** | Repository, training pipeline, model, evaluation harness, calibration, CI. All GPU work. |
| **Cai Haitong** (lead) | Problem framing, plan and interface contracts, C2 protected-set discipline, independent verification of deliverables, evaluation-protocol review. Authored the submission README, KNOWN_LIMITATIONS, and deliverables 4 and 5. |
| **Xiong Yuxuan** | Hard-case datasets: 100 curated real photographs and 100 generated images across four models, with provenance manifest and validation scripts. |
| **Zhang Xinghan** | R analysis pipeline for calibration curves, ECE, decision-curve analysis and threshold drift, built and validated against a mock dataset. |
| **Yang Zihao** | Degradation estimator (DCT-based blind JPEG quality-factor estimation); provenance-survival measurement of C2PA declarations under the 15 transforms. |

## Licence

**Code: MIT** — see [LICENSE](LICENSE).

**Weights: DINOv3 License, not MIT** — see [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md).
The submitted checkpoint is a full fine-tune of DINOv3
(`facebook/dinov3-vitl16-pretrain-lvd1689m`), developed by Meta AI and used
under the DINOv3 License. A DINOv3-derived checkpoint must never be relicensed
as MIT, so the two licences are kept separate and the agreement travels with
the weights in the release.

Everything else on the critical path is Apache-2.0 or MIT.
