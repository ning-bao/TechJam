# Epoch 1 evaluation — DINOv3-L/16 @ 448px

**Model:** `runs/dinov3l448_d4/epoch1_best.pt` — step 7,000, config `4ed11edd6769`, model `80b5fdf5d548`
**Backbone:** `facebook/dinov3-vitl16-pretrain-lvd1689m`, 303,130,625 params (15% of the 2 B budget)
**Evaluated:** 2026-08-27, after epoch 1 (training stopped at step 8,500; epoch boundary 8,261)

Epoch 1's best checkpoint is **step 7,000, not the final step** — worst-case bAcc peaked there (0.9685) and was flat-to-down for the remaining 1,500 steps.

---

## 1. Headline — protected set (val2017 × DALL·E 3 Advanced)

The organiser's demonstration benchmark. **The model has never seen a DALL·E image of any kind**: the entire family (64,482 paths) is denylisted from training, and DALL·E 3 is an unseen generator architecture. This is a genuine out-of-distribution generalization test.

| condition | bAcc | 95% CI | AUROC | AP | FPR@95TPR | ECE |
|---|---|---|---|---|---|---|
| clean | **0.9891** | 0.9876–0.9907 | 0.9998 | 0.9999 | 0.000 | 0.013 |
| jpeg_30 | 0.9643 | 0.9612–0.9672 | 0.9931 | 0.9963 | 0.024 | 0.025 |
| blur_20 | 0.9667 | 0.9639–0.9694 | 0.9967 | 0.9982 | 0.012 | 0.035 |
| **resize_025** | **0.9584** | 0.9548–0.9617 | 0.9909 | 0.9953 | 0.032 | 0.029 |
| noise_010 | 0.9614 | 0.9585–0.9647 | 0.9938 | 0.9966 | 0.026 | 0.032 |

**Mean bAcc 0.968 · mean AUROC 0.9949 · worst case 0.9584 · maximum degradation from clean 3.1 points.**

n = 13,843 per condition (5,000 real + 8,843 fake).

---

## 2. Generalization gap: dev vs protected

| condition | dev (in-distribution) | protected (unseen generator) | gap |
|---|---|---|---|
| clean | 0.9995 | 0.9891 | 0.0104 |
| jpeg_30 | 0.9820 | 0.9643 | 0.0177 |
| blur_20 | 0.9920 | 0.9667 | 0.0253 |
| resize_025 | 0.9685 | 0.9584 | 0.0101 |
| noise_010 | 0.9770 | 0.9614 | 0.0156 |

**The gap is 1.0–2.5 points.** That is the single most important number in this report. Dev shares all five generator families with train; the protected set shares none. A model that had learned "what FLUX and SD VAE decoders look like" rather than a general generation artifact would fall off a cliff here. It does not.

The concern raised earlier in the project — that dev is structurally blind to generator-family overfitting — is now answered with evidence rather than argument.

---

## 3. Full 15-condition dev matrix

`reports/matrix_dev15_epoch1_step7000.csv`, n = 2,000 per condition.

| condition | bAcc | AUROC | Δclean | condition | bAcc | AUROC | Δclean |
|---|---|---|---|---|---|---|---|
| clean | 0.9995 | 1.0000 | — | resize_050 | 0.9975 | 1.0000 | −0.002 |
| jpeg_90 | 0.9985 | 1.0000 | −0.001 | **resize_025** | **0.9685** | 0.9961 | −0.031 |
| jpeg_70 | 0.9965 | 1.0000 | −0.003 | noise_002 | 0.9975 | 1.0000 | −0.002 |
| jpeg_50 | 0.9945 | 0.9998 | −0.005 | noise_005 | 0.9935 | 0.9998 | −0.006 |
| jpeg_30 | 0.9820 | 0.9987 | −0.018 | noise_010 | 0.9770 | 0.9984 | −0.023 |
| blur_05 | 0.9990 | 1.0000 | −0.001 | jitter_pm20 | 0.9995 | 1.0000 | 0.000 |
| blur_10 | 0.9980 | 1.0000 | −0.002 | crop_80 | 0.9990 | 1.0000 | −0.001 |
| blur_20 | 0.9920 | 0.9997 | −0.008 | | | | |

**Mean bAcc 0.9928 · mean AUROC 0.9995 · worst case 0.9685.**

`jitter_pm20` (−0.000) and `crop_80` (−0.001) are essentially free: the model is not relying on colour statistics or framing. The three hardest cells — resize_025, noise_010, jpeg_30 — are all high-frequency-destroying operations, exactly as PLAN D9 predicted, and consistent with a detector reading generator artifacts rather than semantics.

---

## 4. Gate G2 — PASSED

The Day-0 floor was measured on the old `floor_pool` dev and was **not a valid reference** for the current data. It was re-measured here: frozen DINOv3-L backbone + linear head, identical train/dev, identical crop and normalization (`configs/probe_floor_current.yaml`, hash `43b0e32d586d`). Only the head learns, so this isolates what end-to-end fine-tuning actually bought.

| condition | floor (frozen) | e2e (step 7,000) | gain |
|---|---|---|---|
| clean | 0.845 | 0.9995 | +0.1545 |
| jpeg_30 | 0.763 | 0.9820 | +0.2190 |
| blur_20 | 0.829 | 0.9920 | +0.1630 |
| resize_025 | 0.777 | 0.9685 | +0.1910 |
| noise_010 | 0.737 | 0.9770 | +0.2400 |
| **worst-case** | **0.737** | **0.9685** | **+0.2315** |

**G2 requires e2e ≥ floor + 0.10 = 0.8370. Achieved 0.9685 — PASS with a margin of +0.1315.**

This also validates PLAN D2 empirically: the research report cited B-Free's +26.7 bAcc for end-to-end over a linear probe; we measure **+23.2** on our own data. Full fine-tuning was worth the GPU time, and the frozen-probe fallback lane would have cost ~23 points.

---

## 5. Calibration

**No calibration has been applied.** `calibration` in the checkpoint is `{temperature: None, alpha: None, threshold: None}`; every figure above uses a naive 0.5 threshold.

Despite that, ECE is already low — 0.013 clean and 0.035 worst on the protected set, 0.0007–0.024 on dev. D7's temperature + α fit on the calib split should therefore be a small correction rather than a rescue. That is a good sign: a badly miscalibrated model usually indicates the decision boundary is being propped up by something fragile.

FPR@95TPR on the protected set is 0.000 clean, rising to 0.032 at resize_025 — comfortably inside the FPR ≤ 5% operating point D7 specifies for threshold freezing.

---

## 6. Caveats

1. **Protected-set duplicates.** 5,124 of the 13,843 rows are byte-identical duplicates (same sha256 and pHash at different archive paths), all on the fake side: 6,932 DALL·E rows across 1,808 groups, some appearing 5×. The fake class has 8,843 rows but only ~3,700 distinct images. Scoring every row is correct — that is the benchmark as delivered — but the effective sample is smaller than n suggests, so the CIs above are narrower than the distinct-image count would justify. This is a property of the organiser's data, verified against the source denylist (1,808 duplicate sha256 lines, 0 duplicate paths).
2. **5,000 reals, not 4,998.** The organiser's demo subset is identified by WildFake's internal ids (`img158957.jpg`), which have zero overlap with canonical val2017 filenames (`000000212226.jpg`), and WildFake's COCO copies are not downloaded. The 2 omitted images cannot be identified locally, so the full canonical archive is scored — a 0.04% difference, and all 5,000 are denylisted from training regardless.
3. **Five conditions on the protected set, not fifteen.** clean + the four worst-case atoms. The full 15 is ~208k transformed images and belongs with the frozen model.
4. **Dev is a 2,000-image subset** of the 20,023-row dev split, chosen so all 15 atoms could be cached and evaluated repeatedly.
5. **~1% of protected fakes are below the 448 crop** (1st percentile short side 326 px) and are reflect-padded at inference. No training image was ever padded, so this is a small out-of-domain slice worth checking in error analysis.

---

## 7. Adjustments made during this evaluation

- **Epoch-1 checkpoints preserved outside the pruning rotation.** The trainer keeps only the newest 2 recovery checkpoints and overwrites `best.pt` whenever a later step wins, so epoch 1 would have been destroyed by epoch 2. Now saved as `epoch1_best.pt` (step 7,000, weights), `epoch1_end_step8500.pt` and `epoch1_resumable_step8000.pt` (full optimizer/EMA/RNG state, so epoch 2 can be rolled back).
- **Matrix filename collision fixed.** `eval_matrix.py` names its CSV `matrix_<config_hash>_<model_hash>.csv`, so the protected run overwrote the dev run — same checkpoint, same name. Both are now preserved under explicit names (`matrix_dev15_…`, `matrix_protected_…`).
- **`--resume` added to `scripts/train.py`** so epoch 2 continues in the same code path that started the run.
- **Floor re-measured** on current data, making G2 evaluable for the first time.

---

## 8. Verdict

Epoch 1 is a submission-viable model. On an unseen generator under the benchmark's own transforms it holds **0.958–0.989 bAcc with a 3.1-point maximum degradation**, passes G2 with a wide margin, and shows a dev→protected gap of only 1–2.5 points — evidence that it learned a generation artifact rather than the five generator families it was trained on.

Training was stopped at step 8,500 with worst-case gains of +0.001 per 500 steps over the last 2,000. Epoch 2 is unlikely to move the headline much; its value is as a check on whether more passes help or begin to overfit, with the epoch-1 state preserved for rollback either way.

**Still unratified:** the 448-crop decision, which excludes the adm/ddpm/vqdm/gan families from training.
