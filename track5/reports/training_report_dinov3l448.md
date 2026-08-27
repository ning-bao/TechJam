# Training examination report — DINOv3-L/16 @ 448px

**Run:** `runs/dinov3l448_d4` · **Config:** `configs/dinov3l448_d4.yaml` (hash `4ed11edd6769`)
**Status: IN PROGRESS** — step 6,000 / 16,524 (36%)
Started 2026-08-27 13:43 · snapshot 16:25 · projected finish ~20:50

This is the first end-to-end run of the project. It is a **Day-2 deliverable produced on Day 1**, because the data pipeline and gate G1 landed a day early.

---

## 1. What is being trained

| | |
|---|---|
| Backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` — **primary, not the fallback** |
| Parameters | **303,130,625** (0.303 B) — 15% of the 2 B budget (constraint C1) |
| Head | linear on CLS pooling |
| Adaptation | full end-to-end fine-tune (PLAN D2), backbone unfrozen |
| Input | random **448×448 native-resolution crop**, never resized (PLAN D6) |
| Precision | bf16 + activation checkpointing |
| Batch | micro-batch 2 × grad-accum 8 = **effective 16** |
| Optimiser | AdamW, lr_head 1e-3 / lr_backbone 1e-5, wd 0.05, 5% warmup |
| Loss | BCE |
| EMA | 0.999 (SWA off) |
| Schedule | 2 epochs = **16,524 optimiser steps** |

The gated DINOv3 checkpoint resolved successfully, so the DINOv2-with-registers fallback declared in the config was never used. Verified by reloading `best.pt` and reading `config.model.backbone` plus the parameter count.

### Why 448 and not the 512 of `configs/dinov3l512.yaml`

The training pool is filtered to `min(W,H) >= crop` so **nothing is ever padded**. Padding would apply almost exclusively to fakes (WildFake generators emit 128–512 px; reals are continuous, median short side 428) and would hand the model a free "padded implies fake" rule. COCO reals are overwhelmingly 640×480, so a 512 crop would pad them; 448 fits COCO (480), WildFake sd/other (512) and SID_Set (1024) alike, and is a multiple of the 16 px patch.

**Cost of that decision:** the adm / ddpm / vqdm / gan families (128–256 px) cannot supply a 448 crop and are excluded, leaving five fake families. This is a deliberate deviation from PLAN D3's "all non-DALL·E families" and is **still awaiting ratification**. The excluded families are retained as a natural held-out-generator OOD set.

---

## 2. Data the model is seeing

**train 132,186 rows — 66,093 real / 66,093 fake (exactly balanced).**

| fake family | n | real source | n |
|---|---|---|---|
| flux (SID_Set) | 18,556 | coco_train2017 | 32,262 |
| vae_sd15 | 13,919 | sid_set | 28,416 |
| vae_sdxl | 13,513 | wildfake | 5,415 |
| sd (WildFake) | 10,281 | | |
| other (WildFake) | 9,824 | | |

Built from 416,093 candidate rows: 186,009 dropped below the 448 crop, 50 exact duplicates, then per-bucket capping, class balancing, and size-distribution matching (160,076 → 132,186; 27,890 dropped across 80 container×size strata).

**Integrity, verified on the final manifests:**

- **0** denylist hits — no COCO val2017, no WildFake DALL·E of any kind (C2 holds)
- **0** sha256 overlap between train / dev / calib
- **0** cross-split content leaks across all 28,413 VAE-reconstruction/source pairs. A reconstruction is pixel-content-identical to its source real, so the pair is split as one unit via 161,638 content groups
- **0** parquet-row paths (each would cost an ~844-image row-group decode per read)

Each sample passes through PLAN D4 container normalization (`data.normalize: true`) and then the §9.2 distortion sampler, in that order — normalization models "the container it was delivered in", distortion models "what happened to it in the wild".

---

## 3. Gate G1 — passed before training was allowed to start

Probes measured on **post-crop, post-normalization** metadata, i.e. what the model actually receives, not the delivered container.

| probe | raw corpus | after D4 treatment |
|---|---|---|
| file_size | 0.589 | **0.497** |
| dimensions | 0.732 | **0.500** |
| jpeg_quality | **0.974** | **0.500** |

The raw 0.974 is the "Fake or JPEG?" trap measured on our own data: untreated, a classifier scores ~97% from the JPEG quantization table alone, and collapses under the benchmark's re-encodes.

**Probe 4 — frozen DINOv3-L embeddings** (bf16; fp16 returns NaN for this backbone). The 0.60 threshold deliberately does not gate this one — a strong backbone *should* separate the classes; that is the task.

| probe | bAcc |
|---|---|
| label_all | 0.900 |
| **label_matched** (1,552 content-matched pairs) | **0.920** |
| real_source | 0.829 |
| fake_family | 0.835 |
| **content-reliance gap** | **−0.021** |

`label_matched` pits a COCO real against **its own VAE reconstruction** — identical content, so separability there can only come from the generation artifact. It scores *higher* than the mixed set, which is the evidence that the model is not keying on "COCO photo vs generated art". Provenance remains readable (real_source 0.829) but is not needed.

---

## 4. Results so far

Balanced accuracy on the 2,000-image dev subset, evaluated every 500 steps against pre-cached, byte-deterministic transforms. `worst_case_bacc` — the minimum across all five conditions — is the model-selection metric (PLAN D5).

| step | % | clean | jpeg_30 | blur_20 | resize_025 | noise_010 | **worst-case** |
|---|---|---|---|---|---|---|---|
| 500 | 3% | 0.793 | 0.659 | 0.760 | 0.715 | 0.625 | **0.625** |
| 1,000 | 6% | 0.911 | 0.759 | 0.865 | 0.833 | 0.727 | **0.727** |
| 1,500 | 9% | 0.964 | 0.833 | 0.913 | 0.873 | 0.841 | **0.833** |
| 2,000 | 12% | 0.988 | 0.900 | 0.956 | 0.922 | 0.894 | **0.894** |
| 2,500 | 15% | 0.994 | 0.930 | 0.972 | 0.936 | 0.934 | **0.930** |
| 3,000 | 18% | 0.996 | 0.950 | 0.982 | 0.948 | 0.943 | **0.943** |
| 3,500 | 21% | 0.998 | 0.961 | 0.983 | 0.946 | 0.952 | **0.946** |
| 4,000 | 24% | 0.998 | 0.967 | 0.986 | 0.953 | 0.961 | **0.953** |
| 4,500 | 27% | 0.999 | 0.973 | 0.987 | 0.953 | 0.961 | **0.953** |
| 5,000 | 30% | 0.999 | 0.975 | 0.989 | 0.955 | 0.964 | **0.955** |
| 5,500 | 33% | 0.9995 | 0.978 | 0.990 | 0.962 | 0.969 | **0.962** |
| 6,000 | 36% | 0.9995 | 0.980 | 0.990 | 0.964 | 0.972 | **0.964** |

**Best checkpoint: step 6,000, worst-case bAcc 0.964.**

Observations:

1. **Monotone improvement on every condition** — no instability, no divergence, no sign of the clean/degraded gap widening.
2. **Clean is saturated** (0.9995) from ~step 4,000. The remaining headroom is entirely in the degraded conditions, which is what the worst-case selection metric is designed to chase.
3. **`resize_025` is the binding constraint** at 0.964, exactly as PLAN D9 predicted ("expect Q30 / noise .10 / resize .25"). A 0.25× downscale-then-upscale destroys the high-frequency artifacts a detector relies on.
4. **The clean-to-worst gap has closed from 0.168 (step 500) to 0.036 (step 6,000)** — the number that matters for a robustness track. The model is not simply learning clean-image detection and degrading gracefully.
5. **Diminishing returns are clearly visible.** Worst-case gained +0.302 over the first 6,000 steps but only +0.011 over the last 1,500. The remaining 10,524 steps are unlikely to deliver much beyond ~0.97.

---

## 5. Throughput and cost

| | |
|---|---|
| Rate | 763 s per 500 optimiser steps (1.525 s/step) |
| Images seen | 96,000 of 264,372 |
| GPU | RTX 5070 Ti — **100% utilisation**, 14.4 / 16.3 GiB, 70 °C, 285 W |
| Elapsed | 2 h 41 m · **remaining ≈ 4.5 h, ETA ~20:50** |
| Eval overhead | 33 evals × ~133 s ≈ 1.2 h of the total |

VRAM headroom is only ~1.9 GiB, which is why no second GPU job has been run alongside this one.

---

## 6. Caveats — how NOT to read these numbers

1. **These are in-distribution results.** Dev shares all five generator families and all three real sources with train. They are not an estimate of protected-set (DALL·E Advanced) performance. The honest generalization reads are the held-out DALL·E shadow set (2 queries budgeted) and the excluded 256 px families.
2. **The dev subset is 2,000 images**, not the full 20,023-row dev split. It was chosen so the five worst-case transforms could be pre-cached and evaluated 33 times without dominating the run. Final numbers must come from the full 15-condition matrix.
3. **The floor baseline is stale.** The Day-0 frozen-probe floor (clean 0.640 / worst-case 0.619) was measured on the old `floor_pool` dev — different fake families (adm/ddpm/other/sd), different resolutions. **Gate G2 ("e2e ≥ floor + 10 bAcc") cannot be evaluated against it.** A frozen probe must be re-run on the current train/dev before G2 is meaningful; it was not run alongside training because of the 1.9 GiB VRAM headroom.
4. **No calibration yet.** `calibration` in the checkpoint is `{temperature: None, alpha: None, threshold: None}`; all bAcc figures use a naive 0.5 threshold. Temperature, α and the frozen threshold are Day-3 work (PLAN D7).
5. **VAE reconstructions are 21% of the fake class.** They are a deliberate anti-shortcut device, not a target distribution. High dev accuracy partly reflects detecting a VAE decoder fingerprint, which is easier than detecting an unseen generator.

---

## 7. Immediate next steps

1. **Re-measure the floor** on the current train/dev (frozen linear probe) so G2 has a valid reference — as soon as the GPU frees.
2. **Full 15-condition robustness matrix** on the final checkpoint. All 15 atom caches are already built (30,000 transformed dev images, 0 errors), so this runs immediately.
3. **Calibration + threshold freeze** (PLAN D7) on the untouched 20,019-row calib split.
4. **Per-generator dev table** to see whether any single family is carrying the average.
5. **Ratify or overturn the 448-crop decision** (§1), which currently excludes four generator families.
