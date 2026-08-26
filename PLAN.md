# Track 5 — Project Structure & Schedule (decided 2026-08-26)

Derived from [research/FINAL-track5-merged-research-report.md](research/FINAL-track5-merged-research-report.md). Every decision below cites the controlled evidence that makes it the best available choice under the four binding constraints:

- **C1 — <2B parameters** (counting rule unconfirmed → stay comfortably under with the *full* checkpoint; report §15 item 5)
- **C2 — Protected demo set** (COCO val2017 × WildFake DALL·E Advanced): never used for training, model selection, or calibration
- **C3 — Hackathon scale**: ~1 GPU (24 GB class), ~200 GB disk, days not weeks
- **C4 — Public repo**: license-clean stack — MIT/Apache preferred; gated assets only with terms read and obligations tracked (DINOv3 License read 2026-08-26 → admissible, see D1/D11)

---

## 1. System decisions

### D1. Backbone: DINOv3 primary, DINOv2-with-registers proven fallback — both end-to-end fine-tuned
**Choice:** `facebook/dinov3-vitl16-pretrain-lvd1689m` (303.1M, DINOv3 License, gated) as primary; `facebook/dinov2-with-registers-base`/`-large` (86.6M/304.4M, Apache-2.0, ungated) as zero-friction fallback and ensemble member. Whichever is available first starts training — DINOv2+reg is never blocked on gate approval.
**Why best:** NTIRE 2026 — functionally the same task — had DINOv3 in ranks 1, 2 and 4 (0.9723 robust AUC), and DINOv3 scales on this task where DINOv2 stays flat (frozen: v2 56–61% across all sizes vs v3 64.6→87.5%, §3.2). Four independent evidence lines put the DINO family above CLIP-family (§3.2). The prior blocker was the *unread* license (report open item 7); terms were read 2026-08-26 and permit our use — obligations in D11. ViT-L/16 @ 512px matches NTIRE 1st's input regime and fits C1 with ~85% headroom; ViT-H+ (840.6M) only as a Day-4 extra if compute allows.
**Why keep DINOv2+reg in the pair:** the best *controlled* ablation in the report is B-Free's — DINOv2+reg end-to-end **99.0 AUC / 95.2 bAcc / 0.040 ECE**, +4.1 bAcc from registers alone (§4.1); it is ungated with no share-alike on weights, and adds diversity mass to the ensemble.
**Rejected:**
- *SigLIP2-Giant + linear head* (NTIRE 5th): simpler, but 0.873 robust AUC vs the 0.92+ tier, and 1.87B leaves no budget headroom under the unresolved counting rule.
- *Tiny CNNs (SAFE/NPR/FerretNet) as primary*: collapse under JPEG (97.1→55.1 at Q75). Ensemble-member candidate only.
- *AIDE off-the-shelf*: MIT but 898M and 92.8→69.6 at JPEG Q50; optional ensemble member at best.

### D2. Adaptation: full fine-tune + heavy distortion augmentation
**Why best:** frozen probes lose 18 pts at JPEG Q70 and hit chance at noise σ10 (§3.2 Table 5); B-Free measures **+18.2 AUC / +26.7 bAcc / −0.10 ECE** for end-to-end over a linear probe on the same backbone (§4.1). Robustness comes from adaptation × augmentation, not the frozen representation (§4.2). ViT-B e2e is hours on one 24 GB GPU.
**Fallback lane** (if compute breaks): Effort-style SVD-residual PEFT — 0.19M trainable, ICML'25 oral, near-e2e results.
**Day-0 floor:** frozen DINOv2+reg linear probe (<5 min class of cost) to validate the pipeline and set the number every later run must beat.

### D3. Training data (~120k real / ~200k fake, <150 GB)
| Component | Source | Size | Note |
|---|---|---|---|
| Real #1 | COCO train2017 | ~118k | denylisted vs val2017 (SHA-256 + pHash) |
| Real #2 | WildFake real half (FFHQ/ImageNet/LSUN/LAION) and/or SID_Set reals (OpenImages) | 30–60k | second source blocks source-bias (§7) |
| Fake #1 | WildFake, **all non-DALL·E families** (SD×3 weights, Midjourney, ADM, DDPM/DDIM, VQDM, GANs, Others) | 15–25k each | the benchmark's own distribution minus the protected family |
| Fake #2 | SID_Set FLUX subset | ~20k | modern high-res generator |
| Fake #3 | **VAE reconstructions of our own COCO-train reals** (SD1.5 + SDXL VAE, encoder→decoder only) | ~50k | Aligned-Datasets recipe: ~10× cheaper than generation, perfectly content/format-matched fakes, 83.4 vs 46.5 TPR@5%FPR at 1k images (§7.2) |
**Held out from training entirely:** the whole DALL·E family (§7.5 rule 2).
**Rejected:** CIFAKE (32×32, irrelevant), full GenImage (655 GB, NC), full Community Forensics (1.1 TB; optional 2–5k/generator HF streaming later — diversity beats volume, §9.2), B-Free training set as dependency (nonprofit-only custom license; fallback only).

### D4. Bias-neutralization protocol = hard CI gate (non-negotiable)
Adopt report §7.5 verbatim: manifest (SHA-256, pHash, source, generator, W/H, format, JPEG q-table, recompression count) → denylist → metadata strip → **match compression/size distributions across classes, balance 0/1/2-JPEG histories** → split by generator family and real source → **four shortcut probes (file size, dimensions, JPEG q-table, source-classifier on frozen embeddings); any probe >60% bAcc blocks training** → compression-sweep sanity check.
**Why non-negotiable:** the demo pairs JPEG reals against PNG fakes — the exact "Fake or JPEG?" trap (§7.1). A shortcut model scores well clean and collapses across the 15-condition matrix; bias-neutral training is also the score-maximizing choice.

### D5. Augmentation: the report's §9.2 sampler, verbatim
30% clean / 55% one corruption / 15% two, identical by class. Families: JPEG .30 (Q25–100, two libjpeg paths, balanced single/double history), resize .20 (0.20–1.00×, 4 kernels), blur .15 (σ 0–2.3), noise .15 (σ 0–0.11, half before/half after JPEG), color jitter .10 (0.75–1.25, **no hue**), crop .10 (0.75–1.00). Train only 10–20% past test severity.
**Explicitly banned:** MixUp/CutMix (B-Free: 78.6 vs 92.2 bAcc), broad chromatic aug (SPAI: 91.0→80.5), hue/solarize.
**Early stopping / model selection metric:** the **minimum** bAcc across {clean, JPEG Q30, blur σ2, resize 0.25×, noise 0.10} on dev — worst-case optimization (§9.2).

### D6. Recipe
BCE loss (focal γ=2/α=0.5 as config toggle), AdamW, cosine + warmup, lower backbone LR than head, EMA + SWA, bf16, grad-accum. Input: **random native-resolution crops ~448–504px, never resize the crop source** (B-Free 504px protocol; crop-vs-resize evidence +6–15% AUC, §6.3); pad images smaller than the crop. 336px fallback if compute is tight.
**Stretch (gate-controlled):** clean/distorted logit-consistency (KL) pairs à la NTIRE 3rd — only if the clean→degraded dev gap stays >5 bAcc.

### D7. Calibration & threshold (§10.2, §10.4)
Temperature + one-scalar logit bias (α) fitted on **our own** deployment-mixture calibration split (equal clean/transformed buckets). One threshold frozen on clean dev (max bAcc s.t. FPR ≤ 5%), **never refit per transform**. Submit `sigmoid(z/T)` = p(AIGC); decode failures to an error log, never a silent 0.5.
**C2 note:** the calibration paper's unsupervised variant needs only 10 unlabeled images — using protected images for it would still be "calibration on the protected set" → banned. Dev mixture only.

### D8. Ensemble: earn membership, don't assume it
Primary submission = single best calibrated model. Members added only past **gate G3 (≥ +0.5 robust-AUC on dev)**, fused by plain probability averaging (NTIRE 1st) — no gating networks (INTSIG-style dual gates = hackathon complexity risk). Candidates in order of value/cost: (a) DINOv3-L + DINOv2+reg average (both exist by Day 3, ~390–610M total), (b) retrained SAFE 1.44M low-level expert (Apache-2.0, 2–3h, paradigm diversity per NTIRE conclusion 6), (c) DINOv3 ViT-H+ or a second seed if compute allows. All combinations stay <2B even summed.

### D9. Evaluation harness is a first-class product
- `transforms_eval` frozen Day 0: the 15 atoms (clean; JPEG 90/70/50/30; blur 0.5/1/2; resize 0.5/0.25×; noise .02/.05/.10; jitter ±20%; crop 80%), seeded, unit-tested for byte determinism, applied **identically to both classes with identical re-encode per condition**.
- Matrix runner over cached transformed sets → per-condition AUROC / AP / bAcc@frozen-τ / FPR@95%TPR / ECE / Δclean, per-generator dev table, bootstrap 95% CIs, model+transform hashes in the CSV (§10.3–10.4).
- **Dev-OOD:** held-out generator families + held-out real slices. **Shadow set:** WildFake DALL·E-*typical* (not Advanced) — queried max twice (Day 3, Day 5), logged, to keep C2 honest while getting a family-proximity read.
- Crop-80% side-vs-area ambiguity: evaluate both, log the convention (§9.2 note).

### D10. Explainability & demo (judge-facing)
Stability strip (p on clean / JPEG70 / 0.5×) + patch evidence map from real local logits + spectrum card — the two high-fidelity options in §12. Gradio app: upload → p(AIGC) + strip + map. Error-analysis notebook per §11.3 (≥12 FP + ≥12 FN, stratified, no cherry-picking). **No generated natural-language "reasons"** (§12: very low fidelity). The robustness-matrix heatmap + the bias-protocol story is the differentiator slide.

### D11. Licensing posture
Repo code MIT (our code is not a derivative of DINO Materials; weights are runtime-fetched from HF, never vendored).
**DINOv3 obligations (license read 2026-08-26):** it is a Llama-style community license — royalty-free, commercial use allowed, fine-tuning and redistribution of derivatives allowed. We must: (1) if we publish fine-tuned DINOv3 weights, distribute them **under the DINOv3 License with a copy of the agreement attached** — never relicense them MIT; (2) acknowledge DINOv3 use in any write-up (§1b.ii); (3) accept the indemnification clause (§5b) and Meta's unilateral-amendment right (§8) — negligible practical exposure at hackathon scale. The weights are **not OSI-open-source** (trade-control field-of-use restriction), which only matters if the track mandates OSI licensing → D12 Q5.
Everything else on the critical path stays Apache/MIT (DINOv2+reg, SAFE, own code); never vendor B-Free weights or NC datasets.

### D12. Questions to organisers (send Day 0, don't block on answers)
1. Does <2B count full checkpoint, vision tower only, or ensemble total? (determines SigLIP2-Giant admissibility; we stay safe regardless)
2. "Center crop 80%" — side or area?
3. Are AI-restored/upscaled authentic photos "real"?
4. Confirm the 8,843-file DALL·E Advanced manifest against the delivered folder (report open item 9) before freezing the denylist.
5. Any license requirements on submitted models/weights (e.g. OSI-only)? A DINOv3-derived checkpoint carries Meta's community license (share-alike, non-OSI); if OSI is required we ship the DINOv2+reg model instead.

---

## 2. Repository layout

```
track5/
├── configs/                  # one YAML per experiment; hash logged into every artifact
├── data/
│   ├── manifests/            # parquet: sha256, phash, source, generator family, W/H,
│   │                         #   format, jpeg qtable, recompression count, split
│   ├── denylist/             # hashes: all COCO val2017 + entire WildFake DALL·E family
│   └── raw/ cache/           # gitignored
├── src/track5/
│   ├── data/                 # manifest builder, denylist, loaders, shortcut_probes.py
│   ├── transforms/
│   │   ├── eval_atoms.py     # FROZEN Day 0 — seeded, deterministic, tested
│   │   └── train_sampler.py  # §9.2 sampler — deliberately separate module
│   ├── models/               # backbone wrappers, heads, ensemble average
│   ├── train/                # loop, EMA/SWA, consistency (stretch)
│   ├── eval/                 # matrix runner, metrics, calibrate.py, threshold.py, bootstrap
│   └── explain/              # patch map, stability strip, spectrum card
├── scripts/                  # download_*.py, build_manifests.py, make_vae_recons.py,
│                             # train.py, eval_matrix.py, predict.py  (stable CLI = §10.4 contract)
├── tests/                    # transform determinism, denylist hit, probe gate, output schema
├── runs/                     # gitignored checkpoints/logs
├── reports/                  # robustness CSV+heatmap, error_analysis.md, model card
└── app/                      # gradio demo
```

Structural principles: eval atoms and training sampler are separate modules (determinism vs stochasticity); `predict.py` is a stable contract from Day 0 so the submission path is never blocked by research code; every artifact carries config + transform hashes; probes run in CI as a merge gate.

---

## 3. Timeline (Day 0 = Wed Aug 26 → submission Tue Sep 1)

Workstreams: **[D]**ata, **[M]**odel, **[E]**val/demo — parallelizable across 2–3 people, serializable solo (order as listed). GPU-idle time always overlaps CPU work.

| Day | Focus | Deliverables | Gate |
|---|---|---|---|
| **0 · Wed 8/26** | Scaffold + floor | Repo + CI; organiser email; **DINOv3 access request**; downloads launched in background (COCO train2017, WildFake subsets, SID_Set sample); `eval_atoms.py` frozen + determinism tests; `predict.py` contract; frozen linear-probe **floor** on a 2k/2k sample through a mini-matrix | **M0:** end-to-end wiring proven; floor logged |
| **1 · Thu 8/27** | Data integrity | Manifests, denylist, metadata strip, class-distribution matching, JPEG-history balancing; VAE reconstructions (~50k, ~2h GPU); splits by family+source; dev / calib / shadow sets; train sampler + visual audit notebook; 30-min ViT-B smoke run | **G1: all 4 shortcut probes <60% bAcc — no training until green** |
| **2 · Fri 8/28** | First real model | First full e2e run (launched Thu night if data ready) — **DINOv3-L if gate approval landed, else DINOv2+reg-B**; dev matrix v1; per-condition table; error analysis v1 → worst cells (expect Q30 / noise .10 / resize .25) | **G2: e2e ≥ floor +10 bAcc on degraded dev**, else fix recipe before scaling |
| **3 · Sat 8/29** | Second backbone + calibrate | Recipe fixes; **second-backbone run** (the one not trained Day 2; overnight OK); calibration (T + α) + threshold-freeze procedure; 5-crop TTA measured; shadow check #1 (logged) | TTA kept only if ≥ +0.5 robust-AUC for the 5× cost |
| **4 · Sun 8/30** | Ensemble decision | DINOv3-L vs DINOv2+reg vs average; optional SAFE expert (2–3h) if error analysis shows missing low-level cues; optional consistency-loss rerun; optional DINOv3 ViT-H+ / second seed; candidate list frozen EOD | **G3: each added member ≥ +0.5 robust-AUC on dev** |
| **5 · Mon 8/31** | Freeze + report + demo | Model freeze (config hash); full dev robustness matrix + CIs + heatmap; §11.3 error analysis (24 cases); model card; Gradio demo; README; shadow check #2 (final); **freeze T, α, τ** | Everything submission-shaped except the protected run |
| **6 · Tue 9/1** | Submission + buffer | **Protected-set inference, once** (clean + transformed matrix ≈ 1–2.5h per §13, budget 3h); package predictions JSON + robustness CSV + repo; demo rehearsal; one-pager with heatmap + bias-protocol story; half-day slack | Submitted |

**If the deadline is tighter than 7 days, cut in this order:** ViT-H+/second-seed extras → ensemble + TTA → second backbone (keep whichever trained first) → consistency loss → spectrum card. **Never cut:** Day-1 data integrity + probes (the trap the whole report warns about), the frozen eval atoms, calibration/threshold freeze, the single protected-set run. 48-hour floor = Day 0 + Day 1 + overnight ViT-B + Day 5-lite + submission.

## 4. Risk register

| Risk | Mitigation |
|---|---|
| WildFake (ModelScope) slow/unreachable | Started Day 0 in background; fallbacks: SID_Set + D3 (MIT) + GenImage HF-streamed samples; last resort B-Free set (nonprofit license, use-only) |
| GPU shortage / OOM at 504px | 336px fallback; Effort PEFT lane (0.19M trainable) |
| DALL·E-Advanced manifest ≠ 8,843 files | Verify delivered folder before freezing denylist (organiser Q4) |
| Disk (<~200 GB needed) | Subset downloads, stream-and-sample, purge transform caches per model |
| DINOv3 HF gate approval delayed | Start on DINOv2+reg (identical recipe, config-only swap); DINOv3 joins when granted — critical path never blocks |
| Track rules require OSI-licensed submission | Ship the DINOv2+reg model (Apache-2.0 base) as the submitted artifact; DINOv3 stays a dev-side ensemble member or is dropped (D12 Q5) |
| Overfitting the shadow set | Hard policy: 2 queries total, both logged |
