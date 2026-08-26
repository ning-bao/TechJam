# Track 5 — Robust Detection of AI-Generated Images Under Real-World Transformations
## Final Merged Research Report

**Compiled:** 2026-08-25
**Sources merged:** (A) *Track 5 Model Viability Research* — backbone/frontier-model and training-strategy survey; (B) *Robust AIGC Image Detection: Hackathon Research and Build Brief* — method-comparison, dataset-risk and build-planning brief.
**Constraint tracked throughout:** models under **2B parameters**.
**Protected demonstration set:** COCO val2017 (4,998 real) vs WildFake DALL·E Advanced (8,843 AIGC). Not for training, model selection, or calibration.

### Reading the evidence

Numbers from different papers are not directly rankable unless they share training data, test data, preprocessing and threshold policy. Three comparisons in this document are internally controlled and therefore carry more weight than cross-paper means:

1. **AIDE's common 16-generator reproduction table** — clean, JPEG Q95/90/75/50, blur σ 1/2/3/4, all methods re-run under one protocol.
2. **B-Free's bias-controlled FakeBench evaluation and architecture ablation** — same training data, same evaluation, backbone and fine-tuning mode varied.
3. **WildFake's within-paper degradation ablations** — CNN vs ViT vs DIRE vs LASTED under one degradation suite.

ACC = accuracy as reported; bAcc = balanced accuracy; AP = average precision; AUC = AUROC; ECE = expected calibration error; NR = not reported. Parameter counts are whole-checkpoint figures from the Hugging Face Hub index unless noted; for dual-tower VLMs the Hub figure **includes the text tower**, so the deployable vision-only count is lower.

---

## Table of Contents

0. [Reconciliation log — conflicts found and resolved](#0-reconciliation-log--conflicts-found-and-resolved)
1. [Field landscape and why papers disagree](#1-field-landscape-and-why-papers-disagree)
2. [NTIRE 2026 — the closest prior art](#2-ntire-2026--the-closest-prior-art)
3. [Backbones under 2B parameters](#3-backbones-under-2b-parameters)
4. [Adaptation strategy: linear probe vs PEFT vs end-to-end](#4-adaptation-strategy-linear-probe-vs-peft-vs-end-to-end)
5. [Method landscape with robustness and licence status](#5-method-landscape-with-robustness-and-licence-status)
6. [Robustness evidence for the exact Track 5 transforms](#6-robustness-evidence-for-the-exact-track-5-transforms)
7. [The dataset-bias trap](#7-the-dataset-bias-trap)
8. [Datasets](#8-datasets)
9. [Training recipe evidence](#9-training-recipe-evidence)
10. [Evaluation, metrics and calibration](#10-evaluation-metrics-and-calibration)
11. [False positives and error analysis](#11-false-positives-and-error-analysis)
12. [Explainability options](#12-explainability-options)
13. [Compute figures](#13-compute-figures)
14. [Off-the-shelf checkpoints](#14-off-the-shelf-checkpoints)
15. [Open items](#15-open-items)
16. [Reference pack](#16-reference-pack)

---

## 0. Reconciliation log — conflicts found and resolved

Nine substantive conflicts or gaps existed between the two source reports. Each was re-verified against primary sources during this pass.

| # | Conflict | Resolution | Verified against |
|---|---|---|---|
| 1 | **B-Free FakeBench table alignment.** Report A extracted FatFormer DALL·E 2 = 52.4/49.5; Report B extracted 45.3/48.1. | **Report B was correct.** Report A's extraction was shifted one column. Full column order is ProGAN, StyleGAN, FuseDream, VQDM, GLIDE, CogView2, DALL·E 2, DALL·E 3, SD, Midjourney, AVG. Corrected table in [§5](#54-b-frees-bias-controlled-fakebench-re-evaluation). | Targeted re-fetch of [arXiv:2412.17671v2](https://arxiv.org/html/2412.17671v2) Appendix E Table 10 with explicit header extraction |
| 2 | **SAFE parameter count / backbone.** Report A hedged "commonly described as ResNet-50 (~25.6M), unverified"; Report B asserted 1.44M. | **Report B was correct.** SAFE adopts the lightweight ResNet from Tan et al. 2024b (the NPR paper) with **1.44M parameters**, explicitly "to meet real-time requirements". It does not introduce a new backbone. Report A's ResNet-50 attribution was wrong. | SAFE paper [arXiv:2408.06741v1](https://arxiv.org/html/2408.06741v1) + [survey 2502.15176](https://arxiv.org/html/2502.15176v2) + [FerretNet comparison table](https://arxiv.org/html/2509.20890) |
| 3 | **SPAI repository and parameters.** Report A linked `mever-team/spai`; Report B linked `kartyg23/spai` and estimated 86–90M. | Both partly right. **`mever-team/spai` is the official repo** (`kartyg23/spai` is a fork). Backbone is **ViT-B/16 + MFM pretraining ≈ 86M** — Report B's estimate confirmed. Licence Apache-2.0; weights public. | [github.com/mever-team/spai](https://github.com/mever-team/spai) |
| 4 | **AIDE parameter count.** Report A did not state one; Report B computed ≈898M. | **Confirmed.** `timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k` = **846.5M**, plus two ResNet-50s (2 × 25.6M) and heads ≈ **898M total**. Fits under 2B but consumes ~45% of the budget. | [HF model index](https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k) |
| 5 | **DINOv3 entirely absent from Report B.** Report B's backbone table stops at DINOv2. | Material gap. NTIRE 2026 ranks 1, 2 and 4 all used DINOv3; two 2026 papers show DINOv3 **scales** on this task where DINOv2 does not. Filled in [§3](#3-backbones-under-2b-parameters). | [NTIRE 2604.11487](https://arxiv.org/html/2604.11487v1), [2602.01738](https://arxiv.org/html/2602.01738), [2511.22471](https://arxiv.org/html/2511.22471v1) |
| 6 | **Linear probe vs end-to-end.** Report A cited *Simplicity Prevails* (frozen linear probe on DINOv3 = 96.4% GenImage) as evidence that simple probes suffice. Report B cited B-Free's ablation (linear probe 80.8 AUC vs end-to-end 99.0) as evidence they do not. | **Both are right about different axes.** Frozen probes generalise well *across generators* on clean data; they are *fragile under degradation and threshold shift*. Newly verified robustness table for frozen DINOv3 + linear probe: 87.50% clean → **69.81% at JPEG Q70**, → **55.31% at Gaussian σ=10**. Synthesis in [§4](#4-adaptation-strategy-linear-probe-vs-peft-vs-end-to-end). | B-Free Table 3; [2511.22471](https://arxiv.org/html/2511.22471v1) Table 5 |
| 7 | **FatFormer parameter count.** Report B: "~307M + adapters". FerretNet's comparison table: 492.59M. | Both figures circulate. 307M is the CLIP ViT-L/14 vision tower alone; **492.59M** is the full FatFormer including text encoder and adapters. Use 492.59M for a parameter-budget calculation. | [FerretNet 2509.20890](https://arxiv.org/html/2509.20890) Table |
| 8 | **WildFake total size.** Report A: ">3.7M images". Report B: paper reports 2.56M fake + 1.01M real in one table but 2.68M fake / 3.69M total in text. | **Report B's flag stands** — the paper is internally inconsistent. Treat ~3.6–3.7M as approximate and verify against the ModelScope file listing. Unresolved. | [arXiv:2402.11843v1](https://arxiv.org/html/2402.11843v1) |
| 9 | **DINOv2-with-registers availability.** Neither report noted licensing. | **`facebook/dinov2-with-registers-base` = 86.6M, Apache-2.0, ungated.** Large = 304.4M, Apache-2.0. This contrasts with DINOv3, which is **gated** under a custom licence. Practical consequence for a public hackathon repo. | HF model index |

**Corrections to Report A:** items 1, 2, 3 (repo), 7. **Corrections to Report B:** items 3 (repo URL), 5 (DINOv3 omission).

---

## 1. Field landscape and why papers disagree

### 1.1 Taxonomy

The 2025 survey *Methods and Trends in Detecting AI-Generated Images* ([arXiv:2502.15176](https://arxiv.org/html/2502.15176v2)) groups methods into seven families: spatial-domain analysis; frequency-domain analysis; fingerprint analysis; patch-based analysis; training-free; multimodal vision-language; commercial tools.

Its three generalisation criteria are **cross-family** (GAN↔diffusion), **cross-category** (image classes) and **cross-scene** (dataset distribution). Stated finding: most methods satisfy the first two and **fail cross-scene**; only RIGID and AIGI-Holmes satisfied all three.

### 1.2 The reproducibility problem, quantified

*How well are open sourced AI-generated image detection models out-of-the-box* ([arXiv:2602.07814](https://arxiv.org/html/2602.07814v1)) evaluated 16 methods / 23 pretrained variants across 12 datasets, 2.6M images, 291 generators:

- Ranking instability across datasets: **Spearman ρ 0.01–0.87**
- **37 percentage-point** spread between best and worst detector
- Best out-of-the-box: **Community-Forensics, 75.0% mean accuracy**; worst: AIGCDetectBenchmark_CNNSpot, 37.5%
- Modern commercial generators (Flux Dev, Firefly v4, Midjourney v7) detected at only **18–30%**
- **Training-data alignment explains 20–60% of performance variance within identical architecture families** (AIDE, DRCT) — often exceeding the variance between different architectures

*Is AI Generated Image Detection a Solved Problem?* (AIGIBench, NeurIPS 2025, [arXiv:2505.12335](https://arxiv.org/abs/2505.12335)) evaluated 11 detectors on four tasks — multi-source generalisation, degradation robustness, augmentation sensitivity, test-time preprocessing — and reports **Fake Image Accuracy often approaching 0%** on in-the-wild social media content, "limited benefits from common augmentations" and "nuanced effects of pre-processing".

### 1.3 Why the rankings flip

Four independent mechanisms explain the apparent contradictions between papers:

1. **Clean-benchmark saturation.** On generator-held-out benchmarks, artifact methods (ESSP, PatchCraft, NPR, FatFormer, C2P-CLIP, Effort) look nearly solved.
2. **Bias control reorders everything.** B-Free shows the same generator can move from "real" to "fake" depending on whether the paired real source is COCO or RAISE. Its bias-controlled benchmark cuts FatFormer, RINE, C2P-CLIP and AIDE fixed-threshold bAcc far below their headline results ([§5.4](#54-b-frees-bias-controlled-fakebench-re-evaluation)).
3. **Local fingerprints are perishable.** AIDE leads both clean and corrupted averages under its own protocol, yet falls **92.77 → 69.60 ACC at JPEG Q50**. The hybrid advantage is real; immunity is not.
4. **Human curation erases apparent progress.** On Chameleon, AIDE reports almost every off-the-shelf detector predicts carefully edited 720p–4K fakes as *real*; AIDE itself reaches only **58.37–65.77 ACC** depending on training set.

**A fifth mechanism, added by the merge:** AUC and fixed-threshold bAcc decouple under distribution shift. B-Free documents several methods retaining high AUC while fixed-threshold bAcc sits near chance — so "robustness" must be measured with a frozen operating point, not ranking alone.

---

## 2. NTIRE 2026 — the closest prior art

**[NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild](https://arxiv.org/html/2604.11487v1)** (arXiv:2604.11487; CVPR 2026 Workshops; Gushchin, Shumitskaya, Filippov, Bychkov, Erofeev, Lavrushkin, Antsiferova, Vatolin, C. Chen, S. Tan, R. Timofte et al.; [Codabench #12761](https://www.codabench.org/competitions/12761/)).

Functionally the same task as Track 5: detect AI-generated images that have been cropped, resized, compressed, blurred, noised and colour-adjusted, and generalise to unseen generators.

### 2.1 Setup

- **Dataset:** 108,750 real + 185,750 AI-generated ≈ 294,500 images
- **Real:** CC12M, CommonPool, RedCaps (~12M filtered to 100k train) via resolution thresholding, CLIP-deduplication, VLM categorisation; ~9k for val/test
- **Generated:** ~177k from 20 open-source generators for training; newer models reserved for val/test
- **42 generators**, released 2022–2026; **36 transformations**
- **Metric:** ROC AUC over the full test set including transformed and untransformed images

### 2.2 The 36 transformations

Gaussian Blur, Lens Blur, Motion Blur, Glass Blur; Color Shift, Color Saturation, Color Jitter, Color Quantization, Color Cast, RGB Channel Shift; JPEG, JPEG 2000, Neural Image Compression (JPEG AI, Cheng2020); White Noise, Impulse Noise, Multiplicative Noise, Shot Noise, Speckle Noise, ISO Noise; Brightness Increase/Decrease; Linear Contrast Change, Random Tone Curve, CLAHE; Pixelation; Random Crop, Random Aspect Crop; Downscale; Perspective Transform; Organic Moire; watermark attacks (adversarial embedding, WMForger, invisible watermark insertion).

**Pipeline:** each transformation has multiple magnitude levels sampled independently; images receive **1 to 5 randomly sampled consecutive distortions** from different groups. Both real and generated images pass through the same pipeline. Different distortion sets per split, progressively complicated in later stages to prevent tuning to specific distortion types.

### 2.3 Leaderboard (ranked by Robust ROC AUC)

| Rank | Team | ROC AUC | Robust ROC AUC | Primary backbone | Under 2B? |
|---|---|---|---|---|---|
| 1 | MICV | 0.9974 | **0.9723** | DINOv3 ensembles (4 + 2 models), 512×512 | Depends on variant |
| 2 | Ant International | 0.9972 | 0.9721 | DINOv3-7B dual-expert (~14B total) | **No** |
| 3 | TeleAI-TeleGuard | 0.9786 | 0.9251 | EVA-CLIP + LoRA | Yes |
| 4 | INTSIG | 0.9897 | 0.9130 | 4× DINOv3-Huge (840.6M) + MetaCLIP2-Giant | Per-model yes |
| 5 | vincentlc | 0.9527 | 0.8730 | SigLIP2-Giant-Opt-Patch16-384, single linear head | Yes (1.87B) |
| 6 | UESTC | 0.9729 | 0.8679 | 2× CLIP ViT-L/14 + 2× SigLIP-So400M | Yes |
| 7 | Reagvis Labs | 0.9452 | 0.8603 | — | — |
| 8 | PSU | 0.9227 | 0.8408 | — | — |
| 9 | Shallow Real | 0.9953 | 0.8336 | — | — |

Note the decoupling: "Shallow Real" ranks near the top on clean AUC (0.9953) but 9th on robust AUC (0.8336) — the same clean/robust divergence B-Free documents for calibration.

### 2.4 Top team methods

**1st — MICV.** DINOv3, 512×512, projection layer → MLP head. Corpus: GenImage, WildFake, AIGIBench, CommunityForensics, So-Fake-Set + self-generated (Qwen-Image, Z-Image, FLUX) + closed-source (Seedream, Kling, GPT-Image, Nano-banana-pro) + challenge set. **Hierarchical stochastic augmentation structured by difficulty level**, simple (blur, noise, shifts) → complex multi-stage. Late-fusion probability averaging across two committees. Focal Loss (γ=2.0, α=0.5), Stochastic Weight Averaging, cosine annealing + linear warmup. 32× A100, 10 epochs, ~8 h.

**2nd — Ant International.** DINOv3-7B dual-expert, ~1M images. **Four-level offline augmentation:** L1 clean; L2 1–3 distortions (μ=0, σ=2.5); L3 3–6 (μ=2.5, σ=2.0); L4 fixed 6 (μ=3.5, σ=1.0). Online: horizontal flip + AugMix (m6-w3-d1). Expert 1 high-resolution (512×512, attention pooling, 1 epoch); Expert 2 robustness (288×288, first_token pooling, 10 epochs). TTA with weighted ensembling. EMA, AMP. Inference 2.21 img/s, 78.25 GB VRAM.

**3rd — TeleAI-TeleGuard.** EVA-CLIP, **LoRA on MHSA and FFN blocks**. Added So-Fake and Chameleon; added Speckle Noise, Color Cast, Organic Moire; Gaussian mean set to 3. **LoRA-based Pairwise Training:** clean + distorted pairs in the same batch, plus a feature-correction network. Loss `L_CE(x,y) + α·L_KL(x,x̂) + β·L_MSE(f_x, f'_x̂)`, α=0.5, β=0.25. AdamW 2e-4, cosine. 8× A800, 5 epochs.

**4th — INTSIG.** M1–M4 DINOv3-Huge variants (baseline / data expansion / enhanced augmentation / 448×448), M5 MetaCLIP2-Giant partial fine-tune. Heads `1280→256→2` and `1664→2048→512→256→2`. Data: official + SoFake-OOD + RRDataset + Chameleon + GenImage_val + AIGIBench_test. **Weighted hierarchical fusion with dual gating:** `0.7[0.7(0.75·M1+0.15·M2+0.1·M3)+0.3·M4]+0.3·M5`, Gate-1 strong-consensus correction, Gate-2 anomaly suppression. Horizontal-flip TTA on M3/M4. Separate LR for backbone vs head.

**5th — vincentlc.** Simplest reported pipeline. **SigLIP2-Giant-Opt-Patch16-384 + single linear layer on globally average-pooled patch tokens.** Official training set only (~277k). **"Squish" preprocessing:** direct resize to 384×384 ignoring aspect ratio. Horizontal flip. `distortion_prob=1.0`, up to 3 ops, 5 severity levels — every training image distorted. No ensemble, no TTA. Team ablation: global average pooling over final-layer patch tokens beat CLS token and attention pooling for robustness/stability.

**6th — UESTC.** 2× CLIP ViT-L/14 (224×224) + 2× SigLIP So400M-patch14-384, probabilities averaged. **Two-stage:** binary classification 2 epochs, then **feature-level self-distillation** using epoch-2 intermediate feature maps as dense targets. ~10 GB peak GPU memory.

### 2.5 Organisers' conclusions

1. Most final solutions used **expert-based ensembles**.
2. **Transformer backbones dominated.**
3. **Aggressive robust augmentation**; constant distortion across batches proved effective.
4. **Large-scale, diverse data** mixing open-source datasets, open generators and closed commercial models significantly improved generalisation.
5. **Model scaling and higher input resolution consistently outperformed** smaller variants.
6. **Paradigm diversity** (vision-language + self-supervised + forensic) was complementary.
7. Clear gap between top-2 (~0.972), tier 2 (~0.925) and tier 3 (<0.88) — *"robustness [is] a key differentiating factor."*
8. *"The problem is not yet solved."*

---

## 3. Backbones under 2B parameters

### 3.1 Self-supervised (DINO family)

| Checkpoint | Params | Arch | Licence | Gated |
|---|---:|---|---|---|
| [`dinov3-vits16`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 21.6M | ViT-S/16 | other | Yes |
| [`dinov3-vitb16`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 85.7M | ViT-B/16 | other | Yes |
| [`dinov3-convnext-large`](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m) | 196.2M | ConvNeXt-L | other | Yes |
| [`dinov3-vitl16`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | 303.1M | ViT-L/16 | other | Yes |
| [`dinov3-vith16plus`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | **840.6M** | ViT-H+/16 | other | Yes |
| `dinov3-vit7b16` | ~6.7B | ViT-7B/16 | other | Yes — **over limit** |
| [`dinov2-with-registers-base`](https://huggingface.co/facebook/dinov2-with-registers-base) | **86.6M** | ViT-B/14+reg | **apache-2.0** | **No** |
| [`dinov2-with-registers-large`](https://huggingface.co/facebook/dinov2-with-registers-large) | 304.4M | ViT-L/14+reg | **apache-2.0** | **No** |
| [`dinov2-large`](https://huggingface.co/facebook/dinov2-large) | 304.4M | ViT-L/14 | apache-2.0 | No |

**DINOv3** ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)): 7B flagship on 1.7B curated images, ViT with RoPE. Headline contribution **Gram anchoring** — a Gram-matrix loss on pairwise patch-feature similarities against an earlier anchor teacher, fixing the degradation of dense feature maps over long training (patch locality degrades after ~200k iterations while global metrics keep improving). Then high-resolution fine-tuning, then multi-student distillation into the smaller family.

**Registers** ([Darcet et al., ICLR 2024, arXiv:2309.16588](https://arxiv.org/abs/2309.16588)): suppress high-norm artifact tokens in attention maps. B-Free's ablation isolates the benefit: DINOv2+reg end-to-end **99.0 AUC / 95.2 bAcc** vs plain DINOv2 end-to-end **98.4 / 91.1** — a **+4.1 bAcc** gain from registers alone.

### 3.2 Evidence for DINO-family on this task

Four independent lines agree:

| Source | Finding |
|---|---|
| ***What Truly Matters?*** ([arXiv:2507.10236](https://arxiv.org/html/2507.10236v1)) | Five encoders under RINE: **DINOv2-L/14 94.90 avg AUC** > BLIP2 94.15 > CLIP-L/14 91.92 > CLIP-H/14 89.60 > OpenCLIP-L/14 83.56. Authors: *"CLIP-based methods' reliance on image-text alignment may introduce semantic shortcuts."* |
| **B-Free** (CVPR 2025) | Architecture ablation under identical training: DINOv2+reg e2e **99.0/95.2/0.040 ECE** > DINOv2 e2e 98.4/91.1 > SigLIP e2e 95.4/89.9 |
| ***Simplicity Prevails*** ([arXiv:2602.01738](https://arxiv.org/html/2602.01738)) | Frozen linear probe: **DINOv3 96.4% GenImage / 94.0% in-the-wild** > PE-CLIP 93.8/89.9 > MetaCLIP2 89.2/84.2. +30.4% in-the-wild over DINOv2 |
| **NTIRE 2026** | Ranks 1, 2, 4 all DINOv3; robust AUC 0.9723 |

**Newly added — DINOv3 scales where DINOv2 does not.** *Rethinking Cross-Generator Image Forgery Detection through DINOv3* ([arXiv:2511.22471](https://arxiv.org/html/2511.22471v1), Huang, J. Li, Wen, T. Li, Yang, Qi, Peng, X. Huang, M.-H. Yang, Cheng; Dec 2025):

- Frozen zero-shot on So-Fake-OOD: **CLIP 55.9% vs DINOv3 73.4%**
- Scaling (Table 8): **DINOv2 S/B/L/H flat at 56.4–61.4%**; **DINOv3 ViT-S 64.6% → ViT-7B 87.5%**
- Method: **Fisher-Guided Token Selection (FGTS)**, two protocols — training-free (cosine similarity to real/fake centroids, zero learnable parameters) and linear probe (single FC 4096→2 on **1,000 real + 1,000 fake from a single generator**)
- Results: So-Fake-OOD 75.06% training-free / **87.53%** linear probe; GenImage 88.2 / **92.6**; AIGCDetectionBenchmark 78.99 / **92.45**
- Cost: **under 5 minutes on a single RTX 5090**, ~300× faster than CNNSpot

**But its robustness table is the important caveat** (Table 5, So-Fake-OOD):

| Perturbation | ACC | AUC | AP |
|---|---:|---:|---:|
| Clean | 87.50 | 95.27 | 95.61 |
| Resize 0.75× | 85.34 | 93.49 | 94.11 |
| Resize 0.5× | 84.07 | 92.25 | 93.93 |
| JPEG QF=80 | 75.14 | 86.12 | 87.13 |
| Gaussian σ=5 | 73.54 | 84.11 | 86.41 |
| **JPEG QF=70** | **69.81** | 76.56 | 81.54 |
| **Gaussian σ=10** | **55.31** | **47.79** | 51.61 |

A frozen backbone with a linear probe and **no degradation augmentation** loses 18 points at JPEG Q70 and falls to chance at σ=10. Note the main results use ViT-7B, which **exceeds the 2B limit** — the in-budget variants are ViT-L (303M) and ViT-H+ (840.6M).

### 3.3 Vision-language backbones

| Checkpoint | Params (full) | Vision tower | Licence |
|---|---:|---:|---|
| [`clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | 427.6M | ~307M | — |
| [`siglip2-base-patch16-384`](https://huggingface.co/google/siglip2-base-patch16-384) | 375.5M | ~86M | apache-2.0 |
| [`siglip2-large-patch16-384`](https://huggingface.co/google/siglip2-large-patch16-384) | 881.9M | ~303M | apache-2.0 |
| [`siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384) | 1,136M | ~400M | apache-2.0 |
| [`siglip2-giant-opt-patch16-384`](https://huggingface.co/google/siglip2-giant-opt-patch16-384) | **1,871.9M** | ~1.0B | apache-2.0 |
| [`metaclip-2-worldwide-huge`](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) | **1,858.8M** | — | **cc-by-nc-4.0** |
| [`PE-Core-L14-336`](https://huggingface.co/facebook/PE-Core-L14-336) | ~0.3B vision | 0.3B | apache-2.0 |
| `PE-Core-B16-224` | ~86M vision | 86M | apache-2.0 |
| `PE-Core-G14-448` | 1.88B vision + 0.47B text | 1.88B | apache-2.0 |
| [`eva02_large_patch14_448`](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) | 305.1M | — | mit |

**SigLIP 2** ([arXiv:2502.14786](https://arxiv.org/pdf/2502.14786)): ViT-B (86M), L (303M), So400m (400M), g (~1B). NaFlex (native aspect ratio, variable resolution) exists for B/L/So400m but **not** for giant. Fixed-resolution models use `SiglipModel`; NaFlex needs `Siglip2Model`.

**Perception Encoder** ([arXiv:2504.13181](https://arxiv.org/pdf/2504.13181)): B ≈86M, L ≈0.3B, G ≈1.88B vision; 5.4B image-alt-text pairs curated with MetaCLIP; B and L distilled from G. **SSAFE** ([arXiv:2606.08634](https://arxiv.org/html/2606.08634)) selected PE-Core-G14-448, reporting it *"provides the clearest real/fake separation and the most structured generator clusters"* vs CLIP, SigLIP, DINO variants.

### 3.4 Parameter budget against the <2B rule

| Comfortably under | Near the ceiling | Over the limit |
|---|---|---|
| DINOv3 S/B/L/H+ (21.6M–840.6M) · DINOv3 ConvNeXt-L (196M) · DINOv2+reg B/L (86.6M/304.4M) · CLIP ViT-L/14 (427.6M) · SigLIP2 B/L/So400M (375M–1.14B) · PE-Core-B/L · EVA-02 B/L · AIDE full stack (~898M) · Swin/SwinV2 · all tiny CNNs (NPR/SAFE 1.44M, FerretNet 1.06M, CoDE ViT-T 5.7M) | SigLIP2-Giant-Opt (1.87B) · MetaCLIP2-Huge (1.86B, **non-commercial**) · PE-Core-G vision tower alone (1.88B) | DINOv3 ViT-7B (~6.7B) · PE-Core-G full dual-tower (~2.35B) · any 7B MLLM detector (LEGION, SIDA, FakeVLM) |

**Licensing notes.** DINOv3 checkpoints are **gated** under a custom "other" licence — access request required, and terms unread. **DINOv2-with-registers is Apache-2.0 and ungated**, which is the practical difference for a public repo. MetaCLIP 2 worldwide-huge is cc-by-nc-4.0. SigLIP2, PE-Core, EVA-02 (MIT) are permissive.

### 3.5 Head and pooling design

- **TAP** ([arXiv:2604.26772](https://arxiv.org/abs/2604.26772)) — Tunable Attention Pooling over patch tokens; best model **>+12% accuracy over original CLIP**, SOTA on two in-the-wild benchmarks.
- **Counter-datapoint (NTIRE 5th):** global average pooling over final-layer patch tokens beat CLS token *and* attention pooling for robustness/stability.
- **RINE** — learned importance weighting over intermediate CLIP blocks; +10.6 avg points across 20 test sets.

---

## 4. Adaptation strategy: linear probe vs PEFT vs end-to-end

This was the sharpest disagreement between the two source reports. Resolved below.

### 4.1 B-Free's controlled ablation (Table 3)

Same training data, same evaluation, only architecture and fine-tuning mode varied:

| Architecture | Fine-tuning | AUC | bAcc | NLL | ECE |
|---|---|---:|---:|---:|---:|
| DINOv2+reg | **Linear probe** | 80.8 | 68.5 | 0.58 | .141 |
| DINOv2+reg | **End-to-end** | **99.0** | **95.2** | **0.14** | **.040** |
| DINOv2 | End-to-end | 98.4 | 91.1 | 0.24 | .077 |
| SigLIP | End-to-end | 95.4 | 89.9 | 0.28 | .066 |

Gap between linear probe and end-to-end on the *same backbone*: **+18.2 AUC, +26.7 bAcc, −0.101 ECE.**

### 4.2 The reconciliation

Frozen-feature linear probes and end-to-end fine-tuning are being measured on different axes:

| Axis | Frozen + linear probe | End-to-end / PEFT |
|---|---|---|
| **Cross-generator generalisation, clean data** | Strong — DINOv3-Linear 96.4% GenImage, 94.0% in-the-wild; FGTS 92.45% on 16 generators from 1k+1k training images | Strong |
| **Robustness to degradation** | **Weak without augmentation** — frozen DINOv3 87.50% → 69.81% at JPEG Q70 → 55.31% at σ=10 | Strong when trained with degradation augmentation (NTIRE 0.9723 robust AUC) |
| **Fixed-threshold bAcc / calibration** | **Weak** — 68.5 bAcc, 0.141 ECE | Strong — 95.2 bAcc, 0.040 ECE |
| **Training cost** | Minutes; <5 min on one RTX 5090 | Hours to days |
| **Data required** | 1k–10k images | 100k–1M images |

The two source reports were both correct within their own scope. The merged reading: **frozen probes are an excellent fast baseline and generalise across generators; degradation robustness and calibration come from adaptation plus augmentation, not from the frozen representation.**

### 4.3 Full strategy comparison

| Strategy | Trainable params | Headline result | Source |
|---|---|---|---|
| Linear probe (CLIP) | ~1–2K | 81.38% mean UnivFD benchmark; AIDE-common 78.43 ACC | [Ojha CVPR'23](https://arxiv.org/abs/2302.10174) |
| Linear probe (modern VFM) | ~1–2K | DINOv3 96.4% GenImage / 94.0% wild | [2602.01738](https://arxiv.org/html/2602.01738) |
| Linear probe + curated 10K | ~1–2K | 89.4% AIGIBench, **98.3% TNR** | [SSAFE](https://arxiv.org/html/2606.08634) |
| Training-free centroid (FGTS) | **0** | 78.99% mean, 16 generators | [2511.22471](https://arxiv.org/html/2511.22471v1) |
| Intermediate-block agg. (RINE) | small MLP + TIE | +10.6 abs. avg; **1 epoch ≈ 8 min** | [ECCV'24](https://arxiv.org/pdf/2402.19091) |
| Attention pooling head (TAP) | small | >+12% over CLIP | [2604.26772](https://arxiv.org/abs/2604.26772) |
| **SVD residual (Effort)** | **0.19M** | 95.19 mAcc / 99.41 mAP, 19 subsets | [ICML'25 Oral](https://arxiv.org/html/2411.15633v3) |
| LoRA (DeeCLIP) | low-rank | 84.53% → **89.00%**; +10.36 vs C2P-CLIP | [2504.19876](https://arxiv.org/pdf/2504.19876) |
| LoRA (C2P-CLIP) | low-rank | +12.41% over CLIP, **no test-time params** | [AAAI'25](https://arxiv.org/abs/2408.09647) |
| Adapter + language align (FatFormer) | adapter | 98% unseen GAN / 95% unseen diffusion | [CVPR'24](https://arxiv.org/abs/2312.16649) |
| One-class dual-head (DRIFT) | 2 small MLPs | 98.1% AUC, **real images only** | [2606.06918](https://arxiv.org/html/2606.06918v1) |
| **End-to-end (B-Free)** | full | **99.0 AUC / 95.2 bAcc / 0.040 ECE** | [CVPR'25](https://arxiv.org/html/2412.17671v2) |
| End-to-end + ensemble (NTIRE 1st) | full | 0.9723 robust AUC | [NTIRE](https://arxiv.org/html/2604.11487v1) |
| Post-hoc calibration | **1 scalar** | +7–16% acc on existing detectors | [2602.01973](https://arxiv.org/html/2602.01973) |

**Effort** ([arXiv:2411.15633](https://arxiv.org/html/2411.15633v3), ICML 2025 Oral) deserves separate note. It diagnoses the **asymmetry phenomenon** — a naively trained detector overfits to limited fake patterns, making the feature space **low-ranked**. Fix: SVD-decompose each weight into orthogonal subspaces, **freeze the principal components, adapt only the residual**. Unlike LoRA, orthogonality is explicitly enforced, raising the rank of the whole feature space. **0.19M trainable parameters** (~1000× fewer than LSDA 133M, ProDet 96M); lr 2e-4, Adam, batch 32–48, 224px. Plug-and-play into any ViT. Validated on CLIP ViT-L/14, BEiT-v2, SigLIP.

**DeeCLIP** ([arXiv:2504.19876](https://arxiv.org/pdf/2504.19876)) is the most robustness-targeted PEFT method: CLIP-ViT-L/14 + LoRA, plus **DeeFuser** cross-attention fusing deep features (queries) with shallow features (keys/values) — explicitly *"improving robustness against degradations such as compression and blurring"* — plus triplet loss.

### 4.4 Post-hoc calibration

**[Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated](https://arxiv.org/html/2602.01973)** (Feb 2026). Problem: **threshold misalignment** — models trained on balanced data systematically misclassify *fake* as *real* under test-time shift, from class-conditional input shift plus label prior shift. Fix: **one learnable scalar bias on the logits**, `f̃(x) := f(x) − α`. Supervised (KDE on labelled validation) or unsupervised (distributional symmetry, as few as **10 unlabelled samples**). Backbone frozen.

| Benchmark | Detector | Gain |
|---|---|---|
| AIGCDetectBenchmark | CNNSpot | +7.39% |
| | Fusing | +9.57% sup / +6.75% unsup |
| | Effort | +10.13% sup |
| GenImage | RINE | **+16.16% sup / +15.80% unsup** |
| | Fusing | +14.03% sup |
| **JPEG QF=90** | AIDE | **+15.39%** |

Consistent across 9 detectors; ~1% of test data (**100 images**) sufficed.

---

## 5. Method landscape with robustness and licence status

### 5.1 AIDE's common 16-generator protocol (the most comparable robustness table available)

All methods re-run under one protocol. Rows: clean; JPEG Q95/Q90/Q75/Q50; blur σ 1/2/3/4. Accuracy %.

| Method | Clean | JPEG 95 | 90 | 75 | 50 | Blur 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **AIDE** | **92.77** | 75.54 | 74.21 | 70.64 | 69.60 | **81.88** | **80.35** | **80.05** | **79.86** |
| PatchCraft | 89.31 | 72.48 | 71.41 | 69.43 | 67.78 | 75.99 | 74.90 | 73.53 | 72.28 |
| UnivFD | 78.43 | 74.10 | 74.02 | 69.92 | 68.68 | 70.31 | 68.29 | 64.62 | 61.18 |
| LGrad | 75.34 | 51.55 | 51.39 | 50.00 | 50.00 | 71.73 | 69.12 | 68.43 | 66.22 |
| CNNDetection | 70.78 | 64.03 | 62.26 | 60.65 | 59.66 | 68.39 | 67.26 | 67.13 | 65.85 |
| DIRE | 68.68 | 66.49 | 66.12 | 65.28 | 64.34 | 64.00 | 63.09 | 62.21 | 61.91 |
| FreDect | 64.03 | 66.95 | 67.45 | 66.64 | 65.33 | 65.75 | 66.48 | 68.58 | 69.64 |

Readings: **LGrad is at chance by JPEG Q95.** AIDE leads clean and both corrupted averages but still loses **23.2 points** from clean to Q50. FreDect's apparent *improvement* under corruption is protocol-specific, not evidence of universal robustness. UnivFD degrades most gently under JPEG (−9.75 clean→Q50) but worst under blur (−17.25 clean→σ4).

### 5.2 Low-level / frequency / patch methods

| Method | Cue | Cross-generator | Params | Code / licence | Note |
|---|---|---|---:|---|---|
| CNNDetection (CVPR'20) | ResNet-50 RGB + blur/JPEG aug | AIDE-common 70.78 | 25.6M | [repo](https://github.com/PeterWang512/CNNDetection) · **CC BY-NC-SA 4.0** | Foundational augmentation result |
| FreDect (2020) | DCT spectrum classifier | AIDE-common 64.03 | impl-dependent | [repo](https://github.com/RUB-SysSec/GANDCTAnalysis) · MIT | Diagnostic |
| LGrad (CVPR'23) | Gradients through a pretrained generator | AIDE-common 75.34 | 25.6M + extractor | [repo](https://github.com/chuangchuangtan/LGrad) · no explicit licence | Collapses at Q95 |
| **NPR (CVPR'24)** | Neighbouring-pixel relations from upsampling | +11.6 pts over prior across 28 generators; AIDE-repro 82.91 clean; SD1.4→GenImage 71.6 (per ESSP) | **1.44M** | [repo](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) · no explicit licence | Fragile local pixel statistics |
| PatchCraft (2024) | Smash-and-reconstruct texture patches | AIDE-common 89.31; SD1.4→GenImage 82.30 | ~25.6M | [repo](https://github.com/cvlcgabriel/PatchCraft) · no explicit licence | Strong local reference |
| SSP / ESSP (2024) | Simplest patch + SRM; ESSP adds degradation perception | ESSP 90.6 mean ACC on 8 GenImage generators | ~25.6M + enhancer | [repo](https://github.com/bcmi/SSP-AI-Generated-Image-Detection) · MIT | ESSP: blur 81.4, compression 73.8. Authors state **all methods poor below JPEG Q90**. Checkpoint availability issues reported |
| **SAFE (KDD'25)** | Crop-not-resize; ColorJitter; RandomRotation; patch masking; DWT | +4.5 ACC / +2.9 AP across 26 generators; 99.0 ACC_M / 99.8 AP_M on GAN sets | **1.44M** (lightweight ResNet borrowed from NPR) | [repo](https://github.com/Ouxiang-Li/SAFE) · **Apache-2.0** | GPT-4o gens: 98.92% GenEval, 96.32% ReasoningEdit |
| **FerretNet (2025)** | Local pixel dependencies via neighbourhood-median reconstruction | 97.1% ACC / 99.6% AP across 22 generators | **1.06M** | [arXiv:2509.20890](https://arxiv.org/html/2509.20890) | **JPEG Q75 → 55.1%** — near chance. Trained on 4-class ProGAN only. 772 FPS on RTX 4090 |
| SPAI (CVPR'25) | Masked spectral learning + spectral reconstruction similarity | Mean AUC 91.0 over 13 generators; DALL·E 2/3 91.1/90.2 | ~86M (ViT-B/16 + MFM) | [mever-team/spai](https://github.com/mever-team/spai) · **Apache-2.0** | Removing distortion aug: 91.0 → 84.2. Broad chromatic aug: → **80.5** |

**The 1.44M / 1.06M cluster is the clearest robustness lesson in the table:** these models are excellent on clean cross-generator tests and collapse under JPEG. FerretNet's 97.1% → 55.1% at Q75 is the extreme case.

### 5.3 Foundation-model and hybrid methods

| Method | Mechanism | Cross-generator | Params | Code / licence |
|---|---|---|---:|---|
| UnivFD (CVPR'23) | Frozen CLIP ViT-L/14 + linear probe | 93.38 mean AP on its suite; AIDE-common 78.43 | 307M vision + head | [repo](https://github.com/WisconsinAIVision/UniversalFakeDetect) · **MIT** |
| DIRE (ICCV'23) | Diffusion reconstruction error | AIDE-common 68.68 | ~550M diffusion + 25.6M, iterative | [repo](https://github.com/zhendongwang6/dire) · archived, no licence |
| DRCT (ICML'24) | Diffusion-reconstruction contrastive training; one-pass at test | >10-pt cross-set improvement | 89M ConvNeXt-B or 307M CLIP | [repo](https://github.com/beibuwandeluori/DRCT) · licence unverified |
| AEROBLADE (CVPR'24) | Training-free nearest-autoencoder reconstruction error | Strong when test fakes share a known VAE | 3 VAEs + LPIPS, <400M | [repo](https://github.com/jonasricker/aeroblade) · no licence |
| FatFormer (CVPR'24) | CLIP + forgery-aware image/frequency adapters + language alignment | Own protocol: 98.4 ACC unseen GAN, 95.0 unseen diffusion. **Bias-controlled: DALL·E 2 45.3/48.1** | **492.59M** total (307M vision tower) | [repo](https://github.com/Michel-liu/FatFormer) · **Apache-2.0** |
| RINE (ECCV'24) | Importance over intermediate CLIP blocks | +10.6 avg over 20 sets. Bias-controlled GenImage: 95.0 AUC but **69.1 bAcc** | ~307M + head | [repo](https://github.com/mever-team/rine) · **Apache-2.0** |
| CoDE (ECCV'24) | Contrastive global-local embeddings, ViT-Tiny from scratch on D3 | 94.7 avg fake accuracy over 12 unseen diffusion generators | **5.7M** (ViT-T/16) | [repo](https://github.com/aimagelab/CoDE) · **MIT** |
| C2P-CLIP (AAAI'25) | Category-common prompt + CLIP LoRA, merged at inference | 93.79 mean ACC / 98.66 mAP UFD; 95.8 SD1.4→GenImage | ~307M, no test-time additions | [repo](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) · licence unverified |
| Effort (ICML'25 Oral) | SVD orthogonal residual adaptation | 99.41 mAP / 95.19 mACC over 19 UFD sets vs 97.95/86.22 CLIP baseline | 307M base + **0.19M trainable** | [repo](https://github.com/YZY-stack/Effort-AIGI-Detection) · no explicit licence |
| B-Free (CVPR'25) | Bias-free self-conditioned + inpainted training, end-to-end DINOv2+reg | 99.0 AUC / 95.2 bAcc / 0.14 NLL / **0.040 ECE** | ~86M if ViT-B | [repo](https://github.com/grip-unina/B-Free) · **custom: informational/nonprofit only** |
| AIDE (ICLR'25) | DCT-selected hi/lo-freq patches + SRM + frozen OpenCLIP ConvNeXt-XXL | AIDE-common 92.77; DALL·E 2 96.60; GenImage SD1.4 86.88 | **~898M** (846.5M XXL + 2×25.6M) | [repo](https://github.com/shilinyan99/AIDE) · **MIT** |
| DeeCLIP (2025) | CLIP-L/14 + LoRA + DeeFuser deep/shallow cross-attention + triplet | 84.53 → 89.00 with LoRA; +10.36 vs C2P-CLIP | ~307M | [repo](https://github.com/Mamadou-Keita/DeeCLIP) |
| WaRPAD (2025) | Training-free wavelet high-frequency perturbation + RRC-style TTA | 0.834 AUROC Synthbuster (DINOv2) vs RIGID 0.587, MINDER 0.518 | DINOv2 86–304M | [repo](https://github.com/sungikchoi/WaRPAD) · no licence |
| DRIFT (2026) | One-class robust/fragile subspace decomposition, real images only | 98.1% AUC; ForenSynths 97.8 ACC / 99.8 AP | frozen DINOv2 ViT-B + 2 MLPs | [arXiv:2606.06918](https://arxiv.org/html/2606.06918v1) |
| GlobalForge (2026) | Local Information Bottleneck + Global Structural Reasoning + degradation-aware contrastive | 85.93 avg BAcc in-the-wild, +5.89 over prior SOTA | — | [arXiv:2607.14684](https://arxiv.org/html/2607.14684v1) |

### 5.4 B-Free's bias-controlled FakeBench re-evaluation

**Corrected table** (Appendix E, Table 10). Format AUC/bAcc. Column order verified: ProGAN, StyleGAN, FuseDream, VQDM, GLIDE, CogView2, DALL·E 2, DALL·E 3, SD, Midjourney, AVG.

| Method | ProGAN | StyleGAN | FuseDream | VQDM | GLIDE | CogView2 | DALL·E 2 | DALL·E 3 | SD | Midjourney | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UnivFD | 99.9/98.6 | 96.0/83.4 | 99.2/96.3 | 94.6/77.3 | 86.5/62.8 | 84.7/63.1 | 88.0/65.9 | 69.6/55.8 | 76.8/56.4 | 86.1/71.5 | 91.6/74.0 |
| AIDE | 89.4/64.3 | 89.4/70.0 | 71.7/47.3 | 90.7/78.1 | 79.7/68.3 | 85.5/60.0 | 84.1/52.6 | 88.0/61.9 | 86.0/64.6 | 88.0/71.5 | 85.2/63.9 |
| FatFormer | 100./97.6 | 99.3/97.1 | 90.7/81.8 | 96.8/88.5 | 74.2/69.0 | 47.1/53.3 | **45.3/48.1** | **52.4/49.5** | 50.4/51.0 | 79.6/64.6 | 73.6/70.0 |
| RINE | 100./99.6 | 99.3/95.1 | 99.8/96.6 | 98.8/88.6 | 95.4/70.2 | 86.7/59.2 | 93.0/60.9 | 75.1/52.6 | 85.5/55.9 | 82.2/61.1 | 91.6/74.0 |
| C2P-CLIP | 100./99.5 | 99.4/98.0 | 98.2/93.0 | 97.1/86.7 | 91.9/76.4 | 67.3/61.7 | 72.6/56.9 | 74.7/55.5 | 74.9/59.9 | 88.1/58.0 | 86.4/74.6 |
| CoDE | 64.3/52.5 | 53.0/49.5 | 73.4/56.3 | 78.4/61.7 | 91.6/78.0 | 97.7/93.7 | **93.8/82.8** | **95.8/89.2** | 99.5/96.2 | 89.7/76.7 | 83.7/73.7 |
| DIRE | 90.4/89.5 | 56.6/55.4 | 23.7/40.0 | 91.3/89.2 | 53.2/63.7 | 36.7/41.0 | 44.2/43.0 | 76.6/74.5 | 47.7/49.7 | 83.4/81.2 | 60.4/62.7 |
| **B-Free** | 99.3/96.4 | 97.7/88.5 | 99.3/95.2 | 96.5/86.5 | 95.5/87.7 | 100./98.9 | **98.7/94.9** | **99.9/98.7** | 98.0/94.5 | 98.5/94.0 | **98.5/94.0** |

The DALL·E columns matter most for Track 5, whose AIGC validation half is DALL·E Advanced.

Three observations: **FatFormer** goes from 100./97.6 on ProGAN to **45.3/48.1 on DALL·E 2** — worse than chance on AUC. **CoDE** shows the inverse profile: weak on GANs (53.0/49.5 StyleGAN), strong on modern diffusion and DALL·E. Across the board, **AUC–bAcc gaps of 20–35 points** are common (RINE DALL·E 2: 93.0 AUC but 60.9 bAcc), confirming that a fixed threshold is where these methods actually fail.

> ⚠️ *Caveat:* UnivFD and RINE show identical AVG (91.6/74.0). This may be an extraction artifact; verify against the published PDF before quoting.

---

## 6. Robustness evidence for the exact Track 5 transforms

### 6.1 Per-transform evidence

| Transform | Published evidence | Expected feature behaviour | Engineering consequence |
|---|---|---|---|
| **JPEG Q90/70/50/30** | AIDE-common: LGrad ≈ chance by Q95; PatchCraft 89.31→67.78; AIDE 92.77→69.60 by Q50. WildFake: ViT 92.4 ACC at Q70, 86.6 at Q35; DIRE 75.6 and 65.4. FerretNet **97.1→55.1 at Q75**. Frozen DINOv3+linear **87.50→69.81 at Q70**. GlobalForge holds 87.79 BAcc under JPEG. ESSP authors: *all methods generally poor below Q90*. | Block-DCT quantisation destroys weak high-frequency and local-correlation traces. Semantic/mid-frequency transformers retain ranking longer, but fixed thresholds still shift. **Q30 is outside most published training envelopes.** | Route away from residual branches as Q falls; balance single/double JPEG histories; train Q25–100. |
| **Gaussian blur σ 0.5/1/2** | AIDE 92.77→81.88 (σ1) →80.35 (σ2); PatchCraft 75.99/74.90; UnivFD 70.31/68.29. GlobalForge 84.65 BAcc. *Simplicity Prevails*: DINOv3 and MetaCLIP2 retain capability at σ=2.0 better than legacy models. | Blur removes sensor/noise/upsampling evidence but leaves global semantics and lower/mid-frequency decoder structure. | Train to σ≈2.3; preserve a global branch; consider clean→blurred distillation (DINO-Detect). |
| **Resize 0.5× / 0.25×** | WildFake: ViT 91.8 ACC at 128px, **79.3 at 64px**; ResNet-50 91.1 / 71.3; DIRE 85.8 / 61.6. GenImage: ResNet-50 96.2 at 112px but **57.4 at 64px**. Frozen DINOv3: 84.07 at 0.5×. GlobalForge 89.63 BAcc (its *best* single condition). B-Free avoids resizing forensic crops entirely. | Resampling changes the very interpolation/correlation statistics detectors use. Transformers retain more at moderate downscale, **not at 0.25×**. | Use native-resolution local crops; augment with multiple interpolation kernels; report effective input resolution. |
| **Gaussian noise σ 0.02/0.05/0.10** | GlobalForge 82.06 BAcc — its **worst** single operator. Frozen DINOv3: σ=5 → 73.54; **σ=10 → 55.31 ACC / 47.79 AUC** (below chance on AUC). AEROBLADE and WaRPAD evaluate noise; reconstruction/frequency methods degrade sharply. | Added noise can mimic camera residuals, mask generator traces, or create false "real" evidence. **σ=0.10 normalized RGB is severe.** | Train noise-before-JPEG and noise-after-JPEG; gate monotonically against residual confidence at high noise. |
| **Colour jitter ±20%** | WildFake: ViT **98.5 ACC** under colour transform vs ResNet-50 87.9, DIRE 81.3. GlobalForge lists brightness/contrast/saturation among its 7 operators. **SPAI counter-example: over-broad chromatic augmentation drops mean AUC 91.0 → 80.5.** | Semantics survives modest photometric change. Spectral/colour-specific detectors can be *harmed* by training distributions broader than deployment. | Match ±20%, extend only to ±25%; do not add hue/solarize unless tested. |
| **Center crop 80%** | WildFake: ViT 98.9, LASTED 98.0, ResNet-50 91.3, DIRE 86.3. TextureCrop: **+6.1% AUC vs center crop, +15% vs resizing**. *What Truly Matters?*: texture crop lifts DMID 78.78→89.46, RINE 91.26→94.90, but **hurts NPR 70.11→67.48**. | Whole-image semantic models tolerate moderate crop; single-location forensic cues can vanish. | Aggregate several patches; include crop consistency in confidence. |

### 6.2 Compound degradation chains

**GlobalForge / RealDeg-Bench** ([arXiv:2607.14684](https://arxiv.org/html/2607.14684v1)) is the only benchmark that measures chains. 7 operators (JPEG, blur, resize, noise, brightness, contrast, saturation) × chains N∈{1..5}; 13 conditions, 95,589 images. 12 baselines: NPR, UnivFD, C2P-CLIP, FatFormer, SAFE, AIDE, Effort, DRCT, Aligned, B-Free, DDA, GAPL.

| Condition | GlobalForge BAcc |
|---|---:|
| Clean | 87.77 |
| Resize | 89.63 |
| JPEG | 87.79 |
| Blur | 84.65 |
| Noise | 82.06 |
| Avg single operator | 87.35 |
| **5-step compound chain** | **79.53** |

Comparison: **DDA drops 88.90 (clean) → 70.30 (compound)**; GlobalForge holds 79.53.

Stated methodological critique: existing protocols apply **one perturbation at a time at fixed strength**, so decay along realistic chains is uncharacterised, while in-the-wild benchmarks have uncontrollable degradation. *"Detectors near saturation on clean benchmarks routinely collapse after real propagation chains."* NTIRE 2026 independently chose 1–5 chained distortions.

### 6.3 Preprocessing — crop vs resize

**TextureCrop** ([arXiv:2407.15500](https://arxiv.org/abs/2407.15500), WACVW 2025, [code](https://github.com/mever-team/texture-crop)): sliding-window analysis cropping texture-rich regions, filtering low-texture-variability areas. **+6.1% AUC vs center cropping, +15% vs resizing** across detectors on ForenSynths, Synthbuster, TWIGMA.

*What Truly Matters?* cropping comparison (avg AUC): DMID 78.78→**89.46**; RINE 91.26→**94.90**; NPR **70.11**→67.48. Resizing was abandoned because it *"erases subtle high-frequency traces."*

**Counter-datapoint:** NTIRE 5th place deliberately avoided random resized cropping (arguing it removes localised forensic cues) in favour of **"squish"** — direct resize to 384×384 ignoring aspect ratio — and reached 0.8730 robust AUC single-model.

### 6.4 Test-time augmentation and patch aggregation

- **Majority voting vs logit averaging** — both improve over single-patch; majority voting consistently better, though near-chance datasets slightly degrade.
- **Any-patch (OR) rule** — all patches real → real; ≥1 patch synthetic → synthetic. Chosen where missed detection is the critical error.
- **Selective patches** — uniform aggregation dilutes evidence since traces are spatially non-uniform. AIDE's DCT selection of 2 high-freq + 2 low-freq patches is canonical.
- **Token pooling** — NTIRE: global average pooling over final-layer patch tokens beat CLS and attention pooling.
- **top-k median** — DRIFT's aggregation.
- **Horizontal-flip TTA** — NTIRE 4th applied to 2 of 5 models.
- **RRC-simulating TTA** — WaRPAD deterministically simulates multiple random-resized-crop instances and averages; core observation is that *AI-generated images lose robustness to wavelet high-frequency perturbation when examined in patches*.

### 6.5 Adversarial robustness

- **[Robustness of AI-Image Detectors](https://openreview.net/pdf?id=dLoAdIKENc)** (ICLR 2024) — fundamental trade-off between evasion and spoofing error rates; theory extended to classifier-based detectors.
- **[Backbone is All You Need](https://arxiv.org/pdf/2605.13381)** (May 2026) — **SIAA**, a gray-box attack exploiting only knowledge of the detector backbone, achieving success rates *"often approaching white-box performance"*. Directly relevant given the field's convergence on a handful of public backbones.
- **[Exploring the Adversarial Robustness of CLIP](https://arxiv.org/pdf/2407.19553)** (WIFS 2024) and **[Adversarial Robustness in the Real World](https://arxiv.org/pdf/2410.01574)** — attacks remain effective even after lifecycle degradation such as social-media re-upload.

---

## 7. The dataset-bias trap

Track 5 asks teams to **create their own transformed test cases**. The literature documents specific ways this goes wrong.

### 7.1 Fake or JPEG?

**[Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/abs/2403.17608)** (Grommelt, Weiss, Pfreundt, Keuper; [code](https://github.com/gendetection/UnbiasedGenImage)).

**Mechanism:** most ImageNet real images are JPEG (modal quality ≈96, range 70–100); generated fakes are stored losslessly as PNG. A detector exploits the compression difference as a shortcut. Parallel **size bias**: generator outputs have fixed model-specific sizes; real ImageNet images have a broad multimodal size distribution.

**Evidence:** ResNet-50 on raw GenImage declines even at Q95. Precision on AI-generated images stayed near 1 while **recall dropped sharply** — compressing a generated image makes the model call it natural. Controlled for the "compression destroys artefacts" objection using uncompressed FFHQ PNGs.

**Bias direction is configuration-dependent:** stronger compression shifts predictions toward *real* in one configuration and toward *fake* in the reverse; on uncompressed images the shift inverts — the detector treats *absence* of JPEG artefacts as evidence either way.

**Bias-controlled protocol:** recompress **both** classes to uniform **JPEG Q=96**; align native size distributions (both center-cropped to **450×450** before resizing); enforce symmetric preprocessing (resize, crop, resize).

**Result: +11.06 to +11.74 cross-generator accuracy points** for ResNet-50 and Swin-T on GenImage.

### 7.2 Aligned datasets

**[Aligned Datasets Improve Detection of Latent Diffusion-Generated Images](https://arxiv.org/html/2410.11835v3)** (Sundara Rajan, Ojha, Schloesser, Y. J. Lee).

Reconstruct real images using **only the LDM VAE encoder-decoder, skipping the U-Net**: `𝒱 = {φ_dec(φ_enc(x))}`. The "fake" preserves resolution, aspect ratio and semantics, differing **almost exclusively in decoder artefacts** — and costs ~10× less compute.

Named spurious feature: existing methods *"mistakenly learn that downsampling correlates with real images due to resolution mismatches."*

Setup: ResNet-50 (ImageNet-pretrained); BCE, Adam 1e-4; augmentations random JPEG, blur, grayscale, cutout, noise, random resized crop; 179,257 images from MS COCO + LSUN. **Ours-Sync** variant pairs real/fake with *identical* augmentations.

Results: SD 99.31–99.57%; Midjourney 98.50–99.37%; Playground 94.85–99.48% (+12.72/+17.35 over Corvi). **Data efficiency: with 1,000 images, 83.37% TPR@5%FPR vs Corvi's 46.51%.** Detects Kandinsky (different VAE) at 99.57–99.92%.

Limitations: **FLUX.1-dev (16 latent channels) only 25.87%**; vulnerable to `.webp` artefacts absent from reconstructions. Surprising: training on **OpenGL shader images** achieved competitive performance — alignment matters more than semantics.

### 7.3 B-Free

**[A Bias-Free Training Paradigm](https://arxiv.org/html/2412.17671v2)** (Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva; CVPR 2025).

Fakes generated *from* real images via stable-diffusion conditioning, plus **content augmentation via inpainting**. Training set: **51,517 MS-COCO real + 309,102 generated from SD 2.1**, **504×504 crops, no resizing**.

Data-strategy ablation (Table 2), bAcc average:

| Strategy | SDXL | DALL·E 3 | FLUX | bAcc Avg |
|---|---|---|---|---:|
| Paired text | 100./99.8 | 99.1/75.8 | 98.2/64.0 | 80.7 |
| Reconstructed | 99.9/97.2 | 98.9/76.1 | 95.4/59.4 | 81.4 |
| Self-conditioned | 99.9/97.8 | 99.1/72.7 | 90.4/52.3 | 78.6 |
| Self-cond. + inpainted | 100./99.6 | 99.4/92.8 | 98.7/87.5 | **92.2** |
| Self-cond. + inpainted++ | 100./99.7 | 99.6/96.8 | 97.9/85.3 | **96.4** |

**+17.8 bAcc points** from adding inpainting-based content augmentation to self-conditioning. "inpainted++" adds blurring, JPEG, scaling, cut-out, noise, jittering. Robustness: bAcc **stays above 80** across JPEG, resizing and blur stress plots.

Licence: **custom, informational and nonprofit use only.**

### 7.4 Other bias channels

- **WEBP inside PNG:** LSUN images were WEBP-compressed but stored as lossless PNG; detectors associated WEBP with the real distribution. Excluding WEBP reals restored generalisation.
- **Neural codecs:** [*Three Forensic Cues for JPEG AI Images*](https://arxiv.org/html/2504.03191v1) — **real images compressed with JPEG AI are classified as fake**, with high FPR persisting even after retraining on JPEG AI.
- **Naive symmetric recompression trap:** if a model learned that *absence* of WEBP artefacts is necessary for fakeness, adding WEBP to fakes weakens the fake signal.
- **HiDA-Net** ([arXiv:2508.17346](https://arxiv.org/html/2508.17346v1)) attributes collapse partly to *"mismatched JPEG compression histories between real and fake images, which teaches the model to become a compression detector rather than a synthesis detector"* — and adds an explicit **JPEG Quality Factor Estimation** module to disentangle the two.
- **Historical precedent:** Cattaneo & Roscigno found a tampered-image dataset where untampered images were saved at different quality factors than tampered ones.

### 7.5 Consolidated bias-neutralisation protocol

1. **Manifest before training:** SHA-256, pHash, source dataset, source URL/ID, generator family, original W/H, original format, detected JPEG quantization table, recompression count, semantic bucket.
2. **Protected denylist** from every COCO val2017 image and every WildFake DALL·E folder. Reject exact SHA matches *and* pHash neighbours. Hold out the entire DALL·E family, not just "Advanced".
3. **Strip provenance:** decode EXIF orientation and RGB pixels, strip metadata; never expose filename, extension, EXIF, dimensions or directory source to the model.
4. **Match class distributions** for short-side resolution, aspect ratio, format and compression history. Apply a class-independent canonical decode/encode. Balance zero-, one- and two-JPEG histories within each class. (Q=96 is the GenImage bias-controlled convention.)
5. **Pair semantics** wherever possible — reconstruction/self-conditioning from real images is strongest; otherwise caption/class matching.
6. **Split by generator family and real-image source**, not random image. Entire source sites and generator versions go to exactly one split.
7. **Four shortcut probes:** label from file size only; dimensions only; JPEG quantization only; dataset-source classifier from frozen embeddings. **Any probe above 60% bAcc blocks training.**
8. **Compression sweep sanity check:** if AUROC rises or collapses monotonically with quality factor, you are measuring the codec, not the generator.

**Direct relevance to Track 5:** the validation set pairs COCO val2017 (distributed as JPEG) against DALL·E Advanced (typically PNG). Every mechanism above applies to any self-created transformed test set built on these two sources.

---

## 8. Datasets

### 8.1 The three datasets named in the brief

| Dataset | Size | Real source | Download | Licence | Shortcut risk |
|---|---|---|---|---|---|
| **[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)** | HF: 210K train + 30K val (**123.2 GB + 16.8 GB**); paper says 300K | OpenImages V7, ~1K resolution | Stream or select files | Card says CC BY 4.0; underlying image attribution applies | Card metadata is inconsistent (tagged `arxiv:1505.04870`, i.e. Flickr30k, while the source paper is SIDA `2412.04292`); older card history described a different real source. **Audit the manifest.** COCO contamination must be hashed, not assumed absent |
| **[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)** | 120K (60K CIFAR-10 real + 60K SD1.4 fake), **32×32** | CIFAR-10 | <1 GB | MIT / CIFAR-compatible per author | **Extreme resolution and content shortcut**; unrelated to modern high-res post-processing. 668 duplicate pairs found by one study. Domain shifts (e.g. "ship" rendered as interior scene) |
| **[WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake)** | Paper internally inconsistent: 2.56M fake + 1.01M real in one table vs 2.68M fake / 3.69M total in text | COCO, FFHQ, ImageNet, LSUN, LAION | Not verified; expect hundreds of GB | **Not verified on ModelScope**; paper licence does not license all images | Source, format, resolution and generator-folder shortcuts. **The exact benchmark subset lives here** |

**SID_Set provenance:** from **SIDA** (Huang et al., CVPR 2025, [arXiv:2412.04292](https://arxiv.org/html/2412.04292), [code](https://github.com/hzlsaber/SIDA)) — 100K real OpenImages V7, 100K synthetic from **FLUX**, 100K tampered. GPT-4o generated judgement descriptions for 3,000 images. Only a single `test.zip` is released, deliberately, to limit foundation-model contamination.

**WildFake hierarchy** (4 levels, AAAI 2025, [arXiv:2402.11843](https://arxiv.org/abs/2402.11843)): (1) cross-generator — DMs / GANs / Others; (2) cross-architecture — DALLE, ADM, Imagen, DDPM, DDIM, VQDM, Midjourney, SD; (3) cross-weight — SD fakes in three subsets; (4) cross-version — typical vs advanced. Designed for train-on-one-subset / test-on-another.

### 8.2 Additional datasets

| Dataset | Size | Practical notes | Licence |
|---|---|---|---|
| **GenImage** | 1,331,167 fake + 1,350,000 real; 8 generators | Mirrored Arrow ≈**655 GB** — not hackathon-sized. Stream 2–5k per generator | **CC BY-NC-SA 4.0** + non-commercial terms |
| **[Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics)** | 2.7M images, **4,803 generators** | Full 1.1 TB / Small 278 GB / Eval 206 GB. **Best out-of-the-box detector (75.0%)** in the 2602.07814 benchmark. Detection improves as generator count grows, **saturating ~1,000 generator variants** | Code MIT; **data rights source-specific, not a blanket grant** |
| **DRCT-2M** | ~2M paired real/reconstructed, 16 diffusion configs | Hundreds of GB; strong decoder-family bias | Not verified |
| **ForenSynths** | 720k train, 20 LSUN categories | ~72 GB in UniversalFakeDetect packaging | CC BY-NC-SA 4.0 |
| **B-Free training set** | 51,517 COCO real + 309,102 SD2.1 fakes, 504px crops | **COCO source overlap with the demo's real domain can flatter results even without exact leakage** | Custom informational/nonprofit only |
| **Chameleon** | ~26k curated 720p–4K fakes | Access by request; fake-heavy, not balanced | Academic only |
| **So-Fake-Set / OOD** | 2M+ from 35 generators / 100K Reddit OOD | Used by NTIRE 1st and 4th | [repo](https://github.com/hzlsaber/So-Fake) |
| **AIGIBench** | 23 fake subsets, ~288K train | 4-task robustness protocol | [arXiv:2505.12335](https://arxiv.org/abs/2505.12335) |
| **ITW-SM** | 10,000 images, 50/50, 0.1 MP–8.4K, Facebook/Instagram/LinkedIn/X | Manually verified labels | [arXiv:2507.10236](https://arxiv.org/html/2507.10236v1) |
| **HiRes-50K** | 50,568 images, <1K to >10K long edge, up to 64 MP | Freepik/LiblibAI/Civitai + Unsplash | [arXiv:2508.17346](https://arxiv.org/html/2508.17346v1) |
| **[ELSA_D3](https://huggingface.co/datasets/elsaEU/ELSA_D3)** | 2.3M train rows (2,626 GB) | 4 generated variants per prompt with metadata | EU ELSA project |
| **D3 (CoDE)** | 2.3M records / **11.5M images**; prompt + real + 4 generated | LAION-400M prompts and reals | MIT |
| **NTIRE 2026** | 294,500 images, 42 generators, 36 transforms | [Codabench #12761](https://www.codabench.org/competitions/12761/) | — |

Others cited: Synthbuster, TWIGMA, ArtiFact, Fake2M, RealWorldBench, OpenFake, RRDataset, TrueFake, WildRF, SocialRF, CommunityAI, AIGI-Blur, SynthScars, HPAI-BSC/SuSy-Dataset.

---

## 9. Training recipe evidence

### 9.1 Augmentation recipes with measured effect

| Source | Recipe | Measured effect |
|---|---|---|
| **Wang et al. CVPR 2020** | Gaussian blur p=0.5, **then independently** JPEG p=0.5 (SciPy Gaussian filtering) | *"Data augmentation... is critical for generalisation — even when the target images are not post-processed."* Caveat: hurt on SAN |
| **Cozzolino CVPRW 2024** | With augmentation | *"Basically insensitive to compression (JPEG or WebP) and resizing"*; **+13% on impaired/laundered data** |
| ***What Truly Matters?*** | JPEG + **WEBP**; random crop, rotation, h-flip; Gaussian noise, blur, **sharpening** | DMID **+11.15** AUC, SPAI **+5.59**, RINE +1.74, NPR +0.12; **avg +4.65** |
| **SAFE (KDD 2025)** | RandomCrop (train) / CenterCrop (infer); ColorJitter; RandomRotation; patch masking; DWT | +4.5 ACC / +2.9 AP over 26 generators |
| **B-Free (CVPR 2025)** | Self-conditioned + **inpainting** content augmentation ("inpainted++" adds blur, JPEG, scaling, cut-out, noise, jitter) | Self-cond. 78.6 → +inpainted 92.2 → inpainted++ **96.4** bAcc |
| **Seeing What Matters (NeurIPS 2025)** | **Wavelet-decomposition augmentation** — replacing specific frequency bands | Improves generalisability; competing image methods (UnivFD, FreqNet) not competitive in that setting |
| **Aligned Datasets** | Random JPEG, blur, grayscale, cutout, noise, RRC; "Ours-Sync" pairs real/fake with **identical** augmentations | TPR@5%FPR 46.51 → **83.37** at 1k images |
| **NTIRE 1st (MICV)** | Hierarchical stochastic pipeline by difficulty: simple (blur, noise, shifts) → complex multi-stage | 0.9723 robust AUC |
| **NTIRE 2nd (Ant)** | 4-level offline: clean / 1–3 (μ=0,σ=2.5) / 3–6 (μ=2.5,σ=2.0) / fixed 6 (μ=3.5,σ=1.0); online flip + AugMix m6-w3-d1 | 0.9721 |
| **NTIRE 5th (vincentlc)** | `distortion_prob=1.0`, up to 3 ops, 5 severity levels — **every** training image distorted | 0.8730 with a single model |
| **SPAI — warning** | Broad chromatic augmentation | Removing distortion aug: 91.0 → 84.2. **Broad chromatic aug: 91.0 → 80.5** |
| **B-Free — warning** | Self-conditioned CutMix/MixUp variant | **78.6 bAcc vs 92.2** for semantically valid inpainting. Do not use MixUp/CutMix for whole-image binary labels |
| **AIGIBench — counter-finding** | Common augmentations across 11 detectors | *"Limited benefits from common augmentations"* |

The two warnings matter as much as the positive results: **augmentation breadth must match deployment, not exceed it arbitrarily**, and label-mixing augmentations are actively harmful for this binary task.

### 9.2 A sampler consistent with the evidence

From the build brief, calibrated to the Track 5 transform list. Corruption count: **30% clean, 55% one corruption, 15% two.** Applied **identically by class**.

| Family | P(among corruptions) | Training distribution | Required test atoms |
|---|---:|---|---|
| JPEG | 0.30 | Quality uniform 25–100; two libjpeg paths; balanced one/double history | 90, 70, 50, 30 |
| Resize down/up | 0.20 | Scale uniform 0.20–1.00; area/bilinear/bicubic/Lanczos | 0.50, 0.25 |
| Gaussian blur | 0.15 | σ uniform 0–2.3; kernel ≥ 6σ+1 | 0.5, 1.0, 2.0 |
| Gaussian noise | 0.15 | σ uniform 0–0.11 in RGB [0,1]; half before and half after JPEG | 0.02, 0.05, 0.10 |
| Colour jitter | 0.10 | Brightness/contrast/saturation independently uniform 0.75–1.25; **no hue** | ±20% |
| Crop | 0.10 | Center or random retained side/area, uniform 0.75–1.00, resize back | 80% |

**Train only 10–20% beyond test severity** — Q25, σ 2.3, 0.20×, noise 0.11, ±25%, 75% crop. A 5–10% "boundary bucket" is sufficient; SPAI's chromatic ablation is the warning against going wider.

> **Ambiguity to settle with organisers:** "center crop 80%" can mean 80% of each *side* or 80% of *area*. Until confirmed, evaluate both and log the convention.

Additional practices reported as useful: randomise JPEG libraries and interpolation implementations to prevent library signatures; use clean/corrupted consistency on **logits, not feature equality** (forensic features legitimately change); maintain a transform-stratified validation loader and early-stop on the **minimum** across clean, Q30, blur σ2, resize 0.25×, noise 0.10 bAcc. If data is limited, **increase generator/source diversity before images per source** (Community Forensics: gains as generator count grows, saturating ~1,000 variants).

### 9.3 Distillation and consistency for degradation robustness

- **DINO-Detect** ([arXiv:2511.12511](https://arxiv.org/pdf/2511.12511)) — clean-image teacher distils into a student on blurred/degraded images; DINOv3 backbone; introduces **AIGI-Blur**.
- **NTIRE 6th (UESTC)** — 2 epochs binary classification, then **feature-level self-distillation** with epoch-2 intermediate feature maps as dense targets.
- **NTIRE 3rd (TeleAI)** — feature-correction network + KL/MSE consistency between clean/distorted pairs (α=0.5, β=0.25).
- **GlobalForge** — degradation-aware contrastive structural loss aligning clean and degraded representations, plus Local Information Bottleneck (learnable Gaussian smoothing in feature space) and Global Structural Reasoning (masking local attention in 3×3 windows to force distant evidence aggregation).

---

## 10. Evaluation, metrics and calibration

### 10.1 Metrics used across the literature

- **ROC AUC** — NTIRE 2026 primary, computed over transformed **and** untransformed images
- **Balanced Accuracy** — GlobalForge/RealDeg-Bench, B-Free
- **mAP / AP** — UnivFD lineage, Effort, DRIFT; useful under the 8,843 fake / 4,998 real imbalance
- **Decomposed R.Acc / F.Acc** — AIGIBench separates Real from Fake Image Accuracy; this is what exposed the "F.Acc → 0%" failure
- **TPR@5%FPR** — Aligned Datasets
- **TNR on real images** — SSAFE reports separately (98.3% for curated 10K)
- **NLL, Brier, 15-bin ECE** — B-Free
- **Localization IoU** — So-Fake-R1, LEGION, SIDA

### 10.2 Threshold policy

Documented methods: fixed FPR budget (naive thresholding produces unacceptably high FPR); **Youden index** on a calibration set; calibration using **regenerated real images** through a known generator so the threshold is set without touching unseen test data; learnable scalar logit shift with 10–100 samples.

**Noted caveat:** benchmarks report AUROC/AP precisely *because* accuracy varies with thresholding — convenient for papers, unhelpful for an analyst who must pick one operating point. Threshold sweeps show **most models below 30% TPR at 5% FPR**.

**Protocol from the build brief:** choose one threshold once on clean validation by maximising bAcc subject to a declared FPR constraint, then **freeze it for every transform and generator**. Fit one temperature on a separate deployment-mixture calibration set with equal clean and transformed buckets. Never refit per transform. Report bootstrap 95% CIs by image for overall metrics and by generator for macro metrics.

### 10.3 Required clean-vs-transform matrix

| Condition | AUROC | AP | bAcc @ fixed threshold | FPR@95%TPR | ECE | Score drift | Δ bAcc |
|---|---|---|---|---|---|---|---|
| Clean | | | | | | | — |
| JPEG 90 / 70 / 50 / 30 | one row each | | | | | | |
| Blur 0.5 / 1.0 / 2.0 | one row each | | | | | | |
| Resize 0.5× / 0.25× | one row each | | | | | | |
| Noise 0.02 / 0.05 / 0.10 | one row each | | | | | | |
| Colour jitter ±20% | | | | | | | |
| Center crop 80% | | | | | | | |

Add a second table with one row per fake generator on **development** data: count, AUROC vs each real source, TPR at the global threshold, mean probability, worst transform.

### 10.4 Inference contract

Submit `sigmoid(z/T)` — **p(AIGC), never a raw logit**. Temperature scaling preserves ranking, so AUROC/AP are unchanged while NLL/ECE improve. Recursively enumerate images in deterministic sorted order, apply EXIF orientation, decode to RGB, batch in fp16, write JSON with `image_path` and `pred`. Decode failures go to a separate error log — **do not silently emit 0.5**. Seed all transform evaluation, save the transform manifest, include model/transform hashes in the robustness CSV.

---

## 11. False positives and error analysis

### 11.1 Documented false-positive magnets

| Authentic type | Why it attracts false AIGC scores |
|---|---|
| **Computational phone photos** (HDR, night mode, portrait blur, multi-frame denoise/sharpen) | ISP pipelines suppress sensor noise, hallucinate detail, create local correlations resembling generative decoders |
| **AI-upscaled or aggressively denoised photographs** | Restoration removes camera traces and may introduce diffusion/GAN decoder traces |
| **Screenshots, memes, social posts, repeated JPEG** | No camera noise, hard text edges, resampling, overlays, double compression |
| **Digital illustrations, vector graphics, 3D renders, game imagery** | Style overlap with text-to-image outputs; no physical camera pipeline |
| **Scans, old photos, thumbnails, WEBP conversions** | Non-camera noise, low effective resolution, unusual spectra |
| **Heavily filtered/beautified portraits** | Skin smoothing and synthetic bokeh erase local sensor evidence |
| **JPEG AI-compressed real images** | JPEG AI leaves frequency artefacts similar to synthetic images; **high FPR persists even after retraining on JPEG AI** |

**Independent audit:** a 2026 NewsGuard audit ran 15 authentic news photographs through 5 leading detectors — **3 of 5 misclassified real images, worst tool flagging 6 of 15 (40%)**. Disproportionately flagged: studio portraits with smooth skin and controlled lighting; heavily processed landscapes.

Research separately notes real images that are easily misclassified tend to be **simpler or visually incoherent** — minimalistic scenes, unusual facial features.

**Mechanism, not just correlation:** B-Free documents score reversals for the same generator when the real source changes between RAISE and COCO, tracing them to content, format and resolution biases. Aligned-dataset work shows a resizing shortcut can make a detector associate downsampling with *real* and upsampling with *fake*.

### 11.2 Documented false-negative categories

- Modern commercial generators (Flux Dev, Firefly v4, Midjourney v7) at **18–30% detection**
- Chameleon-style images curated by photographers and AI artists — *"almost every off-the-shelf detector predicts them as real"*
- Images after **5-step compound degradation chains** (DDA 88.90 → 70.30)
- **VAE-reconstructed and locally-edited images** — explicit blind spot noted in *Simplicity Prevails*
- FLUX.1-dev under VAE-alignment training (25.87% for Aligned Datasets)

### 11.3 Error-analysis note template

Select ≥12 FPs and ≥12 FNs: three clean, three moderate, three severe, three branch-disagreement. For each: thumbnail, source/generator, transform and severity, final probability, per-branch probabilities, gate weight, nearest real/fake neighbours, patch heatmap, one-sentence hypothesised cause. Aggregate error rates by style/content, original format, short-side resolution, estimated degradation. Do not cherry-pick only visually amusing failures.

**Policy item to settle with organisers:** whether "authentic but AI-restored/upscaled" counts as real. Define before labelling; keep as a separate analysis bucket regardless.

---

## 12. Explainability options

| Option | Fidelity | Notes |
|---|---|---|
| **Patch evidence map** | High | Run the local expert over a coarse native-resolution grid, colour each patch by its fake logit. Most faithful because the branch actually aggregates those logits |
| **Counterfactual stability strip** | High | Probabilities for original, JPEG 70, resize 0.5×. A stable score is more persuasive than a single saliency map |
| **Residual/spectrum card** | Medium | Image + fixed SRM residual + log-DCT magnitude + radial power profile against median real/fake training envelopes |
| **Nearest-neighbour evidence** | Medium | Three real and three fake neighbours from the frozen embedding with source/generator and cosine similarity. UnivFD found CLIP NN structure informative — but neighbours are context, not provenance |
| **Attention rollout / Grad-CAM** | Low | Label it "model attention", **not** proof of manipulation or localization ground truth |
| Generated natural-language "reasons" | Very low | Overstate forensic certainty; avoid |

**Published explainability datasets and models:** **LEGION** ([arXiv:2503.15264](https://arxiv.org/pdf/2503.15264), ICCV 2025 Highlight) with **SynthScars** — 12,236 synthetic images, expert pixel-level masks, textual explanations, artefact category labels, compared against 19 methods. **SIDA** (CVPR 2025) — authenticity + tampered-region mask + textual explanation. **AIDE/Chameleon** — DCT+SRM patch selection. **NPR** — heatmaps showing artefacts on hair, eyes, beard. **CIFAKE** — Grad-CAM. **C2P-CLIP** — decodes detection features into text; concludes CLIP recognises *concepts*, not real/fake semantics. **Effort** — a tool for quantifying degree of model overfitting.

---

## 13. Compute figures

| Source | Setup | Reported cost |
|---|---|---|
| **DINOv3 + FGTS linear probe** | Frozen DINOv3, 1k+1k images | **<5 min on one RTX 5090**; ~300× faster than CNNSpot |
| **RINE** | CLIP-L + intermediate-block aggregation | **1 epoch ≈ 8 minutes** |
| **Effort** | CLIP ViT-L/14, 224px, batch 32–48 | 0.19M trainable params; "very little training cost" |
| **Simplicity Prevails** | Linear probe on frozen VFM | AdamW 1e-3, batch 128, **2 epochs** |
| **SSAFE** | Frozen PE-Core-G + linear head | AdamW 1e-3, batch 40, **10K images** |
| **DRIFT** | Frozen DINOv2 ViT-B/14 + 2 MLPs | AdamW 3e-4, batch 64, 50 epochs |
| **SPAI** | ViT-B/16 + MFM | **Inference <8 GB**; training needs ~48 GB (L40S-class) |
| **FerretNet** | 1.06M CNN | 772 FPS on RTX 4090 |
| **NTIRE 6th (UESTC)** | 4-expert CLIP-L + SigLIP ensemble | **~10 GB peak GPU memory** |
| **NTIRE 4th (INTSIG)** | 5-model ensemble | 8× H800, DDP, batch 16 |
| **NTIRE 3rd (TeleAI)** | EVA-CLIP + LoRA | 8× A800, 5 epochs |
| **NTIRE 2nd (Ant)** | DINOv3-7B ×2 | B200 training; A100 inference **2.21 img/s, 78.25 GB VRAM** |
| **NTIRE 1st (MICV)** | DINOv3 ensemble, 512×512 | **32× A100, 10 epochs, ~8 h** |
| **Aligned Datasets** | ResNet-50 on VAE reconstructions | VAE-only reconstruction **~10× cheaper** than full denoising |

**Evaluation-cost arithmetic for Track 5:** 13,841 images × 15 conditions = **207,615 image-condition evaluations**. Planning estimates for a T4/RTX-class GPU with fp16, batched decoding and cached transformed files: a ~90M-parameter hybrid finishes the clean set in roughly 4–10 minutes and the full matrix in about 1–2.5 hours. Iterative-inversion methods (DIRE, AEROBLADE) can take hours for the clean set alone.

---

## 14. Off-the-shelf checkpoints

| Model | Params | Arch | Downloads | Licence |
|---|---:|---|---:|---|
| [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) | 86.8M | Swin | 877.4K | **cc-by-nc-3.0** |
| [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) | ~86M | Swin | 826.3K | cc-by-4.0 |
| [`Ateeqq/ai-vs-human-image-detector`](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) | 92.9M | SigLIP | 340.2K | apache-2.0 |
| [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | 85.8M | ViT | 283.3K | apache-2.0 |
| [`haywoodsloan/ai-image-detector-deploy`](https://huggingface.co/haywoodsloan/ai-image-detector-deploy) | 195.2M | SwinV2 | 248.7K | apache-2.0 |
| [`NYUAD-ComNets/NYUAD_AI-generated_images_detector`](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector) | 85.8M | ViT | 58.4K | apache-2.0 |
| [`HPAI-BSC/SuSy`](https://huggingface.co/HPAI-BSC/SuSy) | — | CNN + patch | 1.0K | apache-2.0 |

**Research repos with released weights, ranked by licence clarity:**

| Licence | Methods |
|---|---|
| **MIT** | UniversalFakeDetect, AIDE, CoDE, FreDect/GANDCTAnalysis, SSP, Community Forensics (code) |
| **Apache-2.0** | SAFE, SPAI (`mever-team/spai`), RINE, FatFormer |
| **CC BY-NC-SA 4.0** | CNNDetection / ForenSynths |
| **Custom, nonprofit only** | B-Free |
| **No explicit licence found** | NPR, PatchCraft, LGrad, DIRE, DRCT, AEROBLADE, Effort, C2P-CLIP, WaRPAD |
| **Gated, custom** | DINOv3 checkpoints |

Additional repos: [TextureCrop](https://github.com/mever-team/texture-crop), [DeeCLIP](https://github.com/Mamadou-Keita/DeeCLIP), [Bi-LORA](https://github.com/Mamadou-Keita/VLM-DETECT), [ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection), [So-Fake](https://github.com/hzlsaber/So-Fake), [SIDA](https://github.com/hzlsaber/SIDA), [LEGION](https://github.com/opendatalab/LEGION), [UnbiasedGenImage](https://github.com/gendetection/UnbiasedGenImage). Curated lists: [ant-research/Awesome-AIGC-Image-Video-Detection](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection), [nxZhai/Awesome-AI-generated-Image-Detection](https://github.com/nxZhai/Awesome-AI-generated-Image-Detection), [Awesome-AIGCDetection](https://fdmas.github.io/AIGCDetect/Awesome-AIGCDetection).

---

## 15. Open items

Carried forward from both reports; items resolved during the merge are struck through.

1. ~~SAFE backbone and parameter count~~ — **resolved: 1.44M lightweight ResNet borrowed from NPR.**
2. ~~B-Free FakeBench table column alignment~~ — **resolved: full column order recovered.**
3. ~~SPAI official repo and parameters~~ — **resolved: `mever-team/spai`, ViT-B/16 + MFM ≈86M, Apache-2.0.**
4. ~~AIDE parameter count~~ — **resolved: ≈898M (ConvNeXt-XXL 846.5M verified).**
5. **Whether the <2B rule counts the full dual-tower checkpoint or the vision tower only.** Determines admissibility of SigLIP2-Giant-Opt (1.87B full), MetaCLIP2-Huge (1.86B full), PE-Core-G (1.88B vision / ~2.35B full). **Ask the organisers.**
6. **"Center crop 80%" — side or area?** Affects evaluation and training distribution. **Ask the organisers.**
7. **DINOv3 licence terms.** Gated under a custom "other" licence; terms unread. Redistribution/usage rights for a public hackathon repo unverified. DINOv2-with-registers (Apache-2.0, ungated) is the unambiguous alternative.
8. **WildFake internal size inconsistency** (2.56M+1.01M vs 2.68M/3.69M) and per-subset archive sizes. Authoritative source is the ModelScope Files tab plus supplementary material.
9. **The exact 8,843-file DALL·E Advanced manifest** was not independently recovered by either research pass. Treat 8,843 as organiser-provided and verify against the delivered directory before freezing the denylist.
10. **WildFake / DRCT-2M dataset licences** unverified on ModelScope.
11. **SID_Set card metadata** is erroneous (paper reference mismatched); real-source history changed between card versions. Audit the manifest before use.
12. **NTIRE 2026 distortion pipeline source code** — described in prose; no public repo surfaced. Codabench #12761 is the likely host of a starter kit.
13. **RA-Det exact robustness values** under JPEG QF 95/90/85 and blur σ 0.8/1.0/1.5 vs NPR/FerretNet/UniFD — PDF text layer did not extract.
14. **DINO-Detect exact DINOv3 variant** and full AIGI-Blur specifications — PDF text layer did not extract.
15. **UnivFD and RINE identical AVG (91.6/74.0)** in B-Free Table 10 — possible extraction artifact; verify against the published PDF.
16. **FatFormer parameter count** — 307M (vision tower) vs 492.59M (full) both circulate; confirm which the organisers would count.

---

## 16. Reference pack

### Challenges and benchmarks
NTIRE 2026 — [arXiv:2604.11487](https://arxiv.org/html/2604.11487v1) · [CVF](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Gushchin_NTIRE_2026_Challenge_on_Robust_AI-Generated_Image_Detection_in_the_CVPRW_2026_paper.pdf) · [Codabench](https://www.codabench.org/competitions/12761/) · AIGIBench — [arXiv:2505.12335](https://arxiv.org/abs/2505.12335) · Out-of-the-box benchmark — [arXiv:2602.07814](https://arxiv.org/html/2602.07814v1) · GlobalForge/RealDeg-Bench — [arXiv:2607.14684](https://arxiv.org/html/2607.14684v1) · GenImage — [arXiv:2306.08571](https://arxiv.org/abs/2306.08571) · AI-GenBench — [arXiv:2504.20865](https://arxiv.org/pdf/2504.20865) · AI-Synthesized Face benchmark — [arXiv:2402.08750](https://arxiv.org/html/2402.08750v1) · BIAS-ID — [arXiv:2605.31153](https://arxiv.org/pdf/2605.31153)

### Surveys
[arXiv:2502.15176](https://arxiv.org/html/2502.15176v2) · [arXiv:2502.05240](https://arxiv.org/pdf/2502.05240) · [arXiv:2405.00196](https://arxiv.org/pdf/2405.00196) · [arXiv:2409.14128](https://arxiv.org/pdf/2409.14128)

### Backbones
DINOv3 — [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) · Registers — [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) · DINOv2 — [arXiv:2304.07193](https://huggingface.co/facebook/dinov2-large) · SigLIP 2 — [arXiv:2502.14786](https://arxiv.org/pdf/2502.14786) · Perception Encoder — [arXiv:2504.13181](https://arxiv.org/pdf/2504.13181) · MetaCLIP 2 — [arXiv:2507.22062](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) · CLIP — [arXiv:2103.00020](https://huggingface.co/openai/clip-vit-large-patch14) · EVA-02 — [arXiv:2303.11331](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k)

### Detection methods
CNNDetection — [arXiv:1912.11035](https://arxiv.org/abs/1912.11035), [reproduction arXiv:2104.02984](https://arxiv.org/pdf/2104.02984) · FreDect — [arXiv:2003.08685](https://arxiv.org/abs/2003.08685) · UnivFD — [arXiv:2302.10174](https://arxiv.org/abs/2302.10174) · DIRE — [arXiv:2303.09295](https://arxiv.org/abs/2303.09295) · LGrad — [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.html) · PatchCraft — [arXiv:2311.12397](https://arxiv.org/abs/2311.12397) · NPR — [arXiv:2312.10461](https://arxiv.org/html/2312.10461v2) · FatFormer — [arXiv:2312.16649](https://arxiv.org/abs/2312.16649) · RINE — [arXiv:2402.19091](https://arxiv.org/pdf/2402.19091) · SSP/ESSP — [arXiv:2402.01123](https://arxiv.org/abs/2402.01123) · AEROBLADE — [arXiv:2401.17879](https://arxiv.org/abs/2401.17879) · Raising the Bar with CLIP — [arXiv:2312.00195](https://arxiv.org/abs/2312.00195) · DRCT — [PMLR ICML 2024](https://proceedings.mlr.press/v235/chen24ay.html) · CoDE — [arXiv:2407.20337](https://arxiv.org/abs/2407.20337) · TextureCrop — [arXiv:2407.15500](https://arxiv.org/abs/2407.15500) · SAFE — [arXiv:2408.06741](https://arxiv.org/abs/2408.06741) · C2P-CLIP — [arXiv:2408.09647](https://arxiv.org/abs/2408.09647) · AIDE/Chameleon — [arXiv:2406.19435](https://arxiv.org/abs/2406.19435) · SPAI — [arXiv:2411.19417](https://arxiv.org/abs/2411.19417) · Effort — [arXiv:2411.15633](https://arxiv.org/html/2411.15633v3) · B-Free — [arXiv:2412.17671](https://arxiv.org/html/2412.17671v2) · Community Forensics — [arXiv:2411.04125](https://arxiv.org/html/2411.04125v2) · DeeCLIP — [arXiv:2504.19876](https://arxiv.org/pdf/2504.19876) · Bi-LORA — [arXiv:2404.01959](https://arxiv.org/abs/2404.01959) · Seeing What Matters — [arXiv:2506.16802](https://arxiv.org/abs/2506.16802) · What Truly Matters? — [arXiv:2507.10236](https://arxiv.org/html/2507.10236v1) · HiDA-Net — [arXiv:2508.17346](https://arxiv.org/html/2508.17346v1) · FerretNet — [arXiv:2509.20890](https://arxiv.org/html/2509.20890) · DINO-Detect — [arXiv:2511.12511](https://arxiv.org/pdf/2511.12511) · WaRPAD — [arXiv:2511.14030](https://arxiv.org/abs/2511.14030) · DINOv3 cross-generator — [arXiv:2511.22471](https://arxiv.org/html/2511.22471v1) · Simplicity Prevails — [arXiv:2602.01738](https://arxiv.org/html/2602.01738) · Calibration — [arXiv:2602.01973](https://arxiv.org/html/2602.01973) · RA-Det — [arXiv:2603.01544](https://arxiv.org/pdf/2603.01544) · TAP — [arXiv:2604.26772](https://arxiv.org/abs/2604.26772) · DRIFT — [arXiv:2606.06918](https://arxiv.org/html/2606.06918v1) · SSAFE — [arXiv:2606.08634](https://arxiv.org/html/2606.08634)

### Bias, robustness limits and attacks
Fake or JPEG? — [arXiv:2403.17608](https://arxiv.org/abs/2403.17608) · Aligned Datasets — [arXiv:2410.11835](https://arxiv.org/html/2410.11835v3) · Dual Data Alignment — [arXiv:2505.14359](https://arxiv.org/pdf/2505.14359) · Robustness fundamental limits — [arXiv:2310.00076](https://arxiv.org/abs/2310.00076) · Backbone is All You Need (SIAA) — [arXiv:2605.13381](https://arxiv.org/pdf/2605.13381) · CLIP adversarial robustness — [arXiv:2407.19553](https://arxiv.org/pdf/2407.19553) · Real-world adversarial robustness — [arXiv:2410.01574](https://arxiv.org/pdf/2410.01574) · JPEG AI forensic cues — [arXiv:2504.03191](https://arxiv.org/html/2504.03191v1) · MMFusion (SRM/Bayar/NoisePrint++) — [arXiv:2312.01790](https://arxiv.org/html/2312.01790)

### Datasets
SID_Set/SIDA — [arXiv:2412.04292](https://arxiv.org/html/2412.04292) · [HF](https://huggingface.co/datasets/saberzl/SID_Set) · WildFake — [arXiv:2402.11843](https://arxiv.org/abs/2402.11843) · [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32363) · [ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake) · CIFAKE — [arXiv:2303.14126](https://arxiv.org/abs/2303.14126) · [IEEE](https://ieeexplore.ieee.org/abstract/document/10409290) · [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) · Community Forensics — [HF](https://huggingface.co/datasets/OwensLab/CommunityForensics) · So-Fake — [arXiv:2505.18660](https://arxiv.org/abs/2505.18660) · ELSA_D3 — [HF](https://huggingface.co/datasets/elsaEU/ELSA_D3) · LEGION/SynthScars — [arXiv:2503.15264](https://arxiv.org/pdf/2503.15264) · GenImage — [project](https://genimage-dataset.github.io/)
