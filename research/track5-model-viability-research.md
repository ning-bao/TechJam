# Track 5 — Robust Detection of AI-Generated Images Under Real-World Transformations
## Viability Research: Frontier Backbones (<2B params) and Training Approaches

**Compiled:** 2026-08-25
**Scope:** Literature and resource gathering only. This document reports what published papers, benchmarks, challenge reports and model cards say. It does not recommend a solution or evaluate options against each other.
**Hard constraint tracked throughout:** the task rules require models with **< 2B parameters**.

---

## Table of Contents

1. [How the field is organised](#1-how-the-field-is-organised)
2. [The single most relevant prior art: NTIRE 2026](#2-the-single-most-relevant-prior-art-ntire-2026-challenge)
3. [Part A — Candidate frontier backbones under 2B parameters](#part-a--candidate-frontier-backbones-under-2b-parameters)
4. [Part B — Adaptation / fine-tuning strategies](#part-b--adaptation--fine-tuning-strategies)
5. [Part C — Training a model from scratch / lightweight architectures](#part-c--training-a-model-from-scratch--lightweight-architectures)
6. [Part D — Robustness engineering: what the papers report](#part-d--robustness-engineering-what-the-papers-report)
7. [Part E — The dataset-bias trap (JPEG / resolution shortcuts)](#part-e--the-dataset-bias-trap-jpeg--resolution-shortcuts)
8. [Part F — Datasets](#part-f--datasets)
9. [Part G — Evaluation protocols and metrics used in the literature](#part-g--evaluation-protocols-and-metrics-used-in-the-literature)
10. [Part H — Off-the-shelf checkpoints on Hugging Face](#part-h--off-the-shelf-checkpoints-on-hugging-face)
11. [Part I — Reported compute figures](#part-i--reported-compute-figures)
12. [Part J — Explainability / error-analysis material](#part-j--explainability--error-analysis-material)
13. [Consolidated reference list](#consolidated-reference-list)

---

## 1. How the field is organised

The 2025 survey *Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review* ([arXiv:2502.15176](https://arxiv.org/html/2502.15176v2)) groups detection methods into seven families:

| # | Family | Representative methods |
|---|---|---|
| 1 | **Spatial-domain analysis** | Wang et al. CNNDetection, GASE-Net, DIRE |
| 2 | **Frequency-domain analysis** | Fourier/DCT/DWT spectra, TwoStreamNet, Synthbuster, AIDE |
| 3 | **Fingerprint analysis** | GAN-specific artefact signatures, gradient-based (LGrad) |
| 4 | **Patch-based analysis** | Local region classification, inter-patch dependency |
| 5 | **Training-free** | AEROBLADE, RIGID, HFI, ZED, WaRPAD |
| 6 | **Multimodal vision-language** | UnivFD (Ojha), GenDet, FatFormer, ForenX, AIGI-Holmes |
| 7 | **Commercial tools** | Proprietary detectors |

Survey-reported comparison figures (accuracy %):

- **UnivFD benchmark:** Ojha et al. 81.38 · MoLE 92.45 · HyperDet 92.10 · GenDet 94.42
- **GenImage, training-free methods:** RIGID 0.812 · AEROBLADE 0.935 · HFI 0.977 · ForenX (MLLM) 0.978

The survey introduces three generalisation criteria — **cross-family** (GAN↔diffusion), **cross-category** (image classes), and **cross-scene** (dataset distribution). Its stated finding: most methods satisfy the first two and **fail cross-scene**; only RIGID and AIGI-Holmes satisfied all three.

Documented robustness-to-post-processing claims in the survey: GASE-Net (tested vs JPEG, Gaussian blur, resize), AdaptedMultiLID ("resilient to JPEG compression and Gaussian blur"), DIRE ("maintains robustness under Gaussian blur and JPEG"), TwoStreamNet (incorporates blur + JPEG as augmentation).

### Field-level context on "state of the art"

A February 2026 out-of-the-box benchmark study — *How well are open sourced AI-generated image detection models out-of-the-box* ([arXiv:2602.07814](https://arxiv.org/html/2602.07814v1), Ren et al.) — evaluated **16 detection methods (23 pretrained variants) across 12 datasets, 2.6M images, 291 generators**:

- Ranking instability across datasets: Spearman ρ **0.01–0.87**
- **37 percentage-point** accuracy gap between best and worst detector
- Best: **Community-Forensics, 75.0% mean accuracy**; worst: AIGCDetectBenchmark_CNNSpot, 37.5%
- Top-5 range: 67.5–78.0% mean accuracy
- Modern commercial generators (Flux Dev, Firefly v4, Midjourney v7) detected at only **18–30%**
- **Key stated finding:** within identical architecture families (AIDE, DRCT), *training-data alignment explains 20–60% of performance variance, often exceeding the variance between different architectures.*

A NeurIPS 2025 Datasets & Benchmarks paper, *Is AI Generated Image Detection a Solved Problem?* ([arXiv:2505.12335](https://arxiv.org/abs/2505.12335), Li et al.), introduced **AIGIBench** with four core tasks: multi-source generalisation, robustness to image degradation, sensitivity to data augmentation, impact of test-time pre-processing. It evaluated 11 SOTA detectors; reported conclusion is that all showed substantial drops on challenging subsets, with **Fake Image Accuracy often approaching 0%** on in-the-wild social media content. Its degradation protocol uses JPEG compression, noise interference and up/down-sampling; augmentation assessment uses RandomRotation, ColorJitter, RandomMask averaged over 25 subsets; pre-processing assessment uses cropping vs resizing strategies over 25 subsets. Training set ≈ 288K images.

---

## 2. The single most relevant prior art: NTIRE 2026 Challenge

**[NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild](https://arxiv.org/html/2604.11487v1)** (arXiv:2604.11487; CVPR 2026 Workshops; Gushchin, Shumitskaya, Filippov, Bychkov, Erofeev, Lavrushkin, Antsiferova, Vatolin, C. Chen, S. Tan, R. Timofte et al.)

This challenge is functionally the same task as Track 5: detect AI-generated images that have been cropped, resized, compressed, blurred, noised and colour-adjusted, plus generalise to unseen generators.

### Challenge setup

- **Dataset:** 108,750 real + 185,750 AI-generated = ~294,500 images
- **Real sources:** CC12M, CommonPool, RedCaps (~12M filtered to 100k train) via resolution thresholding, CLIP-deduplication, VLM-based categorisation; ~9k for val/test
- **Generated:** ~177k from 20 open-source generators for training; newer models reserved for val/test
- **42 generators total**, released 2022–2026
- **36 image transformations**
- **Metric:** ROC AUC on the full test set including transformed and untransformed images

### Reported transformation list

Gaussian Blur, Lens Blur, Motion Blur, Glass Blur; Color Shift, Color Saturation, Color Jitter, Color Quantization, Color Cast, RGB Channel Shift; JPEG Compression, JPEG 2000, Neural Image Compression (JPEG AI, Cheng2020); White Noise, Impulse Noise, Multiplicative Noise, Shot Noise, Speckle Noise, ISO Noise; Brightness Increase/Decrease; Linear Contrast Change, Random Tone Curve, CLAHE; Pixelation; Random Crop, Random Aspect Crop; Downscale; Perspective Transform; Organic Moire; watermark attacks (adversarial embedding, WMForger, invisible watermark insertion).

**Pipeline structure:** "Each transformation has multiple magnitude levels, which are sampled independently"; images receive **1 to 5 randomly sampled consecutive distortions** from different groups. Both real and generated images pass through the same degradation pipeline. Different distortion sets were used per split, progressively complicated in later stages to prevent tuning to specific distortion types.

### Final leaderboard (ranked by Robust ROC AUC)

| Rank | Team | ROC AUC | Robust ROC AUC | Primary backbone |
|---|---|---|---|---|
| 1 | MICV | 0.9974 | **0.9723** | DINOv3 ensembles (4 + 2 models), 512×512 |
| 2 | Ant International | 0.9972 | 0.9721 | DINOv3-7B dual-expert (~14B total) |
| 3 | TeleAI-TeleGuard | 0.9786 | 0.9251 | EVA-CLIP + LoRA |
| 4 | INTSIG | 0.9897 | 0.9130 | 4× DINOv3-Huge + 1× MetaCLIP2-Giant |
| 5 | vincentlc | 0.9527 | 0.8730 | SigLIP2-Giant-Opt-Patch16-384, single linear head |
| 6 | UESTC | 0.9729 | 0.8679 | 2× CLIP ViT-L/14 + 2× SigLIP-So400M |
| 7 | Reagvis Labs | 0.9452 | 0.8603 | — |
| 8 | PSU | 0.9227 | 0.8408 | — |
| 9 | Shallow Real | 0.9953 | 0.8336 | — |

Note the divergence between clean AUC and robust AUC — e.g. "Shallow Real" ranked near the top on clean AUC (0.9953) but 9th on robust AUC (0.8336).

### Top team method details as reported

**1st — MICV.** DINOv3 detection framework, 512×512 input, projection layer → MLP head. Training corpus: GenImage, WildFake, AIGIBench, CommunityForensics, So-Fake-Set + self-generated (Qwen-Image, Z-Image, FLUX) + closed-source (Seedream, Kling, GPT-Image, Nano-banana-pro) + challenge set. **Hierarchical stochastic augmentation pipeline structured by difficulty levels**, progressing from simple (blur, noise, shifts) to complex multi-stage degradations. Late-fusion probability averaging across two model committees. Focal Loss (γ=2.0, α=0.5), Stochastic Weight Averaging, cosine annealing with linear warmup. 32× A100, 10 epochs, ~8 hours.

**2nd — Ant International.** DINOv3-7B dual-expert. ~1M images. **Four-level offline augmentation:** L1 clean; L2 1–3 distortions (mean 0, std 2.5); L3 3–6 distortions (mean 2.5, std 2.0); L4 fixed 6 distortions (mean 3.5, std 1.0). Online: random horizontal flip + AugMix (m6-w3-d1). Expert 1 = high-resolution specialist (512×512, attention pooling, 1 epoch); Expert 2 = robustness specialist (288×288, first_token pooling, 10 epochs). TTA with weighted ensembling. Model EMA, AMP. B200 training; A100 inference at 2.21 img/s, 78.25 GB VRAM.

**3rd — TeleAI-TeleGuard.** EVA-CLIP with **LoRA-adapted linear layers in MHSA and FFN blocks**. Added So-Fake and Chameleon; added Speckle Noise, Color Cast, Organic Moire beyond the organiser pipeline; Gaussian distribution mean set to 3. **LoRA-based Pairwise Training (LPT):** clean + distorted image pairs in the same batch, plus a feature correction network for distorted samples. Loss: `L_CE(x,y) + α·L_KL(x,x̂) + β·L_MSE(f_x, f'_x̂)` with α=0.5, β=0.25. AdamW lr 2e-4, cosine annealing. 8× A800, 5 epochs.

**4th — INTSIG.** Five models: M1–M4 DINOv3-Huge variants (baseline / dataset expansion / enhanced augmentation / 448×448 high-res), M5 MetaCLIP2-Giant partial fine-tune. Heads: M1–M4 `1280→256→2`; M5 `1664→2048→512` then `512→256→2`. Data: official + SoFake-OOD + RRDataset + Chameleon + GenImage_val + AIGIBench_test. Augmentation: M1–M3 random resized crop 256×256 bicubic scale 0.08–1.0; M3 enhanced to 5 distortions × 5 levels; M4 distortion prob 0.5, up to 3 distortions, 5 levels; M5 lighter (prob 0.2, 3 distortions, 3 levels). **Weighted hierarchical fusion with dual-gating:** `0.7[0.7(0.75·M1+0.15·M2+0.1·M3)+0.3·M4]+0.3·M5`, Gate-1 strong-consensus correction, Gate-2 anomaly suppression. Horizontal-flip TTA on M3/M4. Separate LR for backbone vs head. 8× H800.

**5th — vincentlc.** *Simplest reported pipeline.* **SigLIP2-Giant-Opt-Patch16-384**, single linear layer on **globally average-pooled patch tokens**. Official training set only (~277k images). **"Squish" preprocessing**: direct resize to 384×384 ignoring aspect ratio. Random horizontal flip. Competition distortions with `distortion_prob=1.0`, up to 3 operations, 5 severity levels — i.e. *every* training image is distorted. No ensembling, no TTA. The team's reported ablation preference: global average pooling over final-layer patch tokens beat CLS token and attention pooling for robustness/stability.

**6th — UESTC.** Four-expert ensemble: 2× CLIP ViT-L/14 (224×224) + 2× SigLIP So400M-patch14-384 (384×384), probabilities simply averaged. **Two-stage training:** Stage 1 standard binary classification (2 epochs); Stage 2 **feature-level self-distillation** using intermediate feature maps from epoch 2 as dense targets. ~10 GB peak GPU memory.

### Organisers' stated conclusions

1. Most final solutions used an **expert-based (ensemble) architecture** with several different models.
2. **Transformer backbones dominated** the top of the leaderboard.
3. Aggressive **robust augmentation** during training; constant distortion across batches proved effective.
4. **Large-scale, diverse data** — mixing open-source datasets, open-source generators and closed-source commercial models — significantly improved generalisation.
5. **Model scaling and higher input resolution consistently outperformed** smaller variants.
6. **Paradigm diversity** (vision-language + self-supervised + forensic) was complementary.
7. Clear gap between top-2 (~0.972 robust AUC), tier 2 (~0.925) and tier 3 (<0.88) — *"robustness [is] a key differentiating factor."*
8. *"The problem is not yet solved, and there remains room for advances in design, training strategies and data curation."*

---

## Part A — Candidate frontier backbones under 2B parameters

Parameter counts below are taken from the Hugging Face model index (whole-checkpoint parameter count as reported by the Hub) unless noted. **Where a model is a dual-tower VLM (CLIP/SigLIP/MetaCLIP/PE-Core), the Hub figure includes the text tower**; only the vision tower is normally needed for a detector, so the effective count is lower.

### A.1 Self-supervised vision backbones (DINO family)

| Checkpoint | Params | Arch | License | Notes |
|---|---:|---|---|---|
| [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 21.6M | ViT-S/16 | other (gated) | |
| [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 85.7M | ViT-B/16 | other (gated) | |
| [`facebook/dinov3-convnext-large-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m) | 196.2M | ConvNeXt-L | other (gated) | |
| [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | 303.1M | ViT-L/16 | other (gated) | |
| [`facebook/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | **840.6M** | ViT-H+/16 | other (gated) | The "DINOv3-Huge" used by NTIRE 4th place |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | ~6.7B | ViT-7B/16 | other (gated) | **Exceeds the 2B limit** |
| [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large) | 304.4M | ViT-L/14 | apache-2.0 | |

**DINOv3 paper** ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104), Meta AI, Aug 2025): 7B flagship trained on 1.7B curated images (from ~17B Instagram images via hierarchical k-means + retrieval-based selection + ImageNet/Mapillary), improved ViT with RoPE. Headline contribution is **Gram anchoring** — a Gram-matrix loss on pairwise patch-feature similarities between student and an earlier "anchor teacher", fixing the known degradation of dense feature maps over long training (patch-level locality degrades after ~200k iterations while global metrics keep improving). Followed by high-resolution fine-tuning, then multi-student distillation into the smaller family members. Meta's stated positioning: frozen backbone + lightweight task head, "train once, apply everywhere."

**Evidence for DINO-family in this task:**

- *Navigating the Challenges of AI-Generated Image Detection in the Wild: What Truly Matters?* ([arXiv:2507.10236](https://arxiv.org/html/2507.10236v1), Konstantinidou, Karageorgiou, Koutlis, Papadopoulou, Schinas, Papadopoulos; 5th ACM Workshop on Multimedia AI against Disinformation) directly compared five encoders under the RINE method across Synthbuster / Chameleon / ITW-SM:

  | Backbone | Synthbuster AUC | Chameleon AUC | ITW-SM AUC | Avg AUC |
  |---|---:|---:|---:|---:|
  | CLIP L/14 | 96.98 | 82.25 | 96.53 | 91.92 |
  | OpenCLIP L/14 | 74.82 | 85.86 | 90.01 | 83.56 |
  | CLIP H/14 | 97.02 | 81.22 | 90.56 | 89.60 |
  | BLIP2 | 99.37 | 86.58 | 96.49 | 94.15 |
  | **DINOv2-L/14** | **99.14** | **87.33** | **98.23** | **94.90** |

  Authors' stated interpretation: *"CLIP-based methods' reliance on image-text alignment may introduce semantic shortcuts,"* reducing sensitivity to synthetic artefacts.

- *Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models* ([arXiv:2602.01738](https://arxiv.org/html/2602.01738), Zhou, He, Lin, Fan, Ding, Li; Feb 2026, rev. Apr 2026) trained **a single linear classifier on frozen VFM features**, with AdamW lr 1e-3, batch 128, 2 epochs, on the GenImage SD v1.4 subset only:

  | Model | GenImage avg acc | In-the-wild avg (Chameleon, WildRF, SocialRF, CommunityAI) | AIGIHolmes |
  |---|---:|---:|---:|
  | **DINOv3-Linear** | **96.4%** | **94.0%** | 97.2% |
  | PE-CLIP-Linear (ViT-L/14 @336) | 93.8% | 89.9% | **97.8%** |
  | MetaCLIP2-Linear | 89.2% | 84.2% | 94.2% |
  | OMAT (specialist SOTA) | 94.6% | — | — |

  Reported +30.4% in-the-wild improvement of DINOv3 over DINOv2. Robustness note: under JPEG q=65 and Gaussian blur σ=2.0, DINOv3 and MetaCLIP2 retained detection capability better than legacy models; under real-world recapture DINOv3 reached ~64.7% on severely degraded images. Stated limitations: degradation under recapture and transmission; blind to VAE reconstruction and localized editing. Claimed emergence mechanism: massive-scale pretraining data now *contains* synthetic content, so VLMs develop explicit forgery concepts and SSL models acquire implicit forensic features.

- **NTIRE 2026:** ranks 1, 2 and 4 all built on DINOv3.

- *DINO-Detect: A Simple yet Effective Framework for Blur-Robust AI-Generated Image Detection* ([arXiv:2511.12511](https://arxiv.org/pdf/2511.12511), Shen, Zheng, Xue, Chen, Yao, Kang, Liu, Gong, Wang, Yang, Wang, T. Liu; Nov 2025) uses DINOv3 with a **teacher-student distillation** where a clean-image teacher transfers to a student on blurred/degraded images. Introduces the **AIGI-Blur** dataset (AI-generated + real motion-blurred images). *DailyBench* independently describes DINO-Det as improving robustness under degradation by "distilling feature and logit responses from sharp images to blurred images with a frozen teacher model."

- *DRIFT* ([arXiv:2606.06918](https://arxiv.org/html/2606.06918v1), Samsung Research India, Jun 2026) uses **frozen DINOv2 ViT-B/14** (768-dim) with two 2-layer MLP heads.

### A.2 Vision-language backbones

| Checkpoint | Params (full) | Vision tower | License | Notes |
|---|---:|---:|---|---|
| [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | 427.6M | ~304M | — | The UnivFD / RINE / Effort default |
| [`google/siglip2-base-patch16-384`](https://huggingface.co/google/siglip2-base-patch16-384) | 375.5M | ~86M | apache-2.0 | |
| [`google/siglip2-large-patch16-384`](https://huggingface.co/google/siglip2-large-patch16-384) | 881.9M | ~303M | apache-2.0 | |
| [`google/siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384) | 1,136M | ~400M | apache-2.0 | SigLIP So400M family used by NTIRE 6th place |
| [`google/siglip2-giant-opt-patch16-384`](https://huggingface.co/google/siglip2-giant-opt-patch16-384) | **1,871.9M** | ~1.0B | apache-2.0 | Used by NTIRE 5th place; **under 2B but close to the ceiling** |
| [`facebook/metaclip-2-worldwide-huge-quickgelu`](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) | **1,858.8M** | — | **cc-by-nc-4.0** | Non-commercial licence |
| [`facebook/PE-Core-L14-336`](https://huggingface.co/facebook/PE-Core-L14-336) | ~0.3B vision | 0.3B | apache-2.0 | |
| `facebook/PE-Core-B16-224` | ~86M vision | 86M | apache-2.0 | |
| `facebook/PE-Core-G14-448` | 1.88B vision + 0.47B text | 1.88B | apache-2.0 | Vision tower alone under 2B; **full dual-tower ≈2.35B exceeds it** |
| [`timm/eva02_large_patch14_448...`](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) | 305.1M | — | mit | EVA-02 L; EVA-CLIP used by NTIRE 3rd place |
| `timm/eva02_base_patch14_448...` | 87.1M | — | mit | |

**SigLIP 2** ([arXiv:2502.14786](https://arxiv.org/pdf/2502.14786), Tschannen et al., Feb 2025): four sizes — ViT-B (86M), L (303M), So400m (400M), g (~1B). NaFlex variants (native aspect ratio, variable resolution) exist for B, L, So400m but **not** for giant. Fixed-resolution models use `SiglipModel`; NaFlex requires `Siglip2Model`.

**Perception Encoder** ([arXiv:2504.13181](https://arxiv.org/pdf/2504.13181), Meta, Apr 2025): PE-Core-B ≈86M, PE-Core-L ≈0.3B, PE-Core-G ≈1.88B vision. Trained on 5.4B image-alt-text pairs curated with MetaCLIP; B and L distilled from PE-core-G. Meta reports PE core outperforming SigLIP2 on image benchmarks and PE spatial outperforming DINOv2 on dense prediction. Central claim of the paper title: *"the best visual embeddings are not at the output of the network"* — they are in intermediate layers, requiring alignment to extract.

**Evidence for PE in this task:** *SSAFE: Simple and Strong AI-Generated Image Detection via Frozen Vision Encoders* ([arXiv:2606.08634](https://arxiv.org/html/2606.08634), Lee, Kim, Nam, K. Lee, Shin; Jun 2026) selected **PE-Core-G14-448**, reporting it *"provides the clearest real/fake separation and the most structured generator clusters"* versus CLIP, SigLIP, DINO variants. Head is a **single linear layer + sigmoid** on L2-normalised embeddings, threshold 0.5. Detail in [Part B.9](#b9-representation-aware-data-curation-ssafe).

### A.3 Parameter-budget summary against the <2B rule

| Comfortably under 2B | Near the ceiling | Over the limit |
|---|---|---|
| DINOv3 S/B/L/H+ (21.6M–840.6M) · DINOv3 ConvNeXt-L (196M) · DINOv2-L (304M) · CLIP ViT-L/14 (428M) · SigLIP2 B/L/So400M (375M–1.14B) · PE-Core-B/L · EVA-02 B/L · Swin/SwinV2 variants | SigLIP2-Giant-Opt (1.87B) · MetaCLIP2-Huge (1.86B, non-commercial licence) · PE-Core-G vision tower alone (1.88B) | DINOv3 ViT-7B (~6.7B) · PE-Core-G full dual-tower (~2.35B) · any 7B MLLM-based detector (LEGION, SIDA, FakeVLM) |

**Licensing notes gathered:** DINOv3 checkpoints are **gated** on the Hub and carry a custom "other" licence (DINOv3 Licence) — access request required. MetaCLIP 2 worldwide-huge is **cc-by-nc-4.0** (non-commercial). SigLIP2 and PE-Core are apache-2.0. EVA-02 timm weights are MIT. `Organika/sdxl-detector` is cc-by-nc-3.0.

### A.4 Head/pooling design evidence

*TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated Image Detection* ([arXiv:2604.26772](https://arxiv.org/abs/2604.26772), Abdullah, Ebert, Wasenmüller; Apr 2026) benchmarks multiple VFM families (varied pretraining objectives, input resolutions, model scales) and proposes **TAP = Tunable Attention Pooling**, described as *"a simple redesign of the classifier head... which aggregates output tokens into a refined global representation."* Reported: best model **outperforms original CLIP by >12% accuracy**, SOTA on two in-the-wild benchmarks for generated and inpainted images.

Counter-datapoint from NTIRE 2026: the 5th-place team evaluated CLS-token extraction, attention pooling, and multi-layer feature concatenation, and reported **global average pooling over all final-layer patch tokens** gave the most robust and stable results.

---

## Part B — Adaptation / fine-tuning strategies

### B.1 Frozen backbone + linear probe (UnivFD / UniFD)

**[Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://github.com/WisconsinAIVision/UniversalFakeDetect)** — Ojha, Li, Lee, CVPR 2023 ([arXiv:2302.10174](https://arxiv.org/abs/2302.10174)).

Core argument: detectors trained *only* on fake-specific features introduce bias and lose generalisability; instead use a feature space **not trained for this task** — CLIP-ViT's. Instantiated as nearest-neighbour and linear probing in the frozen CLIP:ViT-L/14 space. Trained on ProGAN fakes only. Reported +15.07 mAP and +25.90% accuracy over prior SOTA on unseen diffusion and autoregressive models, with ablations on internet-scale pretraining and robustness to JPEG/Gaussian blur.

Training invocation from the official repo:
```
python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ \
  --data_mode=wang2020 --arch=CLIP:ViT-L/14 --fix_backbone
```
`--fix_backbone` ensures only the linear layer trains.

Current standing: a 2025 prompt-learning method reports 95.61% mAcc / 99.32% mAP across 19 sub-test sets — +14.23% mAcc, +9.18% mAP over UniFD.

**Few-shot variant:** *Raising the Bar of AI-generated Image Detection with CLIP* ([arXiv:2312.00195](https://arxiv.org/abs/2312.00195), Cozzolino, Poggi, Corvi, Nießner, Verdoliva, CVPRW 2024; [code](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)). Central claim: *a large domain-specific training dataset is neither necessary nor convenient* — with only a handful of example images from a single generator, a CLIP-based detector generalises well, including to DALL·E 3, Midjourney v5, Firefly. Reported +6% AUC on OOD data and **+13% on impaired/laundered data**. Supplementary robustness analysis: both the 1k+ and 10k+ **versions with augmentation are "basically insensitive to compression (JPEG or WebP) and resizing"**; versions without augmentation lose performance but far less dramatically than reference methods.

### B.2 Intermediate-layer aggregation (RINE)

**[Leveraging Representations from Intermediate Encoder-Blocks for Synthetic Image Detection](https://arxiv.org/pdf/2402.19091)** — Koutlis & Papadopoulos, ECCV 2024 ([code](https://github.com/mever-team/rine)).

Hypothesis: final-layer foundation-model features encode high-level semantics, but **fine-grained low-level detail — which shallow layers encode — matters more for SID**.

Architecture: CLS tokens from each intermediate transformer block concatenated as `K = ⊕(l=1..n) Z_l^[0] ∈ ℝ^(b×n×d)`; projected, multiplied by block scores from a **Trainable Importance Estimator (TIE)**, summed across the block dimension → one feature vector per image; second projection + classification head. Two losses: binary cross-entropy + a contrastive loss forming a dense per-class cluster.

Reported: +10.6% absolute average improvement over SOTA across 20 test datasets. **Best models require just a single epoch (~8 minutes).**

### B.3 LoRA / parameter-efficient fine-tuning

**DeeCLIP** ([arXiv:2504.19876](https://arxiv.org/pdf/2504.19876), Keita et al.; [code](https://github.com/Mamadou-Keita/DeeCLIP); also Springer 10.1007/978-3-032-07343-3_12) — explicitly targets robustness.

- Backbone: **CLIP-ViT-L/14 fine-tuned with LoRA** (`W' = W + BA`). Rationale given: full fine-tuning on limited data risks overfitting and distorts CLIP's pre-existing visual-world knowledge, reducing performance under distribution shift.
- **DeeFuser**: fuses deep features (as queries) with shallow features (keys/values) from multiple CLIP-ViT layers via cross-attention + MLP — *"improving robustness against degradations such as compression and blurring."*
- **Triplet loss** to refine the embedding space.
- Reported: DeeFuser multi-scale fusion gives **+10.36% over C2P-CLIP**; LoRA tuning improves accuracy **84.53% → 89.00%**. Benchmarked vs C2P-CLIP, RINE, FatFormer, AntifakePrompt, Bi-LORA.

**Bi-LORA** ([arXiv:2404.01959](https://arxiv.org/abs/2404.01959), Keita, Hamidouche et al.; *Expert Systems* 2025, doi 10.1111/exsy.13829; [code](https://github.com/Mamadou-Keita/VLM-DETECT)) — reframes binary classification as **image captioning** with BLIP2 + LoRA. Reported 93.41% average accuracy on unseen generators with far fewer tuned parameters than prior work. Stated limitation: uneven cross-generator transfer — trained on ADM/IDDPM and tested on SD v1.4, precision drops to ~48–50%; SD v1.4 / GLIDE training subsets generalise poorly (~50% across unconditional diffusion and LDM).

**C2P-CLIP** ([arXiv:2408.09647](https://arxiv.org/abs/2408.09647), Tan, Tao, Liu, Gu, Wu, Zhao, Wei; AAAI 2025, 39(7):7184–7192). Analysis method: decode the detection feature into text plus word-frequency analysis; **finding — CLIP detects deepfakes by recognising similar *concepts*, not by comprehending "real"/"fake" semantics.** Method: inject a category common prompt into the text encoder to embed category-related concepts into the image encoder; text encoder frozen, **image encoder trained with LoRA**. Reported +12.41% detection accuracy over original CLIP, **no additional parameters at test time**.

**NTIRE 3rd place** used LoRA on EVA-CLIP in MHSA and FFN blocks, combined with pairwise clean/distorted training (see [Section 2](#2-the-single-most-relevant-prior-art-ntire-2026-challenge)).

### B.4 Orthogonal subspace decomposition (Effort) — lowest reported trainable-parameter count

**[Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://arxiv.org/html/2411.15633v3)** — Yan, Wang, Wang, Jin, Zhang, Chen, Yao, Ding, Wu, Yuan; **ICML 2025 Oral**, PMLR 267:70268–70288 ([code](https://github.com/YZY-stack/Effort-AIGI-Detection), [OpenReview](https://openreview.net/forum?id=GFpjO8S8Po)).

Diagnosed problem — the **"asymmetry phenomenon"**: a naively trained detector overfits to limited, monotonous fake patterns, making the feature space highly constrained and **low-ranked**, limiting expressivity and generalisation.

Method: **SVD-decompose the feature space into two orthogonal subspaces; freeze the principal components, adapt only the residual components.** Preserves pretrained knowledge while learning fake patterns; unlike full-parameter and LoRA tuning, orthogonality is explicitly ensured, enabling a **higher rank** of the whole feature space. Implementation is `torch.linalg.svd` on a layer's weight, top-*r* singular components fixed as "main" weight, residual trainable. Plug-and-play into any ViT-based model.

Stated implicit prior learned: *fakes are derived from the real — a hierarchical, not independent, relationship.*

| Property | Value |
|---|---|
| Backbone | CLIP ViT-L/14 (also validated: BEiT-v2, SigLIP) |
| **Trainable params** | **0.19M** (~1000× fewer than LSDA 133M, ProDet 96M) |
| Training data | ProGAN (synthetic detection) / FF++ c23 (deepfake) |
| Hyperparameters | lr 2e-4 fixed, Adam, batch 32 (deepfake) / 48 (synthetic), 224px |
| Synthetic-image result | **95.19% mAcc, 99.41% mAP** across 19 test subsets (vs FatFormer 90.86% mAcc) |
| Deepfake cross-dataset | 0.917 avg AUC (vs CDFA 0.878, ProDet 0.828) |
| Robustness | Fig. 8 — video-level AUC across 5 degradation levels × 3 perturbation types (block-wise distortion, contrast, JPEG) |

Also contributes a tool for quantifying degree of model overfitting.

### B.5 Adapter + language alignment (FatFormer)

**[Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection](https://arxiv.org/abs/2312.16649)** — Liu, Tan, Tan, Wei, Wang, Zhao, CVPR 2024, pp. 10770–10780 ([code](https://github.com/Michel-liu/FatFormer)).

Diagnosis: the fixed paradigm of frozen CLIP-ViT + learnable linear layer (UniFD) *"tends to yield detectors with insufficient learning of forgery representations"* — the key challenge is **lack of forgery adaptation**.

Two designs:
1. **Forgery-aware Adapter (FAA)** — adapts image features to discern and integrate local forgery traces in **both image and frequency domains**.
2. **Language-guided Alignment (LGA)** — contrastive objectives between adapted image features and text prompt embeddings; described as "a previously overlooked aspect" yielding non-trivial generalisation improvement.

Reported: tuned on 4-class ProGAN data → **98% average accuracy on unseen GANs, 95% on unseen diffusion models.** Forgery adaptation consistently boosts various architectures and pretraining strategies.

### B.6 Distillation for degradation robustness

- **DINO-Detect** ([arXiv:2511.12511](https://arxiv.org/pdf/2511.12511)) — teacher trained on clean images distils into a student operating on blurred/degraded images; DINOv3 backbone; AIGI-Blur dataset.
- **NTIRE 6th (UESTC)** — two-stage: standard binary classification for 2 epochs, then **feature-level self-distillation with dense supervision**, using intermediate feature maps from epoch 2 as dense targets.
- **NTIRE 3rd (TeleAI)** — feature correction network + KL/MSE consistency between clean and distorted pairs (α=0.5, β=0.25).
- **GlobalForge** — degradation-aware contrastive structural loss aligning clean and degraded representations ([Part D](#part-d--robustness-engineering-what-the-papers-report)).

### B.7 One-class training on real images only (DRIFT)

**[DRIFT: From Robustness Gaps to Invariance Manifolds for AI-Generated Image Detection](https://arxiv.org/html/2606.06918v1)** — Ameta, Banerjee, Pandith, Harshit, Chatterjee, Bankar, Unde (Samsung Research India), Jun 2026.

Models real images as samples from a low-dimensional manifold shaped by natural image statistics and physical imaging processes. Trains two lightweight projection heads decomposing representation space into:

- **Robust subspace** — suppresses variation from physically plausible transforms (mild blur, JPEG, resampling, photometric jitter)
- **Fragile subspace** — retains sensitivity to edit-like/structurally inconsistent perturbations (pixelation, defocus blur, heavy compression, strong photometric distortion)

Detection = margin-violation test: `S(x) = D_R(x) + γ − D_F(x)`.

| Property | Value |
|---|---|
| Backbone | Frozen **DINOv2 ViT-B/14** (768-d) |
| Trainable | Two 2-layer MLPs (→512→256, GELU) |
| Training data | **Real images only** — MIT-5K, LSUN, RAISE. No AI-generated images during training. |
| Stabilisation | EMA teacher λ=0.996, reconstruction anchor loss, ordering margin |
| Optimisation | AdamW lr 3e-4, batch 64, 50 epochs, cosine + linear warmup |
| Loss weights | λ_rob=1.0, λ_frag=1.0, λ_ord=0.5, λ_rec=0.1, margin γ=0.3 |
| Inference | Patch-wise drift scores, aggregated by **top-k median** |

Results: ForenSynths 97.8% ACC / 99.8% AP mean; Diffusion-6cls 92.1–98.6% ACC; PromptWorld-1K (Gemini/ChatGPT) 93.2–94.8% ACC. Ablation: robust-only 96.3% AUC, fragile-only 61.2%, shared single head 94.6%, **dual-head 98.1%**. Individual transform families: robust 85.7–87.3% AUC, fragile 59.3–68.7%.

### B.8 Training-free methods

**WaRPAD** — *[Training-free Detection of AI-generated images via Cropping Robustness](https://arxiv.org/pdf/2511.14030)* (Choi, Lee, Lee; Nov 2025). Combines wavelet high-frequency perturbation, cropping robustness, and **RRC-inspired TTA** — deterministically simulating multiple random-resized-crop instances by explicitly rescaling and patchifying the image. Core observation: *AI-generated images lose robustness to wavelet-based high-frequency perturbation when examined in patches*; discrimination is strengthened by **averaging the score function across patches**. Uses pretrained DINOv2, CLIP, SwAV. Evaluated on Synthbuster/Raise-1K and GenImage at resolutions 256×256 up to 4928×3264.

**SPAI** — *[Any-Resolution AI-Generated Image Detection by Spectral Learning](https://arxiv.org/abs/2411.19417)* (Karageorgiou, Papadopoulos, Kompatsiaris, Gavves; CVPR 2025 pp. 18706–18717; [code](https://github.com/mever-team/spai), [project](https://mever-team.github.io/spai/)). Key idea: **the spectral distribution of real images is both invariant and highly discriminative.** Modelled self-supervised via **masked spectral learning** (pretext task = frequency reconstruction); generated images are OOD for this model, captured by **Spectral Reconstruction Similarity (SRS)**; **Spectral Context Attention (SCA)** enables any-resolution processing without pre-processing. Reported **+5.5% absolute AUC** over prior SOTA across 13 generators, *"while exhibiting robustness against common online perturbations."* Inference runs under 8 GB GPU RAM; reproducing training needs ~48 GB.

Others catalogued by the survey: AEROBLADE (0.935 mean acc on GenImage), HFI (0.977), RIGID (0.812), ZED.

### B.9 Representation-aware data curation (SSAFE)

**[SSAFE: Simple and Strong AI-Generated Image Detection via Frozen Vision Encoders](https://arxiv.org/html/2606.08634)** — Lee, Kim, Nam, K. Lee, Shin; Jun 2026.

Central finding: *"frozen multimodal encoders naturally separate real and synthetic images in their embedding space,"* enabling classification without task-specific backbone training.

**Curation pipeline (50K → 10K):**
1. Extract D-dim normalised embeddings from the frozen encoder
2. Pairwise **Maximum Mean Discrepancy (MMD)** between generator distributions → distance matrix
3. **Hierarchical clustering** (average linkage) → hyperclusters
4. **Greedy Farthest-Point Sampling** for diversity → ~8 representative generator combinations from 28

| Setting | Value |
|---|---|
| Encoder | PE-Core-G14-448 (frozen) |
| Head | single linear + sigmoid on L2-normalised embeddings, threshold 0.5 |
| Optimiser | AdamW, lr 1e-3, batch 40, BCE |

| Benchmark | Result |
|---|---|
| AIGIBench | 89.4% acc, 95.7% AP — with **29× fewer samples** than AIGIBench's 288K |
| AIGI-Holmes (10 frontier T2I generators) | 99.9% acc, 100.0% AP |
| OpenFake subsampling | 5K → 99.0% F1; 10K → 98.8%; 30K → 99.3% (vs 4M-image SwinV2 baseline 99.2%) |
| RealWorldBench | Curated 10K: **98.3% TNR**, 94.4% avg TPR · Universal 50K: 96.8% TNR, 95.3% TPR |
| Curation ablation | Curated 10K 96.4% vs Random 10K 94.9% (+1.5%) |

### B.10 Post-hoc calibration

**[Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated](https://arxiv.org/html/2602.01973)** — Yang, Goenawan, H. Wang, Qin, Xu, Yang, Fang, Sun, Lim, Zhu; Feb 2026.

Problem identified — **threshold misalignment**: models trained on balanced datasets systematically misclassify *fake* images as *real* under test-time distribution shift. Two causes: (1) class-conditional input shift — different generators produce "coherent and systematic deviations in visual statistics"; (2) label prior shift.

Method: a **single learnable scalar bias α** on output logits, `f̃(x) := f(x) − α`. Two variants — supervised (KDE on labelled validation data) and unsupervised (distributional symmetry, as few as **10 unlabelled samples**). Backbone stays frozen.

Reported gains (avg accuracy):

| Benchmark | Detector | Gain |
|---|---|---|
| AIGCDetectBenchmark (16 gen.) | CNNSpot | +7.39% |
| | Fusing | +9.57% sup / +6.75% unsup |
| | Effort | +10.13% sup |
| GenImage (8 gen.) | RINE | **+16.16% sup / +15.80% unsup** |
| | Fusing | +14.03% sup |
| **Robustness — JPEG QF=90** | AIDE | **+15.39%** |

Consistent across 9 detectors, strongest for CLIP-feature-based models. Required only ~1% of test data (**100 images**) for calibration.

### B.11 Summary comparison of adaptation strategies

| Strategy | Trainable params | Backbone frozen? | Reported headline | Source |
|---|---|---|---|---|
| Linear probe | ~1–2K | Yes | 81.38% mean (UnivFD benchmark) | Ojha CVPR'23 |
| Linear probe on modern VFM | ~1–2K | Yes | DINOv3 96.4% GenImage / 94.0% in-the-wild | Simplicity Prevails |
| Linear probe + curated 10K | ~1–2K | Yes | 89.4% acc AIGIBench, 98.3% TNR | SSAFE |
| Intermediate-block agg. (RINE) | small MLP + TIE | Yes | +10.6% abs. over SOTA; 1 epoch ≈8 min | Koutlis ECCV'24 |
| Attention-pooling head (TAP) | small | Yes | >+12% acc over original CLIP | TAP 2026 |
| **SVD residual (Effort)** | **0.19M** | Partially | 95.19% mAcc / 99.41% mAP | Yan ICML'25 Oral |
| LoRA (DeeCLIP) | low-rank | No (LoRA) | 84.53%→89.00%; +10.36% vs C2P-CLIP | DeeCLIP 2025 |
| LoRA (C2P-CLIP) | low-rank | No (LoRA) | +12.41% over CLIP | AAAI'25 |
| Adapter + language align (FatFormer) | adapter | No | 98% unseen GAN / 95% unseen DM | CVPR'24 |
| One-class dual-head (DRIFT) | 2 small MLPs | Yes | 98.1% AUC | Samsung 2026 |
| Full fine-tune (NTIRE top teams) | full | No | 0.9723 robust AUC | NTIRE 2026 |
| Post-hoc calibration | **1 scalar** | Yes | +7–16% acc on existing detectors | Yang 2026 |

---

## Part C — Training a model from scratch / lightweight architectures

### C.1 CNNDetection (the foundational augmentation result)

**[CNN-generated images are surprisingly easy to spot... for now](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf)** — Wang, Wang, Zhang, Owens, Efros, CVPR 2020 ([code](https://github.com/peterwang512/CNNDetection)). ResNet-50, ProGAN training data.

Findings relevant to Track 5:

- **Data augmentation in the form of common post-processing operations is critical for generalisation — even when the target test images are not post-processed.**
- Rationale: post-processing occurs downstream of image creation (storage, distribution); with the correct steps, classifiers are robust to JPEG compression, blurring and resizing.
- Training-image diversity matters, up to a point.
- Figure 5 reports AP under test-time Gaussian blur (left) and JPEG (right).
- Caveat: for SAN, applying data augmentation actually *hurts*.

**Critical implementation detail** from the reproducibility study ([arXiv:2104.02984](https://arxiv.org/pdf/2104.02984), Frank et al.): "Blur+JPEG (0.5)" means blur is applied with p=0.5 and **then** JPEG independently with p=0.5 — *not* jointly at p=0.5. The reproduction team initially applied them jointly and got divergent results (SAN differed by 30.2%). They also note Wang et al. implemented Gaussian filtering with SciPy, not PyTorch built-ins. Reproduced mAP across 11 generators: no augmentation — Wang 90.1 vs reproduced 89.7; blur only — 84.4 vs 83.8.

### C.2 NPR — lightweight CNN on pixel-relationship features

**[Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection](https://arxiv.org/html/2312.10461v2)** — Tan et al., CVPR 2024 ([code](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)).

Idea: up-sampling operators (universal in GANs and diffusion) create **local interdependence among neighbouring pixels**. NPR = **Neighbouring Pixel Relationships**, computed by *subtraction* within a local grid (window size *l*=2, index *j*=1), over the whole image. The NPR representation feeds a **lightweight CNN**.

Why it is claimed to generalise: the relationship is *relative* and *local*, deriving from the upsampling layer and benefiting from convolutional translation invariance. Heatmaps show it captures artefacts around hair, eyes, beard. Reported **+11.6% over existing methods** across 28 generative models, trained only on ProGAN, generalising to unseen GAN and diffusion sources.

**Documented robustness weakness:** RA-Det ([arXiv:2603.01544](https://arxiv.org/pdf/2603.01544)) states there is *"a clear robustness gap between artifact-driven detectors and feature-level modeling, since methods like NPR rely on fragile local pixel statistics easily disrupted by compression and blur."* Independently, in *What Truly Matters?*, NPR barely benefits from augmentation (67.36 → 67.48 avg AUC) and *loses* accuracy under texture cropping (70.11 → 67.48) — opposite to the trend for CLIP-based methods.

### C.3 SAFE — preprocessing/augmentation-centric CNN

**[Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective](https://arxiv.org/abs/2408.06741)** — Li, Cai, Hao, Jiang, Hu, Feng; **KDD 2025** (submitted Aug 2024, revised Jan 2025).

Diagnosed biases: weakened and overfitted artefact features. Four strategies:

1. **Replace conventional down-sampling with a crop operator** — RandomCrop for training, CenterCrop for inference — to preserve local correlations and prevent artefact distortion.
2. **Invariant augmentations** — ColorJitter and RandomRotation — to mitigate colour discrepancies and irrelevant rotation-related features.
3. **Patch-based random masking** — to enhance sensitivity to local regions and subtle artefacts.
4. **Discrete Wavelet Transform (DWT)** — to extract high-frequency features.

Reported: +4.5% accuracy and +2.9% average precision over existing methods across 26 generative models. Commonly described as using a lightweight **ResNet-50** backbone — the paper's selling point being that a simple CNN with the right preprocessing beats far heavier CLIP-based detectors. (Backbone/venue detail was not confirmable in the abstract excerpt retrieved; verify against the paper directly.)

### C.4 CIFAKE baseline CNN

Bird & Lotfi, 2023 ([IEEE 10409290](https://ieeexplore.ieee.org/abstract/document/10409290)). Optimal architecture reported: **two convolutional layers with 32 filters each + two fully connected layers**, achieving **92.93% accuracy**, BCE loss 0.18, on 32×32 images. Emphasis on explainability via Gradient Class Activation Mapping.

### C.5 High-resolution architecture (HiDA-Net)

**[No Pixel Left Behind: A Detail-Preserving Architecture for Robust High-Resolution AI-Generated Image Detection](https://arxiv.org/html/2508.17346v1)** — Mu et al., Aug 2025.

Attributes performance collapse to (1) **Input Degradation** — an architectural bottleneck from resize/center-crop, and (2) **Limited Generalization** — shortcut learning, *"exacerbated by mismatched JPEG compression histories between real and fake images, which teaches the model to become a compression detector rather than a synthesis detector."*

Three modules:
- **FAM (Feature Aggregation Module)** — fuses features from multiple **full-resolution local tiles** with a down-sampled global view; the entire image is processed as full-coverage native-resolution tiles.
- **TFL (Token-wise Forgery Localization)** — fine-grained spatial awareness, robust to localized forgeries like inpainting.
- **QFE (JPEG Quality Factor Estimation)** — explicitly **disentangles generative artefacts from compression noise**.

New benchmark **HiRes-50K**: 50,568 images from Freepik, LiblibAI, Civitai (AIGI) and Unsplash (real), long-edge resolutions from <1K to >10K pixels, some up to 64 MP. Reported: **+13% accuracy on Chameleon, +10% on HiRes-50K**.

### C.6 Forensic front-end filters (feature engineering)

- **SRM (Spatial Rich Model) filters** — fixed high-pass kernels from steganalysis; suppress content, highlight forensic noise. Of ~30 original filters, most works use ~3; outputs truncated and combined into a noise descriptor. Used inside **AIDE**.
- **Bayar & Stamm constrained convolution** — learnable prediction-error kernel with the constraint `w_k(0,0) = −1` and remaining weights summing to 1, re-applied after each update. Argument for it over SRM: static high-pass filters are vulnerable to attack; the constrained layer *adaptively learns manipulation traces from data*.
- **Complementarity finding:** filters are not redundant. MMFusion ([arXiv:2312.01790](https://arxiv.org/html/2312.01790)) uses **NoisePrint++, SRM and Bayar convolution as auxiliary inputs to RGB** — SRM mostly extracts edge features, NoisePrint++ is self-supervised camera-fingerprint, Bayar is supervised.
- **Structural critique** noted in recent work: high-frequency cues are suppressed/isolated/aggregated but never *coupled to semantic content* during training, so natural low–high frequency co-modulation is not explicitly modelled; two-stream designs fuse late without enforcing statistical relation.
- **DWT** — used by SAFE (feature extraction) and by *Seeing What Matters* (augmentation).

### C.7 AIDE — hybrid low-level + semantic

**[A Sanity Check for AI-generated Image Detection](https://arxiv.org/abs/2406.19435)** — Yan et al., ICLR 2025 ([code](https://github.com/shilinyan99/AIDE)).

AIDE = **AI-generated Image DEtector with hybrid features**:
- **DCT scoring module** selects **two high-frequency and two low-frequency patches** per image
- **SRM filters** extract high-frequency residual responses from those four patches; residual cues averaged
- Fused in parallel with **global semantic features** from a pretrained OpenCLIP encoder on the full image
- Features from various levels fused in the channel dimension

Reported: +3.5% accuracy on AIGCDetectBenchmark, +4.6% on GenImage. Also introduces the **Chameleon** benchmark (images from online sites, adjusted by photographers and AI artists, 720P–4K). **Finding: evaluating 9 off-the-shelf detectors on Chameleon, almost all misclassify AI-generated images as real** — and AIDE itself still drops significantly.

---

## Part D — Robustness engineering: what the papers report

### D.1 The robustness-benchmark landscape

| Benchmark | Protocol | Source |
|---|---|---|
| **GenImage** degraded-image task | Low-resolution, blurred, compressed | [arXiv:2306.08571](https://arxiv.org/abs/2306.08571), NeurIPS 2023 D&B |
| **Towards Detection of AI-Synthesized Human Face Images** | JPEG QF {10,20,…,90}; Gaussian blur kernel {3,5,…,15}; also Gaussian noise + resize; on ProGAN and DDIM test sets | [arXiv:2402.08750](https://arxiv.org/html/2402.08750v1) |
| **AIGIBench** | 4 tasks incl. degradation robustness, augmentation sensitivity, test-time preprocessing | [arXiv:2505.12335](https://arxiv.org/abs/2505.12335), NeurIPS 2025 |
| **RealDeg-Bench (GlobalForge)** | 7 operators × compound chains N∈{1..5}; 13 conditions; 95,589 images | [arXiv:2607.14684](https://arxiv.org/html/2607.14684v1) |
| **NTIRE 2026** | 36 transformations, 1–5 chained per image, multiple magnitude levels | [arXiv:2604.11487](https://arxiv.org/html/2604.11487v1) |
| **AIGI-Blur** | Curated AI-generated + real motion-blurred images | [arXiv:2511.12511](https://arxiv.org/pdf/2511.12511) |
| **BIAS-ID** | Framework for analysing transformation biases in AIGI detectors | [arXiv:2605.31153](https://arxiv.org/pdf/2605.31153) |

**Stated methodological critique** (GlobalForge): existing robustness protocols typically apply **one perturbation at a time** at a fixed strength, making it impossible to characterise decay along realistic degradation chains; meanwhile in-the-wild benchmarks have uncontrollable degradation type, strength and depth. *"Detectors near saturation on clean benchmarks routinely collapse after real propagation chains."*

Also noted (re: ImageNet-C as a template): many of its corruptions are no longer OOD for web-scale models, since JPEG/blur/noise already appear in pretraining data, causing near-saturation scores that mask true weaknesses.

### D.2 GlobalForge — reported numbers under the Track-5-style transform set

**[GlobalForge: Towards Robust AI-Generated Image Detection](https://arxiv.org/html/2607.14684v1)** — Cui, R. Liu, Zou, Qin, Xu, Z. Wang, Wei, Zhou, Y. Liu, Y. Wang, Wu (HUST / CASIA / Jilin / Tsinghua), Jul 2026.

Core argument: existing methods rely on **local generator artefacts easily destroyed by JPEG compression and blur**; the proposal is to shift focus to **robust global structural patterns**.

Three components:
1. **Local Information Bottleneck (LIB)** — learnable Gaussian smoothing in the *feature* domain to suppress high-frequency artefacts
2. **Global Structural Reasoning (GSR)** — masks local attention within 3×3 windows, forcing tokens to aggregate distant evidence
3. **Degradation-aware Contrastive Structural loss** — aligns clean and degraded representations

**RealDeg-Bench**: 7 operators — JPEG compression, Gaussian blur, resize, Gaussian noise, brightness, contrast, saturation — matching the Track 5 transform list almost exactly; plus compound chains N∈{1,2,3,4,5}; 13 conditions total, 95,589 images. 12 baselines evaluated: NPR, UnivFD, C2P-CLIP, FatFormer, SAFE, AIDE, Effort, DRCT, Aligned, B-Free, DDA, GAPL.

| Condition | GlobalForge BAcc |
|---|---:|
| Clean | 87.77% |
| JPEG | 87.79% |
| Resize | 89.63% |
| Gaussian blur | 84.65% |
| Gaussian noise | 82.06% |
| Average, single operators | 87.35% |
| **5-step compound chains** | **79.53%** |
| In-the-wild (8 groups) | 85.93% avg BAcc, +5.89% over prior SOTA |

Comparison datapoint: the **DDA baseline drops 88.90% (clean) → 70.30% (compound chains)**, while GlobalForge holds 79.53%.

### D.3 Preprocessing: crop vs resize

This is one of the most consistently reported robustness levers.

**TextureCrop** ([arXiv:2407.15500](https://arxiv.org/abs/2407.15500), Konstantinidou, Koutlis, Papadopoulos, WACVW 2025 pp. 1459–1468; [code](https://github.com/mever-team/texture-crop)). A plug-in preprocessing component: **sliding-window analysis, cropping texture-rich regions, filtering out low-texture-variability areas.** Reported improvement in AUC across various detectors: **+6.1% vs center cropping and +15% vs resizing**, on high-resolution images from ForenSynths, Synthbuster and TWIGMA. (The earlier preprint reported +5.7% vs center crop; revised upward in the WACVW version.)

**From *What Truly Matters?*** — cropping method comparison (avg AUC):

| Method | Center cropping | Texture cropping | Δ |
|---|---:|---:|---:|
| DMID | 78.78 | **89.46** | +10.68 |
| RINE | 91.26 | **94.90** | +3.64 |
| NPR | **70.11** | 67.48 | −2.63 |

The authors state resizing was abandoned because it *"erases subtle high-frequency traces left by the generation process."*

**General forensic recipe** as summarised in the TTA literature: train on randomly cropped patches, decide on the whole image via a fusion strategy, and avoid down-sampling in early network layers. Rationale: resizing suppresses artefact information; CenterCrop only retains artefacts at the centre, overlooking traces away from centre.

**Counter-datapoint:** the NTIRE 5th-place team deliberately avoided random resized cropping (arguing it may remove localised forensic cues) in favour of a **"squish"** strategy — direct resize to 384×384 ignoring aspect ratio — and still reached 0.8730 robust AUC with a single model.

### D.4 Augmentation recipes reported to help

| Source | Recipe |
|---|---|
| **Wang et al. CVPR 2020** | Gaussian blur p=0.5, **then independently** JPEG p=0.5 (SciPy Gaussian filtering) |
| **Cozzolino CVPRW 2024** | With augmentation → "basically insensitive to compression (JPEG or WebP) and resizing" |
| ***What Truly Matters?*** | JPEG and **WEBP** compression; random cropping, rotation, horizontal flipping; Gaussian noise, blur, **sharpening** |
| **SAFE (KDD 2025)** | RandomCrop (train) / CenterCrop (infer); ColorJitter; RandomRotation; patch-based random masking; DWT |
| **B-Free (CVPR 2025)** | Content augmentation via **inpainting** for semantic alignment |
| **Seeing What Matters (NeurIPS 2025)** | **Wavelet-decomposition augmentation** — replacing specific frequency-related bands |
| **NTIRE 1st (MICV)** | Hierarchical stochastic pipeline by difficulty level: simple (blur, noise, shifts) → complex multi-stage |
| **NTIRE 2nd (Ant)** | 4-level offline: clean / 1–3 distortions (μ=0, σ=2.5) / 3–6 (μ=2.5, σ=2.0) / fixed 6 (μ=3.5, σ=1.0); online flip + AugMix m6-w3-d1 |
| **NTIRE 5th (vincentlc)** | `distortion_prob=1.0`, up to 3 ops, 5 severity levels — every training image distorted |
| **Aligned Datasets** | Random JPEG, blur, grayscale, cutout, noise, random resized crop; "Ours-Sync" variant pairs real/fake with **identical** augmentations |

**Quantified augmentation gain** from *What Truly Matters?* (avg AUC):

| Method | Without aug | With aug | Gain |
|---|---:|---:|---:|
| DMID | 78.31 | 89.46 | **+11.15** |
| SPAI | 89.66 | 95.25 | +5.59 |
| RINE | 93.16 | 94.90 | +1.74 |
| NPR | 67.36 | 67.48 | +0.12 |
| **Average** | | | **+4.65** |

**Counter-finding:** AIGIBench reports "limited benefits from common augmentations" across the 11 detectors it evaluated, and *"nuanced effects of pre-processing."* Wang et al. also note augmentation *hurt* on SAN.

### D.5 Test-time augmentation and patch aggregation

Aggregation rules documented in the literature:

- **Majority voting vs logit averaging** — one study found both improve over single-patch inference, with **majority voting consistently outperforming**; high-performing datasets improved further, poorly-performing ones saw minimal gain, near-chance ones slightly degraded.
- **Any-patch (OR) rule** — in an orthogonal-CNN ensemble ([arXiv:2203.02246](https://arxiv.org/pdf/2203.02246)): if all patches are real → real; if **at least one** patch is synthetic → synthetic. Chosen because missed detection on synthetic images was deemed the most critical parameter.
- **Selective patches** — uniform aggregation can dilute evidence because forgery traces are minor and spatially non-uniform; AIDE's DCT-based selection of 2 high-frequency + 2 low-frequency patches is the canonical instance.
- **Token pooling** — NTIRE 2026 team evaluation: global average pooling over all final-layer patch tokens beat CLS-token and attention pooling for robustness/stability.
- **top-k median** — DRIFT aggregates patch-wise drift scores this way.
- **Horizontal-flip TTA** — NTIRE 4th place applied it to 2 of 5 models.
- **RRC-simulating TTA** — WaRPAD deterministically simulates multiple random-resized-crop instances and averages scores across patches.

### D.6 Adversarial / attack-side robustness

- **[Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks](https://openreview.net/pdf?id=dLoAdIKENc)** — Saberi, Sadasivan, Rezaei, Kumar, Chegini, W. Wang, Feizi, ICLR 2024 ([arXiv:2310.00076](https://arxiv.org/abs/2310.00076), [code](https://github.com/mehrdadsaberi/watermark_robustness)). Establishes a **fundamental trade-off between evasion error rate and spoofing error rate** for low-perturbation watermarking under diffusion purification; model-substitution attack for high-perturbation watermarks; spoofing attacks that make real images be flagged as watermarked with only black-box access. Theory extended to characterise trade-offs for **classifier-based detectors**.
- **[Backbone is All You Need: Assessing Vulnerabilities of Frozen Foundation Models in Synthetic Image Forensics](https://arxiv.org/pdf/2605.13381)** — Musso, Battocchio, Montibeller, Boato, May 2026. Introduces **SIAA (Surrogate Iterative Adversarial Attack)**, a **gray-box** attack exploiting knowledge of the detector backbone. Finding: *"backbone knowledge alone is sufficient to undermine detector reliability"* — high attack success rates approaching white-box performance across few-shot and complete training-misalignment scenarios. Directly relevant given the convergence of the field on a small set of public backbones.
- **Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection** — De Rosa et al., [arXiv:2407.19553](https://arxiv.org/pdf/2407.19553), WIFS 2024.
- **Adversarial Robustness of AI-Generated Image Detectors in the Real World** — [arXiv:2410.01574](https://arxiv.org/pdf/2410.01574). Notes attacks remain effective even when images are degraded through their lifecycle (e.g. social-media upload post-processing).

---

## Part E — The dataset-bias trap (JPEG / resolution shortcuts)

This section is flagged because Track 5 explicitly asks teams to **create their own transformed test cases** — the literature documents specific ways this goes wrong.

### E.1 Fake or JPEG?

**[Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/abs/2403.17608)** — Grommelt, Weiss, Pfreundt, Keuper, Mar 2024 (also [Springer 10.1007/978-3-031-92089-9_6](https://link.springer.com/chapter/10.1007/978-3-031-92089-9_6)).

**The mechanism:** most ImageNet real images are JPEG-encoded (modal quality ≈96, range 70–100), while generated fakes are typically stored losslessly as PNG. A detector can trivially exploit the compression difference as a shortcut. A parallel **size bias** exists: generator outputs have fixed model-specific sizes while real ImageNet images have a broad multimodal size distribution.

**Evidence the shortcut is real:** a ResNet-50 trained on raw GenImage shows strong accuracy decline even at high quality factors like QF=95. Precision on AI-generated images stayed near 1 while **recall dropped sharply** — compressing a generated image considerably increases the chance the model calls it natural. The authors controlled for the "compression just destroys artefacts" objection using uncompressed natural PNGs from FFHQ.

**Bias direction is symmetric and configuration-dependent:** stronger compression shifts predictions toward *real* in one configuration and toward *fake* in the reverse configuration; on uncompressed images predictions shift the opposite way — the detector treats the *absence* of JPEG artefacts as evidence of fakeness or realness depending on the training setup.

**Bias-controlled protocol:** recompress **both** real and synthetic images to a uniform **JPEG Q = 96**; control real-image selection so native size distributions align with the generative split (both **center-cropped to 450×450 before resizing**); enforce symmetric preprocessing operations (resize, crop, resize) for both classes.

**Result: >11 percentage-point increase in cross-generator performance** for ResNet-50 and Swin-T detectors on GenImage.

### E.2 Aligned datasets

**[Aligned Datasets Improve Detection of Latent Diffusion-Generated Images](https://arxiv.org/html/2410.11835v3)** — Sundara Rajan, Ojha, Schloesser, Y. J. Lee (UW-Madison), Feb 2025.

Idea: instead of generating fakes by iterative denoising, **reconstruct real images using only the LDM's VAE encoder-decoder, skipping the U-Net**: `𝒱 = {φ_dec(φ_enc(x)) | ∀x ∈ ℛ}`. The "fake" preserves resolution, aspect ratio and semantic content, differing **almost exclusively in decoder artefacts** — and costs ~10× less compute.

Concrete spurious feature they cite: existing methods *"mistakenly learn that downsampling correlates with real images due to resolution mismatches in training data."*

| Setting | Value |
|---|---|
| Architecture | ResNet-50 (ImageNet-pretrained), binary head |
| Loss / optimiser | BCE (real=0, fake=1) / Adam lr 1e-4 |
| Augmentations | Random JPEG, blur, grayscale, cutout, noise, random resized crop |
| Training data | 179,257 images from MS COCO and LSUN |
| Variants | Standard batching vs **Ours-Sync** (paired real-fake, *identical* augmentations) |

Results: Stable Diffusion 99.31–99.57% acc; Midjourney 98.50–99.37%; Playground 94.85–99.48% (+12.72/+17.35 over Corvi). **Data efficiency: with only 1,000 images, 83.37% TPR@5%FPR vs Corvi's 46.51%.** Resizing robustness: gradual decline under extreme downsampling vs Corvi's drastic drops. Detects Kandinsky (different VAE) at 99.57–99.92%.

Stated limitations: fails on major VAE architecture changes — **FLUX.1-dev (16 latent channels) only 25.87% accuracy**; vulnerable to format-specific artefacts like `.webp` compression absent from reconstructed training images. Surprising ancillary finding: training on **algorithmically-generated OpenGL shader images** achieved competitive performance, indicating dataset *alignment* matters more than training-image semantics.

### E.3 B-Free

**[A Bias-Free Training Paradigm for More General AI-generated Image Detection](https://arxiv.org/abs/2412.17671)** — Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva, CVPR 2025 pp. 18685–18694 ([project](https://grip-unina.github.io/B-Free/), [code](https://github.com/grip-unina/B-Free)).

**Self-conditioned reconstruction:** fake images are generated *from* real ones using the conditioning procedure of stable diffusion, ensuring semantic alignment so any difference stems solely from AI-generation artefacts. Paired with **content augmentation via inpainting**. Together these aim to eliminate format, content and resolution biases.

Reported: significant improvements in generalisation *and* robustness over SOTA, plus **more calibrated results across 27 generative models** including FLUX and SD 3.5. One summary reports generalisation AUC >99% and balanced accuracy 95.2%. Noted gap: video generation models (e.g. Sora) not evaluated.

### E.4 Other bias channels documented

- **WEBP inside PNG:** LSUN images were WEBP-compressed but saved as lossless PNG; fake training images lacked those artefacts, so detectors associated WEBP compression with the real distribution. Excluding WEBP reals during training restored generalisation.
- **Neural codecs:** *Three Forensic Cues for JPEG AI Images* ([arXiv:2504.03191](https://arxiv.org/html/2504.03191v1)) — **real images compressed with JPEG AI are classified as fake** by synthetic-image detectors, because JPEG AI and synthetic images leave similar frequency-domain artefacts; high false-positive rates persist even after retraining on JPEG AI images.
- **Naive symmetric recompression trap:** if a model has learned that *absence* of WEBP artefacts is necessary for fakeness, adding WEBP to fakes weakens the fake signal.
- **Historical precedent:** Cattaneo & Roscigno showed a widely used tampered-image dataset had a JPEG bias where untampered images were saved at different quality factors than tampered ones, with significant performance drops once removed.
- **Reconstruction-alignment residual leak:** aligning via the LDM autoencoder assumes most real-image properties transfer to the reconstruction, but some low-level properties do not.

**Practical checklist assembled from these sources:**
1. Audit quantization tables / quality factors per class before training — a Q-factor histogram split by label is the cheapest bias detector available.
2. Resave everything through one encoder at a fixed Q (Q=96 is the GenImage bias-controlled convention), including already-JPEG images — double compression applied to both classes is safer than single-vs-none.
3. Align resolution and crop/resize order — size is an independent leakage channel.
4. Watch for non-JPEG codecs (WEBP, H.264, JPEG AI) hiding inside PNG-wrapped datasets.
5. Sanity-check with a compression sweep: if AUROC rises or collapses monotonically with quality factor, you are measuring the codec, not the generator.

**Note on Track 5's provided validation set:** the AIGC half is DALL·E Advanced (8,843 images) and the non-AIGC half is COCO val2017 (4,998 images). COCO val2017 is distributed as JPEG; DALL·E outputs are typically PNG. The bias literature above applies directly to any self-created transformed test set built on these two sources.

---

## Part F — Datasets

### F.1 The three datasets named in the brief

| Dataset | Scale | Format / access | Notes |
|---|---|---|---|
| **[saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)** | 240K rows on the Hub (210K train / 30K validation); paper describes 300K total | Parquet, **123.2 GB train + 16.8 GB validation** (283 files) | Schema: `img_id, image, mask, width, height, label`. From **SIDA** (Huang et al., CVPR 2025, [arXiv:2412.04292](https://arxiv.org/html/2412.04292), [code](https://github.com/hzlsaber/SIDA)). Composition: 100K real from OpenImages V7, 100K synthetic from **FLUX**, 100K tampered (objects/regions replaced). GPT-4o generated textual judgement descriptions for 3,000 images. Only a single `test.zip` is provided for test, deliberately, to limit contamination. cc-by-4.0. |
| **[CIFAKE (Kaggle)](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)** | 120,000 images (60K real CIFAR-10 + 60K synthetic) | **32×32 px** | Bird & Lotfi 2023. Synthetic from **Stable Diffusion 1.4**, prompts "A photograph of [object]" plus context modifiers, mirroring the 10 CIFAR-10 classes. MIT licence. Caveats documented: one study found **668 duplicate image pairs**; domain shifts exist (e.g. "ship" generated as an interior scene). Extended variants CIFAKE-SD2.1 and CIFAKE-SD3.0 exist. |
| **[WildFake (ModelScope)](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)** | **>3.7M images** | ModelScope; [GitHub index](https://github.com/hy-zpg/AIGC-Image-Detection-Dataset) | Hong, Feng, H. Chen, Lan, Zhu, W. Wang, J. Zhang — **AAAI 2025**, 39(4):3500–3508 ([arXiv:2402.11843](https://arxiv.org/abs/2402.11843)). **Four-level hierarchy:** (1) cross-generator — DMs / GANs / Others; (2) cross-architecture — DALLE, ADM, Imagen, DDPM, DDIM, VQDM, Midjourney, SD; (3) cross-weight — SD fakes split into three subsets; (4) cross-version — typical vs advanced classes. Fakes from open-source communities plus the authors' own pipeline; reals from existing captioning/classification datasets. Designed for train-on-one-subset / test-on-another evaluation. |

**Validation set specified by the brief (demonstration only, not for training):** COCO val2017 (4,998 non-AIGC) + DALL·E Advanced (8,843 AIGC) — a WildFake subset.

### F.2 Additional public datasets referenced in the literature

| Dataset | Scale | Source |
|---|---|---|
| **[OwensLab/CommunityForensics](https://huggingface.co/datasets/OwensLab/CommunityForensics)** | **2.7M images from 4,803 generators**; Hub splits: Systematic 1.9M (780 GB), Manual 774K (257 GB), PublicEval 51.8K (27.7 GB), Commercial 14.9K (16.6 GB) | Park & Owens, **CVPR 2025** pp. 8245–8257 ([arXiv:2411.04125](https://arxiv.org/html/2411.04125v2), [code](https://github.com/JeongsooP/Community-Forensics)). **Community Forensics-Small** ≈11% of base, paired with redistributable-licence real data (278 GB); **Community Forensics-Eval** is the recommended comprehensive eval set (206 GB); Full 1.1 TB. Finding: detection performance improves as the *number* of models in the training set increases, even when architectures are similar; increasing model diversity also improves performance. cc-by-4.0. **Best out-of-the-box mean accuracy (75.0%) in the 2602.07814 benchmark.** |
| **GenImage** | >1M ⟨fake, real⟩ pairs, ImageNet's 1000 classes | Zhu et al., **NeurIPS 2023 D&B** ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571), [code](https://github.com/GenImage-Dataset/GenImage)). Generators: Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM, BigGAN. Two tasks: cross-generator classification and **degraded image classification**. |
| **Chameleon** | 720P–4K, online-sourced, adjusted by photographers and AI artists | AIDE / ICLR 2025 ([arXiv:2406.19435](https://arxiv.org/abs/2406.19435)) |
| **AIGIBench** | 23 fake subsets; ~288K training images | NeurIPS 2025 ([arXiv:2505.12335](https://arxiv.org/abs/2505.12335)) |
| **So-Fake-Set / So-Fake-OOD** | So-Fake-Set >2M images from **35 generative models**; So-Fake-OOD 100K from real Reddit content | [arXiv:2505.18660](https://arxiv.org/abs/2505.18660), [code](https://github.com/hzlsaber/So-Fake). So-Fake-OOD released 2025-05-23; So-Fake-Set 2025-10-29. |
| **ITW-SM** | 10,000 images, 50/50 real/AI, 0.1 MP to 8.4K, from Facebook, Instagram, LinkedIn, X | *What Truly Matters?* ([arXiv:2507.10236](https://arxiv.org/html/2507.10236v1)). Real from verified trusted accounts; synthetic from public AI-content accounts. Filtered for text overlays, watermarks, duplicates; labels manually verified. |
| **HiRes-50K** | 50,568 images, <1K to >10K long edge, up to 64 MP | HiDA-Net ([arXiv:2508.17346](https://arxiv.org/html/2508.17346v1)). Freepik, LiblibAI, Civitai (AIGI) + Unsplash (real). |
| **[elsaEU/ELSA_D3](https://huggingface.co/datasets/elsaEU/ELSA_D3)** | 2.3M train rows (2,626 GB), 4.8K validation | EU ELSA project, [benchmarks.elsa-ai.eu](https://benchmarks.elsa-ai.eu/). 4 generated variants per prompt with model/size/step metadata. |
| **AI-GenBench** | 36 generators, temporal/incremental protocol | [arXiv:2504.20865](https://arxiv.org/pdf/2504.20865), Verimedia workshop @ IJCNN 2025 ([site](https://mi-biolab.github.io/aigenbench-website/), [code](https://github.com/MI-BioLab/AI-GenBench)) |
| **NTIRE 2026 set** | 294,500 images, 42 generators, 36 transformations | [Codabench #12761](https://www.codabench.org/competitions/12761/) |
| Others cited | Synthbuster, TWIGMA, ForenSynths, DRCT-2M, ArtiFact, Fake2M, RealWorldBench, OpenFake, RRDataset, TrueFake, WildRF, SocialRF, CommunityAI, AIGI-Blur, HPAI-BSC/SuSy-Dataset | |

---

## Part G — Evaluation protocols and metrics used in the literature

**Metrics observed across papers:**

- **ROC AUC** — NTIRE 2026 primary metric, computed over the full test set *including both transformed and untransformed images*; teams additionally scored on "Robust ROC AUC"
- **Balanced Accuracy (BAcc)** — GlobalForge / RealDeg-Bench
- **mAP / AP** — UnivFD lineage, Effort, DRIFT
- **Accuracy (mAcc)** — most cross-generator tables
- **Decomposed R.Acc / F.Acc** — AIGIBench separates Real Image Accuracy from Fake Image Accuracy, which exposed the "F.Acc → 0%" failure mode
- **TPR@5%FPR** — Aligned Datasets; also the text-detection convention of fixing an FPR budget
- **TNR on real images** — SSAFE reports this separately (98.3% for curated 10K)
- **Localization IoU** — So-Fake-R1, LEGION, SIDA

**Threshold-setting methods documented:**

- Fixed FPR budget (e.g. calibrate so FPR stays at 5%), noting naive thresholding produces unacceptably high false-positive rates
- **Youden index** on a calibration set for balanced TPR/FPR trade-off
- Calibration using **regenerated real images** passed through a known generator, so the threshold is set without access to unseen test data
- Learnable scalar logit shift with 10–100 samples ([B.10](#b10-post-hoc-calibration))

**Noted caveat:** benchmarks often report AUROC/AP precisely *because* accuracy varies with the thresholding protocol — which is convenient for papers but unhelpful for an analyst who must pick one operating point. Threshold sweeps show **most models below 30% TPR at 5% FPR**.

**Reported false-positive evidence:** a 2026 NewsGuard audit ran 15 authentic news photographs through 5 leading detectors; **3 of 5 misclassified real images, worst tool flagging 6 of 15 (40%) as AI-generated.** Disproportionately flagged categories: studio portraits with smooth skin and controlled lighting; heavily processed landscapes. Research separately notes real images that are easily misclassified tend to be simpler or visually incoherent — minimalistic scenes, unusual facial features.

---

## Part H — Off-the-shelf checkpoints on Hugging Face

All are well under the 2B limit and are used across the community as baselines or Space backends.

| Model | Params | Arch | Downloads | Licence |
|---|---:|---|---:|---|
| [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) | 86.8M | Swin | 877.4K | **cc-by-nc-3.0** |
| [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) | ~86M (Swin) | Swin | 826.3K | cc-by-4.0 |
| [`Ateeqq/ai-vs-human-image-detector`](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) | 92.9M | **SigLIP** | 340.2K | apache-2.0 |
| [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | 85.8M | ViT (from `vit-base-patch16-224-in21k`) | 283.3K | apache-2.0 |
| [`haywoodsloan/ai-image-detector-deploy`](https://huggingface.co/haywoodsloan/ai-image-detector-deploy) | 195.2M | **SwinV2** | 248.7K | apache-2.0 |
| [`NYUAD-ComNets/NYUAD_AI-generated_images_detector`](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector) | 85.8M | ViT | 58.4K | apache-2.0 |
| [`HPAI-BSC/SuSy`](https://huggingface.co/HPAI-BSC/SuSy) | — | CNN + patch | 1.0K | apache-2.0 |

**SuSy** is documented in *Present and Future Generalization of Synthetic Image Detectors* ([arXiv:2409.14128](https://arxiv.org/pdf/2409.14128)); trained on COCO, dalle-3-images, diffusiondb, midjourney-images, duchaiten-realistic-sdxl.

**Research code repositories with released weights:**

| Method | Repository |
|---|---|
| UnivFD | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| CNNDetection | [peterwang512/CNNDetection](https://github.com/peterwang512/CNNDetection) |
| NPR | [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) |
| RINE | [mever-team/rine](https://github.com/mever-team/rine) |
| SPAI | [mever-team/spai](https://github.com/mever-team/spai) |
| TextureCrop | [mever-team/texture-crop](https://github.com/mever-team/texture-crop) |
| Effort (ICML'25 Oral) | [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) |
| FatFormer | [Michel-liu/FatFormer](https://github.com/Michel-liu/FatFormer) |
| DeeCLIP | [Mamadou-Keita/DeeCLIP](https://github.com/Mamadou-Keita/DeeCLIP) |
| Bi-LORA | [Mamadou-Keita/VLM-DETECT](https://github.com/Mamadou-Keita/VLM-DETECT) |
| B-Free | [grip-unina/B-Free](https://github.com/grip-unina/B-Free) |
| CLIP-based SID (Cozzolino) | [grip-unina/ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) |
| AIDE | [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) |
| Community Forensics | [JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics) |
| So-Fake | [hzlsaber/So-Fake](https://github.com/hzlsaber/So-Fake) |
| SIDA / SID_Set | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) |
| LEGION (ICCV'25 Highlight) | [opendatalab/LEGION](https://github.com/opendatalab/LEGION) |
| Curated survey list | [ant-research/Awesome-AIGC-Image-Video-Detection](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection) · [Awesome-AIGCDetection](https://fdmas.github.io/AIGCDetect/Awesome-AIGCDetection) |

---

## Part I — Reported compute figures

| Source | Setup | Reported cost |
|---|---|---|
| **RINE** | CLIP-L + intermediate-block aggregation | *"best performing models require just a single epoch for training (~8 minutes)"* |
| **Effort** | CLIP ViT-L/14, 224px, batch 32–48 | 0.19M trainable params; described as "very little training cost" |
| **Simplicity Prevails** | Linear probe on frozen VFM | AdamW lr 1e-3, batch 128, **2 epochs**, GenImage SD1.4 subset only |
| **SSAFE** | Frozen PE-Core-G + linear head | AdamW lr 1e-3, batch 40, **10K training images** |
| **DRIFT** | Frozen DINOv2 ViT-B/14 + 2 small MLPs | AdamW lr 3e-4, batch 64, 50 epochs |
| **SPAI** | Full method | **Inference under 8 GB GPU RAM**; reproducing training requires ~48 GB (L40S-class) |
| **NTIRE 6th (UESTC)** | 4-expert CLIP-L + SigLIP-So400M ensemble | **~10 GB peak GPU memory** |
| **NTIRE 5th (vincentlc)** | SigLIP2-Giant + single linear layer | Hardware not specified; single model, no TTA |
| **NTIRE 4th (INTSIG)** | 5-model ensemble | 8× H800, DDP, batch 16 |
| **NTIRE 3rd (TeleAI)** | EVA-CLIP + LoRA | 8× A800, 5 epochs |
| **NTIRE 2nd (Ant)** | DINOv3-7B ×2 | B200 training; A100 inference **2.21 img/s, 78.25 GB VRAM** |
| **NTIRE 1st (MICV)** | DINOv3 ensemble, 512×512 | **32× A100, 10 epochs, ~8 hours** |
| **Aligned Datasets** | ResNet-50 on VAE reconstructions | VAE-only reconstruction is **~10× cheaper** than full denoising generation |

---

## Part J — Explainability / error-analysis material

The brief asks for an error-analysis note and mentions explainability ideas as in scope. Relevant published material:

- **LEGION** ([arXiv:2503.15264](https://arxiv.org/pdf/2503.15264), ICCV 2025 Highlight, [code](https://github.com/opendatalab/LEGION)) — artefact localization + explanation generation + forgery detection. Dataset **SynthScars**: 12,236 fully synthetic images with expert annotations, pixel-level segmentation masks, textual explanations, artefact category labels, irregular-polygon masks; 4 content types, 3 artefact categories. Compared against 19 existing methods. Positioned both as "Defender" and as "Controller" guiding higher-quality generation.
- **SIDA** ([arXiv:2412.04292](https://arxiv.org/html/2412.04292), CVPR 2025) — the source of SID_Set; predicts authenticity, tampered-region mask, and textual explanation of judgement criteria.
- **AIDE / Chameleon** — heatmap and patch-level evidence via DCT+SRM patch selection.
- **NPR** — heatmaps showing artefacts concentrated on image details (hair, eyes, beard).
- **CIFAKE** — Gradient Class Activation Mapping for explainability, motivated by humans being unable to distinguish the classes.
- **C2P-CLIP** — decodes detection features into text plus word-frequency analysis as an interpretability probe; conclusion that CLIP recognises *concepts*, not real/fake semantics.
- **Effort** — contributes a tool for quantifying degree of model overfitting.
- **Documented false-positive categories** — studio portraits with smooth skin and controlled lighting; heavily processed landscapes; minimalistic or visually incoherent real scenes; real images compressed with JPEG AI.
- **Documented false-negative categories** — modern commercial generators (Flux Dev, Firefly v4, Midjourney v7) at 18–30% detection; Chameleon-style images curated by photographers/AI artists; images after 5-step compound degradation chains; VAE-reconstructed and locally-edited images (explicit blind spot noted in *Simplicity Prevails*).

---

## Consolidated reference list

### Challenge / benchmark reports
- NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild — [arXiv:2604.11487](https://arxiv.org/html/2604.11487v1) · [CVF](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Gushchin_NTIRE_2026_Challenge_on_Robust_AI-Generated_Image_Detection_in_the_CVPRW_2026_paper.pdf) · [Codabench](https://www.codabench.org/competitions/12761/)
- Is AI Generated Image Detection a Solved Problem? (AIGIBench) — [arXiv:2505.12335](https://arxiv.org/abs/2505.12335), NeurIPS 2025
- How well are open sourced AIGI detection models out-of-the-box — [arXiv:2602.07814](https://arxiv.org/html/2602.07814v1)
- GlobalForge / RealDeg-Bench — [arXiv:2607.14684](https://arxiv.org/html/2607.14684v1)
- GenImage — [arXiv:2306.08571](https://arxiv.org/abs/2306.08571), NeurIPS 2023
- AI-GenBench — [arXiv:2504.20865](https://arxiv.org/pdf/2504.20865)
- Towards the Detection of AI-Synthesized Human Face Images — [arXiv:2402.08750](https://arxiv.org/html/2402.08750v1)
- BIAS-ID — [arXiv:2605.31153](https://arxiv.org/pdf/2605.31153)

### Surveys
- Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review — [arXiv:2502.15176](https://arxiv.org/html/2502.15176v2)
- Survey on AI-Generated Media Detection: From Non-MLLM to MLLM — [arXiv:2502.05240](https://arxiv.org/pdf/2502.05240)
- Synthetic Image Verification in the Era of Generative AI — [arXiv:2405.00196](https://arxiv.org/pdf/2405.00196)
- Present and Future Generalization of Synthetic Image Detectors — [arXiv:2409.14128](https://arxiv.org/pdf/2409.14128)

### Backbones
- DINOv3 — [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- SigLIP 2 — [arXiv:2502.14786](https://arxiv.org/pdf/2502.14786) · [HF blog](https://huggingface.co/blog/siglip2)
- Perception Encoder — [arXiv:2504.13181](https://arxiv.org/pdf/2504.13181)
- MetaCLIP 2 — [arXiv:2507.22062](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu)
- CLIP — [arXiv:2103.00020](https://huggingface.co/openai/clip-vit-large-patch14) · DINOv2 — [arXiv:2304.07193](https://huggingface.co/facebook/dinov2-large) · EVA-02 — [arXiv:2303.11331](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k)

### Detection methods
- CNNDetection — [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf) · reproduction [arXiv:2104.02984](https://arxiv.org/pdf/2104.02984)
- UnivFD — [arXiv:2302.10174](https://arxiv.org/abs/2302.10174), CVPR 2023
- NPR — [arXiv:2312.10461](https://arxiv.org/html/2312.10461v2), CVPR 2024
- FatFormer — [arXiv:2312.16649](https://arxiv.org/abs/2312.16649), CVPR 2024
- RINE — [arXiv:2402.19091](https://arxiv.org/pdf/2402.19091), ECCV 2024
- Raising the Bar with CLIP — [arXiv:2312.00195](https://arxiv.org/abs/2312.00195), CVPRW 2024
- C2P-CLIP — [arXiv:2408.09647](https://arxiv.org/abs/2408.09647), AAAI 2025
- SAFE — [arXiv:2408.06741](https://arxiv.org/abs/2408.06741), KDD 2025
- AIDE / Chameleon — [arXiv:2406.19435](https://arxiv.org/abs/2406.19435), ICLR 2025
- SPAI — [arXiv:2411.19417](https://arxiv.org/abs/2411.19417), CVPR 2025
- Effort — [arXiv:2411.15633](https://arxiv.org/html/2411.15633v3) · [PMLR 267:70268](https://proceedings.mlr.press/v267/yan25b.html), ICML 2025 Oral
- B-Free — [arXiv:2412.17671](https://arxiv.org/abs/2412.17671), CVPR 2025
- Community Forensics — [arXiv:2411.04125](https://arxiv.org/html/2411.04125v2), CVPR 2025
- DeeCLIP — [arXiv:2504.19876](https://arxiv.org/pdf/2504.19876)
- Bi-LORA — [arXiv:2404.01959](https://arxiv.org/abs/2404.01959), Expert Systems 2025
- HiDA-Net — [arXiv:2508.17346](https://arxiv.org/html/2508.17346v1)
- DINO-Detect — [arXiv:2511.12511](https://arxiv.org/pdf/2511.12511)
- WaRPAD — [arXiv:2511.14030](https://arxiv.org/pdf/2511.14030)
- Simplicity Prevails — [arXiv:2602.01738](https://arxiv.org/html/2602.01738)
- SSAFE — [arXiv:2606.08634](https://arxiv.org/html/2606.08634)
- TAP — [arXiv:2604.26772](https://arxiv.org/abs/2604.26772)
- DRIFT — [arXiv:2606.06918](https://arxiv.org/html/2606.06918v1)
- RA-Det — [arXiv:2603.01544](https://arxiv.org/pdf/2603.01544)
- Calibration — [arXiv:2602.01973](https://arxiv.org/html/2602.01973)
- Seeing What Matters (wavelet augmentation) — [arXiv:2506.16802](https://arxiv.org/abs/2506.16802), NeurIPS 2025
- What Truly Matters? — [arXiv:2507.10236](https://arxiv.org/html/2507.10236v1), ACM MAD 2025
- TextureCrop — [arXiv:2407.15500](https://arxiv.org/abs/2407.15500), WACVW 2025

### Bias, robustness limits and attacks
- Fake or JPEG? — [arXiv:2403.17608](https://arxiv.org/abs/2403.17608)
- Aligned Datasets — [arXiv:2410.11835](https://arxiv.org/html/2410.11835v3)
- Robustness of AI-Image Detectors (fundamental limits) — [arXiv:2310.00076](https://arxiv.org/abs/2310.00076), ICLR 2024
- Backbone is All You Need (SIAA gray-box attack) — [arXiv:2605.13381](https://arxiv.org/pdf/2605.13381)
- Adversarial Robustness of CLIP for AIGI Detection — [arXiv:2407.19553](https://arxiv.org/pdf/2407.19553), WIFS 2024
- Adversarial Robustness of AIGI Detectors in the Real World — [arXiv:2410.01574](https://arxiv.org/pdf/2410.01574)
- Three Forensic Cues for JPEG AI Images — [arXiv:2504.03191](https://arxiv.org/html/2504.03191v1)
- MMFusion (SRM / Bayar / NoisePrint++) — [arXiv:2312.01790](https://arxiv.org/html/2312.01790)

### Datasets
- SID_Set / SIDA — [arXiv:2412.04292](https://arxiv.org/html/2412.04292) · [HF](https://huggingface.co/datasets/saberzl/SID_Set)
- WildFake — [arXiv:2402.11843](https://arxiv.org/abs/2402.11843) · [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32363) · [ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
- CIFAKE — [IEEE 10409290](https://ieeexplore.ieee.org/abstract/document/10409290) · [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Community Forensics — [HF](https://huggingface.co/datasets/OwensLab/CommunityForensics)
- So-Fake — [arXiv:2505.18660](https://arxiv.org/abs/2505.18660)
- ELSA_D3 — [HF](https://huggingface.co/datasets/elsaEU/ELSA_D3)
- LEGION / SynthScars — [arXiv:2503.15264](https://arxiv.org/pdf/2503.15264), ICCV 2025

---

## Open items not resolved by this research pass

1. **NTIRE 2026 distortion pipeline source code** — the report describes the pipeline in prose but no public GitHub repo for the exact degradation script surfaced. The [Codabench competition page (#12761)](https://www.codabench.org/competitions/12761/) is the most likely host of a starter kit.
2. **SAFE backbone and venue confirmation** — the abstract retrieved confirms KDD 2025 and the four transformations, but the ResNet-50 backbone and DWT component were only confirmed via secondary survey descriptions.
3. **RA-Det exact robustness table values** under JPEG QF 95/90/85 and blur σ 0.8/1.0/1.5 vs NPR/FerretNet/UniFD — the PDF text layer did not extract cleanly.
4. **DINO-Detect exact DINOv3 variant** (S/B/L/H+) and full AIGI-Blur specifications — not extractable from the PDF text layer.
5. **WildFake per-subset image counts and archive sizes** — authoritative source is the ModelScope "Files" tab and the paper's supplementary material.
6. **Whether the <2B constraint counts the full dual-tower checkpoint or the vision tower only** — this determines whether SigLIP2-Giant-Opt (1.87B full), MetaCLIP2-Huge (1.86B full) and PE-Core-G (1.88B vision / ~2.35B full) are admissible.
7. **DINOv3 licence terms** — checkpoints are gated on the Hub under a custom "other" licence; the actual redistribution/usage terms were not read in this pass.
