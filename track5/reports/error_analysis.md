# Error analysis note

Deliverable 5. Where this detector fails, what the failures have in common, and
the trade-offs we chose.

**Status.** Sections 1–3 and 5 are complete and measured. Section 4 — individual
false-positive and false-negative cases — requires scoring the 200-image
hard-case set, which needs the GPU and the frozen checkpoint together. The
evaluation sets, the hypotheses they test, and the acceptance criteria are stated
below so the gap is visible rather than papered over.

---

## 1. Which transforms cause failures, and why those

Failures concentrate in one place. Across the 15 conditions, three account for
almost all the degradation:

| condition | dev bAcc | Δ clean | what it destroys |
|---|---|---|---|
| resize 0.25× | 0.9685 | −0.031 | high-frequency detail, via downsample-then-upsample |
| noise σ0.10 | 0.9770 | −0.023 | high-frequency detail, by burying it in noise |
| JPEG q30 | 0.9820 | −0.018 | high-frequency detail, via aggressive quantization |

The other twelve cost between 0.000 and 0.008. Colour jitter (±20%) and centre
crop (80%) are free.

This ordering is diagnostic, not incidental. The detector's evidence lives in the
high-frequency band — generator upsampling traces, VAE decoder signatures,
resampling artifacts — so every operation that attacks that band hurts, and
operations that leave it intact (colour, framing) do not. A detector that had
instead learned semantic cues would show the opposite pattern.

**Trade-off accepted.** Making the model robust to `resize_025` by training
heavily on downscaled crops would blunt the high-frequency reliance that makes it
work at all. We trained 10–20% past test severity and no further.

---

## 2. The failure we caused ourselves, and how we found it

The most instructive error in this project is one we introduced.

A second training epoch improved every dev condition (worst-case +0.60, 14 wins
and 1 tie across 15 atoms) and lost 4.5–11.6 points on held-out generator
families. Submitting on the dev metric — which is what our own written plan
specified — would have shipped the worse model.

**Root cause.** The augmentation RNG was seeded from `(sha256, index, seed)` with
no epoch term, so epoch 2 replayed epoch 1's byte-identical crops and
distortions. No new diversity; only a tighter fit to the training families.

**What actually degraded.** Out-of-distribution AUROC moved ≤0.003 and improved on
three of five conditions, so the representation was intact. What drifted was the
logit scale: epoch 2 needs T = 1.519 against epoch 1's 1.369, and α = −0.292
against −0.138 — measurably more softening required, which is overconfidence
stated in units of temperature. Because the submitted score is `sigmoid((z+α)/T)`
against a frozen τ, scale drift is precisely the error a frozen operating point
cannot absorb.

**Generalizable lesson.** An in-distribution validation split cannot detect
overfitting to the generators it shares with training. It will actively reward it.
This is why the held-out-families split exists, and it is the reason the decision
went against our own selection metric (documented as a deviation in
KNOWN_LIMITATIONS item 2).

**The ablation, run (2026-08-30).** After fixing the RNG we retrained the second
epoch with everything else identical — same config hash, same batch order, fresh
augmentation bytes only. Held-out families, calibrated, n = 4,000 per condition:

| condition | epoch 1 | buggy epoch 2 | salted epoch 2 |
|---|---|---|---|
| clean | 0.8703 | 0.8045 | 0.8295 |
| JPEG q30 | 0.8313 | 0.7205 | 0.7415 |
| blur σ2.0 | 0.8830 | 0.8098 | 0.8275 |
| resize 0.25× | 0.9158 | 0.7995 | 0.8335 |
| noise σ0.10 | 0.8075 | 0.7628 | 0.8028 |
| **worst case** | **0.8075** | **0.7205** | **0.7415** |

Fresh draws recover 2–4 points per condition but close only about a quarter of
the worst-case gap; the rest is the second pass itself. The salted epoch still
beat epoch 1 on dev (worst-case 0.9785 vs 0.9685) while losing held-out by 6.6
points — the in-distribution split rewarded the overfit both times. The lesson
above is a measurement now, not an inference.

---

## 3. Where false positives are structurally likely

A false positive here means calling a real photograph AI-generated. Three
properties of real photographs plausibly mimic generator output:

1. **Smooth studio lighting** — low local variance and few sharp gradients read
   like the over-smoothing typical of diffusion output.
2. **Heavy post-processing** — HDR merging, aggressive denoise and clarity
   sliders erase the sensor-noise floor that distinguishes a camera capture.
3. **Minimal or incoherent composition** — near-empty frames give the detector
   almost no structure to read, so its decision rests on noise statistics alone.

100 real photographs were curated against exactly these three categories (35 / 35
/ 30). All are platform-licensed, provenance-recorded, and verified to carry no
C2PA manifest — so none is a mislabelled AI image.

**Known confounds in that set**, both documented in KNOWN_LIMITATIONS:

- All 100 come from a single platform (Unsplash), so its processing pipeline is a
  competing explanation for any elevated false-positive rate (item 7).
- The images are 3.1–103.8 MP JPEG originals (median 19.9, with 34 of 100 under
  15 MP and 46 over 20 MP), while the generated set is 1–1.6 MP PNG. A 448 px
  centre crop therefore covers a median 1.0% of a real frame and 18.7% of a
  generated one. Measured: median high-frequency energy in the native crop is
  4.3 for the minimal-composition category against 12.8 for the generated set,
  and most of that gap closes when the real images are rescaled to the generated
  set's scale (item 5).

The second confound matters more than it looks. A false positive on a
minimal-composition photograph may be reporting *resolution*, not photography —
so each of these images must be scored twice, native crop and
downscale-then-crop, and both reported. The difference is the part of the
false-positive rate that is an artifact of our own crop policy.

## 3b. Where false negatives are structurally likely

A false negative means calling an AI image real. 100 generated images were
curated to be mundane rather than spectacular — the deliberately unremarkable
output that carries the fewest telltale artifacts — across four models:

| generator | n | declared pixel watermark |
|---|---|---|
| gpt-image-2 (Azure) | 30 | `com.microsoft.invismark.1` |
| gemini-3-pro-image | 25 | SynthID |
| gemini-3.1-flash-image | 20 | SynthID |
| **flux_dev** | **25** | **none — no C2PA manifest at all** |

The Flux slice is the control group and the reason this set can answer anything.
The other 75 images carry vendor watermarks embedded in the pixels by design.
Without a no-watermark slice, a strong result on this set would be unattributable:
we could not tell whether the detector found the generator's fingerprint or a
watermark that every one of those images shares and that has nothing to do with
generation architecture.

10 of the 100 carry visible scene text, tagged as a sub-category: text rendering
is a known generator weakness, so those are the *easiest* cases and should be
reported separately rather than allowed to inflate the aggregate.

**Watermark status is declared, not detected.** We read each file's provenance
manifest; we did not run a watermark detector. An undeclared watermark and an
absent watermark are indistinguishable to us — see KNOWN_LIMITATIONS item 8 for
the near miss this caused.

---

## 4. Individual cases — measured 2026-08-30

One inference pass over the 200 hard-case images with the frozen submission
checkpoint (`epoch1_best_calibrated.pt`, τ = 0.4594), set A scored twice —
native crop, and long-side-to-1024 then crop (lossless PNG, Lanczos; no image
falls below the 448 crop after downscaling, so no padding is involved).
Per-item scores: `reports/hardcase_preds_native.json` and
`reports/hardcase_preds_A_1024.json`.

**The acceptance criteria were fixed in advance** (in the repository history
before the run). Verdicts:

- *Set A FPR at the frozen τ exceeds 5% → headline finding.* **Not triggered:
  2/100 native, 1/100 downscaled.** The operating point survives adversarial
  real photographs.
- *Watermarked 75 vs Flux 25 differ by >~5 points → detector reads watermarks;
  report split.* **Triggered at 93.3 points — in the reverse direction from the
  anticipated mechanism.** Flux, the no-watermark control: 25/25 detected,
  median score 0.9993. The watermarked slice: 5/75, median scores 0.004–0.008.
  A watermark-reading detector would find the watermarked images easy; this
  refutes that mechanism. All set B numbers below are reported split.
- *Crop variants of set A disagree by >~3 points → FPR is about crop policy.*
  **Not triggered in aggregate** (2% vs 1%). Per image the variant is decisive:
  both native false positives fall to ≤0.004 when downscaled, and one image
  flips the other way, 0.044 → 0.956.

### The finding: 2026 consumer endpoints evade the detector

| set B slice | recall @ frozen τ | median score |
|---|---|---|
| flux_dev (seen family, no watermark) | **25/25** | 0.9993 |
| gpt-image-2 (Azure) | **0/30** | 0.0038 |
| gemini-3-pro-image | 3/25 | 0.0076 |
| gemini-3.1-flash-image | 2/20 | 0.0065 |
| text-bearing subset (10 across slices) | 1/10 | — |

These are not threshold-marginal misses: the missed images score around 0.005
where Flux scores 0.999, so ranking fails along with the operating point.
Unseen-family fragility alone does not explain it — DALL·E 3 Advanced is also
unseen and the protected benchmark holds 0.9891 clean (§3 of the robustness
summary). What separates the missed slice is the *generation and delivery
pipeline*: 2026 consumer endpoints (ChatGPT Images 2.0, Gemini 3.x) producing
deliberately mundane 1–1.6 MP PNGs. Candidate explanations, none proven here:
decoder architectures whose artifacts our 2023–24 training families do not
span; vendor-side post-processing — including the pixel watermarks themselves —
overwriting the high-frequency traces the detector reads; the mundane-curation
brief removing semantic tells. Visible text did not rescue detection: 1 of the
10 text-bearing images was caught, against text rendering being a known
generator weakness.

Scope: the track's scored benchmark is the DALL·E protected set, where the
detector holds 0.958–0.989. This finding is measured on our own adversarial
curation of the 2026 frontier, and it is the first thing more time would go to:
add 2026-generation families to the training corpus — the recipe is unchanged,
the corpus ages.

### Worst false positives — the 12 highest-scoring real photographs

| image | native | down-1024 | category |
|---|---|---|---|
| minimal_or_incoherent/027.jpg | **0.8012** | 0.0041 | minimal_or_incoherent |
| heavy_postprocess_landscape/003.jpg | **0.7009** | 0.0012 | heavy_postprocess_landscape |
| heavy_postprocess_landscape/006.jpg | 0.3120 | 0.0052 | heavy_postprocess_landscape |
| heavy_postprocess_landscape/025.jpg | 0.2079 | 0.0006 | heavy_postprocess_landscape |
| heavy_postprocess_landscape/016.jpg | 0.1954 | 0.0052 | heavy_postprocess_landscape |
| minimal_or_incoherent/021.jpg | 0.1681 | 0.0203 | minimal_or_incoherent |
| minimal_or_incoherent/012.jpg | 0.1387 | 0.0049 | minimal_or_incoherent |
| minimal_or_incoherent/030.jpg | 0.1032 | 0.0216 | minimal_or_incoherent |
| minimal_or_incoherent/020.jpg | 0.0438 | **0.9558** | minimal_or_incoherent |
| heavy_postprocess_landscape/024.jpg | 0.0344 | 0.0008 | heavy_postprocess_landscape |
| heavy_postprocess_landscape/034.jpg | 0.0308 | 0.0031 | heavy_postprocess_landscape |
| minimal_or_incoherent/014.jpg | 0.0300 | 0.0044 | minimal_or_incoherent |

Only the top two cross τ natively, and both collapse to noise when the image is
brought to the generated set's scale — those false positives are reporting
resolution, exactly as §3 predicted. `020.jpg` is the counterexample worth
keeping: benign at native resolution, 0.956 downscaled. The studio_portrait
category produced no false positive in either variant.

### Worst false negatives — the 12 lowest-scoring generated images

| image | score | generator | declared watermark |
|---|---|---|---|
| gemini_3_pro_image/021.png | 0.0003 | gemini-3-pro | SynthID |
| gemini_3_pro_image/015.png | 0.0004 | gemini-3-pro | SynthID |
| gpt_image_2_azure/012.png | 0.0004 | gpt-image-2 | invismark |
| gpt_image_2_azure/010.png | 0.0004 | gpt-image-2 | invismark |
| gemini_3_1_flash_image/013.png | 0.0004 | gemini-3.1-flash | SynthID · has_text |
| gemini_3_1_flash_image/019.png | 0.0005 | gemini-3.1-flash | SynthID · has_text |
| gemini_3_1_flash_image/017.png | 0.0005 | gemini-3.1-flash | SynthID · has_text |
| gemini_3_pro_image/004.png | 0.0005 | gemini-3-pro | SynthID |
| gemini_3_pro_image/003.png | 0.0005 | gemini-3-pro | SynthID |
| gemini_3_1_flash_image/016.png | 0.0005 | gemini-3.1-flash | SynthID · has_text |
| gpt_image_2_azure/016.png | 0.0006 | gpt-image-2 | invismark |
| gpt_image_2_azure/013.png | 0.0006 | gpt-image-2 | invismark |

All twelve come from the watermarked 2026 slice — no Flux image appears until
rank 76 — and four carry visible scene text.

---

## 5. Trade-offs, stated plainly

| Choice | Bought | Cost |
|---|---|---|
| 448 px crops at native resolution, never resize the source | +6–15 AUC over resizing; preserves the high-frequency evidence | four low-resolution generator families (adm/ddpm/vqdm/gan) are absent from training |
| Worst-case model selection instead of mean | a model that cannot win by being good at clean images and collapsing under noise | slower apparent progress; the metric is dominated by one condition |
| Calibrate once on a deployment mixture, never per transform | no tuning on evaluation conditions | ECE rises from 0.013 in-distribution to 0.10–0.19 on unseen generators |
| Full fine-tuning over a frozen probe | +23.2 points worst-case | ~6 h of GPU per epoch instead of minutes |
| Protected benchmark excluded from training, selection and calibration | results mean what they claim | no ability to tune against the benchmark; two exploratory reads overran the budget and are disclosed |
| Ban MixUp/CutMix and hue jitter | avoids a documented 13-point collapse on this task | a smaller augmentation space than the defaults would give |

The through-line: every one of these prefers a number that is true over a number
that is high. The detector's value is entirely in what it does to images and
generators it has never seen, and every shortcut available to us would have
inflated the in-distribution figure while destroying that.
