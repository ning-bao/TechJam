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

## 4. Individual cases — pending the scoring run

**What is needed:** one inference pass over the 200 hard-case images with the
frozen checkpoint, scored twice for set A (native crop and
downscale-then-crop-to-1024).

**What will be reported:** the 12 highest-scoring real photographs (worst false
positives) and the 12 lowest-scoring generated images (worst false negatives),
each with its score, its category, and — for set A — both crop variants so the
resolution confound is visible per image.

**Acceptance criteria decided in advance**, so the analysis cannot be shaped by
what comes back:

- If set A's false-positive rate at the frozen τ exceeds 5%, the operating point
  does not survive contact with adversarial real photographs and that is the
  headline finding, not a footnote.
- If the watermarked 75 and the Flux 25 differ by more than ~5 points of bAcc,
  the detector is reading watermarks rather than generator artifacts, and every
  set B number must be reported split rather than pooled.
- If the two crop variants of set A disagree by more than ~3 points, the
  false-positive rate is a statement about our crop policy and must be reported
  as both numbers.

This section is empty rather than estimated. Numbers here would be invented, and
an error-analysis note built on invented cases is worse than one that says which
measurement is missing.

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
