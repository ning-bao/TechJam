# Known limitations

Everything here is measured or decided, not speculative. Each entry says what we
know, how we know it, and what we would do with more time.

Ordered by how much it would change a reader's confidence in the results.

---

## 1. The 448 px crop excludes four generator families from training

**What.** Training reads 448 px crops at native resolution and never pads a
training image. WildFake's `adm`, `ddpm`, `vqdm` and `gan` families are quantized
to 128–256 px, so they are smaller than the crop and are absent from the training
corpus.

**Why it stands.** Downscaling the crop source destroys the high-frequency
evidence the detector depends on; the measured cost of resize-instead-of-crop in
published work is 6–15 AUC points. Padding would introduce a border artifact that
appears in no real image and in no evaluation image.

**Consequence.** The model is trained on five generator families (FLUX, SD VAE
decoders and three others) and tested on families it has never seen. That is the
intended test, but it means low-resolution diffusion output is an untested input
class rather than a held-out one.

**Unratified.** This is the largest design trade-off in the project that was
never explicitly signed off. It should have been a numbered decision with an
owner.

**With more time.** Train a second model at 224 px on the excluded families and
check whether the two disagree; if they do, that disagreement is a more
interesting result than either model alone.

---

## 2. A second training epoch made the model worse, and there is a bug behind it

**What.** Epoch 2 beat epoch 1 on dev (worst-case bAcc +0.60, and ≥ epoch 1 on
all 15 dev conditions) and lost badly on held-out generator families:

| condition | epoch 1 | epoch 2 | Δ |
|---|---|---|---|
| clean | 0.8703 | 0.8045 | −6.58 |
| JPEG q30 | 0.8313 | 0.7205 | −11.08 |
| blur σ2.0 | 0.8830 | 0.8098 | −7.33 |
| resize 0.25× | 0.9158 | 0.7995 | **−11.63** |
| noise σ0.10 | 0.8075 | 0.7628 | −4.48 |

**The bug.** The augmentation RNG was seeded from `(sha256, index, global_seed)`
with no epoch term, so epoch 2 replayed epoch 1's byte-identical crops and
distortions. A second pass over the same augmented bytes adds no diversity — it
can only tighten the fit to the training families.

**What actually degraded.** Not the representation: out-of-distribution AUROC
moved by ≤0.003 and improved on three of five conditions. What drifted is the
logit scale — epoch 2 needs temperature 1.519 against epoch 1's 1.369, and α
−0.292 against −0.138. Since the submitted score is `sigmoid((z+α)/T)` against a
frozen τ, a scale drift is exactly what a frozen operating point cannot absorb.
The bug is an amplifier of that drift, not the sole cause.

**Selection deviation, disclosed.** PLAN D5 specifies model selection by
worst-case bAcc on dev. By that metric epoch 2 wins. We submitted epoch 1 on the
strength of the held-out families instead. Dev shares all five generator families
with training, so it measures degradation on *seen* generators; the track is
scored on unseen ones. Epoch 2's dev lead and its held-out collapse have the same
cause, which makes the dev lead evidence of overfitting rather than of quality.
The held-out split is our own data and contains no protected images.

**We ran it (2026-08-30).** With the salt fixed, the second epoch was retrained
from the step-8000 checkpoint so that every step of it drew fresh augmentations
(run `dinov3l448_e2salt`, same config hash, calibrated by the same D7
procedure). Fresh draws recovered part of the buggy run's held-out loss on every
condition — clean 0.8045 → 0.8295, resize 0.7995 → 0.8335, noise 0.7628 →
0.8028 — and still lost to epoch 1 everywhere: worst-case **0.7415 against
0.8075**, non-overlapping CIs. Temperature says the same thing: the buggy second
epoch needed T = 1.519 and the salted one T = 1.496, against epoch 1's 1.369
(α: −0.138 → −0.292 buggy → −0.814 salted). The replay bug was an amplifier
worth 2–4 points; the dominant cause is the second pass itself — more fitting to
the same five families after dev has saturated. Epoch 1 remains the submission.
Artifacts: `reports/matrix_ood_excluded_epoch2salt_calibrated.csv`,
`reports/calibration_epoch2salt_best.json`.

---

## 3. Calibration does not transfer to unseen generators

Temperature and α are fitted once on our own deployment-mixture calibration
split, and τ is frozen on clean dev at the FPR ≤ 5% operating point. Never
refitted per transform.

ECE in-distribution is 0.0011–0.013. On held-out generator families it is
**0.10–0.19**. The ranking survives the shift; the calibrated operating point
does not.

This is the price of the discipline, not a failure of it. Refitting per condition
would mean tuning on the evaluation conditions, which we consider indefensible —
so we pay the ECE cost and report it.

**Unmeasured.** Our ECE implementation takes an absolute value per bin, so it
reports magnitude and not direction. We do not currently know whether the model
is systematically over- or under-confident on unseen generators, and those have
opposite consequences (false positives vs false negatives). Fixing this needs a
signed-ECE variant and one re-inference pass, because per-item scores are not
persisted — see item 6.

---

## 4. Protected-set discipline was overrun by two reads

The self-imposed rule: the protected benchmark is never training, selection, or
calibration data, and gets exactly one inference run at the end.

Two exploratory reads happened before that: 2026-08-27 19:30 (epoch 1) and
2026-08-28 02:29 (epoch 2), both uncalibrated logits at a naive 0.5 threshold,
recorded in `reports/matrix_protected_epoch1_step7000.csv` and its epoch-2
counterpart.

Neither read selected anything. The epoch-1-vs-epoch-2 decision was made on dev
and held-out families (item 2). The protected numbers corroborate that choice
after the fact; they did not produce it.

We are disclosing the overrun rather than deleting the artifacts before
submission. A repository whose git history contradicts its writeup is a worse
problem than an overrun that is reported.

**Not reproducible from this repo.** The reproduction sequence in the README
deliberately omits the protected run. The numbers in the headline table come from
a run the reader cannot repeat without the benchmark data and an explicit
`--protected-run` flag.

---

## 5. The hard-case evaluation sets are not comparable at `clean`

Two 100-image sets were curated to attack the model from both sides: real
photographs chosen to look synthetic (smooth studio lighting, heavy HDR grading,
minimal composition) and generated images chosen to look mundane.

They differ in ways that are not the thing being tested:

| | real set | generated set |
|---|---|---|
| container | JPEG | PNG |
| resolution | 3.1–103.8 MP (median 19.9) | 1.0–1.6 MP |
| 448 crop covers | 0.19–6.5% of the frame (median 1.0%) | 12.8–19.1% (median 18.7%) |

The real set is also wide *internally*: 34 of its 100 images are under 15 MP and
46 are over 20 MP, so "the real set" is not one resolution regime but two, and a
per-image crop-coverage figure matters more than the set-level median.

A fixed centre crop therefore shows the model a patch of fabric or sky from a
real photograph and most of the composition from a generated one. Measured
consequence: median high-frequency energy in the native crop is 4.3 for the
minimal-composition category against 12.8 for the generated set — and rescaling
the real images to the generated set's scale closes most of that gap. So a false
positive on those images may be reporting resolution, not photography.

Only the JPEG conditions, where every image is re-encoded identically, are clean
comparisons between the two sets.

**With more time.** Score each real image twice — native crop and
downscale-then-crop — and report both. The gap between them is the part of the
false-positive rate that is about resolution.

---

## 6. Per-item scores are not persisted

`eval_matrix` holds per-image scores in a local variable and writes only
aggregate rows. Every new per-item question — signed ECE, per-generator error
rates, false positives by category — needs a fresh inference pass over the whole
matrix.

**With more time.** Persist `(path, label, score, condition)` alongside the
aggregate CSV. That makes the entire class of question free.

---

## 7. Single-platform sourcing in the curated real set

All 100 curated real photographs come from Unsplash. A false-positive rate
measured on them cannot distinguish "this kind of photograph is hard" from "this
platform's processing pipeline leaves a signature". Two or three sources would
have separated those.

---

## 8. Watermark status is what the file declares, not what was detected

Set B splits into 75 images whose C2PA manifest declares a pixel watermark
(`com.microsoft.invismark.1` on 30, SynthID on 45) and 25 Flux Dev images with no
C2PA manifest at all. The Flux slice exists specifically so that "the detector
found the generator's fingerprint" can be separated from "the detector found a
vendor's watermark".

This split comes from reading each file's provenance manifest. We did not run a
watermark detector. An undeclared watermark and an absent watermark are
indistinguishable to us.

**A near miss worth recording.** Our first scan concluded the 45 Gemini images
carried no watermark, because it only looked for a `c2pa.soft-binding`
assertion — which is how Azure declares one. Google declares the same fact inside
a `c2pa.actions` entry: *"Applied imperceptible SynthID watermark."* Two vendors,
one specification, two encodings; checking one of them silently cleared half the
set. The same failure shape — absence of a signal read as absence of the thing —
is what this whole project is about.

---

## 9. Duplicate rows in the protected benchmark

5,124 of the benchmark's 13,843 rows are byte-identical duplicates at different
archive paths, all on the fake side: 6,932 DALL·E rows across 1,808 groups, some
appearing five times. The fake class has 8,843 rows but roughly 3,700 distinct
images.

Scoring every row is correct — that is the benchmark as delivered — but the
effective sample is smaller than n suggests, so the confidence intervals in the
headline table are narrower than the distinct-image count would justify. This is
a property of the organiser's data, verified against the source denylist.

---

## 10. Smaller things

- **5,000 reals scored, not 4,998.** The organiser's subset is identified by
  WildFake's internal ids, which have no overlap with canonical val2017
  filenames, and WildFake's COCO copies are not downloaded. The two omitted
  images cannot be identified locally, so the full canonical archive is scored: a
  0.04% difference, and all 5,000 are denylisted from training regardless.
- **Five protected conditions, not fifteen.** clean plus the four worst-case
  atoms. The full 15 is ~208k transformed images.
- **~1% of protected fakes are smaller than the 448 crop** and are reflect-padded
  at inference. No training image was ever padded, so that slice is out of domain.
- **Dev matrix is a 2,000-image subset** of the 20,023-row dev split, chosen so
  all 15 conditions could be cached and evaluated repeatedly.
- **A CI-deselected environment test.** The lock records a CUDA-local torch
  version that cannot exist on a CPU-only runner, so one of four environment-lock
  assertions is deselected in CI rather than weakened. What that assertion should
  check on a non-CUDA machine is unresolved.
