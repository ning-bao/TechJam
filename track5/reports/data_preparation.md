# Training data preparation (PLAN D3 + D4)

Built 2026-08-26. Reproduce with the commands in each section.

## 1. The "corruption" was a thread-safety bug, not bad data

Strict manifest builds reported ~12% of COCO train2017 and ~17% of WildFake ADM
members as unreadable (`Bad CRC-32`, `Overlapped entries`, zlib `Error -3`).
The archives are intact. `zipfile.ZipFile` owns one file object with one seek
position; `track5.data.resolve` cached a single handle in a module-level dict and
handed it to every thread of an 8-16 worker pool, so concurrent reads interleaved
seeks and returned wrong bytes.

Controlled test on COCO val2017 (5,000 members):

| access pattern | bad members |
|---|---|
| single thread, shared handle | 0 / 5000 |
| 8 threads, **shared** handle | 721 / 5000 |
| 8 threads, per-thread handle | 0 / 5000 |

Independent CRC verification of every archive (`scripts/verify_archives.py`,
one process per archive, single-threaded within): **302 archives, 2,201,339
members, 0 bad.**

Fixed by making the handle caches thread-local in `track5/data/resolve.py`, and
per-thread in `scripts/build_denylist.py` and `_iter_zip`. Regression test:
`tests/test_resolve_threadsafe.py`. Re-running the COCO build strictly then gave
118,287 / 118,287 rows, 0 unreadable.

The one genuinely damaged file is a truncated PNG inside WildFake's own
SDwithAdaptor.zip (1 of 120,000) — damaged upstream, not by the download.

## 2. Protected set (constraint C2)

`scripts/build_denylist.py --coco-val` produces:

- 5,000 content hashes (sha256 + pHash) covering **every** image in the
  canonical val2017 archive; the organiser's demonstration benchmark is the
  4,998-image subset of it, and the other 2 are denied anyway.
- 69,480 protected paths: 4,998 COCO val2017 + 64,482 WildFake DALL·E
  (55,639 Typical + **8,843 Advanced**).

Both organiser-stated counts (8,843 / 4,998) reproduce exactly from our own copy
of the data. DALL·E is refused at manifest-build time on CSV, Architecture,
Category, family and path, so a renamed file cannot leak in. Denylist hits in the
final training set: **0**.

Container check on the protected fakes: all 8,843 carry a `.jpg` extension but
~1/3 are actually PNG payloads (sampled 300: 201 JPEG / 97 PNG / 2 other). So
file extension carries no class signal at test time, and a third of the fakes are
pristine, never-JPEG pixels — which drives the lossless branch in §4.

## 3. Sources and the resolution constraint

| role | source | rows |
|---|---|---|
| real | COCO train2017 | 118,287 |
| real | WildFake afhq/celebahq/church/ffhq/imagenet/laion5b | 48,000 |
| real | SID_Set | 39,821 |
| fake | WildFake non-DALL·E, 20k per family | 119,999 |
| fake | SID_Set FLUX | 40,000 |
| fake | VAE reconstructions of our own COCO reals (SD1.5 + SDXL) | 49,986 |

**Nothing is ever padded.** WildFake fakes are quantized to {128, 200, 256, 512}
px while reals are continuous (median short side 428). At a 448 crop only images
with min(W,H) >= 448 qualify, so the pool is filtered to those and every training
sample is an identical 448x448 of native pixels — native size becomes invisible
downstream. Padding would otherwise apply almost exclusively to fakes and hand
the model a free "padded => fake" rule.

Cost of that rule: the adm / ddpm / vqdm / gan families (128-256 px) cannot
supply a 448 crop and are excluded, leaving sd, other and flux. This is a
deliberate deviation from D3's "all non-DALL·E families" — the alternative was
padding, or resizing, which D6 forbids for the crop source. The excluded families
are a ready-made **held-out-generator OOD set** for D9. Midjourney resolved to no
downloaded archive and contributed 0.

SID_Set rows were extracted to individual files (`scripts/extract_sidset.py`,
76,273 files, ~62 GB) with bytes written verbatim so sha256/pHash and the native
container stay valid. Reading them through the parquet path form decodes a whole
~844-image row group per image, which is fine for one sequential pass and fatal
for shuffled training access.

## 4. Bias neutralization (D4)

Raw, the corpus is trivially separable by container: COCO reals ~100% JPEG q94,
WildFake fakes ~92% PNG.

`track5/data/normalize.py` gives both classes the same delivered-container
statistics. The plan is seeded by sha256 and takes no label:

- 30% of images (both classes) are delivered **lossless** — matching the ~1/3 of
  protected fakes that are pristine PNG payloads.
- the rest get an optional extra pass plus one final JPEG encode at a quality
  drawn from a shared distribution.

The plan is conditional on the *native* container: a real photo arrives already
JPEG-compressed and that cannot be undone, so total history and final-encode
quality cannot both be equalized. We equalize the final encode, because that is
what the q-table probe reads and what a detector most easily latches onto. The
residual (reals carry one extra hidden compression) is a double-compression trace
that is far harder to exploit and is also true of real photos in the wild.

After normalization, file size no longer reflects the container — it reflects
content complexity, and synthetic images are smoother, so they compress smaller.
`scripts/match_distributions.py` equalizes it by selection, stratified by
container then binned by size, keeping equal reals and fakes per stratum.

## 5. Gate G1 result

Probes run on **post-crop, post-normalization** metadata
(`scripts/normalize_manifest.py`), because that is what the model receives —
probing the raw manifest answers the wrong question in both directions.

| probe | raw | normalized | + size-matched |
|---|---|---|---|
| file_size | 0.589 | 0.620 | **0.497** |
| dimensions | 0.732 | 0.500 | **0.500** |
| jpeg_quality | 0.974 | 0.496 | **0.500** |

**GATE: PASS** (all < 0.60). The raw 0.974 is the "Fake or JPEG?" trap measured
on our own data: untreated, a model could score ~97% from the q-table alone and
collapse on the benchmark.

### Probe 4 — frozen embeddings (`scripts/embedding_probe.py`)

Run on **DINOv3-L** (the primary backbone, so this measures the model we ship),
bf16 — fp16 returns NaN embeddings for this backbone.

The 0.60 threshold deliberately does not gate this probe: a strong backbone
*should* separate real from fake, that is the task. `probes.py` defaults its
labels to `label`, which measures detectability, not leakage. What diagnoses the
shortcut is the content-matched contrast — a COCO real against **its own VAE
reconstruction**, where the content is pixel-identical so any separability can
only come from the generation artifact.

| probe | bAcc |
|---|---|
| label_all | 0.900 |
| **label_matched** (1,552 pairs) | **0.920** |
| real_source (reals only) | 0.829 |
| fake_family (fakes only) | 0.835 |
| **content-reliance gap** | **-0.021** |

`label_matched` is *higher* than `label_all`: with content held constant the
backbone separates the classes at least as well, so the signal is a generation
artifact rather than "COCO photo vs generated art". Provenance is still readable
(real_source 0.829), but the model does not need it. This also confirms the VAE
reconstructions are genuinely detectable rather than label noise.

A first run over a random sample caught only 71 matched pairs and read 0.775;
that was small-sample noise, hence `--force-matched`.

## 6. Final splits

| split | rows | real | fake |
|---|---|---|---|
| train | 132,186 | 66,093 | 66,093 |
| dev | 20,023 | 10,006 | 10,017 |
| calib | 20,019 | 10,006 | 10,013 |

Fake families in train: flux 18,556 / vae_sd15 13,919 / vae_sdxl 13,513 /
sd 10,281 / other 9,824. Real sources: coco 32,262 / sid_set 28,416 /
wildfake 5,415.

Verified on the final manifests: 0 sha256 overlap between any two splits,
0 denylist hits, and **0 cross-split leaks across all 28,413 recon/source
pairs** — a reconstruction is content-identical to its source real, so the pair
is split as one unit (`src_sha256` is the grouping key).

**dev and calib are deliberately left un-normalized and un-matched.** At
inference we take images as delivered, so an honest estimate of protected-set
performance needs dev to look like the test distribution. Normalization is a
training-time debiasing intervention, like augmentation. Shortcut reliance still
shows up in evaluation: the matrix re-encodes both classes identically per
condition, so a container-shortcut model shows a large clean->jpeg drop, which is
exactly what `worst_case_bacc` selects against.

Note dev/calib contain no VAE-reconstruction slice worth reading as a generator
score: recons exist to force the model off content cues during training, and the
protected set is DALL-E, not VAE output.

## 7. VAE reconstructions (D3 Fake #3)

25,000 COCO reals with min side >= 448 (deterministic, sha256-sorted) pushed
through the SD1.5 and SDXL VAEs, encoder->decoder only with `latent_dist.mode()`
so there is no sampling and the output is reproducible. 49,986 files, ~4 GB
(7 fewer per VAE than 25,000 because COCO contains a few byte-identical
duplicates, which collide on the sha256-prefix filename).

Fidelity, source vs reconstruction: SD1.5 MAE 13.38 / PSNR 22.93 dB, SDXL
MAE 8.96 / PSNR 25.33 dB. Two distinct artifact strengths, so SDXL is the
subtler and harder positive rather than a duplicate of SD1.5.

These are the only fakes that are content-matched to our reals, which is what
makes them worth the GPU time: normalization removes *container* cues, but
nothing else in the corpus attacks the *content* gap between COCO photographs
and generated art. They also carry a VAE decoder fingerprint, which is the
artifact family DALL-E 3 shares.

Reconstructions inherit their source's container (JPEG), so `native_history` is
1 and normalization treats them exactly like any other row.

## 8. Not yet done

- **4th shortcut probe** (source classifier on frozen embeddings) — needs a
  backbone forward pass, and note `probes.py` defaults its labels to `label`,
  which measures detectability rather than provenance leakage; pass explicit
  source labels via `--emb-labels`.
- `configs/dinov3l448_d4.yaml` is the config matching this data (crop 448,
  `data.normalize: true`). `configs/dinov3l512.yaml` needs a pool rebuilt with
  `--crop 512` before use; at 512 the common 640x480 COCO real would be padded.
