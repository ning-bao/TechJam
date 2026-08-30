# Model weights: licence and provenance

The code in this repository is MIT (see `LICENSE`). **The model weights are
not.** This file states what they are, what licence they carry, and what that
licence requires of anyone who redistributes them.

## What the weights are

| | |
|---|---|
| File | `epoch1_best_calibrated.pt` |
| Base model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (303.1M parameters) |
| Adaptation | full end-to-end fine-tune at 448 px |
| Identity | `model_hash f93b4a45748f`, `config_hash 4ed11edd6769`, step 7000 |
| Calibration carried inside | T = 1.368804671038221, α = −0.13818359375, τ = 0.45936643399060617 |

The checkpoint is a **derivative of DINOv3**. Meta's base weights were never
vendored into this repository; they are fetched at training time from the
Hugging Face Hub. But a fine-tune of those weights is a derivative work, and
the DINOv3 License follows it.

## Licence

**The weights are licensed under the DINOv3 License** (Meta's Llama-style
community licence), *not* MIT and *not* an OSI-approved open-source licence.

Under that licence, royalty-free and commercial use are permitted, and so are
fine-tuning and redistribution of derivatives. Three obligations attach, and we
accept all three:

1. **Distribution.** Fine-tuned DINOv3 weights must be distributed under the
   DINOv3 License **with a copy of the agreement attached**. They must never be
   relicensed under MIT or any other licence.
2. **Acknowledgement.** DINOv3 use must be acknowledged in any write-up. It is
   acknowledged in this repository's `README.md`, in `PLAN.md` D1/D11, and in
   the project's Devpost description.
3. **Indemnification and amendment.** We accept the licence's indemnification
   clause and Meta's unilateral-amendment right.

The weights are not OSI-open-source because the licence carries a
trade-control field-of-use restriction. This track does not mandate OSI
licensing, so it does not affect eligibility.

## Acknowledgement

> This model is a fine-tune of DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`),
> developed by Meta AI and used under the DINOv3 License.

## Where the weights are distributed

GitHub Release on this repository. See the download step in `README.md`
§ Quick start.

## ⚠️ Release checklist — not yet done

Both items below must be completed **before** the repository is made public,
because the second one is a licence obligation, not a nicety.

- [ ] **Create the GitHub Release** and upload `epoch1_best_calibrated.pt`,
      then replace the `<TAG>` placeholder in `README.md` § Quick start with the
      real tag.
- [ ] **Attach the DINOv3 License text to the same release** as a second asset
      (e.g. `DINOv3-LICENSE.txt`). Obligation (1) above requires the agreement
      to travel *with* the weights — a link is not sufficient.

      The canonical text is on the model's Hugging Face page:
      <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
      (also distributed in the model repository as `LICENSE.md`). Copy it
      verbatim; do not paraphrase or retype it.

Neither task can be completed from this repository alone: the checkpoint lives
on the training machine, and the licence text must be taken from Meta's
canonical copy rather than reproduced from memory.
