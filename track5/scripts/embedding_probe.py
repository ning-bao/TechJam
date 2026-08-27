"""PLAN D4 probe 4: what can a linear probe read off FROZEN embeddings?

The other three probes read file metadata. This one asks whether the classes are
separable by *provenance and content* rather than by a generation artifact —
our reals are COCO photographs while our fakes are SD/FLUX renders, and a model
allowed to key on "photo vs art" would score well clean and collapse under the
benchmark's transforms.

`probes.py` defaults its labels to `label`, which measures detectability, not
leakage — a good dataset SHOULD be separable. So this reports several probes and
the one that actually diagnoses the shortcut is the content-matched contrast:

  label_all      real/fake over the whole sample      (detectability ceiling)
  label_matched  COCO reals vs the VAE reconstructions OF THOSE SAME REALS —
                 identical content, so separability here can only come from the
                 generation artifact
  real_source    which real corpus an image came from (reals only)
  fake_family    which generator produced it (fakes only)

A large label_all with a near-chance label_matched would mean the model is
reading content, not generation. The reverse is what we want.

    .venv/Scripts/python.exe scripts/embedding_probe.py --n 4000
"""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/manifests/train.parquet"))
    ap.add_argument("--vae-manifest", default=str(REPO / "data/manifests/fake_vae.parquet"))
    ap.add_argument("--backbone", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--n", type=int, default=4000, help="per class")
    ap.add_argument("--force-matched", type=int, default=0,
                    help="also force N content-matched (real, its VAE recon) "
                         "pairs into the sample; a random sample catches few")
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(REPO / "reports/embedding_probe.json"))
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoModel

    from track5.data.probes import GATE_BACC, _probe
    from track5.data.resolve import resolve_image_bytes
    from track5.models.preprocess import eval_crop, to_tensor

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.manifest)
    vae = pd.read_parquet(args.vae_manifest)[["sha256", "src_sha256"]]
    link = dict(zip(vae.sha256, vae.src_sha256))

    # balanced sample, plus every content-matched pair we can find inside it
    pos = np.concatenate([
        rng.permutation(np.flatnonzero((df["label"] == y).to_numpy()))[:args.n]
        for y in (0, 1)])
    if args.force_matched:
        by_sha = {s: i for i, s in enumerate(df["sha256"])}
        recon_pos = np.flatnonzero((df["source"] == "vae_recon").to_numpy())
        extra = []
        for i in rng.permutation(recon_pos):
            src = link.get(df["sha256"].iloc[i])
            j = by_sha.get(src)
            if j is not None:
                extra += [i, j]
            if len(extra) >= 2 * args.force_matched:
                break
        pos = np.unique(np.concatenate([pos, np.array(extra, dtype=int)]))
    sub = df.iloc[np.sort(pos)].reset_index(drop=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.backbone).to(device).eval()
    # bf16, not fp16: DINOv3-L overflows in fp16 and returns NaN embeddings.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(dtype)
    data_root = REPO / "data/raw"

    feats, keep = [], []
    with torch.no_grad():
        for start in range(0, len(sub), args.batch_size):
            chunk = sub.iloc[start:start + args.batch_size]
            imgs, idx = [], []
            for i, row in zip(chunk.index, chunk.itertuples(index=False)):
                try:
                    im = Image.open(BytesIO(resolve_image_bytes(data_root, row.path)))
                    imgs.append(to_tensor(eval_crop(im.convert("RGB"), args.crop)))
                    idx.append(i)
                except Exception as e:
                    print(f"skip {row.path}: {e}", file=sys.stderr)
            if not imgs:
                continue
            out = model(pixel_values=torch.stack(imgs).to(device, dtype))
            feats.append(out.last_hidden_state[:, 0].float().cpu().numpy())
            keep.extend(idx)
            if start % (args.batch_size * 25) == 0:
                print(f"  {start}/{len(sub)}", flush=True)
    emb = np.concatenate(feats)
    bad = ~np.isfinite(emb).all(axis=1)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} non-finite embeddings",
              file=sys.stderr)
        emb, keep = emb[~bad], [k for k, b in zip(keep, bad) if not b]
    sub = sub.loc[keep].reset_index(drop=True)
    print(f"embedded {len(sub)} images -> {emb.shape}")

    y = sub["label"].to_numpy()
    results = {"label_all": _probe(emb, y)}

    # content-matched contrast: VAE recons present here whose source real is too
    sha_to_row = {s: i for i, s in enumerate(sub["sha256"])}
    pairs = [(sha_to_row[r], sha_to_row[link[r]])
             for r in sub.loc[sub["source"] == "vae_recon", "sha256"]
             if r in sha_to_row and link.get(r) in sha_to_row]
    if len(pairs) >= 20:
        rows = np.array([i for p in pairs for i in p])
        results["label_matched"] = _probe(emb[rows], y[rows])
        results["n_matched_pairs"] = len(pairs)
    else:
        results["label_matched"] = float("nan")
        results["n_matched_pairs"] = len(pairs)

    for name, mask, col in (("real_source", y == 0, "source"),
                            ("fake_family", y == 1, "generator_family")):
        vals = sub.loc[mask, col].to_numpy()
        if len(np.unique(vals)) > 1:
            results[name] = _probe(emb[mask], vals)
        else:
            results[name] = float("nan")

    print(f"\n{'probe':<16} {'bAcc':>7}")
    for k in ("label_all", "label_matched", "real_source", "fake_family"):
        v = results[k]
        print(f"{k:<16} {v:>7.3f}" if v == v else f"{k:<16} {'n/a':>7}")
    print(f"\ncontent-matched pairs used: {results['n_matched_pairs']}")
    gap = results["label_all"] - results["label_matched"]
    if results["label_matched"] == results["label_matched"]:
        print(f"content-reliance gap (label_all - label_matched): {gap:+.3f}")

    results["backbone"] = args.backbone
    results["n_embedded"] = int(len(sub))
    results["gate_bacc_metadata_probes"] = GATE_BACC
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
