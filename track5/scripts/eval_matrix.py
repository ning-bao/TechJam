"""Run the 15-atom robustness matrix for a checkpoint over a manifest split.

.venv/Scripts/python.exe scripts/eval_matrix.py --manifest data/manifests/dev.parquet \
    --checkpoint runs/x/best.pt --atoms all --out reports/ [--limit 500]
"""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--atoms", default="all")
    ap.add_argument("--out", default=str(REPO / "reports"))
    ap.add_argument("--cache", default=str(REPO / "data/cache/eval"))
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from track5.eval.matrix import run_matrix
    from track5.models import build_model
    from track5.models.preprocess import eval_crop, to_tensor
    from track5.transforms.eval_atoms import ATOMS_VERSION, EVAL_15
    from track5.utils.hashing import file_sha256
    from track5.utils.imaging import apply_decode_policy

    # Camera originals routinely exceed PIL's default bomb guard (our own set A
    # peaks at 103.8 MP); without this they warn, and would hard-fail under any
    # warnings-as-errors setting.
    apply_decode_policy()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    cal = ckpt.get("calibration") or {}
    T = cal.get("temperature") or 1.0
    alpha = cal.get("alpha") or 0.0
    threshold = cal.get("threshold") if cal.get("threshold") is not None else 0.5
    crop = int(ckpt["config"].get("data", {}).get("crop", 448))

    manifest = pd.read_parquet(args.manifest)
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"] != "denied"]
    if args.limit:
        manifest = manifest.head(args.limit)

    @torch.no_grad()
    def score_fn(paths):
        out = []
        for i in range(0, len(paths), args.batch_size):
            batch = []
            for p in paths[i:i + args.batch_size]:
                img = Image.open(p).convert("RGB")
                batch.append(to_tensor(eval_crop(img, size=crop)))
            px = torch.stack(batch).to(device)
            z = model(px).float().cpu().numpy()
            out.append(1.0 / (1.0 + np.exp(-(z + alpha) / T)))
        return np.concatenate(out) if out else np.array([])

    atoms = EVAL_15 if args.atoms == "all" else [a.strip() for a in args.atoms.split(",")]
    meta = {
        "data_root": args.data_root,
        "model_hash": file_sha256(args.checkpoint)[:12],
        "config_hash": ckpt.get("meta", {}).get("config_hash", "unknown"),
        "atoms_version": ATOMS_VERSION,
    }
    out_csv = Path(args.out) / f"matrix_{meta['config_hash']}_{meta['model_hash']}.csv"
    df = run_matrix(manifest, score_fn, atoms, args.cache, args.seed, threshold, out_csv, meta)
    print(df.to_string(index=False))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
