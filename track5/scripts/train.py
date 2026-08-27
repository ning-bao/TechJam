"""Train a detector from a config.

.venv/Scripts/python.exe scripts/train.py --config configs/vitb_e2e.yaml \
    --out runs/vitb_e2e [--device auto] [--max-steps N]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def make_eval_fn(cfg, device, batch_size=64):
    """Worst-case bAcc over cfg.eval.worst_case_atoms using pre-cached
    transformed dev dirs (data/cache/eval/<atom>/) when present; falls back to
    the clean dev manifest otherwise."""
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from track5.eval.metrics import bacc, worst_case_bacc
    from track5.models.preprocess import eval_crop, to_tensor

    dev_manifest = pd.read_parquet(REPO / cfg["data"]["dev_manifest"])
    dev_manifest = dev_manifest[dev_manifest["split"] != "denied"]
    label_by_sha = dict(zip(dev_manifest["sha256"], dev_manifest["label"]))
    crop = int(cfg["data"].get("crop", 448))
    cache_root = REPO / "data" / "cache" / "eval"
    atoms = list(cfg.get("eval", {}).get("worst_case_atoms",
                 ["clean", "jpeg_30", "blur_20", "resize_025", "noise_010"]))

    atom_sets = {}
    for atom in atoms:
        d = cache_root / atom
        files = sorted(d.glob("*")) if d.exists() else []
        files = [f for f in files if f.stem in label_by_sha]
        if files:
            atom_sets[atom] = files
    if not atom_sets:
        print("[eval_fn] no cached transformed dev sets found -> clean-manifest "
              "fallback (run scripts/eval_matrix.py once to build caches)",
              file=sys.stderr)

    @torch.no_grad()
    def score_files(model, files):
        scores, labels = [], []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            px = torch.stack([to_tensor(eval_crop(Image.open(f).convert("RGB"), crop))
                              for f in batch]).to(device)
            z = model(px).float().cpu().numpy()
            scores.append(1 / (1 + np.exp(-z)))
            labels.extend(label_by_sha[f.stem] for f in batch)
        return np.asarray(labels), np.concatenate(scores)

    @torch.no_grad()
    def score_manifest(model):
        from io import BytesIO

        from track5.data.resolve import resolve_image_bytes

        scores, labels = [], []
        rows = list(dev_manifest.itertuples(index=False))
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            imgs, ys = [], []
            for r in chunk:
                try:
                    img = Image.open(BytesIO(resolve_image_bytes(
                        REPO / "data" / "raw", r.path))).convert("RGB")
                except Exception:
                    continue
                imgs.append(to_tensor(eval_crop(img, crop)))
                ys.append(int(r.label))
            if not imgs:
                continue
            z = model(torch.stack(imgs).to(device)).float().cpu().numpy()
            scores.append(1 / (1 + np.exp(-z)))
            labels.extend(ys)
        return np.asarray(labels), np.concatenate(scores)

    def eval_fn(model):
        per_atom = {}
        if atom_sets:
            for atom, files in atom_sets.items():
                y, s = score_files(model, files)
                per_atom[atom] = bacc(y, s, 0.5)
        else:
            y, s = score_manifest(model)
            per_atom["clean"] = bacc(y, s, 0.5)
        out = {f"bacc_{a}": v for a, v in per_atom.items()}
        out["worst_case_bacc"] = worst_case_bacc(per_atom)
        return out

    return eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", default="none",
                    help='"auto" (newest in <out>/checkpoints), "none", or a path')
    ap.add_argument("--allow-config-change", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from track5.data.dataset import ManifestDataset
    from track5.models import build_model
    from track5.train.loop import Trainer
    from track5.transforms.train_sampler import TrainDistortionSampler
    from track5.utils.config import load_config

    cfg = load_config(args.config)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    model = build_model(cfg).to(device)
    sampler = TrainDistortionSampler() if cfg.get("distortion", {}).get("enabled", True) else None
    ds = ManifestDataset(REPO / cfg["data"]["train_manifest"], split="train",
                         crop=int(cfg["data"].get("crop", 448)),
                         distortion_sampler=sampler, seed=int(cfg.get("seed", 17)),
                         normalize=bool(cfg["data"].get("normalize", False)))
    workers = int(cfg["data"].get("workers", 0))
    from track5.data.sampler import seed_worker

    loader = DataLoader(ds, batch_size=int(cfg["data"].get("batch_size", 32)),
                        shuffle=True, num_workers=workers, drop_last=True,
                        persistent_workers=workers > 0, pin_memory=device == "cuda",
                        worker_init_fn=seed_worker if workers > 0 else None,
                        generator=torch.Generator().manual_seed(int(cfg.get("seed", 17))))
    eval_fn = make_eval_fn(cfg, device)
    trainer = Trainer(cfg, model, loader, eval_fn, args.out)
    if args.resume != "none":
        resumed = trainer.resume(args.resume,
                                 allow_config_change=args.allow_config_change)
        print(f"[train] resume={args.resume} -> {'resumed' if resumed else 'fresh start'}")
    best = trainer.train()
    print(f"done; best worst_case_bacc={best:.4f}; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
