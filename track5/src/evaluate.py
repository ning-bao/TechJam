"""Per-condition robustness evaluation (TC2 guide section 11).

    python -u -m src.evaluate --config configs/dinov3l512.yaml \
        --checkpoint runs/x/best.pt --manifest data/manifests/dev.parquet \
        --condition jpeg_30 --resume --output runs/x/metrics/jpeg_30.json

    python -u -m src.evaluate ... --condition all    --output runs/x/metrics/
    python -u -m src.evaluate ... --condition group:blur_resize --output <dir>

One named transform per row, generated deterministically from the image sha256
plus the condition (`item_seed(sha256, condition, seed)`), applied identically to
both classes. Scores are calibrated probabilities, never raw logits: the
temperature, alpha and threshold are read from the checkpoint and applied
unchanged to every condition.

Each condition writes its result atomically and then a `.done.json` completion
marker holding a fingerprint of (condition, checkpoint, config, manifest, atoms
version, calibration). `--resume` skips a condition whose marker matches; a
mismatch re-runs it rather than trusting a stale file.

Evaluating the protected COCO-val2017 x DALL-E benchmark requires an explicit
--protected-run, and refuses to overwrite an already completed one.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# TC2 section 11 splits the 15 rows into three group jobs when arrays are not
# available; the names are the sbatch array's task labels.
GROUPS = {
    "clean_jpeg": ["clean", "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30"],
    "blur_resize": ["blur_05", "blur_10", "blur_20", "resize_050", "resize_025"],
    "noise_color_crop": ["noise_002", "noise_005", "noise_010", "jitter_pm20",
                         "crop_80"],
}


def resolve_conditions(spec: str) -> list[str]:
    from track5.transforms.eval_atoms import EVAL_15, canonical_atom

    spec = spec.strip()
    if spec == "all":
        return list(EVAL_15)
    if spec.startswith("group:"):
        name = spec.split(":", 1)[1]
        if name not in GROUPS:
            raise SystemExit(f"unknown group {name!r}; have {sorted(GROUPS)}")
        return list(GROUPS[name])
    if spec in GROUPS:
        return list(GROUPS[spec])
    return [canonical_atom(c.strip()) for c in spec.split(",") if c.strip()]


def fingerprint(parts: dict) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def per_generator_metrics(rows, scores, threshold: float) -> dict:
    """Each fake family is scored against ALL reals; a family on its own has no
    negatives and every metric would be undefined."""
    import numpy as np

    from track5.eval.metrics import all_metrics

    labels = rows["label"].to_numpy(dtype=int)
    fams = rows["generator_family"].astype(str).to_numpy()
    real = labels == 0
    out = {}
    for fam in sorted({f for f, lab in zip(fams, labels) if lab == 1 and f}):
        mask = real | ((labels == 1) & (fams == fam))
        m = all_metrics(labels[mask], scores[mask], threshold)
        m["n_fake_family"] = int(((labels == 1) & (fams == fam)).sum())
        out[fam] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition", required=True,
                    help="a condition name (canonical or TC2 spelling), a comma "
                         "list, 'group:<name>', or 'all'")
    ap.add_argument("--output", required=True,
                    help="a .json file for a single condition, otherwise a directory")
    ap.add_argument("--resume", action="store_true",
                    help="skip conditions whose completion marker still matches")
    ap.add_argument("--data-root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--cache-dir", default=str(REPO / "data" / "cache" / "eval"))
    ap.add_argument("--denylist-dir", default=str(REPO / "data" / "denylist"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--protected-run", action="store_true",
                    help="required when the manifest is the protected benchmark")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from track5.data.denylist import count_protected
    from track5.eval.matrix import prepare_atom_rows
    from track5.eval.metrics import all_metrics
    from track5.infer import calibrated_prob, load_checkpoint
    from track5.models.preprocess import eval_crop, to_tensor
    from track5.train.checkpoint import atomic_write_json
    from track5.transforms.eval_atoms import ATOMS_VERSION
    from track5.utils.config import config_hash, load_config
    from track5.utils.hashing import file_sha256

    conditions = resolve_conditions(args.condition)
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 17))

    out = Path(args.output)
    single_file = len(conditions) == 1 and out.suffix == ".json"
    out_dir = out.parent if single_file else out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_parquet(args.manifest)
    if args.limit:
        manifest_df = manifest_df.head(args.limit)
    protection = count_protected(manifest_df, args.denylist_dir)
    is_protected = protection["total_hits"] > 0 or protection["denied_rows"] > 0
    print(f"[eval] manifest {args.manifest}: {len(manifest_df)} rows, "
          f"protection {protection}", flush=True)
    if is_protected and not args.protected_run:
        print("ERROR: this manifest contains protected-set rows. The protected "
              "benchmark is a single frozen-model event - pass --protected-run "
              "deliberately (constraint C2, TC2 section 11).", file=sys.stderr)
        return 2
    if args.protected_run and not is_protected:
        print("[eval] NOTE --protected-run given but no protected rows were "
              "detected; check --denylist-dir and the manifest", flush=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, calib, crop, ckpt_meta = load_checkpoint(args.checkpoint, device)
    threshold = calib["threshold"]
    manifest_sha = file_sha256(args.manifest)
    base_fp = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_hash": config_hash(cfg),
        "manifest_sha256": manifest_sha,
        "atoms_version": ATOMS_VERSION,
        "calibration": calib, "crop": crop, "seed": seed,
        "limit": args.limit,
    }
    print(f"[eval] checkpoint={args.checkpoint} backbone={ckpt_meta['backbone']} "
          f"crop={crop} device={device} T={calib['temperature']} "
          f"alpha={calib['alpha']} threshold={threshold}", flush=True)
    print(f"[eval] {len(conditions)} condition(s) -> {out_dir}", flush=True)

    @torch.no_grad()
    def score_paths(paths):
        chunks = []
        for i in range(0, len(paths), args.batch_size):
            px = torch.stack([to_tensor(eval_crop(Image.open(p).convert("RGB"), crop))
                              for p in paths[i:i + args.batch_size]]).to(device)
            chunks.append(model(px).float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0)

    n_done = n_skipped = 0
    for cond in conditions:
        res_path = out if single_file else out_dir / f"{cond}.json"
        done_path = res_path.with_name(res_path.stem + ".done.json")
        fp = fingerprint({**base_fp, "condition": cond})

        if done_path.exists():
            try:
                marker = json.loads(done_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                marker = {}
            if marker.get("fingerprint") == fp and res_path.exists():
                if args.resume:
                    print(f"[eval] {cond}: complete, skipping (--resume)", flush=True)
                    n_skipped += 1
                    continue
                if args.protected_run:
                    print(f"ERROR: {cond} already completed for this protected run "
                          f"({done_path}). Refusing to overwrite; pass --resume to "
                          f"continue the run.", file=sys.stderr)
                    return 3
            elif args.resume:
                print(f"[eval] {cond}: marker does not match this run, re-running",
                      flush=True)

        rows, decode_errors = prepare_atom_rows(manifest_df, cond, args.cache_dir,
                                                args.data_root, seed)
        if not len(rows):
            print(f"[eval] {cond}: no usable images", file=sys.stderr, flush=True)
            continue
        logits = score_paths(list(rows["cache_path"]))
        scores = calibrated_prob(logits, calib)
        labels = rows["label"].to_numpy(dtype=int)

        result = {
            "condition": cond,
            "n": int(len(rows)),
            "overall": all_metrics(labels, scores, threshold),
            "per_generator": per_generator_metrics(rows, scores, threshold),
            "threshold": threshold,
            "calibration": calib,
            "score_is_calibrated_probability": True,
            "decode_errors": decode_errors,
            "n_decode_errors": len(decode_errors),
            "protected_run": bool(args.protected_run),
            "manifest_protection": protection,
            "receipt": {
                "fingerprint": fp, **base_fp,
                "manifest": str(args.manifest),
                "checkpoint": str(args.checkpoint),
                "backbone": ckpt_meta["backbone"],
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            "predictions": [
                {"image_id": r.sha256, "path": r.path, "label": int(r.label),
                 "generator_family": r.generator_family, "pred": float(s)}
                for r, s in zip(rows.itertuples(index=False), scores)
            ],
        }
        atomic_write_json(result, res_path)
        # the marker is written last, so a killed job never looks complete
        atomic_write_json({"condition": cond, "fingerprint": fp, "n": int(len(rows)),
                           "result": res_path.name}, done_path)
        m = result["overall"]
        print(f"[eval] {cond}: n={len(rows)} auroc={m['auroc']:.4f} "
              f"ap={m['ap']:.4f} bacc={m['bacc']:.4f} brier={m['brier']:.4f} "
              f"ece={m['ece']:.4f} ({len(decode_errors)} decode errors) "
              f"-> {res_path}", flush=True)
        n_done += 1

    print(f"[eval] done: {n_done} evaluated, {n_skipped} already complete", flush=True)
    return 0 if (n_done or n_skipped) else 4


if __name__ == "__main__":
    sys.exit(main())
