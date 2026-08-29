"""PLAN D7: fit temperature + logit bias, then freeze ONE decision threshold.

Two separate fits, on two separate splits, in this order:

  1. T and alpha are fitted on a DEPLOYMENT MIXTURE built from the calib split --
     half the images clean, half carrying one eval transform each, so the scores
     are calibrated for the conditions the model will actually meet rather than
     for pristine input only.
  2. tau is frozen on CLEAN dev: the threshold maximizing bAcc subject to
     FPR <= 5%. It is never refit per transform -- at inference we do not know
     which transform was applied, so committing to one operating point in
     advance is the only honest choice.

Constraint C2: neither fit may touch the protected set, not even its unlabeled
images. calib (20,019 rows) has never been trained, selected or evaluated on.

    .venv/Scripts/python.exe scripts/calibrate_model.py \
        --checkpoint runs/dinov3l448_d4/epoch1_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib-manifest", default=str(REPO / "data/manifests/calib.parquet"))
    ap.add_argument("--dev-manifest", default=str(REPO / "data/manifests/dev_eval2k.parquet"))
    ap.add_argument("--cache", default=str(REPO / "data/cache/calib"))
    ap.add_argument("--dev-cache", default=str(REPO / "data/cache/eval"))
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--n", type=int, default=6000, help="calib images (balanced)")
    ap.add_argument("--clean-frac", type=float, default=0.5)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="", help="checkpoint to write (default: in place)")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from track5.eval.calibrate import calibrated_scores, fit_temperature_alpha
    from track5.eval.matrix import prepare_atom_rows
    from track5.eval.metrics import all_metrics, bacc
    from track5.eval.threshold import freeze_threshold
    from track5.models import build_model
    from track5.models.preprocess import eval_crop, to_tensor
    from track5.transforms.eval_atoms import EVAL_15

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    crop = int(ckpt["config"].get("data", {}).get("crop", 448))

    @torch.no_grad()
    def logits_for(paths):
        out = []
        for i in range(0, len(paths), args.batch_size):
            batch = [to_tensor(eval_crop(Image.open(p).convert("RGB"), crop))
                     for p in paths[i:i + args.batch_size]]
            out.append(model(torch.stack(batch).to(device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # ---- 1. deployment mixture from calib -----------------------------------
    calib = pd.read_parquet(args.calib_manifest)
    calib = calib[calib["split"] != "denied"]
    rng = np.random.default_rng(args.seed)
    pos = np.concatenate([
        rng.permutation(np.flatnonzero((calib["label"] == y).to_numpy()))[:args.n // 2]
        for y in (0, 1)])
    sub = calib.iloc[np.sort(pos)].reset_index(drop=True)

    # each image carries exactly ONE condition: half clean, half transformed
    transforms = [a for a in EVAL_15 if a != "clean"]
    assign = np.where(rng.random(len(sub)) < args.clean_frac, "clean",
                      rng.choice(transforms, size=len(sub)))
    sub["atom"] = assign
    print(f"[calib] mixture over {len(sub)} images: "
          f"{int((assign=='clean').sum())} clean / "
          f"{int((assign!='clean').sum())} transformed", flush=True)

    all_paths, all_y = [], []
    for atom, grp in sub.groupby("atom"):
        rows, errs = prepare_atom_rows(grp, atom, args.cache, args.data_root, args.seed)
        if not len(rows):
            continue
        all_paths.extend(rows["cache_path"].tolist())
        all_y.extend(rows["label"].tolist())
        print(f"  {atom}: {len(rows)} ({len(errs)} err)", flush=True)
    y_cal = np.asarray(all_y, dtype=int)
    z_cal = logits_for(all_paths)
    print(f"[calib] scored {len(z_cal)} images", flush=True)

    T, alpha = fit_temperature_alpha(z_cal, y_cal)
    print(f"\n[calib] T = {T:.4f}   alpha = {alpha:+.4f}")

    pre = all_metrics(y_cal, 1 / (1 + np.exp(-z_cal)), 0.5)
    post = all_metrics(y_cal, calibrated_scores(z_cal, T, alpha), 0.5)
    print(f"  mixture ECE  {pre['ece']:.4f} -> {post['ece']:.4f}")
    print(f"  mixture bAcc {pre['bacc']:.4f} -> {post['bacc']:.4f} (at 0.5)")

    # ---- 2. freeze tau on CLEAN dev -----------------------------------------
    dev = pd.read_parquet(args.dev_manifest)
    dev = dev[dev["split"] != "denied"]
    rows, errs = prepare_atom_rows(dev, "clean", args.dev_cache, args.data_root, args.seed)
    z_dev = logits_for(rows["cache_path"].tolist())
    y_dev = rows["label"].to_numpy(dtype=int)
    s_dev = calibrated_scores(z_dev, T, alpha)
    tau = freeze_threshold(y_dev, s_dev, max_fpr=args.max_fpr)
    fpr = float((s_dev[y_dev == 0] >= tau).mean())
    tpr = float((s_dev[y_dev == 1] >= tau).mean())
    print(f"\n[calib] tau = {tau:.6f} (clean dev, n={len(y_dev)})")
    print(f"  FPR {fpr:.4f} (cap {args.max_fpr})  TPR {tpr:.4f}  "
          f"bAcc {bacc(y_dev, s_dev, tau):.4f}")

    # ---- 3. write it into the checkpoint ------------------------------------
    ckpt["calibration"] = {"temperature": float(T), "alpha": float(alpha),
                           "threshold": float(tau)}
    out = Path(args.out) if args.out else Path(args.checkpoint)
    torch.save(ckpt, out)
    receipt = {
        "checkpoint": str(out), "temperature": float(T), "alpha": float(alpha),
        "threshold": float(tau), "max_fpr": args.max_fpr,
        "calib_n": int(len(y_cal)), "clean_frac": args.clean_frac,
        "mixture_ece_before": pre["ece"], "mixture_ece_after": post["ece"],
        "clean_dev_n": int(len(y_dev)), "clean_dev_fpr": fpr, "clean_dev_tpr": tpr,
        "fitted_on": "calib split (deployment mixture); tau on clean dev",
        "protected_set_used": False,
    }
    Path(REPO / "reports/calibration.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print(f"\nwrote calibration into {out}")
    print("receipt -> reports/calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
