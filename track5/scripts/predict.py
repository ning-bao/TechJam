"""STABLE submission CLI (INTERFACES §7) — the contract never breaks.

.venv/Scripts/python.exe scripts/predict.py --input <dir|list.txt|list.csv> \
    --checkpoint runs/x/best.pt --out preds.json [--batch-size 32] \
    [--device auto|cuda|cpu] [--tta none|crop5]

Output: score = calibrated sigmoid((z+alpha)/T) = p(AIGC); label = score>=threshold.
Decode failures land in "errors", NEVER as a silent 0.5 prediction.
"""

import argparse
import json
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_inputs(spec: str) -> list[str]:
    p = Path(spec)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if p.suffix.lower() == ".txt":
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if p.suffix.lower() == ".csv":
        import pandas as pd

        return [str(x) for x in pd.read_csv(p)["path"]]
    raise SystemExit(f"--input must be a directory, .txt or .csv (got {spec})")


def five_crops(img, size: int):
    from track5.models.preprocess import _pad_reflect  # padding rule shared with eval_crop
    import numpy as np
    from PIL import Image as PILImage

    arr = _pad_reflect(__import__("numpy").asarray(img.convert("RGB")), size)
    h, w = arr.shape[:2]
    boxes = [((h - size) // 2, (w - size) // 2), (0, 0), (0, w - size),
             (h - size, 0), (h - size, w - size)]
    return [PILImage.fromarray(arr[y:y + size, x:x + size]) for y, x in boxes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", default="none", choices=["none", "crop5"])
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from track5.infer import calibrated_prob, load_checkpoint
    from track5.models.preprocess import eval_crop, to_tensor

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, calib, crop, meta = load_checkpoint(args.checkpoint, device)
    T, alpha, threshold = calib["temperature"], calib["alpha"], calib["threshold"]
    model_hash, config_hash = meta["model_hash"], meta["config_hash"]

    paths = collect_inputs(args.input)
    predictions, errors = [], []
    batch_tensors, batch_meta = [], []  # meta: (path, n_views)

    @torch.no_grad()
    def flush():
        if not batch_tensors:
            return
        px = torch.stack(batch_tensors).to(device)
        z = model(px).float().cpu().numpy()
        probs = calibrated_prob(z, calib)
        i = 0
        for path, n_views in batch_meta:
            p = float(np.mean(probs[i:i + n_views]))
            i += n_views
            predictions.append({"path": path, "score": p, "label": int(p >= threshold)})
        batch_tensors.clear()
        batch_meta.clear()

    for path in paths:
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
            views = five_crops(img, crop) if args.tta == "crop5" else [eval_crop(img, crop)]
            tensors = [to_tensor(v) for v in views]
        except Exception as e:
            errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
            continue
        batch_tensors.extend(tensors)
        batch_meta.append((path, len(tensors)))
        if len(batch_tensors) >= args.batch_size:
            flush()
    flush()

    out = {
        "meta": {"model_hash": model_hash, "config_hash": config_hash,
                 "temperature": T, "alpha": alpha, "threshold": threshold,
                 "n_ok": len(predictions), "n_err": len(errors)},
        "predictions": predictions,
        "errors": errors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(predictions)} predictions, {len(errors)} errors")
    if errors:
        for e in errors[:10]:
            print(f"  ERROR {e['path']}: {e['error']}", file=sys.stderr)
    sys.exit(0 if predictions else 2)


if __name__ == "__main__":
    main()
