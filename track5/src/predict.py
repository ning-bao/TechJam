"""Organiser submission format (TC2 guide §6) — one JSON record per image:

    {"image_path": "relative/or/required/path.jpg", "pred": 0.873214}

`pred` is the calibrated probability of AIGC, never a raw logit.

    python -u -m src.predict --checkpoint runs/x/best.pt \
        --manifest data/manifests/dev.parquet --output preds.json
    python -u -m src.predict --checkpoint runs/x/best.pt \
        --input /path/to/images --output preds.jsonl --format jsonl

DEFAULT OUTPUT IS A JSON ARRAY of those records. The organiser has only said
"JSON", and a JSON array parses with any JSON reader while JSON Lines does not;
`--format jsonl` keeps the one-record-per-line form for streaming consumers.

`image_path` is the manifest `path` value, the filelist line, or — for a
directory input — the path relative to that directory. Exactly one record per
input path.

Decode failures: by default a failed image still emits a record with pred=0.5 so
the row count matches, written loudly to <output>.errors.json and to stderr —
never silently scored. That recovery mode is for exploratory runs only.
`--strict` turns any decode failure into a non-zero exit, and `--protected-run`
implies --strict, refuses to overwrite a completed run, and writes a receipt.

INTERFACES §7 (`scripts/predict.py`) stays the repo-internal contract; this is
the submission-shaped sibling and shares its calibration code.
"""

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ERROR_PRED = 0.5


def collect_from_input(spec: str) -> list[tuple[str, str]]:
    """-> [(image_path as reported, absolute/loadable path)]."""
    p = Path(spec)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        return [(f.relative_to(p).as_posix(), str(f)) for f in files]
    if p.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        return [(ln, ln) for ln in lines]
    if p.suffix.lower() == ".csv":
        import pandas as pd

        return [(str(x), str(x)) for x in pd.read_csv(p)["path"]]
    raise SystemExit(f"--input must be a directory, .txt or .csv (got {spec})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--input", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--format", default="json", choices=["json", "jsonl"],
                    help="json: a JSON array of records (default); "
                         "jsonl: one record per line")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any image fails to decode")
    ap.add_argument("--protected-run", action="store_true",
                    help="final/protected inference: implies --strict, refuses to "
                         "overwrite a completed run, writes a receipt")
    args = ap.parse_args()

    if bool(args.manifest) == bool(args.input):
        print("ERROR: pass exactly one of --manifest / --input", file=sys.stderr)
        return 2

    strict = args.strict or args.protected_run
    out_path = Path(args.output)
    done_path = out_path.with_name(out_path.name + ".done.json")
    if args.protected_run and done_path.exists():
        print(f"ERROR: this protected run already completed ({done_path}). "
              f"Refusing to overwrite - a protected run is a single frozen-model "
              f"event (TC2 section 11).", file=sys.stderr)
        return 5
    if args.protected_run and not args.strict:
        print("[predict] --protected-run implies --strict: any decode failure "
              "fails the job rather than emitting a recovery score", flush=True)

    import torch
    from PIL import Image

    from track5.data.resolve import resolve_image_bytes
    from track5.infer import calibrated_prob, load_checkpoint
    from track5.models.preprocess import eval_crop, to_tensor
    from track5.train.checkpoint import atomic_write_json, atomic_write_text

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, calib, crop, meta = load_checkpoint(args.checkpoint, device)
    print(f"[predict] checkpoint={args.checkpoint} backbone={meta['backbone']} "
          f"crop={crop} device={device} T={calib['temperature']} "
          f"alpha={calib['alpha']} -> {args.output}", flush=True)

    if args.manifest:
        import pandas as pd

        df = pd.read_parquet(args.manifest)
        denied = int((df["split"] == "denied").sum()) if "split" in df else 0
        if denied:
            print(f"[predict] NOTE {denied} denylisted rows present in the manifest "
                  f"- scored as given, they are protected only for train/calib",
                  flush=True)
        items = [(str(r.path), None) for r in df.itertuples(index=False)]
        loader = lambda rel: resolve_image_bytes(args.data_root, rel)  # noqa: E731
    else:
        items = collect_from_input(args.input)
        loader = None

    records, errors = [], []
    batch, batch_keys = [], []

    @torch.no_grad()
    def flush():
        if not batch:
            return
        z = model(torch.stack(batch).to(device)).float().cpu().numpy()
        for key, p in zip(batch_keys, calibrated_prob(z, calib)):
            records.append({"image_path": key, "pred": float(min(1.0, max(0.0, p)))})
        batch.clear()
        batch_keys.clear()

    for reported, concrete in items:
        try:
            if loader is not None:
                img = Image.open(BytesIO(loader(reported)))
            else:
                img = Image.open(concrete)
            img.load()
            batch.append(to_tensor(eval_crop(img.convert("RGB"), crop)))
            batch_keys.append(reported)
        except Exception as e:
            errors.append({"image_path": reported, "error": f"{type(e).__name__}: {e}"})
            records.append({"image_path": reported, "pred": ERROR_PRED})
            print(f"[predict] DECODE FAILURE {reported}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        if len(batch) >= args.batch_size:
            flush()
    flush()

    out = out_path
    if args.format == "jsonl":
        atomic_write_text("".join(json.dumps(r) + "\n" for r in records), out)
    else:
        atomic_write_json(records, out)

    err_path = out.with_name(out.name + ".errors.json")
    atomic_write_json({"n_records": len(records), "n_errors": len(errors),
                       "error_pred": None if strict else ERROR_PRED,
                       "strict": strict, "protected_run": bool(args.protected_run),
                       "checkpoint_meta": meta, "calibration": calib,
                       "errors": errors}, err_path)

    print(f"[predict] wrote {len(records)} records to {out} as "
          f"{'a JSON array' if args.format == 'json' else 'JSON Lines'} "
          f"({len(errors)} decode failures -> {err_path})", flush=True)
    if errors and strict:
        print(f"ERROR: {len(errors)} decode failures in a "
              f"{'protected' if args.protected_run else 'strict'} run; no "
              f"recovery scores are acceptable here", file=sys.stderr)
        return 3
    if not records:
        return 4
    if args.protected_run:
        atomic_write_json(
            {"output": out.name, "format": args.format, "n_records": len(records),
             "n_errors": len(errors), "strict": True,
             "checkpoint_meta": meta, "calibration": calib,
             "input": args.manifest or args.input,
             "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, done_path)
        print(f"[predict] protected run receipt -> {done_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
