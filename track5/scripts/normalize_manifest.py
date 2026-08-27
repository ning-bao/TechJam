"""Materialize post-normalization metadata so the D4 probes measure reality.

The shortcut probes read width/height/format/file_bytes/jpeg_quality. On a raw
manifest those describe the delivered container, which is NOT what the model
sees: the model gets a crop x crop native-pixel crop that has been through the
class-independent re-encode in track5.data.normalize. Probing the raw manifest
answers the wrong question in both directions - it flags leaks the model cannot
reach, and it would miss a leak introduced by normalization itself.

This writes a copy of the manifest whose metadata columns describe the
normalized crop, for scripts/probe_gate.py to consume unchanged.

    .venv/Scripts/python.exe scripts/normalize_manifest.py \
        --manifest data/manifests/train.parquet --sample 20000
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--sample", type=int, default=20000,
                    help="0 = whole manifest; the gate is a statistical check, "
                         "so a sample is enough")
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from PIL import Image

    from track5.data.normalize import normalized_stats
    from track5.data.resolve import resolve_image_bytes
    from track5.models.preprocess import eval_crop

    df = pd.read_parquet(args.manifest)
    df = df[df["split"] != "denied"].reset_index(drop=True)
    if args.sample and len(df) > args.sample:
        rng = np.random.default_rng(args.seed)
        # stratified by label so the probe sees a balanced sample
        pos = np.concatenate([
            rng.permutation(np.flatnonzero((df["label"] == y).to_numpy()))[:args.sample // 2]
            for y in (0, 1)])
        df = df.iloc[np.sort(pos)].reset_index(drop=True)

    data_root = Path(args.data_root)
    failures: list = []

    def work(row):
        try:
            img = Image.open(BytesIO(resolve_image_bytes(data_root, row.path)))
            img = eval_crop(img.convert("RGB"), args.crop)
            return normalized_stats(img, row.sha256, row.format, args.seed)
        except Exception as e:
            failures.append({"path": row.path, "error": f"{type(e).__name__}: {e}"})
            return None

    # Chunked: ex.map buffers every result, which grew past available RAM and
    # got the process killed on the full manifest.
    rows = list(df.itertuples(index=False))
    stats: list = []
    chunk = 4000
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for start in range(0, len(rows), chunk):
            stats.extend(ex.map(work, rows[start:start + chunk]))
            print(f"  {min(start + chunk, len(rows))}/{len(rows)}", flush=True)

    ok = [i for i, s in enumerate(stats) if s is not None]
    out_df = df.iloc[ok].reset_index(drop=True)
    for col in ("width", "height", "format", "file_bytes", "jpeg_quality",
                "n_recompress"):
        out_df[col] = [stats[i][col] for i in ok]

    out = Path(args.out) if args.out else Path(args.manifest).with_name(
        Path(args.manifest).stem + "_normalized.parquet")
    out_df.to_parquet(out, index=False)
    print(f"wrote {len(out_df)} normalized rows to {out} ({len(failures)} failed)")
    for f in failures[:5]:
        print(f"  {f['path']}: {f['error']}", file=sys.stderr)
    print(out_df.groupby("label")[["file_bytes", "jpeg_quality"]].median().to_string())
    print("format by class:",
          out_df.groupby("label")["format"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
