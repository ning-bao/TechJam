"""SID_Set manifest, streamed row-group by row-group.

The generic builder in track5.data.manifest materializes every entry before
hashing; SID_Set entries carry ~1 MB of image bytes each, so 150k of them
exhaust RAM. This walks one row group at a time (a few hundred MB), hashes it
with a thread pool, and drops it before moving on.

Labels follow the dataset's own convention: 0 = real, 1 = fully synthetic
(FLUX), 2 = tampered (skipped - not our real/fake definition).

    .venv/Scripts/python.exe scripts/build_sidset_manifest.py --max-per-label 40000
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--out", default=str(REPO / "data/manifests/sidset.parquet"))
    ap.add_argument("--max-per-label", type=int, default=40000)
    ap.add_argument("--min-side", type=int, default=0,
                    help="skip images whose short side is below this (0 = keep all)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq

    from track5.data.manifest import SCHEMA, row_from_bytes

    data_root = Path(args.data_root)
    shard_dir = data_root / "SID_Set" / "data"
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        print(f"ERROR: no shards under {shard_dir}", file=sys.stderr)
        return 2

    rows: list = []
    kept = {0: 0, 1: 0}
    skipped_small = 0
    failures: list = []
    done = False

    for shard in shards:
        if done:
            break
        rel = shard.relative_to(data_root).as_posix()
        try:
            pf = pq.ParquetFile(shard)
        except Exception as e:
            failures.append({"path": rel, "error": f"{type(e).__name__}: {e}"})
            continue
        row_base = 0
        for g in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(g, columns=["image", "label"])
            labels = tbl.column("label").to_pylist()
            images = tbl.column("image")
            jobs = []
            for i, lab in enumerate(labels):
                if lab == 2:
                    continue
                y = 0 if lab == 0 else 1
                if kept[y] >= args.max_per_label:
                    continue
                kept[y] += 1
                jobs.append((f"{rel}#{row_base + i}", y, images[i].as_py()["bytes"]))

            def work(job):
                path, y, data = job
                try:
                    return row_from_bytes(path, data, y, "sid_set",
                                          "flux" if y == 1 else "")
                except Exception as e:
                    failures.append({"path": path, "error": f"{type(e).__name__}: {e}"})
                    return None

            if jobs:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for r in ex.map(work, jobs):
                        if r is None:
                            continue
                        if args.min_side and min(r["width"], r["height"]) < args.min_side:
                            skipped_small += 1
                            continue
                        rows.append(r)
            del tbl, images, jobs
            row_base += len(labels)
            if all(kept[y] >= args.max_per_label for y in (0, 1)):
                done = True
                break
        print(f"[sidset] {rel}: kept real={kept[0]} fake={kept[1]}", flush=True)

    df = pd.DataFrame(rows, columns=list(SCHEMA)).astype(SCHEMA)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    receipt = {"source": "sid_set", "rows": int(len(df)),
               "kept_by_label": kept, "skipped_below_min_side": skipped_small,
               "min_side": args.min_side, "unreadable": len(failures),
               "unreadable_examples": failures[:20]}
    out.with_name(out.name + ".receipt.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print(df.groupby(["label", "generator_family"]).size().to_string())
    print(f"wrote {len(df)} rows to {out} ({len(failures)} unreadable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
