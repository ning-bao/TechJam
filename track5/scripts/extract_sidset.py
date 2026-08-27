"""Extract the SID_Set rows our manifests use into individual files.

Reading a SID_Set image through the parquet path form decodes a whole row group
(~844 images) per call. That is fine for one sequential manifest pass and fatal
for shuffled training access, where it makes every __getitem__ decode hundreds
of megabytes. Extracted files turn each read back into one plain file read.

Original bytes are written verbatim, so sha256/phash in the manifest stay valid
and the native container (PNG vs JPEG) is preserved - normalization depends on
it. Manifest paths are rewritten in place to the derived location.

    .venv/Scripts/python.exe scripts/extract_sidset.py
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    # Default includes `sidset`: rebuilding the training set reads the SOURCE
    # manifests, so rewriting only train/dev/calib leaves parquet-form paths in
    # sidset.parquet that come straight back on the next rebuild - and each such
    # read decodes a whole ~844-image row group.
    ap.add_argument("--manifests", nargs="*",
                    default=["sidset", "train", "dev", "calib"])
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--out-dir", default=str(REPO / "data/derived/sidset"))
    ap.add_argument("--manifest-dir", default=str(REPO / "data/manifests"))
    args = ap.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq

    mdir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    wanted: dict = {}
    for stem in args.manifests:
        p = mdir / f"{stem}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        frames[stem] = df
        sid = df[df["source"] == "sid_set"]
        for row in sid.itertuples(index=False):
            shard, idx = row.path.split("#", 1)
            ext = "png" if str(row.format).lower() == "png" else "jpg"
            wanted.setdefault(shard, {})[int(idx)] = (row.sha256, ext)
    total = sum(len(v) for v in wanted.values())
    if not total:
        print("no sid_set rows to extract")
        return 0
    print(f"extracting {total} rows from {len(wanted)} shards", flush=True)

    data_root = Path(args.data_root)
    mapping: dict = {}
    written = skipped = 0
    for si, (shard, rows) in enumerate(sorted(wanted.items()), 1):
        pf = pq.ParquetFile(data_root / shard)
        row_base = 0
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            hits = [i for i in rows if row_base <= i < row_base + n]
            if hits:
                tbl = pf.read_row_group(g, columns=["image"])
                col = tbl.column("image")
                for i in hits:
                    sha, ext = rows[i]
                    dest = out_dir / f"{sha}.{ext}"
                    rel = f"sidset/{sha}.{ext}"
                    mapping[f"{shard}#{i}"] = rel
                    if dest.exists():
                        skipped += 1
                        continue
                    dest.write_bytes(col[i - row_base].as_py()["bytes"])
                    written += 1
                del tbl, col
            row_base += n
        if si % 25 == 0 or si == len(wanted):
            print(f"  shard {si}/{len(wanted)}: written={written} reused={skipped}",
                  flush=True)

    for stem, df in frames.items():
        mask = df["source"] == "sid_set"
        df.loc[mask, "path"] = df.loc[mask, "path"].map(
            lambda p: mapping.get(p, p))
        df.to_parquet(mdir / f"{stem}.parquet", index=False)
        print(f"rewrote {int(mask.sum())} paths in {stem}.parquet")

    (out_dir.parent / "sidset_extract.receipt.json").write_text(
        json.dumps({"requested": total, "written": written, "reused": skipped,
                    "unmapped": total - len(mapping)}, indent=1), encoding="utf-8")
    print(f"done: {written} written, {skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
