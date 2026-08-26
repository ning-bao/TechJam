"""Build the protected-set denylist (PLAN C2/D4) in two tables:

  data/denylist/denylist.parquet        (sha256, phash, reason)  -- content match
  data/denylist/protected_paths.parquet (path_key, reason)       -- path match

The content table needs the bytes, so it is built from COCO val2017.zip (present)
plus any protected-source manifest passed in. The path table is built from the
WildFake label CSVs and covers the whole DALL·E family (dalle2 Typical + dalle3
Advanced) plus WildFake's own COCO val2017 copies, none of which are downloaded.

.venv/Scripts/python.exe scripts/build_denylist.py --coco-val \
    [--manifests data/manifests/wildfake.parquet ...] [--workers 8]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def coco_val_rows(data_root: Path, workers: int) -> tuple[list, dict]:
    """(sha256, phash, reason) for every COCO val2017 image, straight from the zip.
    Also returns a coverage record: a partially hashed protected set is a silent
    C2 hole, so the caller refuses to write it by default."""
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from io import BytesIO

    import imagehash
    from PIL import Image

    from track5.utils.hashing import bytes_sha256

    zpath = data_root / "COCO" / "val2017.zip"
    if not zpath.exists():
        print(f"[denylist] {zpath} missing -> no COCO val2017 content hashes",
              file=sys.stderr)
        return [], {"source": str(zpath), "present": False, "expected": 0,
                    "hashed": 0, "failed": [], "complete": False}
    try:
        zf = zipfile.ZipFile(zpath, "r")
        infos = [i for i in zf.infolist()
                 if not i.is_dir() and Path(i.filename).suffix.lower()
                 in {".jpg", ".jpeg", ".png"}]
    except (zipfile.BadZipFile, OSError) as e:
        print(f"[denylist] {zpath} unreadable ({type(e).__name__}: {e}) - still "
              f"downloading?", file=sys.stderr)
        return [], {"source": str(zpath), "present": True, "expected": 0,
                    "hashed": 0, "failed": [], "complete": False,
                    "error": f"{type(e).__name__}: {e}"}

    failed = []

    def one(info):
        try:
            data = zf.read(info)
            img = Image.open(BytesIO(data))
            img.load()
            return {"sha256": bytes_sha256(data),
                    "phash": str(imagehash.phash(img.convert("RGB"))),
                    "reason": "coco_val2017"}
        except Exception as e:
            failed.append({"member": info.filename, "error": f"{type(e).__name__}: {e}"})
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = [r for r in ex.map(one, infos) if r is not None]
    for f in failed[:10]:
        print(f"[denylist] skip {f['member']}: {f['error']}", file=sys.stderr)
    print(f"[denylist] COCO val2017: hashed {len(rows)}/{len(infos)} images",
          file=sys.stderr)
    return rows, {"source": str(zpath), "present": True, "expected": len(infos),
                  "hashed": len(rows), "failed": failed,
                  "complete": bool(infos) and not failed,
                  "member_names": [Path(i.filename).name for i in infos]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="*", default=[])
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--coco-val", action="store_true",
                    help="hash COCO val2017.zip directly (no manifest needed)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "data/denylist/denylist.parquet"))
    ap.add_argument("--out-paths",
                    default=str(REPO / "data/denylist/protected_paths.parquet"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="write an incomplete content denylist anyway (downloads "
                         "still running); the receipt records the gap and the "
                         "training gate keeps refusing until it is re-run")
    args = ap.parse_args()

    import json

    import pandas as pd

    from track5.data.denylist import build_denylist_df, build_protected_paths

    data_root = Path(args.data_root)

    parts = []
    coverage = {"coco_val2017": {"attempted": False}}
    if args.manifests:
        parts.append(build_denylist_df([pd.read_parquet(p) for p in args.manifests]))
    if args.coco_val:
        rows, coverage["coco_val2017"] = coco_val_rows(data_root, args.workers)
        coverage["coco_val2017"]["attempted"] = True
        if rows:
            parts.append(pd.DataFrame(rows))
    deny = (pd.concat(parts, ignore_index=True).drop_duplicates(subset=["sha256"])
            if parts else pd.DataFrame(columns=["sha256", "phash", "reason"]))

    paths = build_protected_paths(data_root)
    complete = (not args.coco_val) or coverage["coco_val2017"]["complete"]

    # Item 5: the denylist covers every image in the canonical val2017 archive;
    # the demonstration benchmark is a 4,998-image subset of it. Images in the
    # archive but outside the benchmark are still denied from training.
    from track5.data.wildfake_csv import coco_val2017_demo_ids, protected_summary

    demo_ids = coco_val2017_demo_ids(data_root)
    archive_ids = coverage["coco_val2017"].get("member_names") or []
    coverage["coco_val2017"].pop("member_names", None)
    coverage["coco_val2017_split"] = {
        "archive_images": len(archive_ids),
        "demonstration_benchmark": len(demo_ids),
        "denied_only_not_evaluated": sorted(set(archive_ids) - demo_ids),
        "in_benchmark_but_absent_from_archive": sorted(demo_ids - set(archive_ids))
        if archive_ids else [],
    }
    coverage["wildfake"] = protected_summary(data_root)

    for out, df, label in ((Path(args.out), deny, "content hashes"),
                           (Path(args.out_paths), paths, "protected paths")):
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"wrote {len(df)} {label} to {out}")
        if len(df):
            print(df.groupby("reason").size().to_string())

    receipt = Path(args.out).with_name(Path(args.out).name + ".receipt.json")
    receipt.write_text(json.dumps(
        {"complete": complete, "allow_partial": args.allow_partial,
         "n_content_hashes": int(len(deny)), "n_protected_paths": int(len(paths)),
         "coverage": coverage}, indent=1), encoding="utf-8")
    print(f"wrote coverage receipt to {receipt}")
    split = coverage["coco_val2017_split"]
    print(f"COCO val2017: {split['archive_images']} archive images denied, "
          f"{split['demonstration_benchmark']} of them are the demonstration "
          f"benchmark, {len(split['denied_only_not_evaluated'])} denied-only")
    print(f"WildFake DALL-E: {coverage['wildfake']}")

    if deny.empty and paths.empty:
        print("ERROR: denylist is empty - C2 cannot be enforced", file=sys.stderr)
        return 1
    if not complete:
        cov = coverage["coco_val2017"]
        msg = (f"COCO val2017 content denylist is INCOMPLETE: hashed "
               f"{cov['hashed']}/{cov['expected']} members. val2017.zip is most "
               f"likely still downloading. Re-run once the archive is final - a "
               f"partial protected-set denylist is a silent C2 hole.")
        if not args.allow_partial:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2
        print(f"WARNING: {msg} (--allow-partial)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
