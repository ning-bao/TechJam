"""CRC-level integrity verification for every raw archive.

data/raw is READ-ONLY: this only reads. `scripts/data_status.py` reads the zip
index (namelist) and therefore reports a truncated/damaged archive as OK; this
decompresses every member and checks its CRC, which is the only way to see the
damage. Writes reports/deep_verification.json.

    .venv/Scripts/python.exe scripts/verify_archives.py [--workers 6]
    .venv/Scripts/python.exe scripts/verify_archives.py --archive COCO/train2017.zip
"""

import argparse
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
CHUNK = 1 << 20
MAX_EXAMPLES = 20


def verify_zip(path_str: str) -> dict:
    """Read every member; a member is bad if decompression or CRC fails."""
    path = Path(path_str)
    out = {
        "archive": path.relative_to(RAW).as_posix(),
        "gb": round(path.stat().st_size / (1 << 30), 2),
        "members": 0, "ok": 0, "bad": 0,
        "bad_names": [], "examples": [], "fatal": None,
    }
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            out["members"] = len(infos)
            for info in infos:
                if info.is_dir():
                    out["ok"] += 1
                    continue
                try:
                    with zf.open(info) as fh:  # CRC is checked on full read
                        while fh.read(CHUNK):
                            pass
                    out["ok"] += 1
                except Exception as e:
                    out["bad"] += 1
                    out["bad_names"].append(info.filename)
                    if len(out["examples"]) < MAX_EXAMPLES:
                        out["examples"].append(
                            {"name": info.filename, "error": f"{type(e).__name__}: {e}"})
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}"
    out["bad_pct"] = round(100.0 * out["bad"] / out["members"], 2) if out["members"] else 0.0
    return out


def verify_parquet(path_str: str) -> dict:
    path = Path(path_str)
    out = {"archive": path.relative_to(RAW).as_posix(),
           "gb": round(path.stat().st_size / (1 << 30), 2),
           "members": 0, "ok": 0, "bad": 0, "bad_names": [], "examples": [], "fatal": None}
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        out["members"] = pf.metadata.num_row_groups
        for i in range(pf.metadata.num_row_groups):
            try:
                pf.read_row_group(i)
                out["ok"] += 1
            except Exception as e:
                out["bad"] += 1
                out["bad_names"].append(f"row_group_{i}")
                if len(out["examples"]) < MAX_EXAMPLES:
                    out["examples"].append(
                        {"name": f"row_group_{i}", "error": f"{type(e).__name__}: {e}"})
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}"
    out["bad_pct"] = round(100.0 * out["bad"] / out["members"], 2) if out["members"] else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--archive", default="", help="verify one archive (path under data/raw)")
    ap.add_argument("--skip-parquet", action="store_true")
    ap.add_argument("--out", default=str(REPO / "reports" / "deep_verification.json"))
    args = ap.parse_args()

    if args.archive:
        targets = [RAW / args.archive]
    else:
        targets = sorted(RAW.rglob("*.zip"))
        if not args.skip_parquet:
            targets += sorted(RAW.rglob("*.parquet"))
    targets = [t for t in targets if t.is_file()]
    if not targets:
        print("no archives found", file=sys.stderr)
        return 2

    print(f"verifying {len(targets)} archives with {args.workers} workers", flush=True)
    results = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify_parquet if t.suffix == ".parquet" else verify_zip,
                          str(t)): t for t in targets}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            flag = "FATAL" if r["fatal"] else ("BAD " if r["bad"] else "ok  ")
            print(f"[{flag}] {r['archive']}: {r['bad']}/{r['members']} bad "
                  f"({r['bad_pct']}%) {r['fatal'] or ''}", flush=True)
            results.sort(key=lambda x: x["archive"])
            out_path.write_text(json.dumps(
                {"archives": results,
                 "summary": {
                     "n_archives": len(results),
                     "n_damaged": sum(1 for x in results if x["bad"] or x["fatal"]),
                     "total_bad_members": sum(x["bad"] for x in results),
                 }}, indent=1), encoding="utf-8")

    damaged = [r for r in results if r["bad"] or r["fatal"]]
    print(f"\n{len(damaged)} damaged of {len(results)} archives; report {out_path}")
    for r in sorted(damaged, key=lambda x: -x["bad_pct"]):
        print(f"  {r['archive']}: {r['bad']}/{r['members']} ({r['bad_pct']}%)")
    return 1 if damaged else 0


if __name__ == "__main__":
    sys.exit(main())
