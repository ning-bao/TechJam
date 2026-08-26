"""Read-only status report of data/raw downloads. Never writes under data/raw.

.venv/Scripts/python.exe scripts/data_status.py
"""

import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


def dir_stats(path: Path):
    files = [f for f in path.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def zip_state(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                return "EMPTY"
            zf.read(names[0])
            zf.read(names[-1])
        return f"OK ({len(names)} members)"
    except (zipfile.BadZipFile, OSError, KeyError, NotImplementedError) as e:
        return f"INCOMPLETE/downloading ({type(e).__name__})"


def main():
    report = {}
    gb = 1 << 30

    coco = RAW / "COCO"
    if coco.exists():
        n, size = dir_stats(coco)
        zips = {z.name: zip_state(z) for z in sorted(coco.glob("*.zip"))}
        report["COCO"] = {"files": n, "gb": round(size / gb, 2), "zips": zips}

    wf = RAW / "WildFake"
    if wf.exists():
        n, size = dir_stats(wf)
        fams = {}
        img_root = wf / "Images"
        if img_root.exists():
            for cat in sorted(p for p in img_root.iterdir() if p.is_dir()):
                inner = [c.name for c in sorted(cat.iterdir())]
                fams[cat.name] = inner
        zstates = {z.relative_to(wf).as_posix(): zip_state(z)
                   for z in sorted(wf.rglob("*.zip"))}
        report["WildFake"] = {"files": n, "gb": round(size / gb, 2),
                              "families": fams, "zips": zstates}

    sid = RAW / "SID_Set"
    if sid.exists():
        n, size = dir_stats(sid)
        shards = sorted((sid / "data").glob("*.parquet")) if (sid / "data").exists() else []
        rows = 0
        bad = 0
        try:
            import pyarrow.parquet as pq

            for s in shards:
                try:
                    rows += pq.ParquetFile(s).metadata.num_rows
                except Exception:
                    bad += 1
        except ImportError:
            rows = -1
        report["SID_Set"] = {"files": n, "gb": round(size / gb, 2),
                             "shards": len(shards), "rows": rows,
                             "bad_shards": bad}

    total_gb = sum(v["gb"] for v in report.values())
    report["_total_gb"] = round(total_gb, 2)

    print(f"{'source':<10} {'GB':>8}  detail")
    for k, v in report.items():
        if k.startswith("_"):
            continue
        detail = ""
        if k == "COCO":
            detail = "; ".join(f"{n}: {s}" for n, s in v["zips"].items())
        elif k == "WildFake":
            ok = sum("OK" in s for s in v["zips"].values())
            detail = f"{ok}/{len(v['zips'])} zips OK; cats: {list(v['families'])}"
        elif k == "SID_Set":
            detail = f"{v['shards']} shards, {v['rows']} rows, {v['bad_shards']} bad"
        print(f"{k:<10} {v['gb']:>8.2f}  {detail}")
    print(f"{'TOTAL':<10} {total_gb:>8.2f}")

    out = REPO / "reports" / "data_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
