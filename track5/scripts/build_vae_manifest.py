"""Manifest for the VAE reconstructions (PLAN D3 Fake #3).

Each reconstruction is content-identical to the real it came from, so the pair
MUST land in the same split — a recon in dev whose source real is in train would
leak content across the split boundary and flatter the dev number. The manifest
therefore carries `src_sha256`, and scripts/build_training_set.py splits on that
as the grouping key.

    .venv/Scripts/python.exe scripts/build_vae_manifest.py \
        --sources data/manifests/vae_sources.parquet
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAMILY = {"sd15": "vae_sd15", "sdxl": "vae_sdxl"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(REPO / "data/manifests/vae_sources.parquet"))
    ap.add_argument("--recon-root", default=str(REPO / "data/derived/vae_recon"))
    ap.add_argument("--out", default=str(REPO / "data/manifests/fake_vae.parquet"))
    ap.add_argument("--vaes", default="sd15,sdxl")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import pandas as pd

    from track5.data.manifest import SCHEMA, row_from_bytes

    src = pd.read_parquet(args.sources)
    by_prefix = {r.sha256[:16]: r.sha256 for r in src.itertuples(index=False)}
    root = Path(args.recon_root)

    jobs = []
    for vae in [v.strip() for v in args.vaes.split(",") if v.strip()]:
        d = root / vae
        if not d.exists():
            print(f"[vae] {d} missing, skipping", file=sys.stderr)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            parent = by_prefix.get(f.stem)
            if parent is None:      # not from this source list
                continue
            jobs.append((f, vae, parent))
    if not jobs:
        print("ERROR: no reconstructions found", file=sys.stderr)
        return 2
    print(f"indexing {len(jobs)} reconstructions", flush=True)

    failures: list = []

    def work(job):
        f, vae, parent = job
        try:
            rel = f"vae_recon/{vae}/{f.name}"   # relative to data/derived
            row = row_from_bytes(rel, f.read_bytes(), 1, "vae_recon", FAMILY[vae])
            row["src_sha256"] = parent
            return row
        except Exception as e:
            failures.append({"path": str(f), "error": f"{type(e).__name__}: {e}"})
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, jobs):
            if r is not None:
                rows.append(r)

    cols = list(SCHEMA) + ["src_sha256"]
    df = pd.DataFrame(rows)[cols].astype({**SCHEMA, "src_sha256": "str"})
    out = Path(args.out)
    df.to_parquet(out, index=False)
    out.with_name(out.name + ".receipt.json").write_text(json.dumps(
        {"rows": int(len(df)), "unreadable": len(failures),
         "by_family": {str(k): int(v) for k, v in
                       df["generator_family"].value_counts().to_dict().items()},
         "unreadable_examples": failures[:20]}, indent=1), encoding="utf-8")
    print(df.groupby("generator_family").size().to_string())
    print(f"wrote {len(df)} rows to {out} ({len(failures)} unreadable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
