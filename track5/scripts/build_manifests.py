"""Build a manifest parquet for one source (read-only scan of data/raw).

STRICT BY DEFAULT: if any requested image is unavailable or unreadable the build
writes its receipt and then FAILS. `--allow-missing` is the explicit
development/sample mode and is never valid for a final manifest.

.venv/Scripts/python.exe scripts/build_manifests.py --source wildfake_csv \
    --out data/manifests/wildfake.parquet \
    --csvs ddim.csv,ddpm.csv,adm.csv --per-family-limit 20000

`wildfake_csv` indexes WildFake from its label CSVs (authoritative labels and
generator families) and resolves each row to a zip member only when the archive
is present. The DALL-E family is refused there by CSV/architecture/category/
family/path (constraint C2) and never reaches the manifest.

`coco_val_demo` builds the 4,998-image demonstration benchmark subset only; the
denylist separately covers all 5,000 images in the canonical val2017 archive.

Scope a strict build with --csvs and/or --per-family-limit: only rows inside the
scope are "requested", so an undownloaded family you did not ask for does not
fail the build.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["coco_train", "coco_val", "coco_val_demo", "wildfake",
                             "wildfake_csv", "sid_set"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-family-limit", type=int, default=0,
                    help="cap per generator family / real source (wildfake_csv); "
                         "PLAN D3 wants ~15-25k per fake family")
    ap.add_argument("--csvs", default="",
                    help="comma-separated label CSV names to include "
                         "(wildfake_csv only), e.g. ddim.csv,ddpm.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="development/sample mode: tolerate unavailable images "
                         "instead of failing the build")
    args = ap.parse_args()

    from track5.data.manifest import MissingImages, build_manifest

    csvs = [c.strip() for c in args.csvs.split(",") if c.strip()] or None
    try:
        df = build_manifest(args.source, args.data_root, args.out,
                            limit=args.limit, workers=args.workers,
                            append=args.append,
                            per_family_limit=args.per_family_limit,
                            csvs=csvs, allow_missing=args.allow_missing)
    except MissingImages as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    print(df.groupby(["source", "label", "generator_family"]).size())
    print(f"wrote {len(df)} rows to {args.out}"
          + ("  [DEVELOPMENT BUILD: --allow-missing]" if args.allow_missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
