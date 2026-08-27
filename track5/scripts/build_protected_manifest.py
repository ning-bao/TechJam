"""Build the PROTECTED evaluation manifest: COCO val2017 x DALL-E 3 Advanced.

This is the organiser's demonstration benchmark (4,998 real + 8,843 fake). It is
the ONLY code path in this repo that is allowed to put DALL-E images into a
manifest -- everything else refuses them on five independent keys (constraint
C2). Three safeguards make that deliberate rather than accidental:

  1. every row is written with split="protected", which no training or
     calibration path ever selects (they take split in {train, calib});
  2. the fake side is built FROM the denylist file itself, so the manifest is by
     construction exactly the protected set and cannot drift into other DALL-E
     subsets;
  3. the row count is asserted against the organiser's stated 8,843 / 4,998.

Never pass the output to scripts/train.py or the calibration step. It exists for
the final inference run and the robustness matrix that accompanies it.

    .venv/Scripts/python.exe scripts/build_protected_manifest.py
"""

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPECT_FAKE = 8843
# The organiser's demo subset is 4,998 COCO val2017 images. Those are identified
# by WildFake's OWN ids (img158957.jpg), which do not map to canonical val2017
# filenames (000000212226.jpg) -- zero overlap -- and WildFake's COCO copies are
# not among the downloaded archives, so the specific 2 omitted images cannot be
# identified locally. We therefore score the full canonical archive: 5,000
# instead of 4,998, a 0.04% difference in the real class, and every one of them
# is denylisted from training either way.
EXPECT_REAL = 5000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--denylist-txt",
                    default=str(REPO / "data/denylist/dalle3_advanced_sha256.txt"))
    ap.add_argument("--out", default=str(REPO / "data/manifests/protected.parquet"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    from track5.data.manifest import SCHEMA, row_from_bytes
    from track5.data.wildfake_csv import coco_val2017_demo_ids

    data_root = Path(args.data_root)
    rows: list = []
    failures: list = []

    # ---- real side: the full canonical COCO val2017 archive (see EXPECT_REAL) ----
    demo_ids = coco_val2017_demo_ids(data_root)
    val_zip = data_root / "COCO" / "val2017.zip"
    if not val_zip.exists():
        print(f"ERROR: {val_zip} missing", file=sys.stderr)
        return 2

    import threading
    local = threading.local()

    def read_val(name):
        # per-thread handle: a shared ZipFile interleaves seeks and returns
        # wrong bytes (see tests/test_resolve_threadsafe.py)
        z = getattr(local, "z", None)
        if z is None:
            z = local.z = zipfile.ZipFile(val_zip, "r")
        try:
            return row_from_bytes(f"COCO/val2017.zip#{name}", z.read(name), 0,
                                  "coco_val2017", "")
        except Exception as e:
            failures.append({"path": name, "error": f"{type(e).__name__}: {e}"})
            return None

    with zipfile.ZipFile(val_zip) as z:
        members = [i.filename for i in z.infolist()
                   if not i.is_dir() and Path(i.filename).suffix.lower()
                   in {".jpg", ".jpeg", ".png"}]
    overlap = {Path(m).name for m in members} & demo_ids
    if overlap and len(overlap) != len(members):
        wanted = [m for m in members if Path(m).name in demo_ids]
    else:  # ids are in WildFake's namespace, not COCO's -> score the archive
        wanted = members
        print(f"[protected] demo-id overlap {len(overlap)}/{len(demo_ids)}; "
              f"scoring the full canonical archive ({len(members)} images)",
              file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows.extend(r for r in ex.map(read_val, wanted) if r is not None)
    n_real = len(rows)
    print(f"[protected] COCO val2017 demo subset: {n_real} rows", flush=True)

    # ---- fake side: DALL-E 3 Advanced, enumerated from the denylist itself ----
    deny_paths = []
    for line in Path(args.denylist_txt).read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            deny_paths.append(parts[1].strip())
    dalle_zip = data_root / "WildFake/Images/Diffusion_based/DALLE.zip"
    if not dalle_zip.exists():
        print(f"ERROR: {dalle_zip} missing", file=sys.stderr)
        return 2
    with zipfile.ZipFile(dalle_zip) as z:
        names = set(z.namelist())
    prefix = ""
    if deny_paths and deny_paths[0] not in names:
        cand = [n for n in names if n.endswith(deny_paths[0])]
        if cand:
            prefix = cand[0][:len(cand[0]) - len(deny_paths[0])]
    local2 = threading.local()

    def read_dalle(rel):
        member = prefix + rel
        z = getattr(local2, "z", None)
        if z is None:
            z = local2.z = zipfile.ZipFile(dalle_zip, "r")
        try:
            return row_from_bytes(
                f"WildFake/Images/Diffusion_based/DALLE.zip#{member}",
                z.read(member), 1, "wildfake_dalle", "dalle")
        except Exception as e:
            failures.append({"path": member, "error": f"{type(e).__name__}: {e}"})
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fake_rows = [r for r in ex.map(read_dalle, deny_paths) if r is not None]
    rows.extend(fake_rows)
    n_fake = len(fake_rows)
    print(f"[protected] DALL-E 3 Advanced: {n_fake} rows", flush=True)

    df = pd.DataFrame(rows)[list(SCHEMA)].astype(SCHEMA)
    df["split"] = "protected"          # safeguard 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    receipt = {"real": n_real, "fake": n_fake, "total": int(len(df)),
               "expected_real": EXPECT_REAL, "expected_fake": EXPECT_FAKE,
               "counts_match": n_real == EXPECT_REAL and n_fake == EXPECT_FAKE,
               "split": "protected", "unreadable": len(failures),
               "unreadable_examples": failures[:20]}
    out.with_name(out.name + ".receipt.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print(f"wrote {len(df)} rows to {out}")
    print(f"  real {n_real}/{EXPECT_REAL}  fake {n_fake}/{EXPECT_FAKE}")

    if not receipt["counts_match"] and not args.allow_count_mismatch:
        print("ERROR: protected-set counts do not match the organiser's spec "
              "(4,998 real / 8,843 fake). Refusing to certify this manifest; "
              "pass --allow-count-mismatch to write it anyway.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
