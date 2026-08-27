"""Assemble the final training/dev/calib manifests (PLAN D3 + D4).

Pipeline, in order:
  1. concat the per-source manifests
  2. drop denylist hits (C2: COCO val2017 + the whole WildFake DALL-E family)
  3. drop min(W,H) < crop  -- nothing is ever padded, so native size cannot leak
  4. de-duplicate on sha256
  5. cap per generator family / real source, and balance the two classes
  6. split by generator family and real source (a family never straddles splits)

    .venv/Scripts/python.exe scripts/build_training_set.py --crop 448
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ["real_coco", "real_wildfake", "fake_wildfake", "sidset",
                   "fake_vae"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="*", default=DEFAULT_SOURCES,
                    help="manifest stems under data/manifests")
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--per-bucket", type=int, default=40000,
                    help="cap per generator family (fakes) / source (reals)")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--calib-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", default=str(REPO / "data" / "manifests"))
    ap.add_argument("--allow-parquet-paths", action="store_true",
                    help="permit parquet-row paths in the output (see the guard "
                         "below); only for a manifest that will not be trained on")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    mdir = Path(args.out_dir)
    frames = []
    for stem in args.manifests:
        p = mdir / f"{stem}.parquet"
        if not p.exists():
            print(f"[build] missing {p}, skipping", file=sys.stderr)
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        print("ERROR: no input manifests", file=sys.stderr)
        return 2
    df = pd.concat(frames, ignore_index=True)
    audit = {"raw_rows": int(len(df))}

    # Content group: a VAE reconstruction is pixel-content-identical to the real
    # it came from, so the pair must share a split or dev sees train content.
    if "src_sha256" not in df.columns:
        df["src_sha256"] = None
    df["group"] = df["src_sha256"].fillna("").astype(str)
    df.loc[df["group"] == "", "group"] = df.loc[df["group"] == "", "sha256"]
    df = df.drop(columns=["src_sha256"])

    # 2. denylist (C2)
    deny_p = REPO / "data" / "denylist" / "denylist.parquet"
    if not deny_p.exists():
        print("ERROR: denylist missing - run scripts/build_denylist.py first",
              file=sys.stderr)
        return 2
    deny = pd.read_parquet(deny_p)
    hit = df["sha256"].isin(set(deny["sha256"]))
    audit["denylist_hits"] = int(hit.sum())
    df = df[~hit]

    # 3. no padding, ever: native size becomes invisible after cropping
    minside = df[["width", "height"]].min(axis=1)
    audit["dropped_below_crop"] = int((minside < args.crop).sum())
    df = df[minside >= args.crop]

    # 4. exact-duplicate images
    before = len(df)
    df = df.drop_duplicates(subset="sha256").reset_index(drop=True)
    audit["dropped_duplicates"] = int(before - len(df))

    # 5. cap per bucket, then balance the classes
    df["bucket"] = np.where(df["label"] == 1,
                            "fake:" + df["generator_family"].astype(str),
                            "real:" + df["source"].astype(str))
    rng = np.random.default_rng(args.seed)
    keep = []
    for bucket, g in df.groupby("bucket"):
        g = g.sort_values("sha256")
        if len(g) > args.per_bucket:
            g = g.iloc[rng.permutation(len(g))[:args.per_bucket]]
        keep.append(g)
    df = pd.concat(keep, ignore_index=True)
    audit["per_bucket_counts"] = df["bucket"].value_counts().to_dict()

    # Positional indexing throughout: groupby().apply() reindexing silently left
    # whole classes unsplit here.
    labels = df["label"].to_numpy()
    buckets = df["bucket"].to_numpy()
    target = min(int((labels == 0).sum()), int((labels == 1).sum()))
    keep = []
    for label in (0, 1):
        pos = np.flatnonzero(labels == label)
        if len(pos) > target:  # trim proportionally across buckets
            uniq, counts = np.unique(buckets[pos], return_counts=True)
            exact = counts * target / len(pos)
            quota = np.floor(exact).astype(int)
            for i in np.argsort(-(exact - quota))[:target - int(quota.sum())]:
                quota[i] += 1
            pos = np.concatenate([
                bp[rng.permutation(len(bp))[:q]]
                for b, q in zip(uniq, quota)
                for bp in (pos[buckets[pos] == b],)])
        keep.append(pos)
    df = df.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    audit["balanced"] = {"real": int((df.label == 0).sum()),
                         "fake": int((df.label == 1).sum())}

    # 6. split by CONTENT GROUP within each bucket, so every family/source keeps
    #    its proportions and a recon never lands away from its source real
    split = np.array(["train"] * len(df), dtype=object)
    buckets = df["bucket"].to_numpy()
    groups = df["group"].to_numpy()
    rows_by_group: dict = {}
    for i, g in enumerate(groups):
        rows_by_group.setdefault(g, []).append(i)
    # a group is placed by the bucket of its first row, so it is decided once
    first_bucket = {g: buckets[idx[0]] for g, idx in rows_by_group.items()}
    assigned: dict = {}
    for b in np.unique(list(first_bucket.values())):
        gs = np.array([g for g, fb in first_bucket.items() if fb == b], dtype=object)
        gs = gs[rng.permutation(len(gs))]
        n_dev = int(round(len(gs) * args.dev_frac))
        n_cal = int(round(len(gs) * args.calib_frac))
        for g in gs[:n_dev]:
            assigned[g] = "dev"
        for g in gs[n_dev:n_dev + n_cal]:
            assigned[g] = "calib"
    for g, name in assigned.items():
        for i in rows_by_group[g]:
            split[i] = name
    df["split"] = split
    audit["content_groups"] = int(len(rows_by_group))
    audit["grouped_rows"] = int(sum(len(v) for v in rows_by_group.values() if len(v) > 1))
    df = df.drop(columns=["bucket", "group"])

    # Guard: a "shard.parquet#row" path costs a whole row-group decode (~844
    # images) per read. Tolerable for one sequential pass, ruinous for shuffled
    # training - it stalled a measurement pass at 48 GB RSS. These come back
    # whenever a source manifest still holds them, so fail loudly here rather
    # than discover it as a mystery slowdown mid-training.
    pq = df["path"].str.contains(".parquet#", regex=False)
    audit["parquet_row_paths"] = int(pq.sum())
    if pq.any() and not args.allow_parquet_paths:
        print(f"ERROR: {int(pq.sum())} rows use parquet-row paths, e.g. "
              f"{df.loc[pq, 'path'].iloc[0]} -- run scripts/extract_sidset.py "
              f"(it rewrites the source manifest too), then rebuild. "
              f"Override with --allow-parquet-paths.", file=sys.stderr)
        return 3

    for name in ("train", "dev", "calib"):
        sub = df[df["split"] == name].reset_index(drop=True)
        out = mdir / f"{name}.parquet"
        sub.to_parquet(out, index=False)
        counts = sub.groupby(["label", "generator_family"]).size().to_dict()
        audit[name] = {"n": int(len(sub)),
                       "real": int((sub.label == 0).sum()),
                       "fake": int((sub.label == 1).sum()),
                       "by_family": {str(k): int(v) for k, v in counts.items()}}
        print(f"{name}: {len(sub)} rows "
              f"({int((sub.label==0).sum())} real / {int((sub.label==1).sum())} fake) -> {out}")

    receipt = mdir / "training_set.receipt.json"
    receipt.write_text(json.dumps({"crop": args.crop, "seed": args.seed, **audit},
                                  indent=1), encoding="utf-8")
    print(f"\naudit -> {receipt}")
    for k in ("raw_rows", "denylist_hits", "dropped_below_crop", "dropped_duplicates"):
        print(f"  {k}: {audit[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
