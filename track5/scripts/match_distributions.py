"""PLAN D4: match the compression/size distribution across classes.

After container normalization the file size of a fixed-size crop no longer
reflects the delivered container - it reflects content complexity, and synthetic
images are smoother, so they compress smaller. That leaves a global "small file
=> fake" cue a detector can ride instead of learning a generator artifact, and
it is exactly the cue that collapses once the benchmark re-encodes everything.

Equalize it by selection: bin the pooled normalized file size and keep the same
number of reals and fakes in every bin, so the marginal distribution carries no
class information. Rows the matching drops are removed from the split.

    .venv/Scripts/python.exe scripts/match_distributions.py \
        --normalized data/manifests/train_normalized.parquet \
        --manifest data/manifests/train.parquet
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="", help="default: overwrite --manifest")
    ap.add_argument("--out-normalized", default="",
                    help="matched copy of the normalized frame (for the gate)")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    norm = pd.read_parquet(args.normalized)
    if "sha256" not in norm.columns:
        print("ERROR: normalized frame needs sha256", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    size = np.log1p(norm["file_bytes"].to_numpy(dtype=np.float64))
    labels = norm["label"].to_numpy()
    # Stratify by delivered container first: lossless rows are far larger than
    # encoded ones, so pooled size bins would mix the two and matching could
    # quietly unbalance the container split that normalization just equalized.
    fmts = norm["format"].to_numpy().astype(str)
    strata = np.empty(len(norm), dtype=object)
    for f in np.unique(fmts):
        m = fmts == f
        edges = np.unique(np.quantile(size[m], np.linspace(0, 1, args.bins + 1)))
        b = np.clip(np.digitize(size[m], edges[1:-1]), 0, max(0, len(edges) - 2))
        strata[m] = [f"{f}:{x}" for x in b]

    keep = []
    for b in np.unique(strata):
        in_bin = np.flatnonzero(strata == b)
        r = in_bin[labels[in_bin] == 0]
        f = in_bin[labels[in_bin] == 1]
        n = min(len(r), len(f))
        if n == 0:
            continue
        keep.append(rng.permutation(r)[:n])
        keep.append(rng.permutation(f)[:n])
    keep = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)

    matched = norm.iloc[keep].reset_index(drop=True)
    kept_sha = set(matched["sha256"])
    df = pd.read_parquet(args.manifest)
    before = len(df)
    df = df[df["sha256"].isin(kept_sha)].reset_index(drop=True)

    out = Path(args.out) if args.out else Path(args.manifest)
    df.to_parquet(out, index=False)
    out_norm = Path(args.out_normalized) if args.out_normalized else Path(
        args.normalized).with_name(Path(args.normalized).stem + "_matched.parquet")
    matched.to_parquet(out_norm, index=False)

    receipt = {
        "bins": args.bins, "strata": int(len(np.unique(strata))), "seed": args.seed,
        "rows_before": int(before), "rows_after": int(len(df)),
        "dropped": int(before - len(df)),
        "real": int((df.label == 0).sum()), "fake": int((df.label == 1).sum()),
        "format_balance": {str(k): int(v) for k, v in
                           matched.groupby(["label", "format"]).size().to_dict().items()},
        "median_bytes": {
            "real": float(matched[matched.label == 0]["file_bytes"].median()),
            "fake": float(matched[matched.label == 1]["file_bytes"].median())},
        "by_family": {str(k): int(v) for k, v in
                      df.groupby(["label", "generator_family"]).size().to_dict().items()},
    }
    Path(out).with_name(Path(out).stem + "_matched.receipt.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print(f"{before} -> {len(df)} rows ({receipt['dropped']} dropped); "
          f"{receipt['real']} real / {receipt['fake']} fake")
    print("median normalized bytes:", receipt["median_bytes"])
    print(f"wrote {out} and {out_norm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
