"""PLAN D4 hard gate: any shortcut probe >= 0.60 bAcc -> exit nonzero, no training.

.venv/Scripts/python.exe scripts/probe_gate.py --manifest data/manifests/train.parquet \
    [--embeddings emb.npy --emb-labels labels.npy]
"""

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--emb-labels", default=None)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    from track5.data.probes import GATE_BACC, gate_passes, run_all_probes

    df = pd.read_parquet(args.manifest)
    df = df[df["split"] != "denied"]
    emb = labels = None
    if args.embeddings:
        try:
            emb = np.load(args.embeddings)
            labels = np.load(args.emb_labels) if args.emb_labels else df["label"].to_numpy()
        except FileNotFoundError as e:
            print(f"WARNING: embeddings not found ({e}); running 3 metadata probes only",
                  file=sys.stderr)
    results = run_all_probes(df, emb, labels)

    print(f"{'probe':<20} {'bAcc':>7}  gate<{GATE_BACC}")
    for name, v in results.items():
        status = "n/a" if v != v else ("PASS" if v < GATE_BACC else "FAIL")
        print(f"{name:<20} {v:>7.3f}  {status}")
    if gate_passes(results):
        print("GATE: PASS — training allowed")
        return 0
    print("GATE: FAIL — a shortcut probe reached the threshold; fix the data "
          "(PLAN D4) before any training run")
    return 1


if __name__ == "__main__":
    sys.exit(main())
