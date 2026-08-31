#!/usr/bin/env python
"""Recompute the held-out operating point and decision curves at the locked tau.

Context: the figure and the three conclusions were computed at tau = 0.7439,
which appears in the repository ONLY as a fixture constant in
track5/tests/test_metrics_json_to_csv.py. The version-locked threshold is
tau = 0.45936643399060617, carried with the weights in WEIGHTS_LICENSE.md and
in reports/calibration.json, and used for the actual hard-case scoring runs.

Step 1 reproduces the published tau = 0.7439 numbers to confirm the pipeline
口径 matches Zhang's; step 2 reports the same quantities at the locked tau.
"""
from pathlib import Path

import pyarrow.parquet as pq

TAU_FIXTURE = 0.7439
TAU_LOCKED = 0.45936643399060617
ITEMS = str(Path(__file__).resolve().parents[1]
            / "matrix_ood_excluded_epoch1_calibrated.items.parquet")
CONDS = ("clean", "jpeg_30", "blur_20", "resize_025", "noise_010")


def load():
    t = pq.read_table(ITEMS).to_pydict()
    by = {c: {"y": [], "s": []} for c in CONDS}
    for lab, sc, cond in zip(t["label"], t["score"], t["condition"]):
        if cond in by:
            by[cond]["y"].append(int(lab))
            by[cond]["s"].append(float(sc))
    return by


def confusion(y, s, tau):
    tp = fp = tn = fn = 0
    for yi, si in zip(y, s):
        pred = si >= tau
        if yi == 1 and pred:
            tp += 1
        elif yi == 1:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def rates(y, s, tau):
    tp, fp, tn, fn = confusion(y, s, tau)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tpr, fpr, (tp, fp, tn, fn)


def net_benefit(y, s, pt):
    """NB = TP/N - (FP/N) * pt/(1-pt), thresholding at p_t itself."""
    n = len(y)
    tp, fp, _, _ = confusion(y, s, pt)
    return tp / n - (fp / n) * (pt / (1.0 - pt))


def report(by, tau, label):
    print(f"\n{'='*72}\n{label}: tau = {tau!r}")
    k = tau / (1.0 - tau)
    print(f"implied C_FP/C_FN = tau/(1-tau) = {k:.6f}")
    print(f"\n{'condition':12s} {'FPR':>8s} {'TPR':>8s}   "
          f"{'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}   {'NB@tau':>9s}")
    for c in CONDS:
        y, s = by[c]["y"], by[c]["s"]
        tpr, fpr, (tp, fp, tn, fn) = rates(y, s, tau)
        nb = net_benefit(y, s, tau)
        print(f"{c:12s} {fpr*100:7.2f}% {tpr*100:7.2f}%   "
              f"{tp:5d} {fp:5d} {tn:5d} {fn:5d}   {nb:9.6f}")
    return k


if __name__ == "__main__":
    by = load()
    for c in CONDS:
        print(f"{c:12s} n={len(by[c]['y']):5d}  "
              f"reals={sum(1 for v in by[c]['y'] if v == 0):4d}  "
              f"fakes={sum(by[c]['y']):4d}")
    report(by, TAU_FIXTURE, "STEP 1  reproduce published figure (fixture tau)")
    report(by, TAU_LOCKED, "STEP 2  version-locked tau")
