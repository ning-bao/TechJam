#!/usr/bin/env python
"""Verify the delivered figure source CSVs against an independent recomputation.

Every numeric cell in ../figures/source_data/ is recomputed from the held-out
per-item scores and compared, including the 0.001-step threshold scan behind
source_heldout_negative_net_benefit.csv. The tau-independent columns (AUROC,
15-bin ECE) are checked against the original matrix CSV instead, since the
threshold revision must not have altered them.

Exits non-zero on any mismatch.
"""
import csv
import sys
from pathlib import Path

from recheck_tau import CONDS, TAU_LOCKED, load, net_benefit, rates

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "figures" / "source_data"
MATRIX = HERE.parent / "matrix_ood_excluded_epoch1_calibrated.csv"
SCAN_STEP = 0.001

fails = []


def chk(what, got, exp, tol=1e-9):
    if abs(float(got) - float(exp)) > tol:
        fails.append(f"{what}: csv={got} recomputed={exp}")


def scan(y, s):
    """Replicate the published 0.001-step scan: first negative NB, and minimum."""
    first = None
    best = best_pt = None
    pt = SCAN_STEP
    while pt <= 0.9995:
        nb = net_benefit(y, s, pt)
        if first is None and nb < 0:
            first = (pt, nb)
        if best is None or nb < best:
            best, best_pt = nb, pt
        pt = round(pt + SCAN_STEP, 3)
    return first, best, best_pt


def main():
    by = load()
    mine = {}
    for c in CONDS:
        y, s = by[c]["y"], by[c]["s"]
        tpr, fpr, (tp, fp, tn, fn) = rates(y, s, TAU_LOCKED)
        mine[c] = dict(tp=tp, fp=fp, tn=tn, fn=fn, tpr=tpr, fpr=fpr,
                       nb=net_benefit(y, s, TAU_LOCKED),
                       bacc=(tpr + (1 - fpr)) / 2)

    for r in csv.DictReader(open(SRC / "source_heldout_condition_metrics.csv")):
        c, m = r["condition"], mine[r["condition"]]
        for col, key in (("tp_at_tau", "tp"), ("fp_at_tau", "fp"),
                         ("tn_at_tau", "tn"), ("fn_at_tau", "fn"),
                         ("tpr_at_tau", "tpr"), ("fpr_at_tau", "fpr"),
                         ("nb_at_tau", "nb"), ("bacc_at_tau", "bacc")):
            chk(f"metrics/{c}/{col}", r[col], m[key])
        counts = sum(int(r[k]) for k in
                     ("tp_at_tau", "fp_at_tau", "tn_at_tau", "fn_at_tau"))
        chk(f"metrics/{c}/n", r["n"], counts)
        chk(f"metrics/{c}/prevalence", r["prevalence"],
            int(r["n_aigc"]) / int(r["n"]))
        chk(f"metrics/{c}/signed==-ece", r["signed_pred_minus_observed"],
            -float(r["ece_15bin"]))

    for r in csv.DictReader(open(SRC / "source_heldout_operating_point.csv")):
        c, m = r["condition"], mine[r["condition"]]
        key = "fpr" if r["metric"] == "False-positive rate" else "tpr"
        chk(f"oppoint/{c}/{r['metric']}", r["value"], m[key])
        for col in ("tp", "fp", "tn", "fn"):
            chk(f"oppoint/{c}/{col}", r[col], m[col])

    for r in csv.DictReader(open(SRC / "source_heldout_frozen_threshold_points.csv")):
        c = r["condition"]
        chk(f"frozen/{c}/pt", r["pt"], TAU_LOCKED, tol=1e-12)
        chk(f"frozen/{c}/nb", r["net_benefit"], mine[c]["nb"])

    for r in csv.DictReader(open(SRC / "source_threshold_audit.csv")):
        t = float(r["threshold"])
        chk(f"audit/{r['threshold_id']}/k", r["implied_C_FP_over_C_FN"], t / (1 - t))

    for r in csv.DictReader(open(SRC / "source_heldout_negative_net_benefit.csv")):
        c = r["condition"]
        first, best, best_pt = scan(by[c]["y"], by[c]["s"])
        if r["first_negative_pt"] == "NA":
            if first is not None:
                fails.append(f"neg/{c}: csv=NA recomputed={first}")
        else:
            chk(f"neg/{c}/first_pt", r["first_negative_pt"], first[0])
            chk(f"neg/{c}/first_nb", r["first_negative_net_benefit"], first[1])
            chk(f"neg/{c}/implied_k", r["implied_k_at_first_negative_pt"],
                first[0] / (1 - first[0]), tol=1e-6)
        chk(f"neg/{c}/min_nb", r["minimum_net_benefit_in_scan"], best)
        chk(f"neg/{c}/min_pt", r["minimum_pt_in_scan"], best_pt)

    # AUROC and 15-bin ECE do not depend on tau: they must be unchanged.
    ref = {r["atom"]: r for r in csv.DictReader(open(MATRIX))}
    for r in csv.DictReader(open(SRC / "source_heldout_condition_metrics.csv")):
        c = r["condition"]
        chk(f"invariant/{c}/auroc", r["auroc"], ref[c]["auroc"])
        chk(f"invariant/{c}/ece", r["ece_15bin"], ref[c]["ece"])

    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        for f in fails:
            print("  !", f)
        return 1
    print("ALL CHECKS PASS -- 5 CSVs verified against independent recomputation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
