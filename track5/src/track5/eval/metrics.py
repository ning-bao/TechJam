"""Per-condition metrics. numpy/sklearn only — no torch in this package."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _arr(y, scores):
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    return y, s


def _single_class(y) -> bool:
    return len(np.unique(y)) < 2


def auroc(y, scores) -> float:
    y, s = _arr(y, scores)
    if _single_class(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def average_precision(y, scores) -> float:
    y, s = _arr(y, scores)
    if _single_class(y):
        return float("nan")
    return float(average_precision_score(y, s))


def bacc(y, scores, threshold: float) -> float:
    y, s = _arr(y, scores)
    pred = (s >= threshold).astype(int)
    accs = []
    for cls in (0, 1):
        mask = y == cls
        if mask.any():
            accs.append(float((pred[mask] == cls).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def fpr_at_95tpr(y, scores) -> float:
    """FPR at the best (lowest-FPR) operating point reaching TPR >= 0.95."""
    y, s = _arr(y, scores)
    if _single_class(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= 0.95
    if not ok.any():
        return 1.0
    return float(fpr[ok].min())


def brier(y, scores) -> float:
    """Mean squared error of the calibrated probability (TC2 section 11)."""
    y, s = _arr(y, scores)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def ece(y, scores, n_bins: int = 15) -> float:
    """Binary ECE on p(class 1): weighted |mean(score) - mean(y)| over equal-width bins."""
    y, s = _arr(y, scores)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(s[mask].mean() - y[mask].mean())
    return float(total)


def all_metrics(y, scores, threshold: float) -> dict:
    y_, _ = _arr(y, scores)
    return {
        "auroc": auroc(y, scores),
        "ap": average_precision(y, scores),
        "bacc": bacc(y, scores, threshold),
        "fpr_at_95tpr": fpr_at_95tpr(y, scores),
        "brier": brier(y, scores),
        "ece": ece(y, scores),
        "n_real": int((y_ == 0).sum()),
        "n_fake": int((y_ == 1).sum()),
    }


def worst_case_bacc(per_atom: dict) -> float:
    return float(min(per_atom.values()))
