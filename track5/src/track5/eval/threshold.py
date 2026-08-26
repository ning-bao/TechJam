import numpy as np


def freeze_threshold(y, scores, max_fpr: float = 0.05) -> float:
    """Threshold maximizing bAcc subject to FPR <= max_fpr.

    PLAN D7: fitted ONCE on CLEAN dev scores, frozen, and NEVER refit per
    transform. If no threshold satisfies the FPR constraint, fall back to the
    minimal-FPR thresholds and take the best bAcc among them.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    uniq = np.unique(s)
    mids = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else np.array([])
    cands = np.concatenate(([uniq.min() - 1e-9] if len(uniq) else [0.5], mids,
                            [uniq.max() + 1e-9] if len(uniq) else []))
    real, fake = s[y == 0], s[y == 1]
    best_t, best_bacc = None, -1.0
    fallback = (np.inf, -1.0, 0.5)  # (fpr, bacc, t)
    for t in cands:
        fpr = float((real >= t).mean()) if len(real) else 0.0
        tpr = float((fake >= t).mean()) if len(fake) else 0.0
        b = ((1.0 - fpr) + tpr) / 2.0
        if fpr <= max_fpr and b > best_bacc:
            best_bacc, best_t = b, float(t)
        if (fpr, -b) < (fallback[0], -fallback[1]):
            fallback = (fpr, b, float(t))
    return best_t if best_t is not None else fallback[2]
