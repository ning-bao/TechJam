"""Temperature + single-scalar logit-bias calibration (PLAN D7).

Fitted ONCE on our own deployment-mixture calibration split (dev-derived,
equal clean/transformed buckets). Never fitted on the protected set.
numpy-only, deterministic.
"""

import numpy as np


def _nll(logits, y, T, alpha):
    z = (logits + alpha) / T
    # stable log(sigmoid) forms
    log_p1 = -np.logaddexp(0.0, -z)
    log_p0 = -np.logaddexp(0.0, z)
    return float(-(y * log_p1 + (1 - y) * log_p0).mean())


def fit_temperature_alpha(logits, y) -> tuple[float, float]:
    """Minimize binary NLL of sigmoid((z + alpha)/T). Coarse grid + coordinate descent."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y).astype(np.float64)

    Ts = np.geomspace(0.05, 20.0, 60)
    alphas = np.linspace(-5.0, 5.0, 41)
    best = (np.inf, 1.0, 0.0)
    for T in Ts:
        for a in alphas:
            v = _nll(logits, y, T, a)
            if v < best[0]:
                best = (v, float(T), float(a))
    _, T, a = best

    # coordinate descent with shrinking step
    step_T, step_a = 0.25, 0.25
    cur = _nll(logits, y, T, a)
    for _ in range(200):
        improved = False
        for dT in (-step_T, step_T):
            nT = T * (1.0 + dT)
            if 1e-3 < nT < 100.0:
                v = _nll(logits, y, nT, a)
                if v < cur - 1e-12:
                    T, cur, improved = nT, v, True
        for da in (-step_a, step_a):
            v = _nll(logits, y, T, a + da)
            if v < cur - 1e-12:
                a, cur, improved = a + da, v, True
        if not improved:
            step_T *= 0.5
            step_a *= 0.5
            if step_T < 1e-4 and step_a < 1e-4:
                break
    return float(T), float(a)


def calibrated_scores(logits, T: float, alpha: float) -> np.ndarray:
    z = (np.asarray(logits, dtype=np.float64) + alpha) / T
    return 1.0 / (1.0 + np.exp(-z))
