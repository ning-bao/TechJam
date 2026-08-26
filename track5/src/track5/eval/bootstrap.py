import numpy as np


def bootstrap_ci(y, scores, metric_fn, n: int = 1000, seed: int = 17, alpha: float = 0.05):
    """Stratified bootstrap percentile CI: resample within each class."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    vals = []
    for _ in range(n):
        r0 = idx0[rng.integers(0, len(idx0), len(idx0))] if len(idx0) else np.array([], int)
        r1 = idx1[rng.integers(0, len(idx1), len(idx1))] if len(idx1) else np.array([], int)
        idx = np.concatenate([r0, r1])
        vals.append(metric_fn(y[idx], s[idx]))
    vals = np.asarray(vals, dtype=np.float64)
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
