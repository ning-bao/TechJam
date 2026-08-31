import numpy as np
import pandas as pd
import pytest

from track5.eval.bootstrap import bootstrap_ci
from track5.eval.calibrate import calibrated_scores, fit_temperature_alpha
from track5.eval.metrics import bacc
from track5.eval.threshold import freeze_threshold


def test_fit_recovers_known_params():
    rng = np.random.Generator(np.random.PCG64(11))
    T_true, a_true = 2.5, 0.7
    z = rng.normal(0, 3.0, 20000)
    p = 1 / (1 + np.exp(-(z + a_true) / T_true))
    y = (rng.uniform(0, 1, len(z)) < p).astype(int)
    T, a = fit_temperature_alpha(z, y)
    assert abs(T - T_true) < 0.3
    assert abs(a - a_true) < 0.15


def test_calibrated_scores_monotone_and_bounded():
    z = np.linspace(-10, 10, 101)
    s = calibrated_scores(z, T=2.0, alpha=0.5)
    assert np.all((s > 0) & (s < 1))
    assert np.all(np.diff(s) > 0)


def test_freeze_threshold_respects_fpr():
    # reals mostly low, a tail of 10% high; fakes high
    real = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.85, 0.95, 10)])
    fake = np.linspace(0.5, 1.0, 100)
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([real, fake])
    t = freeze_threshold(y, s, max_fpr=0.05)
    fpr = (real >= t).mean()
    assert fpr <= 0.05
    # optimality: matches brute-force best bAcc among FPR-feasible thresholds
    grid = np.linspace(0, 1.001, 2003)
    feasible = [g for g in grid if (real >= g).mean() <= 0.05]
    best = max(bacc(y, s, g) for g in feasible)
    assert abs(bacc(y, s, t) - best) < 1e-9


def test_freeze_threshold_infeasible_falls_back():
    # every real above every fake -> FPR<=0.05 impossible except at max threshold
    real = np.linspace(0.6, 1.0, 20)
    fake = np.linspace(0.0, 0.5, 20)
    y = np.array([0] * 20 + [1] * 20)
    s = np.concatenate([real, fake])
    t = freeze_threshold(y, s, max_fpr=0.05)
    assert (real >= t).mean() <= 0.05  # picks minimal-FPR fallback (above all reals)


def test_bootstrap_deterministic_and_brackets():
    rng = np.random.Generator(np.random.PCG64(3))
    y = np.array([0] * 200 + [1] * 200)
    s = np.concatenate([rng.uniform(0, 0.6, 200), rng.uniform(0.4, 1.0, 200)])
    fn = lambda yy, ss: bacc(yy, ss, 0.5)
    lo1, hi1 = bootstrap_ci(y, s, fn, n=200, seed=7)
    lo2, hi2 = bootstrap_ci(y, s, fn, n=200, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    point = fn(y, s)
    assert lo1 <= point <= hi1
    lo3, _ = bootstrap_ci(y, s, fn, n=200, seed=8)
    assert lo3 != lo1  # seed sensitivity


def test_run_matrix_csv_and_cache(tmp_path, monkeypatch):
    import track5.eval.matrix as M

    calls = {"transform": 0}

    def fake_prepare(manifest_df, atom, cache_dir, data_root, seed):
        from pathlib import Path
        d = Path(cache_dir) / atom
        d.mkdir(parents=True, exist_ok=True)
        cache_paths = []
        for r in manifest_df.itertuples(index=False):
            p = d / f"{r.sha256}.png"
            if not p.exists():
                calls["transform"] += 1
                p.write_bytes(b"x")
            cache_paths.append(str(p))
        rows = manifest_df.copy()
        rows["cache_path"] = cache_paths
        return rows, []

    monkeypatch.setattr(M, "prepare_atom_rows", fake_prepare)
    rng = np.random.Generator(np.random.PCG64(1))
    n = 40
    manifest = pd.DataFrame({
        "sha256": [f"h{i:03d}" for i in range(n)],
        "path": [f"p{i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "generator_family": [f"fam{i % 3}" for i in range(n)],
    })
    score_fn = lambda paths: rng.uniform(0, 1, len(paths))
    meta = {"data_root": ".", "model_hash": "m" * 12, "config_hash": "c" * 12,
            "atoms_version": "1.0"}
    out = tmp_path / "matrix.csv"
    df = M.run_matrix(manifest, score_fn, ["jpeg_30", "clean"], tmp_path / "cache",
                      17, 0.5, out, meta)
    got = pd.read_csv(out)
    assert list(got.columns) == M.CSV_COLUMNS
    assert list(got["atom"]) == ["clean", "jpeg_30"]  # clean forced first
    assert got.loc[0, "delta_clean_bacc"] == 0.0
    first = calls["transform"]
    assert first == 2 * n
    items = pd.read_parquet(M.items_path(out))
    assert list(items.columns) == ["path", "label", "score", "condition",
                                   "generator_family"]
    assert len(items) == 2 * n
    assert set(items["condition"]) == {"clean", "jpeg_30"}
    joined = items.merge(manifest[["path", "label", "generator_family"]],
                         on="path", suffixes=("", "_manifest"))
    assert (joined["label"] == joined["label_manifest"]).all()
    assert (joined["generator_family"] == joined["generator_family_manifest"]).all()
    M.run_matrix(manifest, score_fn, ["jpeg_30", "clean"], tmp_path / "cache",
                 17, 0.5, out, meta)
    assert calls["transform"] == first  # cache reused, no recompute
