import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from track5.eval.metrics import (all_metrics, auroc, average_precision, bacc,
                                 brier, ece, fpr_at_95tpr, worst_case_bacc)


def rand_data(n=500, seed=1):
    rng = np.random.Generator(np.random.PCG64(seed))
    y = rng.integers(0, 2, n)
    s = rng.uniform(0, 1, n)
    return y, s


def test_auroc_ap_match_sklearn():
    y, s = rand_data()
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12
    assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-12


def test_bacc_hand_case():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.9, 0.3])
    # t=0.5: real correct 2/4, fake correct 1/2 -> (0.5+0.5)/2
    assert abs(bacc(y, s, 0.5) - 0.5) < 1e-12


def test_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    assert auroc(y, s) == 1.0
    assert fpr_at_95tpr(y, s) == 0.0
    assert bacc(y, s, 0.5) == 1.0


def test_fpr_at_95tpr_constructed():
    # fakes: 95 at 0.9 + 5 low; reals: 10 at 0.95 sit above the t=0.9 operating
    # point needed for TPR>=0.95 -> minimal feasible FPR is 0.10
    fake = np.concatenate([np.full(95, 0.9), np.full(5, 0.05)])
    real = np.concatenate([np.full(90, 0.1), np.full(10, 0.95)])
    y = np.array([1] * 100 + [0] * 100)
    s = np.concatenate([fake, real])
    assert abs(fpr_at_95tpr(y, s) - 0.10) < 1e-9


def test_ece_calibrated_vs_not():
    rng = np.random.Generator(np.random.PCG64(5))
    p = rng.uniform(0, 1, 200000)
    y = (rng.uniform(0, 1, 200000) < p).astype(int)
    assert ece(y, p) < 0.01
    assert ece(1 - y, p) > 0.3


def test_degenerate_single_class():
    y = np.zeros(10, dtype=int)
    s = np.linspace(0, 1, 10)
    assert np.isnan(auroc(y, s))
    assert np.isnan(average_precision(y, s))
    m = all_metrics(y, s, 0.5)
    assert m["n_real"] == 10 and m["n_fake"] == 0


def test_all_metrics_keys_and_worst_case():
    y, s = rand_data()
    m = all_metrics(y, s, 0.5)
    assert set(m) == {"auroc", "ap", "bacc", "fpr_at_95tpr", "brier", "ece",
                      "n_real", "n_fake"}
    assert worst_case_bacc({"clean": 0.9, "jpeg_30": 0.7, "noise_010": 0.8}) == 0.7


def test_brier_hand_cases():
    assert brier([1, 0], [1.0, 0.0]) == 0.0            # perfect
    assert brier([1, 0], [0.0, 1.0]) == 1.0            # maximally wrong
    assert abs(brier([1, 1, 0, 0], [0.5] * 4) - 0.25) < 1e-12   # uninformative
    assert np.isnan(brier([], []))


def test_brier_rewards_calibration_not_just_ranking():
    """Same ranking, different confidence: Brier separates them, AUROC cannot."""
    y = np.array([0] * 50 + [1] * 50)
    sharp = np.concatenate([np.full(50, 0.02), np.full(50, 0.98)])
    timid = np.concatenate([np.full(50, 0.45), np.full(50, 0.55)])
    assert auroc(y, sharp) == auroc(y, timid) == 1.0
    assert brier(y, sharp) < brier(y, timid)
