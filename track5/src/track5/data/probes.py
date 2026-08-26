"""PLAN D4 shortcut probes — the hard CI gate. A probe predicting real/fake
above 0.60 bAcc from metadata means the dataset leaks a shortcut; training is
blocked until fixed."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GATE_BACC = 0.60


def _probe(X: np.ndarray, y: np.ndarray, seed: int = 17) -> float:
    if len(np.unique(y)) < 2 or len(y) < 20:
        return float("nan")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv)
    return float(balanced_accuracy_score(y, pred))


def probe_file_size(df: pd.DataFrame) -> float:
    return _probe(np.log1p(df[["file_bytes"]].to_numpy(dtype=np.float64)),
                  df["label"].to_numpy())


def probe_dimensions(df: pd.DataFrame) -> float:
    return _probe(df[["width", "height"]].to_numpy(dtype=np.float64),
                  df["label"].to_numpy())


def probe_jpeg_quality(df: pd.DataFrame) -> float:
    return _probe(df[["jpeg_quality"]].to_numpy(dtype=np.float64),
                  df["label"].to_numpy())


def probe_embeddings(emb: np.ndarray, y: np.ndarray) -> float:
    return _probe(np.asarray(emb, dtype=np.float64), np.asarray(y))


def run_all_probes(df: pd.DataFrame, embeddings: np.ndarray | None = None,
                   emb_labels: np.ndarray | None = None) -> dict:
    results = {
        "file_size": probe_file_size(df),
        "dimensions": probe_dimensions(df),
        "jpeg_quality": probe_jpeg_quality(df),
    }
    if embeddings is not None and emb_labels is not None:
        results["embedding_source"] = probe_embeddings(embeddings, emb_labels)
    return results


def gate_passes(results: dict) -> bool:
    return all(not (v == v and v >= GATE_BACC) for v in results.values())  # nan-safe
