"""15-atom robustness matrix runner (PLAN D9).

Transforms each dev image per atom (deterministic seed per item), caches the
transformed bytes, scores via a caller-provided score_fn, and writes one CSV
row per atom. Identical pipeline for both classes by construction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from track5.eval.bootstrap import bootstrap_ci
from track5.eval.metrics import all_metrics, bacc

CSV_COLUMNS = [
    "atom", "n", "auroc", "ap", "bacc", "fpr_at_95tpr", "ece",
    "delta_clean_bacc", "ci_lo", "ci_hi", "model_hash", "config_hash",
    "atoms_version",
]


def _cache_ext(atom: str) -> str:
    return "jpg" if atom.startswith("jpeg_") else "png"


def items_path(out_csv) -> Path:
    """Per-item scores parquet written beside an aggregate matrix CSV."""
    return Path(out_csv).with_suffix(".items.parquet")


def prepare_atom_rows(manifest_df, atom: str, cache_dir, data_root, global_seed: int):
    """Transform+cache every manifest row for one atom.

    Single owner of the cache-path convention (<cache>/<atom>/<sha256>.<ext>) and
    of the per-item seed: item_seed(image sha256, condition, global seed), so the
    transform is deterministic from image ID + condition and shared by every
    caller. Returns (rows DataFrame with a cache_path column, decode errors).
    """
    from track5.data.resolve import resolve_image_bytes
    from track5.transforms.eval_atoms import apply_and_encode
    from track5.utils.seed import item_seed

    out_dir = Path(cache_dir) / atom
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _cache_ext(atom)
    keep, cache_paths, errors = [], [], []
    for row in manifest_df.itertuples(index=False):
        out = out_dir / f"{row.sha256}.{ext}"
        if not out.exists():
            try:
                src = resolve_image_bytes(data_root, row.path)
                out.write_bytes(apply_and_encode(
                    src, atom, item_seed(row.sha256, atom, global_seed)))
            except Exception as e:  # partial downloads etc.
                errors.append({"path": row.path, "sha256": row.sha256,
                               "error": f"{type(e).__name__}: {e}"})
                print(f"[matrix] skip {row.path}: {e}", file=sys.stderr)
                continue
        keep.append(row)
        cache_paths.append(str(out))
    rows = pd.DataFrame(list(keep), columns=list(manifest_df.columns))
    rows["cache_path"] = cache_paths
    return rows, errors


def prepare_atom_cache(manifest_df, atom: str, cache_dir, data_root, global_seed: int):
    """(paths, labels, n_skipped) view of prepare_atom_rows."""
    rows, errors = prepare_atom_rows(manifest_df, atom, cache_dir, data_root,
                                     global_seed)
    labels = rows["label"].to_numpy(dtype=int) if len(rows) else np.empty(0, dtype=int)
    return list(rows["cache_path"]) if len(rows) else [], labels, len(errors)


def run_matrix(manifest_df, score_fn, atoms, cache_dir, global_seed: int,
               threshold: float, out_csv, meta: dict) -> pd.DataFrame:
    """meta needs: data_root, model_hash, config_hash, atoms_version.

    Besides the aggregate CSV, writes (path, label, score, condition) for every
    scored item to items_path(out_csv), so per-item questions (signed ECE,
    per-generator error rates, worst false positives) never need a fresh
    inference pass.
    """
    atoms = list(atoms)
    if "clean" in atoms:  # clean first so delta_clean_bacc is computable
        atoms.remove("clean")
        atoms.insert(0, "clean")
    rows, item_frames, clean_bacc = [], [], float("nan")
    for atom in atoms:
        arows, errors = prepare_atom_rows(
            manifest_df, atom, cache_dir, meta["data_root"], global_seed)
        skipped = len(errors)
        if not len(arows):
            print(f"[matrix] no usable images for atom {atom}", file=sys.stderr)
            continue
        paths = list(arows["cache_path"])
        labels = arows["label"].to_numpy(dtype=int)
        scores = np.asarray(score_fn(paths), dtype=np.float64)
        frame = pd.DataFrame({
            "path": arows["path"].to_numpy(), "label": labels,
            "score": scores, "condition": atom})
        # Downstream analysis groups errors by family; ood_excluded.parquet
        # names the column "family", the training-side manifests
        # "generator_family". Emit one canonical name when either exists.
        fam = next((c for c in ("generator_family", "family") if c in arows), None)
        if fam is not None:
            frame["generator_family"] = arows[fam].to_numpy()
        item_frames.append(frame)
        m = all_metrics(labels, scores, threshold)
        if atom == "clean":
            clean_bacc = m["bacc"]
        ci_lo, ci_hi = bootstrap_ci(labels, scores, lambda yy, ss: bacc(yy, ss, threshold))
        rows.append({
            "atom": atom, "n": len(paths), "auroc": m["auroc"], "ap": m["ap"],
            "bacc": m["bacc"], "fpr_at_95tpr": m["fpr_at_95tpr"], "ece": m["ece"],
            "delta_clean_bacc": m["bacc"] - clean_bacc,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "model_hash": meta["model_hash"], "config_hash": meta["config_hash"],
            "atoms_version": meta["atoms_version"],
        })
        if skipped:
            print(f"[matrix] {atom}: skipped {skipped} unreadable images", file=sys.stderr)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if item_frames:
        pd.concat(item_frames, ignore_index=True).to_parquet(
            items_path(out_csv), index=False)
    return df
