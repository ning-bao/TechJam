"""Protected-set denylist (PLAN D4/C2): all COCO val2017 + entire WildFake
DALL·E family. Match = exact sha256 OR pHash Hamming distance <= 4.

Content hashes need the bytes. The WildFake DALL·E archives are not downloaded,
so the family is *also* denied by path from the label CSVs — see
`build_protected_paths` / `data/denylist/protected_paths.parquet`. Both tables
are applied together; the hash table alone cannot protect data we do not have.
"""

import numpy as np
import pandas as pd

from track5.data.wildfake_csv import protected_keys, wildfake_key


def _phash_to_u64(series: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


def build_denylist_df(manifest_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """From manifests of the protected sources, produce (sha256, phash, reason)."""
    parts = []
    for df in manifest_dfs:
        reason = np.where(df["source"].eq("coco_val2017"), "coco_val2017",
                          np.where(df["generator_family"].eq("dalle"),
                                   "wildfake_dalle", ""))
        sub = df.loc[reason != "", ["sha256", "phash"]].copy()
        sub["reason"] = reason[reason != ""]
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["sha256", "phash", "reason"])
    return out.drop_duplicates(subset=["sha256"]).reset_index(drop=True)


def count_protected(manifest_df: pd.DataFrame, denylist_dir) -> dict:
    """How many manifest rows are protected, by each independent gate.

    One implementation shared by the training gate (which must abort) and the
    evaluation entry point (which must demand an explicit --protected-run).
    """
    from pathlib import Path

    denylist_dir = Path(denylist_dir)
    deny = denylist_dir / "denylist.parquet"
    paths = denylist_dir / "protected_paths.parquet"
    out = {"n_rows": int(len(manifest_df)), "family_hits": 0, "source_hits": 0,
           "path_hits": 0, "hash_hits": 0, "denied_rows": 0,
           "tables_present": [p.name for p in (deny, paths) if p.exists()]}

    if "split" in manifest_df:
        out["denied_rows"] = int((manifest_df["split"] == "denied").sum())
    if "generator_family" in manifest_df:
        fam = manifest_df["generator_family"].astype(str).str.lower()
        out["family_hits"] = int(fam.isin(DENIED_FAMILIES).sum())
    if "source" in manifest_df:
        src = manifest_df["source"].astype(str).str.lower()
        out["source_hits"] = int(src.isin(DENIED_SOURCES).sum())
    if deny.exists():
        d = pd.read_parquet(deny)
        if len(d):
            out["hash_hits"] = int(
                manifest_df["sha256"].isin(set(d["sha256"])).sum())
    if paths.exists():
        p = pd.read_parquet(paths)
        if len(p):
            keys = set(p["path_key"])
            out["path_hits"] = int(
                sum(wildfake_key(x) in keys for x in manifest_df["path"]))
    out["total_hits"] = (out["family_hits"] + out["source_hits"] + out["path_hits"]
                         + out["hash_hits"])
    return out


def build_protected_paths(data_root) -> pd.DataFrame:
    """(path_key, reason) for every protected WildFake path, from the label CSVs.
    Works with zero image bytes on disk."""
    keys = protected_keys(data_root)
    return pd.DataFrame({"path_key": list(keys), "reason": list(keys.values())})


DENIED_FAMILIES = frozenset({"dalle"})
DENIED_SOURCES = frozenset({"coco_val2017"})


def apply_denylist(manifest_df: pd.DataFrame, denylist_df: pd.DataFrame,
                   max_hamming: int = 4,
                   protected_paths_df: pd.DataFrame | None = None,
                   denied_families: frozenset = DENIED_FAMILIES,
                   denied_sources: frozenset = DENIED_SOURCES) -> pd.DataFrame:
    """Mark denylisted rows split='denied' (never silently dropped).

    Four independent gates, so no single mislabel lets a protected image through:
    generator family, source dataset, path key, and content hash/pHash. The
    family and source gates come first because they survive a renamed path.
    """
    df = manifest_df.copy()
    if "generator_family" in df.columns and denied_families:
        fam = df["generator_family"].astype(str).str.lower()
        df.loc[fam.isin(denied_families).to_numpy(dtype=bool), "split"] = "denied"
    if "source" in df.columns and denied_sources:
        src = df["source"].astype(str).str.lower()
        df.loc[src.isin(denied_sources).to_numpy(dtype=bool), "split"] = "denied"
    if protected_paths_df is not None and len(protected_paths_df):
        keys = set(protected_paths_df["path_key"])
        hit = df["path"].map(lambda p: wildfake_key(p) in keys).to_numpy(dtype=bool)
        df.loc[hit, "split"] = "denied"
    if denylist_df.empty:
        return df
    denied = np.array(df["sha256"].isin(set(denylist_df["sha256"])), dtype=bool)

    deny_hashes = _phash_to_u64(denylist_df["phash"])
    man_hashes = _phash_to_u64(df["phash"])
    chunk = 2048
    for i in range(0, len(man_hashes), chunk):
        block = man_hashes[i:i + chunk, None] ^ deny_hashes[None, :]
        dist = np.bitwise_count(block).min(axis=1)
        denied[i:i + chunk] |= dist <= max_hamming
    df.loc[denied, "split"] = "denied"
    return df
