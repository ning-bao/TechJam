"""Manifest builder (INTERFACES §4). Scans sources read-only; tolerates the
partial files of in-progress downloads by skipping and counting them."""

import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from track5.data import wildfake_csv as wf
from track5.data.resolve import resolve_image_bytes
from track5.utils.hashing import bytes_sha256

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SOURCES = ("coco_train", "coco_val", "coco_val_demo", "wildfake", "wildfake_csv",
           "sid_set")

SCHEMA = {
    "path": "object", "sha256": "object", "phash": "object", "label": "int8",
    "source": "object", "generator_family": "object", "width": "int32",
    "height": "int32", "format": "object", "file_bytes": "int64",
    "jpeg_quality": "int16", "n_recompress": "int8", "split": "object",
}

# libjpeg standard luminance quantization table (IJG), for quality estimation
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def estimate_jpeg_quality(img: Image.Image) -> int:
    try:
        table = img.quantization
        luma = np.array(table[0], dtype=np.float64)
        if len(luma) != 64:
            return -1
        scale = float(np.mean(100.0 * luma / _STD_LUMA))  # IJG: S=5000/q (q<50) else 200-2q
        q = 5000.0 / scale if scale >= 100.0 else (200.0 - scale) / 2.0
        return int(np.clip(round(q), 1, 100))
    except Exception:
        return -1


def _family_from_path(path: str) -> str:
    p = path.lower()
    for token in ("dalle", "dall-e", "dall_e"):
        if token in p:
            return "dalle"
    rules = [
        ("midjourney", "mj"), ("stable_diffusion", "sd"), ("/sd", "sd"), ("sdxl", "sd"),
        ("adm", "adm"), ("ddpm", "ddpm"), ("ddim", "ddpm"), ("vqdm", "vqdm"),
        ("glide", "glide"), ("flux", "flux"),
        ("gan_based", "gan"), ("gan", "gan"),
    ]
    for token, fam in rules:
        if token in p:
            return fam
    return "other"


def row_from_bytes(path: str, data: bytes, label: int, source: str,
                   generator_family: str) -> dict:
    img = Image.open(BytesIO(data))
    img.load()
    fmt = (img.format or "other").lower()
    quality = estimate_jpeg_quality(img) if fmt == "jpeg" else -1
    rgb = img.convert("RGB")
    return {
        "path": path,
        "sha256": bytes_sha256(data),
        "phash": str(imagehash.phash(rgb)),
        "label": label,
        "source": source,
        "generator_family": generator_family,
        "width": img.width,
        "height": img.height,
        "format": fmt,
        "file_bytes": len(data),
        "jpeg_quality": quality,
        "n_recompress": -1,
        "split": "unassigned",
    }


def _iter_zip(data_root: Path, relzip: str, label: int, source: str, family_fn):
    try:
        zf = zipfile.ZipFile(data_root / relzip, "r")
    except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
        print(f"[manifest] cannot open {relzip}: {e}", file=sys.stderr)
        return
    for info in zf.infolist():
        if info.is_dir() or Path(info.filename).suffix.lower() not in IMG_EXTS:
            continue
        path = f"{relzip}#{info.filename}"
        # read via resolve_image_bytes (per-thread handle), never the shared `zf`
        # used for enumeration: the workers below are threads.
        yield path, label, source, family_fn(path), (
            lambda p=path: resolve_image_bytes(data_root, p))


def _iter_wildfake_csv(data_root: Path, per_family_limit: int = 0,
                       csvs: list[str] | None = None, stats: dict | None = None):
    """CSV-driven WildFake index (PLAN D3). Labels and families come from the
    label CSVs, not from path heuristics.

    Constraint C2 is enforced here by `wf.protected_reason`, which rejects a row
    on ANY of: its source CSV, its Architecture column, its Category column, its
    mapped generator family, or its path - so a renamed path alone cannot get a
    DALL-E image into a manifest. The audit record is
    data/denylist/protected_paths.parquet, built from the same CSVs.

    `stats` collects unresolved rows so the caller can fail a final build
    (strict mode) instead of silently producing a short manifest.
    """
    index = wf.ArchiveIndex(data_root)
    counts: dict[str, int] = {}
    refused: dict[str, int] = {}
    missing: list[str] = []
    for row in wf.iter_csv_rows(data_root, only=csvs):
        reason = wf.protected_reason(row) or (
            "coco_val2017" if wf.is_coco_val2017_key(row.key) else "")
        if reason:
            refused[reason] = refused.get(reason, 0) + 1
            continue
        fam = wf.family_of(row)
        bucket = fam or f"real:{row.architecture}"
        if per_family_limit and counts.get(bucket, 0) >= per_family_limit:
            continue
        path = index.resolve(row.key)
        if path is None:
            missing.append(row.key)
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
        label = 1 if row.is_fake else 0
        yield path, label, "wildfake", fam, (lambda p=path: resolve_image_bytes(data_root, p))
    print(f"[manifest] wildfake_csv: kept {sum(counts.values())} "
          f"({dict(sorted(counts.items()))}), {len(missing)} requested rows "
          f"unavailable, protected refused {dict(sorted(refused.items()))}",
          file=sys.stderr, flush=True)
    if stats is not None:
        stats.update({"kept": sum(counts.values()), "per_bucket": dict(sorted(counts.items())),
                      "protected_refused": dict(sorted(refused.items())),
                      "missing": len(missing), "missing_examples": missing[:20],
                      "unavailable_archives": sorted(index.unavailable)})


def _iter_coco_val_demo(data_root: Path, stats: dict | None = None):
    """The 4,998-image demonstration benchmark subset (item 5).

    Distinct from the denylist, which covers all 5,000 images in the canonical
    val2017 archive. The extra images are denied from training but are NOT part
    of the benchmark, so they must not appear in this manifest.
    """
    demo = wf.coco_val2017_demo_ids(data_root)
    if not demo:
        print("[manifest] real_coco.csv missing -> cannot identify the demo subset",
              file=sys.stderr, flush=True)
        return
    seen, extras = set(), []
    for entry in _iter_zip(data_root, "COCO/val2017.zip", 0, "coco_val2017",
                           lambda p: ""):
        name = entry[0].rsplit("/", 1)[-1]
        if name in demo:
            seen.add(name)
            yield entry
        else:
            extras.append(name)
    absent = sorted(demo - seen)
    print(f"[manifest] coco_val_demo: {len(seen)}/{len(demo)} benchmark images, "
          f"{len(extras)} archive images outside the benchmark subset "
          f"(denied from training, not evaluated)", file=sys.stderr, flush=True)
    if stats is not None:
        stats.update({"demo_expected": len(demo), "demo_found": len(seen),
                      "outside_benchmark": len(extras),
                      "outside_benchmark_examples": sorted(extras)[:20],
                      "missing": len(absent), "missing_examples": absent[:20]})


def iter_source(source: str, data_root: Path, per_family_limit: int = 0,
                csvs: list[str] | None = None, stats: dict | None = None):
    """Yield (path, label, source, family, loader) for every candidate image."""
    if source == "wildfake_csv":
        yield from _iter_wildfake_csv(data_root, per_family_limit, csvs, stats)
    elif source == "coco_val_demo":
        yield from _iter_coco_val_demo(data_root, stats)
    elif source == "coco_train":
        yield from _iter_zip(data_root, "COCO/train2017.zip", 0, "coco_train2017", lambda p: "")
    elif source == "coco_val":
        yield from _iter_zip(data_root, "COCO/val2017.zip", 0, "coco_val2017", lambda p: "")
    elif source == "wildfake":
        base = data_root / "WildFake" / "Images"
        if not base.exists():
            print("[manifest] WildFake/Images missing", file=sys.stderr)
            return
        for f in sorted(base.rglob("*")):
            rel = f.relative_to(data_root).as_posix()
            is_real = "/real/" in f"/{rel.lower()}/"
            label = 0 if is_real else 1
            fam = "" if is_real else _family_from_path(rel)
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                yield rel, label, "wildfake", fam, (lambda ff=f: ff.read_bytes())
            elif f.is_file() and f.suffix.lower() == ".zip":
                yield from _iter_zip(data_root, rel, label, "wildfake",
                                     (lambda p, fam=fam: fam) if not is_real else (lambda p: ""))
    elif source == "sid_set":
        import pyarrow.parquet as pq

        shard_dir = data_root / "SID_Set" / "data"
        for shard in sorted(shard_dir.glob("*.parquet")):
            rel = shard.relative_to(data_root).as_posix()
            try:
                pf = pq.ParquetFile(shard)
            except Exception as e:
                print(f"[manifest] bad shard {rel}: {e}", file=sys.stderr)
                continue
            row_base = 0
            for g in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(g, columns=["image", "label"])
                labels = tbl.column("label").to_pylist()
                images = tbl.column("image")
                for i, lab in enumerate(labels):
                    if lab == 2:  # tampered subset — not our class definition
                        continue
                    path = f"{rel}#{row_base + i}"
                    y = 0 if lab == 0 else 1
                    fam = "" if y == 0 else "flux"
                    data = images[i].as_py()["bytes"]
                    yield path, y, "sid_set", fam, (lambda d=data: d)
                row_base += len(labels)
    else:
        raise ValueError(f"unknown source {source!r}")


class MissingImages(RuntimeError):
    """A final manifest build requested images that are not on disk (item 7)."""


def build_manifest(source: str, data_root, out_path, limit: int = 0,
                   workers: int = 8, append: bool = False,
                   per_family_limit: int = 0, csvs: list[str] | None = None,
                   allow_missing: bool = False) -> pd.DataFrame:
    """Strict by default: any requested image that cannot be read raises
    MissingImages after the receipt is written. `allow_missing=True` is the
    explicit development/sample mode."""
    data_root = Path(data_root)
    out_path = Path(out_path)
    seen: set = set()
    old = None
    if append and out_path.exists():
        old = pd.read_parquet(out_path)
        seen = set(old["path"])

    stats: dict = {}
    entries = []
    for entry in iter_source(source, data_root, per_family_limit, csvs, stats):
        if entry[0] in seen:
            continue
        entries.append(entry)
        if limit and len(entries) >= limit:
            break

    failures: list[dict] = []

    def work(entry):
        path, label, src, fam, loader = entry
        try:
            return row_from_bytes(path, loader(), label, src, fam)
        except Exception as e:
            failures.append({"path": path, "error": f"{type(e).__name__}: {e}"})
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = [r for r in ex.map(work, entries) if r is not None]

    df = pd.DataFrame(rows, columns=list(SCHEMA))
    df = df.astype(SCHEMA)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_missing = int(stats.get("missing", 0))
    n_unreadable = len(failures)
    complete = n_missing == 0 and n_unreadable == 0
    receipt = {
        "source": source, "rows": int(len(df)), "allow_missing": allow_missing,
        "complete": complete, "limit": limit, "per_family_limit": per_family_limit,
        "csvs": csvs, "requested_unavailable": n_missing,
        "unreadable": n_unreadable, "unreadable_examples": failures[:20],
        **{k: v for k, v in stats.items() if k != "missing"},
    }
    receipt_path = out_path.with_name(out_path.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=1, default=str),
                            encoding="utf-8")
    for f in failures[:10]:
        print(f"[manifest] unreadable {f['path']}: {f['error']}", file=sys.stderr)
    print(f"[manifest] {source}: wrote {len(df)} rows to {out_path} "
          f"({n_missing} unavailable, {n_unreadable} unreadable); "
          f"receipt {receipt_path}", file=sys.stderr, flush=True)

    if not complete and not allow_missing:
        raise MissingImages(
            f"{source}: {n_missing} requested images are not on disk and "
            f"{n_unreadable} could not be read. A final manifest must be built "
            f"from a complete corpus; pass allow_missing=True (CLI: "
            f"--allow-missing) only for a development/sample build. "
            f"See {receipt_path}.")
    return df
