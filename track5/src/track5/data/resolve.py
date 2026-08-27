"""Resolve a manifest `path` to raw image bytes. Read-only.

Path grammar (INTERFACES §4 extension):
  plain file:   "WildFake/Images/Real/xxx.jpg"
  zip member:   "COCO/train2017.zip#train2017/000000123.jpg"
  parquet row:  "SID_Set/data/train-00007-of-00283.parquet#123"  (HF image.bytes)
"""

import threading
import zipfile
from pathlib import Path

# Handle caches MUST be per-thread. A ZipFile (and a pyarrow ParquetFile) owns
# one file object with one seek position: sharing it across threads interleaves
# seeks and returns wrong bytes, which surface as "Bad CRC-32" / "Overlapped
# entries" on archives that are perfectly intact. Never make these module-level.
_LOCAL = threading.local()


def _zip_handle(abspath: Path) -> zipfile.ZipFile:
    cache = getattr(_LOCAL, "zips", None)
    if cache is None:
        cache = _LOCAL.zips = {}
    key = str(abspath)
    if key not in cache:
        cache[key] = zipfile.ZipFile(abspath, "r")
    return cache[key]


def _parquet_handle(abspath: Path):
    import pyarrow.parquet as pq

    cache = getattr(_LOCAL, "parquets", None)
    if cache is None:
        cache = _LOCAL.parquets = {}
    key = str(abspath)
    if key not in cache:
        f = pq.ParquetFile(abspath)
        offsets = [0]
        for i in range(f.metadata.num_row_groups):
            offsets.append(offsets[-1] + f.metadata.row_group(i).num_rows)
        cache[key] = (f, offsets)
    return cache[key]


def resolve_image_bytes(data_root, path: str) -> bytes:
    data_root = Path(data_root)
    if "#" not in path:
        direct = data_root / path
        if direct.exists():
            return direct.read_bytes()
        # PLAN: manifest paths are relative to data/raw OR data/derived (VAE
        # reconstructions, extracted shards). data/raw is read-only, so derived
        # products live beside it.
        return (data_root.parent / "derived" / path).read_bytes()
    base, frag = path.split("#", 1)
    abspath = data_root / base
    if base.lower().endswith(".zip"):
        return _zip_handle(abspath).read(frag)
    if base.lower().endswith(".parquet"):
        f, offsets = _parquet_handle(abspath)
        row = int(frag)
        # locate row group containing the row, read only that group
        for g in range(len(offsets) - 1):
            if offsets[g] <= row < offsets[g + 1]:
                tbl = f.read_row_group(g, columns=["image"])
                img = tbl.column("image")[row - offsets[g]].as_py()
                return img["bytes"]
        raise IndexError(f"row {row} out of range for {base}")
    raise ValueError(f"unsupported path form: {path}")
