"""Resolve a manifest `path` to raw image bytes. Read-only.

Path grammar (INTERFACES §4 extension):
  plain file:   "WildFake/Images/Real/xxx.jpg"
  zip member:   "COCO/train2017.zip#train2017/000000123.jpg"
  parquet row:  "SID_Set/data/train-00007-of-00283.parquet#123"  (HF image.bytes)
"""

import zipfile
from pathlib import Path

_ZIP_CACHE: dict = {}
_PQ_CACHE: dict = {}


def _zip_handle(abspath: Path) -> zipfile.ZipFile:
    key = str(abspath)
    if key not in _ZIP_CACHE:
        _ZIP_CACHE[key] = zipfile.ZipFile(abspath, "r")
    return _ZIP_CACHE[key]


def _parquet_handle(abspath: Path):
    import pyarrow.parquet as pq

    key = str(abspath)
    if key not in _PQ_CACHE:
        f = pq.ParquetFile(abspath)
        offsets = [0]
        for i in range(f.metadata.num_row_groups):
            offsets.append(offsets[-1] + f.metadata.row_group(i).num_rows)
        _PQ_CACHE[key] = (f, offsets)
    return _PQ_CACHE[key]


def resolve_image_bytes(data_root, path: str) -> bytes:
    data_root = Path(data_root)
    if "#" not in path:
        return (data_root / path).read_bytes()
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
