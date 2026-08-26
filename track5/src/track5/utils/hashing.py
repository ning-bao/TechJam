import hashlib
from pathlib import Path


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
