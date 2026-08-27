"""Archive handles must be per-thread.

A shared zipfile.ZipFile has one seek position; reading it from several threads
interleaves seeks and returns wrong bytes, which surface as "Bad CRC-32" or
"Overlapped entries" on an intact archive. That looks exactly like data
corruption, so it must stay covered.
"""

import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from track5.data.resolve import resolve_image_bytes


@pytest.fixture(scope="module")
def zip_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    payloads = {}
    with zipfile.ZipFile(root / "imgs.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(120):
            rng = np.random.Generator(np.random.PCG64(i))
            img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            name = f"imgs/{i:04d}.png"
            zf.writestr(name, buf.getvalue())
            payloads[name] = buf.getvalue()
    return root, payloads


def test_concurrent_reads_return_correct_bytes(zip_root):
    root, payloads = zip_root
    names = list(payloads)

    def read(name):
        return name, resolve_image_bytes(root, f"imgs.zip#{name}")

    for _ in range(5):  # the race is probabilistic; repeat to keep it honest
        with ThreadPoolExecutor(max_workers=8) as ex:
            for name, data in ex.map(read, names):
                assert data == payloads[name], f"wrong bytes for {name}"


def test_concurrent_reads_all_decode(zip_root):
    """The failure mode under the old shared handle was an exception storm."""
    root, payloads = zip_root

    def decode(name):
        img = Image.open(BytesIO(resolve_image_bytes(root, f"imgs.zip#{name}")))
        img.load()
        return img.size

    with ThreadPoolExecutor(max_workers=8) as ex:
        sizes = list(ex.map(decode, list(payloads)))
    assert sizes == [(64, 64)] * len(payloads)
