import hashlib
import subprocess
import sys
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from track5.transforms.eval_atoms import ATOMS, EVAL_15, apply_and_encode, apply_atom

VENV_PY = sys.executable


def make_img(w=640, h=480, seed=3):
    rng = np.random.Generator(np.random.PCG64(seed))
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def img_bytes(img, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_registry_shape():
    assert len(EVAL_15) == 15
    assert "crop_80_area" in ATOMS and "crop_80_area" not in EVAL_15
    assert len(ATOMS) == 16


def test_all_atoms_produce_valid_images():
    img = make_img()
    src = img_bytes(img)
    for atom in ATOMS:
        out = apply_and_encode(src, atom, seed=17)
        dec = Image.open(BytesIO(out))
        dec.load()
        assert dec.width >= 1 and dec.height >= 1, atom


def test_byte_determinism_in_process():
    src = img_bytes(make_img())
    for atom in ["clean", "jpeg_30", "noise_010", "jitter_pm20", "crop_80"]:
        a = apply_and_encode(src, atom, seed=99)
        b = apply_and_encode(src, atom, seed=99)
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), atom


def test_byte_determinism_cross_process(tmp_path):
    src = img_bytes(make_img())
    srcfile = tmp_path / "src.png"
    srcfile.write_bytes(src)
    atoms = ["clean", "jpeg_30", "noise_010", "jitter_pm20", "crop_80"]
    local = {a: hashlib.sha256(apply_and_encode(src, a, seed=99)).hexdigest() for a in atoms}
    code = (
        "import hashlib,sys\n"
        "from track5.transforms.eval_atoms import apply_and_encode\n"
        f"src = open(r'{srcfile}','rb').read()\n"
        f"for a in {atoms!r}:\n"
        "    print(a, hashlib.sha256(apply_and_encode(src, a, 99)).hexdigest())\n"
    )
    res = subprocess.run([VENV_PY, "-c", code], capture_output=True, text=True, check=True)
    remote = dict(line.split() for line in res.stdout.strip().splitlines())
    assert remote == local


def test_seed_sensitivity():
    src = img_bytes(make_img())
    assert apply_and_encode(src, "noise_010", 1) != apply_and_encode(src, "noise_010", 2)
    assert apply_and_encode(src, "jpeg_70", 1) == apply_and_encode(src, "jpeg_70", 2)


def test_resize_round_trips_to_original_size():
    img = make_img(640, 480)
    for atom in ("resize_050", "resize_025"):
        out = apply_atom(img, atom, 0)
        assert (out.width, out.height) == (640, 480), atom
    # the round-trip must genuinely degrade: variance drops, more at 0.25x
    v_clean = np.asarray(img, dtype=np.float64).var()
    v_050 = np.asarray(apply_atom(img, "resize_050", 0), dtype=np.float64).var()
    v_025 = np.asarray(apply_atom(img, "resize_025", 0), dtype=np.float64).var()
    assert v_025 < v_050 < v_clean


def test_crop_conventions():
    img = make_img(500, 400)
    side = apply_atom(img, "crop_80", 0)
    area = apply_atom(img, "crop_80_area", 0)
    assert (side.width, side.height) == (400, 320)
    s = 0.8**0.5
    assert (area.width, area.height) == (round(500 * s), round(400 * s))
    # area convention keeps ~80% of pixels; side keeps 64%
    assert area.width * area.height > side.width * side.height


def test_jpeg_atoms_encode_as_jpeg_others_png():
    src = img_bytes(make_img())
    assert apply_and_encode(src, "jpeg_50", 0)[:2] == b"\xff\xd8"
    assert apply_and_encode(src, "blur_10", 0)[:8] == b"\x89PNG\r\n\x1a\n"
    assert apply_and_encode(src, "clean", 0)[:8] == b"\x89PNG\r\n\x1a\n"
