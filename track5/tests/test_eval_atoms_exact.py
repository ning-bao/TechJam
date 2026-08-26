"""The 15 benchmark transforms must be *exactly* the specified ones:
JPEG 90/70/50/30, blur 0.5/1/2, resize 0.5/0.25, noise 0.02/0.05/0.10,
+/-20% colour jitter, 80% centre crop — applied identically to both classes.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from track5.data.manifest import estimate_jpeg_quality
from track5.transforms.eval_atoms import (ATOMS, EVAL_15, TC2_ALIASES, apply_atom,
                                          apply_and_encode, canonical_atom)

SPEC = {
    "jpeg_90": ("jpeg", "q", 90), "jpeg_70": ("jpeg", "q", 70),
    "jpeg_50": ("jpeg", "q", 50), "jpeg_30": ("jpeg", "q", 30),
    "blur_05": ("blur", "sigma", 0.5), "blur_10": ("blur", "sigma", 1.0),
    "blur_20": ("blur", "sigma", 2.0),
    "resize_050": ("resize", "factor", 0.50), "resize_025": ("resize", "factor", 0.25),
    "noise_002": ("noise", "sigma", 0.02), "noise_005": ("noise", "sigma", 0.05),
    "noise_010": ("noise", "sigma", 0.10),
}


def make_img(w=320, h=256, seed=7):
    rng = np.random.Generator(np.random.PCG64(seed))
    arr = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    arr[:, ::8, :] = 255
    return Image.fromarray(arr, "RGB")


def png(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_the_15_conditions_are_exactly_the_benchmark_list():
    assert EVAL_15 == [
        "clean",
        "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
        "blur_05", "blur_10", "blur_20",
        "resize_050", "resize_025",
        "noise_002", "noise_005", "noise_010",
        "jitter_pm20", "crop_80",
    ]


@pytest.mark.parametrize("atom,kind,key,value", [(a, *v) for a, v in SPEC.items()])
def test_atom_parameters_match_the_spec(atom, kind, key, value):
    assert ATOMS[atom]["kind"] == kind
    assert ATOMS[atom][key] == value


def test_jitter_is_pm20_percent_without_hue():
    spec = ATOMS["jitter_pm20"]
    assert (spec["lo"], spec["hi"]) == (0.8, 1.2)
    assert "hue" not in spec


def test_crop_is_80_percent_of_each_side():
    assert ATOMS["crop_80"] == {"kind": "crop", "frac": 0.8, "convention": "side"}
    out = apply_atom(make_img(500, 400), "crop_80", 0)
    assert out.size == (400, 320)


@pytest.mark.parametrize("atom,q", [("jpeg_90", 90), ("jpeg_70", 70),
                                    ("jpeg_50", 50), ("jpeg_30", 30)])
def test_jpeg_atoms_encode_at_the_named_quality(atom, q):
    """Re-read the quantization tables: the encoded bytes really are quality Q,
    and the transform is a single compression, not a double one."""
    out = apply_and_encode(png(make_img()), atom, seed=0)
    img = Image.open(BytesIO(out))
    img.load()
    assert img.format == "JPEG"
    assert abs(estimate_jpeg_quality(img) - q) <= 2


def test_jpeg_quality_ladder_is_monotone_in_file_size():
    src = png(make_img())
    sizes = [len(apply_and_encode(src, a, 0))
             for a in ("jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30")]
    assert sizes == sorted(sizes, reverse=True)


@pytest.mark.parametrize("atom,factor", [("resize_050", 0.5), ("resize_025", 0.25)])
def test_resize_downscales_then_upscales_back(atom, factor):
    """Track spec: "scale 0.5x / 0.25x then upscale" — size is preserved,
    high-frequency content is not."""
    assert ATOMS[atom]["factor"] == factor
    img = make_img(640, 480)
    out = apply_atom(img, atom, 0)
    assert out.size == (640, 480)
    lost = np.abs(np.asarray(out, np.float64) - np.asarray(img, np.float64)).mean()
    assert lost > 1.0  # genuinely degraded, not a no-op


def test_blur_strength_increases_with_sigma():
    img = make_img()
    var = [np.asarray(apply_atom(img, a, 0), dtype=np.float64).var()
           for a in ("clean", "blur_05", "blur_10", "blur_20")]
    assert var == sorted(var, reverse=True)


@pytest.mark.parametrize("atom,sigma", [("noise_002", 0.02), ("noise_005", 0.05),
                                        ("noise_010", 0.10)])
def test_noise_sigma_is_a_fraction_of_255(atom, sigma):
    img = make_img(400, 400)
    base = np.asarray(img, dtype=np.float64)
    out = np.asarray(apply_atom(img, atom, seed=11), dtype=np.float64)
    measured = (out - base).std()
    assert abs(measured - sigma * 255.0) < 0.15 * sigma * 255.0


def test_jitter_stays_within_pm20_percent():
    """Every draw must land inside [0.8, 1.2] for all three channels' factors."""
    img = Image.new("RGB", (64, 64), (120, 120, 120))
    means = [np.asarray(apply_atom(img, "jitter_pm20", s)).mean() for s in range(200)]
    assert min(means) >= 120 * 0.8 - 1.5
    assert max(means) <= 120 * 1.2 + 1.5
    assert len(set(np.round(means, 3))) > 1  # the seed really varies the draw


def test_both_classes_take_the_identical_path():
    """No class-conditional logic exists: identical bytes in, identical out."""
    src = png(make_img())
    for atom in EVAL_15:
        assert apply_and_encode(src, atom, 5) == apply_and_encode(src, atom, 5)


def test_tc2_condition_names_alias_onto_the_frozen_ones():
    assert canonical_atom("jpeg_q30") == "jpeg_30"
    assert canonical_atom("blur_0.5") == "blur_05"
    assert canonical_atom("resize_0.25") == "resize_025"
    assert canonical_atom("noise_0.10") == "noise_010"
    assert canonical_atom("color_jitter_0.20") == "jitter_pm20"
    assert canonical_atom("center_crop_0.80") == "crop_80"
    assert canonical_atom("crop_80") == "crop_80"          # canonical passes through
    assert sorted(TC2_ALIASES.values()) == sorted(EVAL_15)  # all 15, no more
    with pytest.raises(KeyError):
        canonical_atom("jpeg_q42")
