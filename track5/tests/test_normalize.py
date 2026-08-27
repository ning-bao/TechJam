"""PLAN D4 normalization invariants.

The raw corpus is separable by container alone (COCO reals ~100% JPEG q94,
WildFake fakes ~92% PNG), so these properties are what stop a detector from
learning "PNG => fake" instead of a generator artifact.
"""

import hashlib

import numpy as np
import pytest
from PIL import Image

from track5.data.normalize import (QUALITY_RANGE, apply_plan, native_history,
                                   normalization_plan, normalized_stats)


def shas(prefix, n):
    return [hashlib.sha256(f"{prefix}{i}".encode()).hexdigest() for i in range(n)]


def test_plan_is_deterministic():
    s = shas("x", 1)[0]
    assert normalization_plan(s, "png", 17) == normalization_plan(s, "png", 17)
    assert normalization_plan(s, "png", 17) != normalization_plan(s, "png", 18)


def test_plan_never_sees_the_label():
    """Signature carries no label; identity + container fully determine it."""
    s = shas("x", 1)[0]
    assert normalization_plan(s, "jpeg") == normalization_plan(s, "jpeg")


def test_final_quality_distribution_matches_across_containers():
    """The q-table of the delivered file is what the D4 quality probe reads;
    compare the encoded subset, since lossless rows have no q-table."""
    p_jpeg = [normalization_plan(s, "jpeg") for s in shas("r", 4000)]
    p_png = [normalization_plan(s, "png") for s in shas("f", 4000)]
    q_jpeg = [p["final_quality"] for p in p_jpeg if not p["lossless"]]
    q_png = [p["final_quality"] for p in p_png if not p["lossless"]]
    assert abs(np.mean(q_jpeg) - np.mean(q_png)) < 1.0
    for q in q_jpeg + q_png:
        assert QUALITY_RANGE[0] <= q < QUALITY_RANGE[1]


def test_lossless_rate_matches_across_containers():
    """~1/3 of the protected fakes are pristine PNG payloads, so training must
    contain lossless examples - but at the SAME rate for both classes, or the
    container becomes a class cue again."""
    r = np.mean([normalization_plan(s, "jpeg")["lossless"] for s in shas("r", 4000)])
    f = np.mean([normalization_plan(s, "png")["lossless"] for s in shas("f", 4000)])
    assert abs(r - f) < 0.03, (r, f)
    assert 0.2 < r < 0.4


def test_encoded_rows_always_end_with_one_final_encode():
    for fmt in ("jpeg", "png"):
        for s in shas(fmt, 200):
            plan = normalization_plan(s, fmt)
            assert plan["lossless"] or plan["n_extra"] >= 1


def test_native_history_counts_the_container():
    assert native_history("jpeg") == 1 and native_history("jpg") == 1
    assert native_history("png") == 0 and native_history("webp") == 0


@pytest.mark.parametrize("fmt", ["jpeg", "png"])
def test_normalized_output_is_uniform(fmt):
    rng = np.random.Generator(np.random.PCG64(0))
    img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
    st = normalized_stats(img, shas("s", 1)[0], fmt)
    assert st["format"] in {"jpeg", "png"}
    assert st["width"] == 64 and st["height"] == 64
    assert st["file_bytes"] > 0


def test_delivered_container_is_independent_of_class():
    """Same identity + same plan => same container, whatever the native format."""
    fmt_jpeg = [normalized_stats(
        Image.new("RGB", (32, 32), (i % 255, 40, 80)), s, "jpeg")["format"]
        for i, s in enumerate(shas("c", 60))]
    fmt_png = [normalized_stats(
        Image.new("RGB", (32, 32), (i % 255, 40, 80)), s, "png")["format"]
        for i, s in enumerate(shas("c", 60))]
    assert fmt_jpeg == fmt_png


def test_apply_plan_preserves_size_and_mode():
    rng = np.random.Generator(np.random.PCG64(1))
    img = Image.fromarray(rng.integers(0, 256, (48, 32, 3), dtype=np.uint8), "RGB")
    out = apply_plan(img, normalization_plan(shas("t", 1)[0], "png"))
    assert out.size == img.size and out.mode == "RGB"
