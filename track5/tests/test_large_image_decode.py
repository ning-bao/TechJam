"""Large camera originals must be scored, not dropped (track5.utils.imaging).

PIL warns above MAX_IMAGE_PIXELS and raises above 2x it. Set A's largest
delivered image is 8320x12480 = 103.8 MP, which sits in the warning band on
PIL's defaults -- so on a runner with warnings-as-errors it would land in the
error log instead of the predictions.

These tests build their own oversized image rather than depending on that file,
so they run anywhere.
"""

import warnings

import numpy as np
import pytest
from PIL import Image

from track5.utils.imaging import (LARGE_IMAGE_PIXELS, PIL_DEFAULT_MAX_PIXELS,
                                  apply_decode_policy)

# Comfortably past PIL's default (89.5 MP) while staying cheap to synthesise:
# a 1-pixel-tall strip has the pixel count of a 100 MP photo, and PIL's guard
# is on width*height, which is exactly what we want to trip.
OVERSIZE_PIXELS = 100_000_000


@pytest.fixture
def restore_policy():
    previous = Image.MAX_IMAGE_PIXELS
    yield
    Image.MAX_IMAGE_PIXELS = previous


@pytest.fixture
def oversized_png(tmp_path, restore_policy):
    """A file whose pixel count exceeds PIL's default guard.

    The guard has to be lifted to *write* the file, then put back to the PIL
    default before the test body runs -- otherwise the fixture's own `None`
    leaks in and every test passes whether or not the policy works. (It did:
    this test passed against a deliberately neutered apply_decode_policy until
    the restore below was added.)
    """
    Image.MAX_IMAGE_PIXELS = None
    w, h = 10_000, OVERSIZE_PIXELS // 10_000
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, ::7] = 200                        # some structure, still compresses small
    p = tmp_path / "huge.png"
    Image.fromarray(arr).save(p)
    Image.MAX_IMAGE_PIXELS = PIL_DEFAULT_MAX_PIXELS
    return p


def test_default_policy_warns_on_a_real_camera_size(oversized_png):
    """The failure mode being fixed: on PIL defaults this only warns, and a
    warnings-as-errors runner turns it into a dropped image."""
    Image.MAX_IMAGE_PIXELS = PIL_DEFAULT_MAX_PIXELS
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(oversized_png) as im:
            im.load()
    assert any(issubclass(w.category, Image.DecompressionBombWarning)
               for w in caught), "expected the default guard to warn"

    # and with warnings escalated, the same decode fails outright
    Image.MAX_IMAGE_PIXELS = PIL_DEFAULT_MAX_PIXELS
    with pytest.raises(Image.DecompressionBombWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(oversized_png) as im:
                im.load()


def test_policy_decodes_the_same_file_silently(oversized_png):
    apply_decode_policy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # nothing may warn under the policy
        with Image.open(oversized_png) as im:
            im.load()
            assert im.size[0] * im.size[1] == OVERSIZE_PIXELS


def test_policy_still_refuses_an_absurd_file(restore_policy):
    """Not `None`: an unbounded decode can OOM the one protected-set run."""
    apply_decode_policy()
    assert Image.MAX_IMAGE_PIXELS == LARGE_IMAGE_PIXELS
    assert LARGE_IMAGE_PIXELS is not None
    # PIL raises above 2x the limit; confirm the ceiling is where we documented
    assert LARGE_IMAGE_PIXELS > PIL_DEFAULT_MAX_PIXELS
    assert LARGE_IMAGE_PIXELS * 2 < 600_000_000, "ceiling too high to be a guard"


def test_apply_returns_previous_so_callers_can_restore(restore_policy):
    Image.MAX_IMAGE_PIXELS = 123
    assert apply_decode_policy() == 123
    assert Image.MAX_IMAGE_PIXELS == LARGE_IMAGE_PIXELS
