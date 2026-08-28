"""Decode policy for images larger than PIL's default bomb guard.

PIL warns above `MAX_IMAGE_PIXELS` (default 89,478,485) and raises
`DecompressionBombError` above twice that. Real camera originals cross the
warning line routinely -- the largest image in our own hard-case set is
8320x12480 = 103.8 MP -- so on the default policy a legitimate submission image
emits a warning, and any caller that turns warnings into errors (a common CI
setting) would drop it into the error log instead of scoring it.

Leaving the limit at `None` is the other easy mistake: an unbounded decode can
exhaust memory mid-run, and the one run we cannot afford to lose is the single
protected-set inference. So the policy is an explicit, documented ceiling:

    warn  above LARGE_IMAGE_PIXELS      (audit trail, not a failure)
    raise above 2x LARGE_IMAGE_PIXELS   (PIL's own behaviour; ~1.5 GB decoded)

250 MP leaves 2.4x headroom over the largest image we hold, and still refuses a
file engineered to blow up the process.
"""

from __future__ import annotations

from PIL import Image

# 250 MP decoded as RGB is ~750 MB; PIL raises at 2x this (~1.5 GB).
LARGE_IMAGE_PIXELS = 250_000_000

# What PIL shipped with, kept so tests can assert we actually moved it.
PIL_DEFAULT_MAX_PIXELS = 89_478_485


def apply_decode_policy(limit: int | None = LARGE_IMAGE_PIXELS) -> int | None:
    """Set the process-wide decode ceiling. Call once at CLI start.

    Returns the previous value so a caller can restore it (tests do).
    """
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = limit
    return previous
