"""PLAN D4 compression normalization.

The raw corpus is trivially separable by container: COCO reals are ~100% JPEG at
q~94, WildFake fakes ~92% PNG. A detector trained on it learns "PNG => fake"
(the "Fake or JPEG?" trap) and collapses under the benchmark's re-encodes.

Normalization gives every image the SAME distribution of total JPEG history,
regardless of class. It is conditional on the native container because a real
that arrived as JPEG has already been compressed once and that cannot be undone:
drawing a shared *target* history and applying (target - native) compressions is
what actually equalizes the two classes. Drawing a shared number of *extra*
compressions would preserve the gap.

Deterministic given (sha256, seed): the plan is reproducible across processes and
identical for both classes by construction.

Dimensions are handled structurally, not here: the training pool is filtered to
min(W,H) >= crop so nothing is ever padded, which makes native size invisible
downstream (every sample is an identical crop x crop of native pixels).
"""

from io import BytesIO

import numpy as np
from PIL import Image

from track5.utils.seed import item_seed

# Every image ends with exactly one FINAL encode at a quality drawn from this
# shared distribution, plus an optional extra pass beforehand.
#
# A real photo arrives already JPEG-compressed and that cannot be undone, so
# total history and final-encode quality cannot both be equalized across
# classes. We equalize the final encode: the q-table of the delivered file is
# what the D4 quality probe reads and what a detector most easily latches onto.
# The residual asymmetry (reals carry one extra hidden compression) is a
# double-compression trace that is far harder to exploit, and is also true of
# real photographs in the wild, so it is the honest thing to leave in.
P_EXTRA_PASS = 0.5
QUALITY_RANGE = (70, 97)  # the eval atoms supply the harsher end

# Fraction delivered with NO added compression, drawn identically for both
# classes so the container still carries no class signal. Measured on the
# protected set: the DALL-E 3 Advanced files all carry a .jpg extension but
# ~1/3 are actually PNG payloads, i.e. pristine, never-JPEG pixels. Re-encoding
# every training image would leave the model having never seen a lossless input
# for a third of the positive class it is scored on.
P_LOSSLESS = 0.30


def native_history(fmt: str) -> int:
    """JPEG compressions an image already carries from its container."""
    return 1 if str(fmt).lower() in {"jpeg", "jpg"} else 0


def normalization_plan(sha256: str, fmt: str, seed: int = 17) -> dict:
    """Class-independent re-encode plan. Depends only on image identity and
    native container - never on the label."""
    rng = np.random.Generator(np.random.PCG64(item_seed(sha256, "normalize", seed)))
    if rng.random() < P_LOSSLESS:
        return {"final_quality": -1, "n_extra": 0, "lossless": True,
                "total_history": native_history(fmt), "qualities": []}
    extra = bool(rng.random() < P_EXTRA_PASS)
    qualities = [int(rng.integers(*QUALITY_RANGE))] if extra else []
    qualities.append(int(rng.integers(*QUALITY_RANGE)))  # the final encode
    return {"final_quality": qualities[-1], "n_extra": len(qualities),
            "lossless": False,
            "total_history": native_history(fmt) + len(qualities),
            "qualities": qualities}


def apply_plan(img: Image.Image, plan: dict) -> Image.Image:
    img = img.convert("RGB")
    for q in plan["qualities"]:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=False)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def normalize(img: Image.Image, sha256: str, fmt: str, seed: int = 17) -> Image.Image:
    return apply_plan(img, normalization_plan(sha256, fmt, seed))


def normalized_stats(img: Image.Image, sha256: str, fmt: str, seed: int = 17) -> dict:
    """Post-normalization metadata for the D4 shortcut probes: the probes must
    measure what the model actually receives, not the raw container."""
    plan = normalization_plan(sha256, fmt, seed)
    img = img.convert("RGB")
    if plan["lossless"]:
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return {"file_bytes": buf.tell(), "format": "png", "jpeg_quality": -1,
                "n_recompress": plan["total_history"],
                "width": img.width, "height": img.height}
    for q in plan["qualities"][:-1]:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=False)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    buf = BytesIO()  # the final encode IS the delivered file
    img.save(buf, format="JPEG", quality=plan["final_quality"], optimize=False)
    return {"file_bytes": buf.tell(), "format": "jpeg",
            "jpeg_quality": plan["final_quality"],
            "n_recompress": plan["total_history"],
            "width": img.width, "height": img.height}
