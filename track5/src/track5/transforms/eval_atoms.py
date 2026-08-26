"""The 15 frozen eval conditions (+ crop_80_area variant). FROZEN Day 0.

Determinism contract: apply_and_encode(src_bytes, atom, seed) returns identical
bytes for identical inputs across runs and processes (fixed Pillow version).

JPEG atoms: the transformation IS the encode. apply_and_encode does exactly one
compression (decode -> encode at Q). apply_atom(jpeg_*) also degrades pixels
(roundtrip) for in-memory use — do NOT compose apply_atom + encode_atom for
jpeg atoms yourself (double compression); use apply_and_encode.
"""

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ATOMS_VERSION = "2.0"  # 2.0: resize atoms are down-then-up (track spec: "then upscale")

ATOMS = {
    "clean": {"kind": "clean"},
    "jpeg_90": {"kind": "jpeg", "q": 90},
    "jpeg_70": {"kind": "jpeg", "q": 70},
    "jpeg_50": {"kind": "jpeg", "q": 50},
    "jpeg_30": {"kind": "jpeg", "q": 30},
    "blur_05": {"kind": "blur", "sigma": 0.5},
    "blur_10": {"kind": "blur", "sigma": 1.0},
    "blur_20": {"kind": "blur", "sigma": 2.0},
    "resize_050": {"kind": "resize", "factor": 0.50},
    "resize_025": {"kind": "resize", "factor": 0.25},
    "noise_002": {"kind": "noise", "sigma": 0.02},
    "noise_005": {"kind": "noise", "sigma": 0.05},
    "noise_010": {"kind": "noise", "sigma": 0.10},
    "jitter_pm20": {"kind": "jitter", "lo": 0.8, "hi": 1.2},
    "crop_80": {"kind": "crop", "frac": 0.8, "convention": "side"},
    # 16th atom, NOT in the standard 15 — D9 side-vs-area ambiguity check only.
    "crop_80_area": {"kind": "crop", "frac": 0.8, "convention": "area"},
}

EVAL_15 = [
    "clean",
    "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
    "blur_05", "blur_10", "blur_20",
    "resize_050", "resize_025",
    "noise_002", "noise_005", "noise_010",
    "jitter_pm20",
    "crop_80",
]

# The TC2 guide (§11) spells the same 15 conditions differently in its sbatch
# array. Aliases only — the canonical names above stay frozen (INTERFACES §2).
TC2_ALIASES = {
    "clean": "clean",
    "jpeg_q90": "jpeg_90", "jpeg_q70": "jpeg_70",
    "jpeg_q50": "jpeg_50", "jpeg_q30": "jpeg_30",
    "blur_0.5": "blur_05", "blur_1.0": "blur_10", "blur_2.0": "blur_20",
    "resize_0.5": "resize_050", "resize_0.25": "resize_025",
    "noise_0.02": "noise_002", "noise_0.05": "noise_005", "noise_0.10": "noise_010",
    "color_jitter_0.20": "jitter_pm20",
    "center_crop_0.80": "crop_80",
}


def canonical_atom(name: str) -> str:
    """Canonical atom name from either spelling; raises on an unknown one."""
    if name in ATOMS:
        return name
    if name in TC2_ALIASES:
        return TC2_ALIASES[name]
    raise KeyError(f"unknown eval condition {name!r}")


def _jpeg_roundtrip(img: Image.Image, q: int) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_atom(img: Image.Image, atom: str, seed: int) -> Image.Image:
    spec = ATOMS[atom]
    img = img.convert("RGB")
    kind = spec["kind"]
    if kind == "clean":
        return img
    if kind == "jpeg":
        return _jpeg_roundtrip(img, spec["q"])
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=spec["sigma"]))
    if kind == "resize":
        # track spec: "scale 0.5x / 0.25x then upscale" — thumbnail round-trip;
        # output keeps the original size, high frequencies do not survive.
        f = spec["factor"]
        w0, h0 = img.width, img.height
        w = max(1, round(w0 * f))
        h = max(1, round(h0 * f))
        small = img.resize((w, h), Image.Resampling.BICUBIC)
        return small.resize((w0, h0), Image.Resampling.BICUBIC)
    if kind == "noise":
        rng = np.random.Generator(np.random.PCG64(seed))
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, spec["sigma"] * 255.0, size=arr.shape)
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if kind == "jitter":
        rng = np.random.Generator(np.random.PCG64(seed))
        b, c, s = rng.uniform(spec["lo"], spec["hi"], size=3)
        img = ImageEnhance.Brightness(img).enhance(float(b))
        img = ImageEnhance.Contrast(img).enhance(float(c))
        img = ImageEnhance.Color(img).enhance(float(s))  # saturation; NO hue
        return img
    if kind == "crop":
        frac = spec["frac"] if spec["convention"] == "side" else spec["frac"] ** 0.5
        cw = max(1, round(img.width * frac))
        ch = max(1, round(img.height * frac))
        x0 = (img.width - cw) // 2
        y0 = (img.height - ch) // 2
        return img.crop((x0, y0, x0 + cw, y0 + ch))
    raise ValueError(f"unknown atom kind {kind!r}")


def encode_atom(img: Image.Image, atom: str) -> bytes:
    spec = ATOMS[atom]
    buf = BytesIO()
    if spec["kind"] == "jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=spec["q"], optimize=False)
    else:
        img.convert("RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def apply_and_encode(src_bytes: bytes, atom: str, seed: int) -> bytes:
    img = Image.open(BytesIO(src_bytes)).convert("RGB")
    spec = ATOMS[atom]
    if spec["kind"] == "jpeg":
        # exactly one compression: decode -> encode at Q
        return encode_atom(img, atom)
    return encode_atom(apply_atom(img, atom, seed), atom)
