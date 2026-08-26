"""PLAN D5 / report §9.2 training distortion sampler.

Deliberately separate from eval_atoms.py: this module is stochastic, eval atoms
are frozen+deterministic. All randomness flows through a numpy Generator.
Banned by D5 (do not add): MixUp, CutMix, hue, solarize.
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# family -> weight, given that a corruption draw happens
FAMILY_WEIGHTS = {
    "jpeg": 0.30,
    "resize": 0.20,
    "blur": 0.15,
    "noise": 0.15,
    "jitter": 0.10,
    "crop": 0.10,
}
N_CORRUPT_P = (0.30, 0.55, 0.15)  # P(clean), P(one), P(two)

RESIZE_KERNELS = [
    Image.Resampling.NEAREST,
    Image.Resampling.BILINEAR,
    Image.Resampling.BICUBIC,
    Image.Resampling.LANCZOS,
]


def _jpeg_pil(img: Image.Image, q: int) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _jpeg_cv2(img: Image.Image, q: int) -> Image.Image:
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))


class TrainDistortionSampler:
    """Callable: (PIL image, np.random.Generator) -> PIL image."""

    def __init__(self, cfg: dict | None = None, generator: np.random.Generator | None = None):
        self.cfg = cfg or {}
        self.generator = generator
        self._names = list(FAMILY_WEIGHTS)
        self._weights = np.array([FAMILY_WEIGHTS[n] for n in self._names])
        self._weights = self._weights / self._weights.sum()

    def __call__(self, img: Image.Image, rng: np.random.Generator | None = None) -> Image.Image:
        rng = rng if rng is not None else self.generator
        if rng is None:
            raise ValueError("TrainDistortionSampler needs a numpy Generator")
        img = img.convert("RGB")
        n = int(rng.choice(3, p=N_CORRUPT_P))
        if n == 0:
            return img
        fams = list(rng.choice(self._names, size=n, replace=False, p=self._weights))
        # noise half before / half after JPEG when both are drawn
        if "jpeg" in fams and "noise" in fams and rng.random() < 0.5:
            fams.sort(key=lambda f: 0 if f == "noise" else 1)
        for fam in fams:
            img = self._apply(fam, img, rng)
        return img

    def _apply(self, fam: str, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        if fam == "jpeg":
            q = int(rng.integers(25, 101))
            enc = _jpeg_cv2 if rng.random() < 0.5 else _jpeg_pil
            img = enc(img, q)
            if rng.random() < 0.5:  # balanced single/double compression history
                q2 = int(rng.integers(25, 101))
                enc2 = _jpeg_cv2 if rng.random() < 0.5 else _jpeg_pil
                img = enc2(img, q2)
            return img
        if fam == "resize":
            f = float(rng.uniform(0.20, 1.00))
            k = RESIZE_KERNELS[int(rng.integers(len(RESIZE_KERNELS)))]
            w0, h0 = img.width, img.height
            img = img.resize((max(1, round(w0 * f)), max(1, round(h0 * f))), k)
            # 50%: upscale back to original size (eval-atom thumbnail semantics),
            # 50%: stays rescaled — balanced so training sees both regimes.
            if rng.random() < 0.5:
                k2 = RESIZE_KERNELS[int(rng.integers(len(RESIZE_KERNELS)))]
                img = img.resize((w0, h0), k2)
            return img
        if fam == "blur":
            sigma = float(rng.uniform(0.0, 2.3))
            return img.filter(ImageFilter.GaussianBlur(radius=sigma)) if sigma > 1e-3 else img
        if fam == "noise":
            sigma = float(rng.uniform(0.0, 0.11))
            if sigma < 1e-4:
                return img
            arr = np.asarray(img, dtype=np.float32)
            arr = arr + rng.normal(0.0, sigma * 255.0, size=arr.shape)
            return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
        if fam == "jitter":
            b, c, s = rng.uniform(0.75, 1.25, size=3)
            img = ImageEnhance.Brightness(img).enhance(float(b))
            img = ImageEnhance.Contrast(img).enhance(float(c))
            return ImageEnhance.Color(img).enhance(float(s))  # no hue
        if fam == "crop":
            frac = float(rng.uniform(0.75, 1.00))
            cw = max(1, round(img.width * frac))
            ch = max(1, round(img.height * frac))
            x0 = int(rng.integers(0, img.width - cw + 1))
            y0 = int(rng.integers(0, img.height - ch + 1))
            return img.crop((x0, y0, x0 + cw, y0 + ch))
        raise ValueError(f"unknown family {fam!r}")
