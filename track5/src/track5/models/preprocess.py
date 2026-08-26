"""Crop policy (PLAN D6): random/center crops at NATIVE resolution — never
resize the crop source; reflect-pad images smaller than the crop."""

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pad_reflect(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph == 0 and pw == 0:
        return arr
    # numpy reflect requires pad < dim; tile in steps for tiny images
    while ph > 0 or pw > 0:
        h, w = arr.shape[:2]
        step_h = min(ph, h - 1) if h > 1 else ph
        step_w = min(pw, w - 1) if w > 1 else pw
        mode = "reflect" if (h > 1 and w > 1) else "edge"
        arr = np.pad(arr, ((step_h // 2, step_h - step_h // 2),
                           (step_w // 2, step_w - step_w // 2), (0, 0)), mode=mode)
        ph -= step_h
        pw -= step_w
    return arr


def train_crop(img: Image.Image, rng: np.random.Generator, size: int = 448) -> Image.Image:
    arr = _pad_reflect(np.asarray(img.convert("RGB")), size)
    h, w = arr.shape[:2]
    y0 = int(rng.integers(0, h - size + 1))
    x0 = int(rng.integers(0, w - size + 1))
    return Image.fromarray(arr[y0:y0 + size, x0:x0 + size])


def eval_crop(img: Image.Image, size: int = 448) -> Image.Image:
    arr = _pad_reflect(np.asarray(img.convert("RGB")), size)
    h, w = arr.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return Image.fromarray(arr[y0:y0 + size, x0:x0 + size])


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())
