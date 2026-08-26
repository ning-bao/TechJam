"""Shared inference plumbing for both prediction entry points, so the
calibration formula exists exactly once.

  scripts/predict.py -> INTERFACES §7 schema (meta/predictions/errors)
  src/predict.py     -> organiser record schema ({"image_path", "pred"})
"""

import numpy as np
import torch

from track5.utils.hashing import file_sha256


def load_checkpoint(path, device: str = "cpu"):
    """-> (model in eval mode on `device`, calibration, crop, meta)."""
    from track5.models import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    cal = ckpt.get("calibration") or {}

    def _or(key, default):
        v = cal.get(key)
        return float(v) if v is not None else float(default)

    calib = {"temperature": _or("temperature", 1.0), "alpha": _or("alpha", 0.0),
             "threshold": _or("threshold", 0.5)}
    crop = int(ckpt["config"].get("data", {}).get("crop", 448))
    meta = {"model_hash": file_sha256(path)[:12],
            "config_hash": ckpt.get("meta", {}).get("config_hash", "unknown"),
            "backbone": model.backbone_name,
            "step": ckpt.get("meta", {}).get("step", -1)}
    return model, calib, crop, meta


def calibrated_prob(logits, calib: dict) -> np.ndarray:
    """p(AIGC) = sigmoid((z + alpha) / T). Never a raw logit (TC2 §6)."""
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(z + calib["alpha"]) / calib["temperature"]))
