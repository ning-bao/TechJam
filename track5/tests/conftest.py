"""Shared 24-image fixture (TC2 guide §6: make the entry-point interfaces a CI
gate using a small image fixture).

Images are generated deterministically, so nothing binary is committed and the
fixture is byte-identical on every machine. The layout mirrors the real one:
a read-only data root, a parquet manifest with the INTERFACES §4 schema, and
train/dev splits.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from track5.data.manifest import SCHEMA, row_from_bytes

N_IMAGES = 24
N_TRAIN = 16
SIZES = [(96, 96), (120, 96), (96, 128), (160, 112)]


def _image_bytes(i: int) -> tuple[bytes, str]:
    """Half JPEG / half PNG, varied sizes — a stand-in for a mixed corpus."""
    from io import BytesIO

    w, h = SIZES[i % len(SIZES)]
    rng = np.random.Generator(np.random.PCG64(1000 + i))
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    arr[:, ::4, :] = (i * 9) % 256  # some structure so JPEG is not degenerate
    buf = BytesIO()
    if i % 2 == 0:
        Image.fromarray(arr).save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "jpg"
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue(), "png"


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> dict:
    """-> {"root", "raw", "manifest", "files", "labels"} for a 24-image corpus."""
    root = tmp_path_factory.mktemp("fixture")
    raw = root / "raw" / "fixtures"
    raw.mkdir(parents=True)

    rows, files = [], []
    for i in range(N_IMAGES):
        label = i % 2  # 12 real / 12 fake, interleaved
        data, ext = _image_bytes(i)
        name = f"{'fake' if label else 'real'}_{i:02d}.{ext}"
        (raw / name).write_bytes(data)
        rel = f"fixtures/{name}"
        row = row_from_bytes(rel, data, label, "fixture",
                             "sd" if label else "")
        row["split"] = "train" if i < N_TRAIN else "dev"
        rows.append(row)
        files.append(raw / name)

    df = pd.DataFrame(rows, columns=list(SCHEMA)).astype(SCHEMA)
    manifest = root / "manifest.parquet"
    df.to_parquet(manifest, index=False)
    return {"root": root, "raw": root / "raw", "manifest": manifest,
            "files": files, "labels": [i % 2 for i in range(N_IMAGES)], "df": df}


@pytest.fixture(scope="session")
def stub_config(fixture_root, tmp_path_factory) -> dict:
    """A tiny stub-model config wired to the fixture manifest."""
    cfg = {
        "seed": 17,
        "data": {"train_manifest": str(fixture_root["manifest"]),
                 "dev_manifest": str(fixture_root["manifest"]),
                 "crop": 64, "batch_size": 2, "workers": 0},
        "model": {"backbone": "stub", "head": "linear", "pool": "cls",
                  "pretrained": False, "stub": True, "freeze_backbone": False},
        "train": {"epochs": 1, "lr_head": 1e-3, "lr_backbone": 1e-4,
                  "weight_decay": 0.0, "warmup_frac": 0.25, "loss": "bce",
                  "ema_decay": 0.9, "swa": False, "precision": "fp32",
                  "grad_accum": 1, "activation_checkpointing": False},
        "distortion": {"enabled": True},
        "eval": {"every_steps": 4, "dev_limit": 8,
                 "worst_case_atoms": ["clean", "jpeg_30"]},
    }
    path = tmp_path_factory.mktemp("cfg") / "stub.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"cfg": cfg, "path": path}


@pytest.fixture(scope="session")
def stub_checkpoint(stub_config, tmp_path_factory):
    """An INTERFACES §6 checkpoint holding a stub model with real calibration."""
    from track5.models import build_model
    from track5.utils.config import config_hash

    cfg = stub_config["cfg"]
    torch.manual_seed(0)
    model = build_model(cfg)
    path = tmp_path_factory.mktemp("ckpt") / "stub.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "calibration": {"temperature": 1.5, "alpha": 0.25, "threshold": 0.6},
                "meta": {"config_hash": config_hash(cfg), "epoch": 0, "step": 0,
                         "metrics": {}, "code_version": "test"}}, path)
    return path
