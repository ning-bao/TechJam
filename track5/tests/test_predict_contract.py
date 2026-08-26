"""End-to-end contract test for scripts/predict.py (INTERFACES §7) — stub
model, no downloads, run as a real subprocess."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from track5.models import build_model
from track5.utils.config import config_hash

REPO = Path(__file__).resolve().parents[1]

META_KEYS = {"model_hash", "config_hash", "temperature", "alpha", "threshold",
             "n_ok", "n_err"}


def make_ckpt(path: Path, T: float):
    cfg = {"model": {"stub": True, "pretrained": False, "freeze_backbone": False},
           "data": {"crop": 64}, "seed": 1}
    torch.manual_seed(0)  # same stub weights for both checkpoints
    model = build_model(cfg)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg,
        "calibration": {"temperature": T, "alpha": 0.5, "threshold": 0.6},
        "meta": {"config_hash": config_hash(cfg), "epoch": 0, "step": 0,
                 "metrics": {}, "code_version": "test"},
    }, path)


def make_inputs(d: Path):
    d.mkdir()
    rng = np.random.Generator(np.random.PCG64(0))
    for i in range(3):
        Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)).save(
            d / f"img{i}.png")
    (d / "corrupt.png").write_bytes(b"not an image")


def run_predict(ckpt: Path, inputs: Path, out: Path):
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "predict.py"),
         "--input", str(inputs), "--checkpoint", str(ckpt), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO))
    return res


def test_predict_contract(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    make_ckpt(ckpt, T=2.0)
    inputs = tmp_path / "imgs"
    make_inputs(inputs)
    out = tmp_path / "preds.json"

    res = run_predict(ckpt, inputs, out)
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))

    assert META_KEYS <= set(data["meta"])
    assert data["meta"]["n_ok"] == 3 and data["meta"]["n_err"] == 1
    assert len(data["predictions"]) == 3
    for p in data["predictions"]:
        assert 0.0 <= p["score"] <= 1.0
        assert p["label"] == int(p["score"] >= 0.6)
        assert "corrupt" not in p["path"]
    assert len(data["errors"]) == 1
    assert "corrupt.png" in data["errors"][0]["path"]
    assert data["errors"][0]["error"]


def test_calibration_actually_applied(tmp_path):
    inputs = tmp_path / "imgs"
    make_inputs(inputs)
    scores = {}
    for T in (2.0, 8.0):
        ckpt = tmp_path / f"ckpt_{T}.pt"
        make_ckpt(ckpt, T=T)
        out = tmp_path / f"preds_{T}.json"
        res = run_predict(ckpt, inputs, out)
        assert res.returncode == 0, res.stdout + res.stderr
        data = json.loads(out.read_text(encoding="utf-8"))
        scores[T] = [p["score"] for p in sorted(data["predictions"],
                                                key=lambda x: x["path"])]
    assert scores[2.0] != scores[8.0]  # temperature changes scores
