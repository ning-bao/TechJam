"""Atomic, resumable checkpointing — TC2 guide §10 contract.

Two artifact families, deliberately separate:
  * `best.pt` / `last.pt` / `final_weights.pt` keep the INTERFACES §6 shape
    exactly (4 keys) — those are what predict.py / eval consume.
  * `checkpoints/checkpoint_step_<n>.pt` is a *recovery* checkpoint: the same 4
    keys plus a "resume" block with optimizer/EMA/RNG/data position. A file that
    merely exists is not a recoverable checkpoint, so `latest.json` is written
    only after the .pt rename completed and the file re-read.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

REQUIRED_KEYS = ("state_dict", "config", "calibration", "meta")
RESUME_KEYS = ("global_step", "epoch", "batch_in_epoch", "optimizer", "ema",
               "rng", "best", "fingerprint")


def atomic_save(obj, path) -> Path:
    """torch.save via <path>.tmp -> fsync -> rename (same filesystem)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(text: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(obj, path) -> Path:
    return atomic_write_text(json.dumps(obj, indent=1, default=str), path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Best-effort RNG restore. Exact replay can still vary across PyTorch
    releases / nondeterministic kernels (TC2 §10.3)."""
    random.setstate(tuple(state["python"]) if isinstance(state["python"], list)
                    else state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    cuda_states = state.get("cuda") or []
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cuda_states])


def run_fingerprint(cfg: dict, manifest_hash: str = "") -> dict:
    """The compatibility-critical subset (TC2 §10.5). `--resume auto` refuses a
    checkpoint whose fingerprint differs unless the override flag is given."""
    m = cfg.get("model", {})
    t = cfg.get("train", {})
    d = cfg.get("data", {})
    return {
        "backbone": m.get("backbone", "stub" if m.get("stub") else ""),
        "fallback_backbone": m.get("fallback_backbone", ""),
        "head": m.get("head", "linear"),
        "crop": int(d.get("crop", 0)),
        "effective_batch": int(d.get("batch_size", 1)) * int(t.get("grad_accum", 1)),
        "optimizer": "adamw",
        "loss": t.get("loss", "bce"),
        "distortion": bool(cfg.get("distortion", {}).get("enabled", True)),
        "manifest_sha256": manifest_hash,
    }


def fingerprint_diff(want: dict, have: dict) -> dict:
    return {k: (have.get(k), want.get(k)) for k in want if have.get(k) != want.get(k)}


def write_latest(ckpt_dir, path: Path, step: int) -> Path:
    """Point latest.json at a checkpoint only after verifying it loads."""
    validate_checkpoint(path)
    return atomic_write_json(
        {"path": path.name, "step": int(step), "bytes": path.stat().st_size},
        Path(ckpt_dir) / "latest.json")


def read_latest(ckpt_dir) -> Path | None:
    marker = Path(ckpt_dir) / "latest.json"
    if not marker.exists():
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    p = Path(ckpt_dir) / info["path"]
    return p if p.exists() else None


def validate_checkpoint(path, require_resume: bool = False) -> dict:
    """Load on CPU and check the required keys. Raises on a corrupt/partial file."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: checkpoint missing keys {missing}")
    if require_resume:
        r = ckpt.get("resume") or {}
        missing = [k for k in RESUME_KEYS if k not in r]
        if missing:
            raise ValueError(f"{path}: resume block missing keys {missing}")
    return ckpt


def prune_checkpoints(ckpt_dir, keep: int = 2) -> list[Path]:
    """Delete recovery checkpoints beyond the newest `keep`, only once the newest
    one has been verified loadable (TC2 §10.8). Returns the deleted paths."""
    ckpt_dir = Path(ckpt_dir)
    files = sorted(ckpt_dir.glob("checkpoint_step_*.pt"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if len(files) <= keep:
        return []
    validate_checkpoint(files[-1])
    removed = []
    for p in files[:-keep]:
        p.unlink()
        removed.append(p)
    return removed
