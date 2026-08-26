"""src/train.py + Trainer — the TC2 guide §10 checkpoint-and-resume contract:
atomic writes, complete resume state, compatibility refusal, signal handling and
the application wall-clock budget.
"""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from track5.train.checkpoint import (RESUME_KEYS, prune_checkpoints, read_latest,
                                     run_fingerprint, validate_checkpoint)

REPO = Path(__file__).resolve().parents[1]


def run_train(cfg_path, run_dir, fixture_root, cache_dir, *extra):
    return subprocess.run(
        [sys.executable, "-u", "-m", "src.train",
         "--config", str(cfg_path), "--run-dir", str(run_dir),
         "--data-root", str(fixture_root["raw"]), "--cache-dir", str(cache_dir),
         "--device", "cpu", "--skip-denylist-check", *extra],
        capture_output=True, text=True, cwd=str(REPO))


@pytest.fixture(scope="module")
def segment(tmp_path_factory, stub_config, fixture_root):
    """One completed 4-step segment, reused by the read-only assertions."""
    base = tmp_path_factory.mktemp("seg")
    run_dir = base / "run"
    res = run_train(stub_config["path"], run_dir, fixture_root, base / "cache",
                    "--max-steps", "4")
    assert res.returncode == 0, res.stdout + res.stderr
    return {"run_dir": run_dir, "base": base, "res": res}


def test_segment_writes_all_artifacts(segment):
    run = segment["run_dir"]
    for name in ("best.pt", "last.pt", "final_weights.pt", "segment_complete.json",
                 "log.jsonl", "checkpoints/latest.json",
                 "provenance/run_receipt.json"):
        assert (run / name).exists(), f"missing {name}\n{segment['res'].stdout}"
    assert not list(run.rglob("*.tmp")), "atomic write left a .tmp behind"


def test_interfaces_checkpoint_shape_unchanged(segment):
    ckpt = torch.load(segment["run_dir"] / "best.pt", weights_only=False)
    assert set(ckpt) == {"state_dict", "config", "calibration", "meta"}
    assert set(ckpt["calibration"]) == {"temperature", "alpha", "threshold"}


def test_recovery_checkpoint_is_complete(segment):
    latest = read_latest(segment["run_dir"] / "checkpoints")
    assert latest is not None
    ckpt = validate_checkpoint(latest, require_resume=True)
    r = ckpt["resume"]
    for key in RESUME_KEYS:
        assert key in r, key
    for key in ("scaler", "rng", "resume_mode", "precision"):
        assert key in r, key
    assert set(r["rng"]) == {"python", "numpy", "torch", "cuda"}
    assert r["resume_mode"] == "exact"          # loader_factory is wired
    assert r["global_step"] == 4
    assert r["optimizer"]["state"]              # real optimizer state, not empty


def test_latest_json_points_at_a_loadable_file(segment):
    info = json.loads(
        (segment["run_dir"] / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    path = segment["run_dir"] / "checkpoints" / info["path"]
    assert path.exists() and path.stat().st_size == info["bytes"]
    validate_checkpoint(path, require_resume=True)


def test_segment_complete_marker(segment):
    seg = json.loads(
        (segment["run_dir"] / "segment_complete.json").read_text(encoding="utf-8"))
    assert seg["global_step"] == 4
    assert seg["total_steps"] == 4
    assert seg["complete"] is True
    assert seg["stop_reason"] == "max_steps"


def test_resume_auto_continues_from_the_checkpoint(tmp_path_factory, stub_config,
                                                   fixture_root):
    base = tmp_path_factory.mktemp("resume")
    run_dir = base / "run"
    cache = base / "cache"
    first = run_train(stub_config["path"], run_dir, fixture_root, cache, "--max-steps", "4")
    assert first.returncode == 0, first.stdout + first.stderr

    second = run_train(stub_config["path"], run_dir, fixture_root, cache,
                       "--max-steps", "8", "--resume", "auto")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "resumed" in second.stdout
    assert "step=4" in second.stdout

    seg = json.loads((run_dir / "segment_complete.json").read_text(encoding="utf-8"))
    assert seg["global_step"] == 8 and seg["complete"] is True


def test_resume_auto_without_a_checkpoint_starts_fresh(tmp_path_factory, stub_config,
                                                       fixture_root):
    base = tmp_path_factory.mktemp("fresh")
    res = run_train(stub_config["path"], base / "run", fixture_root, base / "cache",
                    "--max-steps", "2", "--resume", "auto")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "no checkpoint yet" in res.stdout


def test_resume_refuses_a_changed_run_definition(tmp_path_factory, stub_config,
                                                 fixture_root):
    base = tmp_path_factory.mktemp("mismatch")
    run_dir = base / "run"
    cache = base / "cache"
    assert run_train(stub_config["path"], run_dir, fixture_root, cache,
                     "--max-steps", "4").returncode == 0

    changed = dict(stub_config["cfg"])
    changed["train"] = {**changed["train"], "grad_accum": 2}   # effective batch changed
    other = base / "changed.yaml"
    other.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")

    res = run_train(other, run_dir, fixture_root, cache, "--max-steps", "8",
                    "--resume", "auto")
    assert res.returncode != 0
    assert "run definition changed" in res.stdout + res.stderr
    assert "effective_batch" in res.stdout + res.stderr

    ok = run_train(other, run_dir, fixture_root, cache, "--max-steps", "8",
                   "--resume", "auto", "--allow-config-change")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "config change overridden" in ok.stdout


def test_wall_budget_stops_the_segment_early(tmp_path_factory, stub_config,
                                             fixture_root):
    base = tmp_path_factory.mktemp("wall")
    run_dir = base / "run"
    res = run_train(stub_config["path"], run_dir, fixture_root, base / "cache",
                    "--max-steps", "1000", "--max-wall-minutes", "0.02",
                    "--grace-minutes", "0")
    assert res.returncode == 0, res.stdout + res.stderr
    seg = json.loads((run_dir / "segment_complete.json").read_text(encoding="utf-8"))
    assert seg["stop_reason"] == "wall_budget"
    assert seg["complete"] is False
    assert 0 < seg["global_step"] < 1000


def test_fingerprint_only_tracks_compat_critical_fields():
    a = {"model": {"backbone": "x"}, "train": {"grad_accum": 2, "epochs": 1},
         "data": {"batch_size": 4, "crop": 448}}
    b = {"model": {"backbone": "x"}, "train": {"grad_accum": 2, "epochs": 99},
         "data": {"batch_size": 4, "crop": 448}}
    assert run_fingerprint(a) == run_fingerprint(b)     # epochs may change
    c = {**b, "data": {"batch_size": 8, "crop": 448}}
    assert run_fingerprint(a) != run_fingerprint(c)     # effective batch may not


# ------------------------------------------------------------------- signals

def _tiny_trainer(tmp_path, on_step=None):
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from track5.train.loop import Trainer

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 2, 3, stride=4)
            self.head = nn.Linear(2, 1)

        def forward(self, x):
            return self.head(self.backbone(x).mean((2, 3))).squeeze(-1)

    class DS(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"pixels": torch.randn(3, 16, 16, generator=g),
                    "label": i % 2, "idx": i}

    cfg = {"seed": 1, "model": {"stub": True},
           "train": {"epochs": 50, "max_steps": 500, "lr_head": 1e-3,
                     "lr_backbone": 1e-4, "weight_decay": 0.0, "warmup_frac": 0.1,
                     "loss": "bce", "ema_decay": 0.9, "precision": "fp32",
                     "grad_accum": 2},
           "eval": {"every_steps": 1000}}
    calls = {"n": 0}

    def eval_fn(_m):
        calls["n"] += 1
        return {"worst_case_bacc": 0.5}

    t = Trainer(cfg, M(), DataLoader(DS(), batch_size=4), eval_fn, tmp_path / "sig")
    if on_step is not None:
        original = t._lr_step

        def hooked(step):
            original(step)
            on_step(t, step)
        t._lr_step = hooked
    return t


def test_signal_handlers_are_armed(tmp_path):
    t = _tiny_trainer(tmp_path)
    armed = t.install_signal_handlers()
    assert "SIGTERM" in armed
    if hasattr(signal, "SIGUSR1"):        # POSIX / TC2; absent on Windows
        assert "SIGUSR1" in armed
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_sigterm_stops_at_an_accumulation_boundary(tmp_path):
    """The handler only sets a flag; the loop must finish the in-flight
    accumulation window, checkpoint, and exit cleanly."""
    def fire(trainer, step):
        if step == 3:
            trainer._on_signal(signal.SIGTERM, None)

    t = _tiny_trainer(tmp_path, on_step=fire)
    t.train()
    seg = json.loads((tmp_path / "sig" / "segment_complete.json").read_text(
        encoding="utf-8"))
    assert seg["stop_reason"] == "signal:SIGTERM"
    assert seg["signal"] == "SIGTERM"
    assert seg["complete"] is False
    assert seg["global_step"] == 4          # step 3 completed, then stopped
    latest = read_latest(tmp_path / "sig" / "checkpoints")
    assert validate_checkpoint(latest, require_resume=True)["resume"]["global_step"] == 4


def test_prune_keeps_the_newest_two(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    payload = {"state_dict": {}, "config": {}, "calibration": {}, "meta": {}}
    for step in (1, 2, 3, 4):
        torch.save(payload, d / f"checkpoint_step_{step}.pt")
    removed = prune_checkpoints(d, keep=2)
    assert {p.name for p in removed} == {"checkpoint_step_1.pt", "checkpoint_step_2.pt"}
    assert sorted(p.name for p in d.glob("*.pt")) == ["checkpoint_step_3.pt",
                                                      "checkpoint_step_4.pt"]
