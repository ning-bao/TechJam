"""Single-device trainer (PLAN D2/D5/D6, TC2 guide §10). Model selection =
worst-case bAcc over the D5 atom set, computed by the injected eval_fn.

Long training is a sequence of resumable segments, not one fragile job: the
trainer polls a monotonic wall-clock budget, answers SIGUSR1/SIGTERM, writes
atomic recovery checkpoints, and exits 0 so a dependent job can continue.
"""

import json
import math
import signal
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn

from track5.train.checkpoint import (atomic_save, atomic_write_json, capture_rng_state,
                                     fingerprint_diff, prune_checkpoints, read_latest,
                                     restore_rng_state, run_fingerprint,
                                     validate_checkpoint, write_latest)
from track5.train.ema import EMA
from track5.utils.config import config_hash
from track5.utils.seed import seed_everything


def focal_loss(logits, targets, gamma: float, alpha: float):
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p)
    w = torch.where(targets > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-w * (1 - pt) ** gamma * torch.log(pt.clamp_min(1e-8))).mean()


def build_optimizer(model, tcfg: dict):
    bb, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if n.startswith("backbone") else rest).append(p)
    groups = []
    if bb:
        groups.append({"params": bb, "lr": float(tcfg["lr_backbone"])})
    if rest:
        groups.append({"params": rest, "lr": float(tcfg["lr_head"])})
    return torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 0.05)))


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def resolve_precision(requested: str, device_type: str) -> tuple[str, str]:
    """TC2 §1: BF16 where supported, else FP16 + GradScaler. Never a silent FP32
    fallback — returns (precision, reason) and the caller logs the reason."""
    if device_type != "cuda":
        return "fp32", f"device={device_type}: autocast off"
    if requested == "fp32":
        return "fp32", "fp32 requested explicitly"
    bf16_ok = torch.cuda.is_bf16_supported()
    if requested in ("bf16", "auto"):
        if bf16_ok:
            return "bf16", "bf16 supported by this GPU"
        return "fp16", "bf16 NOT supported by this GPU -> fp16 + GradScaler"
    if requested == "fp16":
        return "fp16", "fp16 requested explicitly"
    raise ValueError(f"unknown precision {requested!r}")


class Trainer:
    """`loader_factory(epoch, start_index) -> iterable` enables exact data-position
    resume; without it the trainer resumes at an epoch boundary and says so."""

    def __init__(self, cfg: dict, model, train_loader, eval_fn, out_dir,
                 loader_factory=None, max_wall_minutes: float | None = None,
                 grace_minutes: float = 12.0, manifest_hash: str = "",
                 checkpoint_every_minutes: float | None = 35.0,
                 keep_checkpoints: int = 2):
        self.cfg = cfg
        self.model = model
        self.loader = train_loader
        self.loader_factory = loader_factory
        self.eval_fn = eval_fn
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.out / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(int(cfg.get("seed", 17)))

        tcfg = cfg["train"]
        self.device = next(model.parameters()).device
        self.opt = build_optimizer(model, tcfg)
        self.grad_accum = max(1, int(tcfg.get("grad_accum", 1)))
        epochs = float(tcfg.get("epochs", 1))
        self.epochs = epochs
        self.total_steps = int(cfg["train"].get(
            "max_steps", math.ceil(epochs * len(train_loader) / self.grad_accum)))
        self.warmup_steps = int(float(tcfg.get("warmup_frac", 0.0)) * self.total_steps)
        self.precision, self.precision_reason = resolve_precision(
            tcfg.get("precision", "bf16"), self.device.type)
        self.use_bf16 = self.precision == "bf16"
        self.use_fp16 = self.precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)
        self.ema = EMA(model, float(tcfg.get("ema_decay", 0.999)))
        self.swa_model = None
        if tcfg.get("swa", False):
            self.swa_model = torch.optim.swa_utils.AveragedModel(model)
        loss_name = tcfg.get("loss", "bce")
        if loss_name == "focal":
            g, a = float(tcfg.get("focal_gamma", 2.0)), float(tcfg.get("focal_alpha", 0.5))
            self.loss_fn = lambda z, y: focal_loss(z, y, g, a)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.eval_every = int(cfg.get("eval", {}).get("every_steps", 0)) or self.total_steps
        self.best = -float("inf")
        self.cfg_hash = config_hash(cfg)
        self.fingerprint = run_fingerprint(cfg, manifest_hash)

        self.max_wall_seconds = None if max_wall_minutes is None else max_wall_minutes * 60.0
        self.grace_seconds = grace_minutes * 60.0
        self.checkpoint_every = (None if checkpoint_every_minutes is None
                                 else checkpoint_every_minutes * 60.0)
        self.keep_checkpoints = keep_checkpoints
        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._stop_reason = None
        self._signal_name = None
        self._t0 = time.monotonic()
        self._last_ckpt_t = self._t0

    # ---------------------------------------------------------------- signals

    def install_signal_handlers(self) -> list[str]:
        """SIGUSR1 (Slurm `--signal=USR1@300`) and SIGTERM. SIGUSR1 does not
        exist on Windows; installed handlers are returned so callers can log
        what is actually armed."""
        if threading.current_thread() is not threading.main_thread():
            return []
        armed = []
        for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                continue
            armed.append(name)
        return armed

    def _on_signal(self, signum, frame):
        self._signal_name = signal.Signals(signum).name
        self._stop_reason = f"signal:{self._signal_name}"
        print(f"[train] {self._signal_name} received -> graceful stop at the next "
              f"accumulation boundary", flush=True)

    def _wall_exhausted(self) -> bool:
        if self.max_wall_seconds is None:
            return False
        return (time.monotonic() - self._t0) >= (self.max_wall_seconds - self.grace_seconds)

    # ------------------------------------------------------------ checkpoints

    def _base_ckpt(self, step: int, metrics: dict) -> dict:
        return {
            "state_dict": self.model.state_dict(),
            "config": self.cfg,
            "calibration": {"temperature": None, "alpha": None, "threshold": None},
            "meta": {"config_hash": self.cfg_hash, "epoch": self.epoch, "step": step,
                     "metrics": metrics, "code_version": "day0"},
        }

    def _checkpoint(self, path: Path, step: int, metrics: dict):
        atomic_save(self._base_ckpt(step, metrics), path)

    def _save_recovery(self, metrics: dict, tag: str = "") -> Path:
        payload = self._base_ckpt(self.step, metrics)
        payload["resume"] = {
            "global_step": self.step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "micro_in_accum": 0,  # only saved at accumulation boundaries
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_fp16 else None,
            "ema": self.ema.state_dict(),
            "swa": self.swa_model.state_dict() if self.swa_model is not None else None,
            "rng": capture_rng_state(),
            "best": self.best,
            "fingerprint": self.fingerprint,
            "resume_mode": "exact" if self.loader_factory else "epoch_boundary",
            "precision": self.precision,
            "tag": tag,
        }
        path = self.ckpt_dir / f"checkpoint_step_{self.step}.pt"
        atomic_save(payload, path)
        write_latest(self.ckpt_dir, path, self.step)
        for gone in prune_checkpoints(self.ckpt_dir, self.keep_checkpoints):
            print(f"[train] pruned {gone.name}", flush=True)
        self._last_ckpt_t = time.monotonic()
        print(f"[train] recovery checkpoint {path.name} ({tag or 'periodic'})", flush=True)
        return path

    def resume(self, spec: str = "auto", allow_config_change: bool = False) -> bool:
        """spec: "auto" (newest in checkpoints/), "none", or an explicit path."""
        if spec in ("none", "", None):
            return False
        path = read_latest(self.ckpt_dir) if spec == "auto" else Path(spec)
        if path is None or not Path(path).exists():
            if spec != "auto":
                raise FileNotFoundError(f"--resume {spec}: no such checkpoint")
            print("[train] --resume auto: no checkpoint yet, starting fresh", flush=True)
            return False

        ckpt = validate_checkpoint(path, require_resume=True)
        r = ckpt["resume"]
        diff = fingerprint_diff(self.fingerprint, r["fingerprint"])
        if diff:
            msg = "; ".join(f"{k}: checkpoint={old!r} config={new!r}"
                            for k, (old, new) in diff.items())
            if not allow_config_change:
                raise SystemExit(
                    f"--resume refused: run definition changed ({msg}). Use a new "
                    f"--run-dir, or pass --allow-config-change to override (logged).")
            print(f"[train] WARNING config change overridden: {msg}", flush=True)

        self.model.load_state_dict(ckpt["state_dict"])
        self.opt.load_state_dict(r["optimizer"])
        if self.use_fp16 and r.get("scaler"):
            self.scaler.load_state_dict(r["scaler"])
        self.ema.load_state_dict(r["ema"])
        if self.swa_model is not None and r.get("swa"):
            self.swa_model.load_state_dict(r["swa"])
        restore_rng_state(r["rng"])
        self.step = int(r["global_step"])
        self.epoch = int(r["epoch"])
        self.batch_in_epoch = int(r["batch_in_epoch"]) if self.loader_factory else 0
        self.best = float(r["best"])
        mode = "exact data position" if self.loader_factory else "epoch boundary"
        print(f"[train] resumed {Path(path).name}: step={self.step} epoch={self.epoch} "
              f"batch_in_epoch={self.batch_in_epoch} best={self.best:.4f} ({mode})",
              flush=True)
        return True

    # ------------------------------------------------------------------- eval

    def _lr_step(self, step: int):
        scale = cosine_warmup(step, self.total_steps, self.warmup_steps)
        for g, base in zip(self.opt.param_groups, self._base_lrs):
            g["lr"] = base * scale

    def _evaluate(self, step: int) -> dict:
        self.ema.copy_to(self.model)
        self.model.eval()
        try:
            metrics = self.eval_fn(self.model)
        finally:
            self.model.train()
            self.ema.restore(self.model)
        metrics = dict(metrics)
        metrics["step"] = step
        with open(self.out / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        wc = metrics.get("worst_case_bacc")
        if wc is None:
            raise ValueError("eval_fn must return 'worst_case_bacc' (PLAN D5)")
        # save best with EMA weights applied
        self.ema.copy_to(self.model)
        try:
            if wc > self.best:
                self.best = wc
                self._checkpoint(self.out / "best.pt", step, metrics)
            self._checkpoint(self.out / "last.pt", step, metrics)
        finally:
            self.ema.restore(self.model)
        return metrics

    # ------------------------------------------------------------------ train

    def _epoch_batches(self):
        if self.loader_factory is None:
            return self.loader
        bs = int(self.cfg.get("data", {}).get("batch_size", 1))
        return self.loader_factory(self.epoch, self.batch_in_epoch * bs)

    def train(self):
        self._base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.model.train()
        micro = 0
        self.opt.zero_grad(set_to_none=True)
        last_metrics: dict = {}
        done = False
        empty_epochs = 0
        while not done:
            consumed = 0
            for batch in self._epoch_batches():
                consumed += 1
                pixels = batch["pixels"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True).float()
                if self.use_bf16 or self.use_fp16:
                    dtype = torch.bfloat16 if self.use_bf16 else torch.float16
                    with torch.autocast("cuda", dtype=dtype):
                        loss = self.loss_fn(self.model(pixels), labels) / self.grad_accum
                else:
                    loss = self.loss_fn(self.model(pixels), labels) / self.grad_accum
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro += 1
                self.batch_in_epoch += 1
                if micro % self.grad_accum != 0:
                    continue

                if self.use_fp16:
                    self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._lr_step(self.step)
                if self.use_fp16:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.ema.update(self.model)
                if self.swa_model is not None and self.step > self.total_steps // 2:
                    self.swa_model.update_parameters(self.model)
                self.step += 1

                if self.step % self.eval_every == 0 or self.step >= self.total_steps:
                    last_metrics = self._evaluate(self.step)
                    self._save_recovery(last_metrics, tag="eval")
                elif (self.checkpoint_every is not None
                      and time.monotonic() - self._last_ckpt_t >= self.checkpoint_every):
                    self._save_recovery(last_metrics, tag="periodic")

                if self.step >= self.total_steps:
                    self._stop_reason = self._stop_reason or "max_steps"
                    done = True
                    break
                if self._wall_exhausted():
                    self._stop_reason = "wall_budget"
                    done = True
                    break
                if self._stop_reason is not None:
                    done = True
                    break
            else:
                self.epoch += 1
                self.batch_in_epoch = 0
                empty_epochs = empty_epochs + 1 if consumed == 0 else 0
                if empty_epochs >= 2:
                    # e.g. dataset smaller than batch_size with drop_last
                    self._stop_reason = "empty_loader"
                    print("[train] the loader yielded no batches twice in a row - "
                          "dataset smaller than the batch size?", flush=True)
                    done = True
                elif self.loader_factory is None and self.epoch >= math.ceil(self.epochs):
                    self._stop_reason = self._stop_reason or "epochs_exhausted"
                    done = True
                continue
            break

        if not (self.out / "last.pt").exists():
            last_metrics = self._evaluate(self.step)
        self._save_recovery(last_metrics, tag=self._stop_reason or "end")
        atomic_save(
            {"state_dict": self.model.state_dict(), "config": self.cfg,
             "calibration": {"temperature": None, "alpha": None, "threshold": None},
             "meta": {"config_hash": self.cfg_hash, "epoch": self.epoch,
                      "step": self.step, "metrics": last_metrics,
                      "code_version": "day0"}},
            self.out / "final_weights.pt")
        atomic_write_json(
            {"stop_reason": self._stop_reason or "end", "signal": self._signal_name,
             "global_step": self.step, "epoch": self.epoch,
             "batch_in_epoch": self.batch_in_epoch, "best_worst_case_bacc": self.best,
             "total_steps": self.total_steps,
             "complete": self.step >= self.total_steps,
             "elapsed_seconds": round(time.monotonic() - self._t0, 3),
             "precision": self.precision, "config_hash": self.cfg_hash},
            self.out / "segment_complete.json")
        return self.best
