"""DINOv3-L/16 qualification benchmark (TC2 guide §8).

The only valid answer to "how long will DINOv3-L/512 take?" is a measurement on
the allocated card, so this sweeps the real graph over resolution x precision x
micro-batch x activation-checkpointing, in both inference and full
forward+backward mode, and in two data regimes that are never mixed up:

  compute_only_random_tensor  — a device-resident random batch, pure compute
  end_to_end_dataloader       — decode + D5 augmentation + crop + collate + H2D

Every configuration is timed with torch.cuda.synchronize() around the compute
region, 20 warm-up steps and >=100 measured optimizer steps by default. A
configuration that OOMs is recorded and the sweep continues with smaller ones;
results are flushed to JSON after every configuration.

    python -u scripts/benchmark_gpu.py --out reports/bench.json
    python -u scripts/benchmark_gpu.py --resolutions 512 --modes train \
        --precisions bf16 --micro-batches 1 --checkpointing on --data dataloader
"""

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

PRIMARY = "facebook/dinov3-vitl16-pretrain-lvd1689m"
FALLBACK = "facebook/dinov2-with-registers-large"
SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
              "SLURM_MEM_PER_NODE")
COMPUTE_ONLY = "compute_only_random_tensor"
END_TO_END = "end_to_end_dataloader"


# --------------------------------------------------------------- environment

def nvidia_driver() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else ""
    except (OSError, IndexError, subprocess.SubprocessError):
        return ""


def environment_report() -> dict:
    import torch

    rep = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": nvidia_driver(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "slurm": {k: os.environ.get(k) for k in SLURM_KEYS if os.environ.get(k)},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rep.update({"gpu": p.name,
                    "vram_gib": round(p.total_memory / 2**30, 3),
                    "compute_capability": f"{p.major}.{p.minor}",
                    "multi_processor_count": p.multi_processor_count,
                    "bf16_supported": torch.cuda.is_bf16_supported()})
    return rep


def device_uuid() -> str:
    """`GPU-<uuid>` for the visible device, so utilization is attributed to the
    card this process actually got under CUDA_VISIBLE_DEVICES."""
    import torch

    if not torch.cuda.is_available():
        return ""
    try:
        return "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    except Exception:
        return ""


class GpuSampler:
    """One long-lived `nvidia-smi -lms` process for the whole run; each measured
    window is sliced out of the sample stream afterwards. Cheaper and less
    perturbing than spawning nvidia-smi per configuration, and it degrades to
    "unavailable" rather than failing when nvidia-smi is absent."""

    FIELDS = "uuid,utilization.gpu,utilization.memory,memory.used"

    def __init__(self, interval_ms: int = 250, uuid: str = ""):
        self.interval_ms = interval_ms
        self.uuid = uuid
        self.samples: list[tuple[float, int, int, int]] = []
        self.proc = None
        self.thread = None
        self.error = ""

    def _consume(self, line: str) -> bool:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            return False
        if self.uuid and parts[0] != self.uuid:
            return False
        try:
            self.samples.append((time.monotonic(), int(parts[1]), int(parts[2]),
                                 int(parts[3])))
        except ValueError:
            return False
        return True

    def sample_once(self) -> bool:
        """One blocking query, so the series is never empty even if the loop is
        slower than a short measured window."""
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={self.FIELDS}",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as e:
            self.error = self.error or f"{type(e).__name__}: {e}"
            return False
        return any(self._consume(ln) for ln in r.stdout.splitlines())

    def start(self):
        # take a datum before the caller's warm-up begins
        self.sample_once()
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.FIELDS}",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                bufsize=1)
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"
            return self
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            self._consume(line)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def window(self, t0: float, t1: float) -> dict:
        if self.error and not self.samples:
            return {"available": False, "reason": self.error}
        vals = [s for s in self.samples if t0 <= s[0] <= t1]
        widened = False
        if not vals and self.samples:
            # a window shorter than the sampling interval: fall back to the
            # nearest samples on either side and say so, rather than reporting
            # nothing for a configuration that did run
            before = [s for s in self.samples if s[0] <= t0]
            after = [s for s in self.samples if s[0] >= t1]
            vals = ([before[-1]] if before else []) + ([after[0]] if after else [])
            widened = bool(vals)
        if not vals:
            return {"available": False, "reason": "no samples in the measured window",
                    "interval_ms": self.interval_ms,
                    "window_s": round(t1 - t0, 4)}
        gpu = [v[1] for v in vals]
        mem = [v[2] for v in vals]
        used = [v[3] for v in vals]
        return {
            "available": True, "n_samples": len(vals), "interval_ms": self.interval_ms,
            "window_s": round(t1 - t0, 4), "widened_to_nearest_samples": widened,
            "gpu_util_pct": {"median": statistics.median(gpu),
                             "mean": round(statistics.fmean(gpu), 1),
                             "min": min(gpu), "max": max(gpu)},
            "memory_util_pct": {"median": statistics.median(mem),
                                "mean": round(statistics.fmean(mem), 1),
                                "max": max(mem)},
            "memory_used_mib": {"median": statistics.median(used), "max": max(used)},
        }


# ---------------------------------------------------------------------- data

def synthetic_images(out_dir: Path, n: int, size: int, seed: int = 17) -> list[Path]:
    """Deterministic source images, half JPEG half PNG, larger than any crop so
    the crop policy never pads."""
    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        ext = "jpg" if i % 2 == 0 else "png"
        f = out_dir / f"synth_{size}_{i:05d}.{ext}"
        if not f.exists():
            rng = np.random.Generator(np.random.PCG64(seed + i))
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            # some low-frequency structure so JPEG/blur costs are not degenerate
            arr[::8, :, :] = 0
            img = Image.fromarray(arr)
            img.save(f, quality=92) if ext == "jpg" else img.save(f)
        files.append(f)
    return files


def build_dataset(source, crop, train, seed, data_root):
    """`source` is a list of image files (synthetic) or a manifest parquet path.

    Everything reachable from the returned object must be picklable: Windows
    spawns worker processes and pickles the dataset, its sampler and
    worker_init_fn. Module objects, lambdas and locally defined classes fail with
    `TypeError: cannot pickle 'module' object`.
    """
    from track5.data.dataset import FolderDataset, ManifestDataset
    from track5.transforms.train_sampler import TrainDistortionSampler

    distortion = TrainDistortionSampler() if train else None
    if isinstance(source, (str, Path)):
        return ManifestDataset(source, split=None, crop=crop,
                               distortion_sampler=distortion, seed=seed,
                               data_root=data_root, train=train)
    return FolderDataset(source, crop=crop, distortion_sampler=distortion,
                         seed=seed, train=train)


def make_loader(source, crop, batch_size, workers, train, seed, data_root,
                multiprocessing_context=None):
    import torch
    from torch.utils.data import DataLoader

    from track5.data.sampler import seed_worker

    ds = build_dataset(source, crop, train, seed, data_root)
    kwargs = {}
    if workers > 0 and multiprocessing_context:
        kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True,
                      num_workers=workers, persistent_workers=workers > 0,
                      worker_init_fn=seed_worker if workers > 0 else None,
                      generator=torch.Generator().manual_seed(seed), **kwargs)


def cycle_batches(loader):
    while True:
        for b in loader:
            yield b


# ------------------------------------------------------------------ sweeping

def build_configs(args) -> list[dict]:
    """`--checkpointing auto` plans the off rows only; the on row for a given
    (mode, resolution, micro-batch) is inserted at run time, and only if the off
    row OOMed or exceeded the VRAM headroom."""
    auto = getattr(args, "checkpoint_auto", False)
    out = []
    for mode in args.modes:
        for data_path in args.data:
            for res in args.resolutions:
                for prec in args.precisions:
                    for mb in args.micro_batches:
                        ckpts = args.checkpointing if mode == "train" else [False]
                        for ck in ckpts:
                            workers = args.workers if data_path == END_TO_END else [0]
                            for w in workers:
                                out.append({
                                    "mode": mode, "data_path": data_path,
                                    "resolution": res, "precision": prec,
                                    "micro_batch": mb, "grad_accum": args.grad_accum,
                                    "effective_batch": mb * args.grad_accum,
                                    "activation_checkpointing": ck, "workers": w,
                                    "escalate_ckpt": auto and mode == "train" and not ck,
                                })
    # most memory-hungry first, so an OOM is followed by smaller settings
    out.sort(key=lambda c: (c["resolution"], c["micro_batch"],
                            c["mode"] == "train", not c["activation_checkpointing"]),
             reverse=True)
    return out


def headroom_pct_of(result: dict):
    """Peak allocation as a share of the memory this process can actually get.

    Prefers the available-based figure: on a workstation several GiB are already
    held by the desktop, so "% of total VRAM" overstates the headroom and would
    let a configuration look safe right up to the OOM.
    """
    pct = result.get("peak_allocated_pct_of_available")
    return pct if pct is not None else result.get("peak_allocated_pct")


def escalate_reason(result: dict, headroom_pct: float) -> str:
    """Why the activation-checkpointing-on row is worth running after this
    checkpointing-off row. Empty string = not needed."""
    if result["status"] == "oom":
        return "OOM without activation checkpointing"
    pct = headroom_pct_of(result)
    if pct is not None and pct > headroom_pct:
        basis = ("of available VRAM" if "peak_allocated_pct_of_available" in result
                 else "of total VRAM")
        return f"peak allocated {pct}% {basis} > {headroom_pct}% headroom"
    return ""


DEFINITIVE = ("ok", "oom", "skipped")


def config_key(cfg: dict) -> str:
    return "|".join(str(cfg[k]) for k in (
        "mode", "data_path", "resolution", "precision", "micro_batch",
        "grad_accum", "activation_checkpointing", "workers"))


def grid_fingerprint(model_cfg: dict, env: dict, args) -> str:
    """Rows from a different GPU, model or step budget are not comparable, so a
    resume against a mismatched fingerprint starts fresh instead of mixing them."""
    import hashlib

    payload = json.dumps({
        "model": model_cfg, "gpu": env.get("gpu"), "torch": env.get("torch"),
        "steps": args.steps, "warmup_steps": args.warmup_steps,
        "grad_accum": args.grad_accum,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_completed(out_path: Path, fingerprint: str) -> dict:
    """-> {config_key: preserved result} from a partially completed run."""
    if not out_path.exists():
        return {}
    try:
        prior = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[bench] --resume: cannot read {out_path} ({e}); starting fresh",
              flush=True)
        return {}
    if prior.get("settings", {}).get("grid_fingerprint") != fingerprint:
        print("[bench] --resume: the existing results were produced by a "
              "different model/GPU/step budget; starting fresh", flush=True)
        return {}
    return {config_key(r): r for r in prior.get("results", [])
            if r.get("status") in DEFINITIVE}


def summarize(times: list[float]) -> dict:
    s = sorted(times)
    return {
        "median": round(statistics.median(s), 6),
        "p95": round(s[max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))], 6),
        "mean": round(statistics.fmean(s), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def run_config(model, cfg: dict, args, device, source, sampler=None) -> dict:
    import torch

    from track5.models.backbone import set_gradient_checkpointing

    res, mb, prec = cfg["resolution"], cfg["micro_batch"], cfg["precision"]
    train = cfg["mode"] == "train"
    cuda = device.type == "cuda"
    result = dict(cfg)

    if prec == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        result.update(status="skipped", reason="bf16 not supported by this GPU")
        return result
    if not cuda and prec in ("bf16", "fp16"):
        result.update(status="skipped",
                      reason=f"{prec} autocast needs CUDA; device is {device.type}")
        return result

    def sync():
        if cuda:
            torch.cuda.synchronize()

    model.train(train)
    result["activation_checkpointing_applied"] = set_gradient_checkpointing(
        model, cfg["activation_checkpointing"])

    opt = scaler = None
    loader = batches = None
    fixed: dict = {}
    try:
        if train:
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16" and cuda))

        if cfg["data_path"] == END_TO_END:
            loader = make_loader(source, res, mb, cfg["workers"], train, args.seed,
                                 args.data_root)
            batches = cycle_batches(loader)
            fetch = lambda: next(batches)  # noqa: E731
        else:
            fixed["x"] = torch.randn(mb, 3, res, res, device=device)
            fixed["y"] = torch.randint(0, 2, (mb,), device=device).float()
            fetch = lambda: (fixed["x"], fixed["y"])  # noqa: E731

        def to_device(batch):
            if isinstance(batch, tuple):
                return batch
            return (batch["pixels"].to(device, non_blocking=True),
                    batch["label"].to(device, non_blocking=True).float())

        autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)

        def compute(x, y):
            ctx = (torch.autocast("cuda", dtype=autocast_dtype)
                   if autocast_dtype is not None else _null())
            with ctx:
                z = model(x)
                return torch.nn.functional.binary_cross_entropy_with_logits(z, y)

        def step():
            """One timed unit = grad_accum micro-batches + one optimizer step."""
            data_s = 0.0
            if train:
                opt.zero_grad(set_to_none=True)
            for _ in range(cfg["grad_accum"]):
                t = time.perf_counter()
                batch = fetch()
                x, y = to_device(batch)
                data_s += time.perf_counter() - t
                if train:
                    loss = compute(x, y) / cfg["grad_accum"]
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    with torch.no_grad():
                        compute(x, y)
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
            return data_s

        for _ in range(args.warmup_steps):
            step()
        sync()
        other_b = free_before_b = total_b = 0
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            free_before_b, total_b = torch.cuda.mem_get_info()
            # what the desktop compositor / other processes hold right now:
            # PyTorch's own peak_allocated says nothing about this, and on a
            # WDDM workstation it is several GiB.
            other_b = max(0, total_b - free_before_b - torch.cuda.memory_reserved())

        data_times, step_times = [], []
        window0 = time.perf_counter()
        mono0 = time.monotonic()
        n = 0
        deadline = window0 + args.min_measure_seconds
        # --min-measure-seconds is bounded by time, not by a step multiple: a
        # 32px config needs thousands of steps to fill a second, a 512px one
        # needs a handful. STEP_BACKSTOP only guards against a stalled clock.
        STEP_BACKSTOP = 1_000_000
        while n < args.steps or (time.perf_counter() < deadline
                                 and n < STEP_BACKSTOP):
            t0 = time.perf_counter()
            data_s = step()
            sync()
            t1 = time.perf_counter()
            data_times.append(data_s)
            step_times.append(t1 - t0)
            n += 1
        window = time.perf_counter() - window0
        mono1 = time.monotonic()

        imgs = mb * cfg["grad_accum"]
        med = max(statistics.median(step_times), 1e-9)  # guard trivially fast configs
        window = max(window, 1e-9)
        result.update({
            "status": "ok",
            "steps": args.steps,
            "steps_measured": n,
            "warmup_steps": args.warmup_steps,
            "views_per_image": 1,
            "step_time_s": summarize(step_times),
            "data_time_s": summarize(data_times),
            "images_per_s": {
                "from_median_step": round(imgs / med, 3),
                "from_total_window": round(imgs * n / window, 3),
            },
            "total_window_s": round(window, 3),
            "torch_compile": False,
            "gpu_utilization": (sampler.window(mono0, mono1) if sampler is not None
                                else {"available": False, "reason": "no sampler"}),
        })
        if cuda:
            free_after_b, _ = torch.cuda.mem_get_info()
            alloc = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            usable_b = max(1, total_b - other_b)
            result.update({
                "peak_allocated_gib": round(alloc / 2**30, 3),
                "peak_reserved_gib": round(reserved / 2**30, 3),
                "vram_total_gib": round(total_b / 2**30, 3),
                # honest headroom: total minus what other processes already hold
                "vram_usable_gib": round(usable_b / 2**30, 3),
                "device_used_by_others_gib": round(other_b / 2**30, 3),
                "device_free_before_gib": round(free_before_b / 2**30, 3),
                "device_free_after_gib": round(free_after_b / 2**30, 3),
                "peak_allocated_pct": round(100.0 * alloc / total_b, 2),
                "peak_reserved_pct": round(100.0 * reserved / total_b, 2),
                "peak_allocated_pct_of_available": round(100.0 * alloc / usable_b, 2),
                "peak_reserved_pct_of_available": round(100.0 * reserved / usable_b, 2),
            })
    except torch.cuda.OutOfMemoryError as e:
        result.update(status="oom", reason=str(e).splitlines()[0])
    except RuntimeError as e:
        kind = "oom" if "out of memory" in str(e).lower() else "error"
        result.update(status=kind,
                      reason=f"{type(e).__name__}: {str(e).splitlines()[0]}")
        if kind == "error":
            result["traceback"] = traceback.format_exc()
    except Exception as e:  # never let one configuration kill the grid
        result.update(status="error",
                      reason=f"{type(e).__name__}: {str(e).splitlines()[0]}",
                      traceback=traceback.format_exc())
    finally:
        fixed.clear()
        del opt, scaler, loader, batches
        if train:
            model.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return result


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------- main

def csv_list(cast):
    def parse(s):
        return [cast(x) for x in str(s).split(",") if str(x).strip() != ""]
    return parse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="",
                    help="optional YAML; --model/--resolutions override it")
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--fallback", default=FALLBACK)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="build the exact architecture without downloading weights")
    ap.add_argument("--stub", action="store_true", help="tiny stand-in model (CI)")
    ap.add_argument("--resolutions", type=csv_list(int), default=[512, 448, 384])
    ap.add_argument("--precisions", type=csv_list(str), default=["bf16", "fp16"])
    ap.add_argument("--micro-batches", type=csv_list(int), default=[1, 2])
    ap.add_argument("--modes", type=csv_list(str), default=["train", "inference"])
    ap.add_argument("--data", type=csv_list(str), default=["compute", "dataloader"],
                    help="compute = random device tensor; dataloader = end-to-end")
    ap.add_argument("--checkpointing", type=csv_list(str), default=["on", "off"],
                    help="on / off / both (e.g. off,on), or 'auto': run off first "
                         "and add the on row only when off OOMs or exceeds "
                         "--vram-headroom-pct")
    ap.add_argument("--vram-headroom-pct", type=float, default=90.0,
                    help="TC2 section 8: peak allocated VRAM must stay under this")
    ap.add_argument("--workers", type=csv_list(int), default=[4])
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--min-measure-seconds", type=float, default=0.0,
                    help="keep timing past --steps until the measured window "
                         "lasts this long, so short configurations still produce "
                         "meaningful utilization samples")
    ap.add_argument("--gpu-sample-ms", type=int, default=250,
                    help="nvidia-smi utilization sampling interval")
    ap.add_argument("--resume", action="store_true",
                    help="keep completed rows from an existing --out file and "
                         "only run the configurations still outstanding")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--manifest", default="", help="real data for the dataloader mode")
    ap.add_argument("--data-root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--synthetic-dir", default=str(REPO / "data" / "cache" / "bench_synth"))
    ap.add_argument("--synthetic-images", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    args.data = [{"compute": COMPUTE_ONLY, "dataloader": END_TO_END}.get(d, d)
                 for d in args.data]
    args.checkpoint_auto = [c.lower() for c in args.checkpointing] == ["auto"]
    args.checkpointing = ([False] if args.checkpoint_auto
                          else [c in ("on", "true", "1", "True")
                                for c in args.checkpointing])

    import torch

    from track5.models import build_model
    from track5.train.checkpoint import atomic_write_json
    from track5.utils.config import config_hash, load_config

    notes = []
    if args.steps < 100:
        notes.append(f"steps={args.steps} is below the TC2 section 8 minimum of 100 "
                     f"measured steps - treat this run as a smoke test")
        print(f"WARNING: {notes[-1]}", file=sys.stderr, flush=True)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type != "cuda":
        notes.append("no CUDA device: VRAM figures and bf16/fp16 rows are unavailable")

    model_cfg = {"model": {"backbone": args.model, "fallback_backbone": args.fallback,
                           "head": "linear", "pool": "cls",
                           "pretrained": not args.no_pretrained and not args.stub,
                           "stub": args.stub, "freeze_backbone": False}}
    if args.config:
        loaded = load_config(args.config)
        model_cfg["model"].update(loaded.get("model", {}))
        model_cfg["model"]["stub"] = args.stub or model_cfg["model"].get("stub", False)
        if args.no_pretrained:
            model_cfg["model"]["pretrained"] = False

    t0 = time.time()
    model = build_model(model_cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[bench] model {model.backbone_name} params={params / 1e6:.1f}M "
          f"(C1 <2B: {params < 2_000_000_000}) built in {time.time() - t0:.1f}s "
          f"on {device}", flush=True)

    source = args.manifest or []
    if END_TO_END in args.data:
        if args.manifest:
            notes.append(f"end-to-end rows decode real data via {args.manifest} "
                         f"(data-root {args.data_root})")
        else:
            src = max(args.resolutions) + 128
            source = synthetic_images(Path(args.synthetic_dir),
                                      args.synthetic_images, src, args.seed)
            notes.append(f"end-to-end rows use {len(source)} synthetic {src}px images "
                         f"(no --manifest given): decode+augment cost is real, image "
                         f"statistics are not")

    configs = build_configs(args)
    out_path = Path(args.out) if args.out else (
        REPO / "reports" /
        f"benchmark_{socket.gethostname()}_{os.environ.get('SLURM_JOB_ID', os.getpid())}.json")

    env = environment_report()
    fingerprint = grid_fingerprint(model_cfg, env, args)
    completed = load_completed(out_path, fingerprint) if args.resume else {}
    if completed:
        print(f"[bench] --resume: preserving {len(completed)} completed rows from "
              f"{out_path}", flush=True)

    report = {
        "benchmark": "dinov3_qualification",
        "started_unix": int(t0),
        "environment": env,
        "model": {"requested": args.model, "fallback": args.fallback,
                  "resolved": model.backbone_name, "params_total": params,
                  "params_under_2b": params < 2_000_000_000,
                  "pretrained": model_cfg["model"]["pretrained"],
                  "stub": model_cfg["model"]["stub"]},
        "config": model_cfg,
        "config_hash": config_hash(model_cfg),
        "settings": {"steps": args.steps, "warmup_steps": args.warmup_steps,
                     "min_measure_seconds": args.min_measure_seconds,
                     "gpu_sample_ms": args.gpu_sample_ms,
                     "grad_accum": args.grad_accum, "seed": args.seed,
                     "device": str(device), "n_configs": len(configs),
                     "torch_compile": False, "grid_fingerprint": fingerprint,
                     "resumed_rows": len(completed),
                     "checkpointing_mode": "auto" if args.checkpoint_auto else "explicit",
                     "vram_headroom_pct": args.vram_headroom_pct},
        "notes": notes,
        "results": [],
    }
    atomic_write_json(report, out_path)
    print(f"[bench] {len(configs)} configurations -> {out_path}", flush=True)

    # started before any warm-up, and it takes an immediate sample, so even a
    # sub-interval measured window has utilization data
    sampler = (GpuSampler(interval_ms=args.gpu_sample_ms, uuid=device_uuid()).start()
               if device.type == "cuda" else None)
    if sampler is not None and sampler.error:
        notes.append(f"GPU utilization unavailable: {sampler.error}")

    queue = list(configs)
    i = 0
    try:
        while i < len(queue):
            cfg = queue[i]
            label = (f"{cfg['mode']}/{cfg['data_path']} {cfg['resolution']}px "
                     f"{cfg['precision']} mb={cfg['micro_batch']} "
                     f"ckpt={cfg['activation_checkpointing']} workers={cfg['workers']}")
            key = config_key(cfg)
            if key in completed:
                print(f"[bench] {i + 1}/{len(queue)} {label} -> preserved from a "
                      f"previous run ({completed[key]['status']})", flush=True)
                res = {**completed[key], "resumed": True}
                report["results"].append(res)
                report["settings"]["n_configs_run"] = len(report["results"])
                atomic_write_json(report, out_path)
                i += 1
                continue
            print(f"[bench] {i + 1}/{len(queue)} {label}", flush=True)
            res = run_config(model, cfg, args, device, source, sampler)

            if cfg.get("escalate_ckpt"):
                pct = headroom_pct_of(res)
                why = escalate_reason(res, args.vram_headroom_pct)
                if why:
                    res["escalated_to_checkpointing"] = why
                    queue.insert(i + 1, {**cfg, "activation_checkpointing": True,
                                         "escalate_ckpt": False,
                                         "escalated_because": why})
                elif res["status"] == "ok":
                    res["escalated_to_checkpointing"] = (
                        f"not needed: peak allocated {pct}% within "
                        f"{args.vram_headroom_pct}% headroom" if pct is not None
                        else "not evaluated: no VRAM figure on this device")

            report["results"].append(res)
            report["settings"]["n_configs_run"] = len(report["results"])
            atomic_write_json(report, out_path)  # flush after every configuration
            if res["status"] == "ok":
                util = res.get("gpu_utilization", {})
                util_s = (f" gpu_util {util['gpu_util_pct']['median']}%"
                          if util.get("available") else "")
                avail = res.get("peak_allocated_pct_of_available")
                mem_s = (f"peak_alloc {res.get('peak_allocated_gib', 'n/a')} GiB "
                         f"({res.get('peak_allocated_pct', 'n/a')}% of total"
                         + (f", {avail}% of available)" if avail is not None else ")"))
                print(f"        median {res['step_time_s']['median'] * 1000:.1f} ms  "
                      f"p95 {res['step_time_s']['p95'] * 1000:.1f} ms  "
                      f"{res['images_per_s']['from_median_step']:.2f} img/s  "
                      f"{mem_s}{util_s}", flush=True)
            else:
                print(f"        {res['status'].upper()}: {res.get('reason', '')}",
                      flush=True)
                if res.get("traceback"):
                    print(res["traceback"], file=sys.stderr, flush=True)
            if res.get("escalated_to_checkpointing"):
                print(f"        -> {res['escalated_to_checkpointing']}", flush=True)
            i += 1
    finally:
        if sampler is not None:
            sampler.stop()

    atomic_write_json(report, out_path)
    ok = sum(r["status"] == "ok" for r in report["results"])
    oom = sum(r["status"] == "oom" for r in report["results"])
    errors = [r for r in report["results"] if r["status"] == "error"]
    skipped = len(report["results"]) - ok - oom - len(errors)
    print(f"[bench] done: {ok} ok, {oom} OOM, {len(errors)} error, "
          f"{skipped} skipped -> {out_path}", flush=True)
    if errors:
        # OOM and skipped are expected outcomes of a sweep; an error is a bug
        for r in errors:
            print(f"[bench] ERROR {config_key(r)}: {r.get('reason')}",
                  file=sys.stderr, flush=True)
        return 6
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
