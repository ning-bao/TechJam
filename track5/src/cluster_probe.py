"""Hardware/software probe — first GPU job on TC2 (guide §7).

    python -u -m src.cluster_probe [--config configs/dinov3l512.yaml]
                                   [--resolution 512] [--output probe.json]
                                   [--allow-cpu]

Pass gate: CUDA visible, the backbone loads (offline cache is fine), BF16 or
FP16 works, a forward+backward pass completes, and nothing secret is logged.
Exits non-zero if any of that fails.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Allow-listed only: never dump the environment wholesale (TC2 §14).
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "HF_HUB_OFFLINE",
            "OMP_NUM_THREADS")


def nvidia_driver() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
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
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "env": {k: os.environ.get(k) for k in ENV_KEYS if os.environ.get(k)},
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rep.update({
            "gpu": p.name,
            "vram_gib": round(p.total_memory / 2**30, 2),
            "compute_capability": f"{p.major}.{p.minor}",
            "multi_processor_count": p.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "dinov3l512.yaml"))
    ap.add_argument("--resolution", type=int, default=0,
                    help="override data.crop for the forward/backward check")
    ap.add_argument("--output", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="do not fail when no GPU is allocated (local smoke only)")
    args = ap.parse_args()

    import torch

    from track5.models import build_model
    from track5.train.checkpoint import atomic_write_json
    from track5.train.loop import resolve_precision
    from track5.utils.config import config_hash, load_config

    report = {"probe": "cluster_probe", "config_path": str(args.config)}
    report["environment"] = environment_report()
    print(json.dumps(report["environment"], indent=2), flush=True)

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    if device != "cuda" and not args.allow_cpu:
        print("FAIL: PyTorch cannot see an allocated GPU", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if args.resolution:
        cfg.setdefault("data", {})["crop"] = args.resolution
    res = int(cfg.get("data", {}).get("crop", 512))
    report["config_hash"] = config_hash(cfg)
    report["requested_backbone"] = cfg["model"].get("backbone")
    report["fallback_backbone"] = cfg["model"].get("fallback_backbone")

    precision, reason = resolve_precision(cfg["train"].get("precision", "bf16"), device)
    report["precision"] = precision
    report["precision_reason"] = reason
    print(f"[probe] precision -> {precision} ({reason})", flush=True)

    try:
        model = build_model(cfg).to(device)
    except Exception as e:
        report["status"] = "fail"
        report["error"] = f"backbone load failed: {type(e).__name__}: {e}"
        print(f"FAIL: {report['error']}", file=sys.stderr)
        if args.output:
            atomic_write_json(report, args.output)
        return 3

    report["resolved_backbone"] = model.backbone_name
    report["params_total"] = sum(p.numel() for p in model.parameters())
    report["params_trainable"] = sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)
    report["params_under_2b"] = report["params_total"] < 2_000_000_000
    print(f"[probe] backbone={model.backbone_name} "
          f"params={report['params_total'] / 1e6:.1f}M "
          f"(C1 <2B: {report['params_under_2b']})", flush=True)

    model.train()
    x = torch.randn(1, 3, res, res, device=device)
    y = torch.zeros(1, device=device)
    try:
        if precision in ("bf16", "fp16"):
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
    except Exception as e:
        report["status"] = "fail"
        report["error"] = f"forward/backward at {res}px failed: {type(e).__name__}: {e}"
        print(f"FAIL: {report['error']}", file=sys.stderr)
        if args.output:
            atomic_write_json(report, args.output)
        return 4

    report["forward_backward_ok"] = True
    report["resolution"] = res
    report["loss"] = float(loss.detach().float())
    if device == "cuda":
        report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        report["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    report["status"] = "pass"
    print(json.dumps({k: v for k, v in report.items() if k != "environment"},
                     indent=2, default=str), flush=True)
    if args.output:
        p = atomic_write_json(report, args.output)
        print(f"[probe] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
