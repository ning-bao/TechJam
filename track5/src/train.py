"""Resumable single-GPU training segment (TC2 guide §6/§9/§10).

    python -u -m src.train --config configs/dinov3l512.yaml \
        --train-manifest data/manifests/train.parquet \
        --run-dir runs/dinov3l512_seed0 --resume auto --max-wall-minutes 330

Long training is a chain of these segments. The segment stops itself on
SIGUSR1/SIGTERM or when the wall budget minus a grace window is spent, writes an
atomic recovery checkpoint plus segment_complete.json, and exits 0 so an
`--dependency=afterok` successor picks up exactly where it stopped.

Loss is real binary cross-entropy on a linear classifier head over the backbone
CLS token; the head and the loss are the same ones the eval/predict paths use.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST",
              "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_ARRAY_TASK_ID")


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def assert_no_protected_overlap(df, repo: Path) -> dict:
    """TC2 section 5: a training job must abort if its manifest intersects the
    protected set, by family, source, path or content hash."""
    from track5.data.denylist import count_protected

    deny_dir = repo / "data" / "denylist"
    report = count_protected(df, deny_dir)
    report["denylist_tables_present"] = report.pop("tables_present")
    if not report["denylist_tables_present"]:
        raise SystemExit(
            "refusing to train: no denylist table found. Run "
            "scripts/build_denylist.py --coco-val first (constraint C2).")

    deny = deny_dir / "denylist.parquet"
    receipt = deny.with_name(deny.name + ".receipt.json")
    if receipt.exists():
        cov = json.loads(receipt.read_text(encoding="utf-8"))
        report["denylist_complete"] = bool(cov.get("complete"))
        if not cov.get("complete"):
            raise SystemExit(
                f"refusing to train: the protected-set denylist is incomplete "
                f"({cov.get('coverage')}). Re-run scripts/build_denylist.py once "
                f"the archives are final (constraint C2).")
    if report["total_hits"]:
        raise SystemExit(
            f"refusing to train: manifest intersects the protected set "
            f"({report['family_hits']} family, {report['source_hits']} source, "
            f"{report['path_hits']} path, {report['hash_hits']} hash hits). "
            f"Constraint C2 - fix the manifest, never the denylist.")
    return report


def make_eval_fn(cfg, model_device, data_root, repo: Path, cache_dir: Path,
                 batch_size: int = 32):
    """Worst-case bAcc over cfg.eval.worst_case_atoms (PLAN D5), transforming the
    dev manifest through the frozen eval atoms and caching the results."""
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from track5.eval.matrix import prepare_atom_cache
    from track5.eval.metrics import bacc, worst_case_bacc
    from track5.models.preprocess import eval_crop, to_tensor

    ecfg = cfg.get("eval", {})
    atoms = list(ecfg.get("worst_case_atoms",
                          ["clean", "jpeg_30", "blur_20", "resize_025", "noise_010"]))
    crop = int(cfg["data"].get("crop", 448))
    seed = int(cfg.get("seed", 17))
    dev = pd.read_parquet(repo / cfg["data"]["dev_manifest"])
    dev = dev[dev["split"] != "denied"].reset_index(drop=True)
    limit = int(ecfg.get("dev_limit", 0))
    if limit and len(dev) > limit:
        rng = np.random.Generator(np.random.PCG64(seed))
        dev = dev.iloc[np.sort(rng.choice(len(dev), limit, replace=False))]
    cache = Path(cache_dir)

    prepared = {}
    for atom in atoms:
        paths, labels, skipped = prepare_atom_cache(dev, atom, cache, data_root, seed)
        if not paths:
            print(f"[eval] atom {atom}: no usable images", file=sys.stderr, flush=True)
            continue
        prepared[atom] = (paths, labels)
        print(f"[eval] atom {atom}: {len(paths)} images ({skipped} skipped)", flush=True)
    if not prepared:
        raise SystemExit("no dev images could be prepared - check the dev manifest")

    @torch.no_grad()
    def score(model, paths):
        out = []
        for i in range(0, len(paths), batch_size):
            px = torch.stack([to_tensor(eval_crop(Image.open(f).convert("RGB"), crop))
                              for f in paths[i:i + batch_size]]).to(model_device)
            out.append(torch.sigmoid(model(px).float()).cpu().numpy())
        return np.concatenate(out)

    def eval_fn(model):
        per_atom = {a: bacc(y, score(model, p), 0.5) for a, (p, y) in prepared.items()}
        out = {f"bacc_{a}": v for a, v in per_atom.items()}
        out["worst_case_bacc"] = worst_case_bacc(per_atom)
        return out

    return eval_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-manifest", default="")
    ap.add_argument("--dev-manifest", default="")
    ap.add_argument("--data-root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--cache-dir", default=str(REPO / "data" / "cache" / "eval"),
                    help="transformed-dev cache; must be on persistent storage")
    ap.add_argument("--resume", default="none",
                    help="auto | none | path to a recovery checkpoint")
    ap.add_argument("--max-wall-minutes", type=float, default=None)
    ap.add_argument("--grace-minutes", type=float, default=12.0,
                    help="start the graceful save this long before the budget ends")
    ap.add_argument("--checkpoint-every-minutes", type=float, default=35.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--allow-config-change", action="store_true")
    ap.add_argument("--skip-denylist-check", action="store_true",
                    help="smoke tests on synthetic fixtures only; never for real data")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from track5.data.dataset import ManifestDataset
    from track5.data.sampler import EpochPermutationSampler, seed_worker
    from track5.models import build_model
    from track5.models.backbone import set_gradient_checkpointing
    from track5.train.checkpoint import atomic_write_json
    from track5.train.loop import Trainer
    from track5.transforms.train_sampler import TrainDistortionSampler
    from track5.utils.config import config_hash, load_config
    from track5.utils.hashing import file_sha256

    t_start = time.monotonic()
    cfg = load_config(args.config)
    if args.train_manifest:
        cfg["data"]["train_manifest"] = args.train_manifest
    if args.dev_manifest:
        cfg["data"]["dev_manifest"] = args.dev_manifest
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run-dir  {run_dir.resolve()}", flush=True)
    print(f"[train] config   {config_hash(cfg)}\n{json.dumps(cfg, indent=2, default=str)}",
          flush=True)

    train_manifest = REPO / cfg["data"]["train_manifest"]
    manifest_hash = file_sha256(train_manifest)
    if args.skip_denylist_check:
        print("[train] WARNING denylist check skipped (--skip-denylist-check)",
              flush=True)
        deny_report = {"skipped": True}
    else:
        deny_report = assert_no_protected_overlap(pd.read_parquet(train_manifest), REPO)
        print(f"[train] denylist gate PASS {deny_report}", flush=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model = build_model(cfg).to(device)
    ck_on = bool(cfg["train"].get("activation_checkpointing", False))
    applied = set_gradient_checkpointing(model, ck_on)
    print(f"[train] backbone={model.backbone_name} device={device} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"activation_checkpointing={ck_on} (applied={applied})", flush=True)

    crop = int(cfg["data"].get("crop", 448))
    seed = int(cfg.get("seed", 17))
    sampler = (TrainDistortionSampler()
               if cfg.get("distortion", {}).get("enabled", True) else None)
    ds = ManifestDataset(train_manifest, split="train", crop=crop,
                         distortion_sampler=sampler, seed=seed,
                         data_root=args.data_root)
    if len(ds) == 0:
        print("ERROR: no rows with split='train' in the manifest", file=sys.stderr)
        return 2
    bs = int(cfg["data"].get("batch_size", 32))
    workers = int(cfg["data"].get("workers", 0))

    def loader_factory(epoch: int, start_index: int):
        # dataset, sampler and worker_init_fn are all pickled into Windows spawn
        # workers: keep them importable module-level objects, never closures.
        return DataLoader(
            ds, batch_size=bs, drop_last=True, num_workers=workers,
            pin_memory=(device == "cuda"),
            worker_init_fn=seed_worker if workers > 0 else None,
            sampler=EpochPermutationSampler(len(ds), seed, epoch, start_index))

    eval_fn = make_eval_fn(cfg, device, args.data_root, REPO, Path(args.cache_dir),
                           batch_size=max(1, bs))
    trainer = Trainer(cfg, model, loader_factory(0, 0), eval_fn, run_dir,
                      loader_factory=loader_factory,
                      max_wall_minutes=args.max_wall_minutes,
                      grace_minutes=args.grace_minutes,
                      manifest_hash=manifest_hash,
                      checkpoint_every_minutes=args.checkpoint_every_minutes)
    armed = trainer.install_signal_handlers()
    print(f"[train] precision={trainer.precision} ({trainer.precision_reason}); "
          f"grad_accum={trainer.grad_accum}; total_steps={trainer.total_steps}; "
          f"signals armed: {armed or 'none'}", flush=True)

    resumed = trainer.resume(args.resume, allow_config_change=args.allow_config_change)

    atomic_write_json({
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "config_hash": config_hash(cfg),
        "resolved_config": cfg,
        "train_manifest": str(train_manifest),
        "manifest_sha256": manifest_hash,
        "denylist_gate": deny_report,
        "seed": seed,
        "resumed": resumed,
        "backbone": model.backbone_name,
        "precision": trainer.precision,
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "slurm": {k: os.environ.get(k) for k in SLURM_KEYS if os.environ.get(k)},
        "max_wall_minutes": args.max_wall_minutes,
    }, run_dir / "provenance" / "run_receipt.json")

    best = trainer.train()
    seg = json.loads((run_dir / "segment_complete.json").read_text(encoding="utf-8"))
    print(f"[train] segment done in {time.monotonic() - t_start:.1f}s: "
          f"stop_reason={seg['stop_reason']} step={seg['global_step']}/"
          f"{seg['total_steps']} complete={seg['complete']} "
          f"best_worst_case_bacc={best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
