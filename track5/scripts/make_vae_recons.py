"""Fake #3 (PLAN D3): VAE reconstructions of our own real images —
Aligned-Datasets recipe, encoder->decoder only, no sampling. GPU job; launch
deliberately, resumable.

.venv/Scripts/python.exe scripts/make_vae_recons.py --manifest data/manifests/coco_train.parquet \
    --source-filter coco_train2017 --vae sd15 --limit 25000 --device cuda --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

VAES = {
    "sd15": [("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"),
             ("runwayml/stable-diffusion-v1-5", "vae")],
    "sdxl": [("madebyollin/sdxl-vae-fp16-fix", None)],
}


def load_vae(name: str, device: str, dtype):
    from diffusers import AutoencoderKL

    last = None
    for repo, sub in VAES[name]:
        try:
            vae = (AutoencoderKL.from_pretrained(repo, subfolder=sub, torch_dtype=dtype)
                   if sub else AutoencoderKL.from_pretrained(repo, torch_dtype=dtype))
            return vae.to(device).eval()
        except Exception as e:
            last = e
            print(f"[vae] {repo} failed: {e}", file=sys.stderr)
    raise SystemExit(f"could not load VAE {name}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source-filter", default="coco_train2017")
    ap.add_argument("--vae", required=True, choices=list(VAES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=str(REPO / "data/raw"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from io import BytesIO

    import pandas as pd
    import torch
    from PIL import Image

    from track5.data.resolve import resolve_image_bytes

    out_dir = Path(args.out) if args.out else REPO / "data/derived/vae_recon" / args.vae
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "recon_log.jsonl"

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    vae = load_vae(args.vae, args.device, dtype)
    vae.enable_tiling()

    df = pd.read_parquet(args.manifest)
    if args.source_filter:
        df = df[df["source"] == args.source_filter]
    if "split" in df.columns:
        df = df[df["split"] != "denied"]
    if args.limit:
        df = df.head(args.limit)

    done = skipped = 0
    with open(log_path, "a", encoding="utf-8") as log:
        for row in df.itertuples(index=False):
            is_jpeg = row.format == "jpeg"
            ext = "jpg" if is_jpeg else "png"
            out_path = out_dir / f"{row.sha256[:16]}.{ext}"
            if args.resume and out_path.exists():
                done += 1
                continue
            t0 = time.time()
            try:
                img = Image.open(BytesIO(resolve_image_bytes(args.data_root, row.path)))
                img = img.convert("RGB")
                w, h = img.size
                pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
                if pw or ph:
                    padded = Image.new("RGB", (w + pw, h + ph))
                    padded.paste(img, (0, 0))
                    img = padded
                import numpy as np

                x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
                x = x.permute(2, 0, 1).unsqueeze(0).to(args.device, dtype)
                with torch.no_grad():
                    lat = vae.encode(x).latent_dist.mode()  # no sampling
                    y = vae.decode(lat).sample
                y = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte()
                arr = y[0].permute(1, 2, 0).cpu().numpy()[:h, :w]
                rec = Image.fromarray(arr)
                if is_jpeg:
                    rec.save(out_path, format="JPEG", quality=90)
                else:
                    rec.save(out_path, format="PNG")
                done += 1
                log.write(json.dumps({"src": row.path, "out": str(out_path),
                                      "secs": round(time.time() - t0, 3)}) + "\n")
            except Exception as e:
                skipped += 1
                print(f"[vae] skip {row.path}: {type(e).__name__}: {e}", file=sys.stderr)
            if done and done % 500 == 0:
                print(f"[vae] {done} done, {skipped} skipped", file=sys.stderr)
    print(f"[vae] finished: {done} reconstructions, {skipped} skipped -> {out_dir}")


if __name__ == "__main__":
    main()
