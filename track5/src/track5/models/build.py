"""Model builder (INTERFACES §5) for the PLAN D1 backbone pair:
DINOv3-L/16 primary, DINOv2-with-registers fallback.

`model.fallback_backbone` makes the fallback automatic: if the gated primary
cannot be loaded (no gate approval, HF_HUB_OFFLINE with an empty cache, no
network) the fallback is built instead and the resolved name is recorded on the
Detector. The critical path is never blocked on gate approval.
"""

import sys

import torch.nn as nn

from track5.models.backbone import Detector, StubBackbone

# Exact architectures for gated checkpoints, so the benchmark can build the real
# graph before/without gate approval (`pretrained: false`). Verified param count
# for dinov3-vitl16: 303,129,600.
LOCAL_ARCH = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": (
        "DINOv3ViTConfig",
        {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
         "num_attention_heads": 16, "patch_size": 16, "num_register_tokens": 4,
         "image_size": 224, "layerscale_value": 1.0, "rope_theta": 100.0},
    ),
}


def _load_hf(name: str, pretrained: bool):
    from transformers import AutoModel  # heavy import only on the real path

    if pretrained:
        return AutoModel.from_pretrained(name)

    # Architecture-only: prefer the local spec so this path needs no network at
    # all (HF_HUB_OFFLINE, no gate approval). Only ever used for shape/throughput
    # work — a real run loads pretrained weights and their own config.
    if name in LOCAL_ARCH:
        import transformers

        cls_name, kwargs = LOCAL_ARCH[name]
        return AutoModel.from_config(getattr(transformers, cls_name)(**kwargs))

    from transformers import AutoConfig

    return AutoModel.from_config(AutoConfig.from_pretrained(name))


def build_model(cfg: dict) -> Detector:
    m = cfg["model"]
    frozen = bool(m.get("freeze_backbone", False))
    if m.get("stub", False):
        bb = StubBackbone()
        head = nn.Linear(bb.dim, 1)
        return Detector(bb, head, frozen_backbone=frozen, hf_backbone=False,
                        backbone_name="stub")

    name = m.get("backbone", "facebook/dinov2-with-registers-base")
    fallback = m.get("fallback_backbone")
    pretrained = bool(m.get("pretrained", True))
    try:
        bb = _load_hf(name, pretrained)
    except Exception as e:
        if not fallback:
            raise
        print(f"[models] primary backbone {name!r} unavailable "
              f"({type(e).__name__}: {e}) -> PLAN D1 fallback {fallback!r}",
              file=sys.stderr, flush=True)
        bb = _load_hf(fallback, pretrained)
        name = fallback
    head = nn.Linear(bb.config.hidden_size, 1)
    return Detector(bb, head, frozen_backbone=frozen, hf_backbone=True,
                    backbone_name=name)
