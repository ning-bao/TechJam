import torch
import torch.nn as nn


class Detector(nn.Module):
    """backbone -> CLS pool -> linear head -> logits [B]."""

    def __init__(self, backbone: nn.Module, head: nn.Module, frozen_backbone: bool = False,
                 hf_backbone: bool = True, backbone_name: str = "stub"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.frozen_backbone = frozen_backbone
        self.hf_backbone = hf_backbone
        self.backbone_name = backbone_name
        if frozen_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()  # keep BN/dropout of frozen backbone fixed
        return self

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.hf_backbone:
            out = self.backbone(pixel_values=pixels)
            return out.last_hidden_state[:, 0]  # CLS token
        return self.backbone(pixels)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.frozen_backbone:
            with torch.no_grad():
                feats = self.features(pixels)
        else:
            feats = self.features(pixels)
        return self.head(feats).squeeze(-1)


class StubBackbone(nn.Module):
    """Tiny stand-in with the plain-tensor API; no downloads. For tests/CI."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=3, stride=4, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim = dim

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.conv(pixels))).flatten(1)


def set_gradient_checkpointing(model: nn.Module, enable: bool) -> bool:
    """Activation checkpointing on the HF backbone (TC2 §8 memory strategy).

    Returns True when the toggle was actually applied — the stub backbone and
    any non-HF module report False so callers can log the real state instead of
    assuming it took effect.
    """
    bb = getattr(model, "backbone", model)
    if enable:
        fn = getattr(bb, "gradient_checkpointing_enable", None)
        if fn is None:
            return False
        try:
            fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:  # older signature without kwargs
            fn()
        return True
    fn = getattr(bb, "gradient_checkpointing_disable", None)
    if fn is None:
        return False
    fn()
    return True
