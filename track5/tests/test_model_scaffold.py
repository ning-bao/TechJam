"""PLAN D1 backbone pair: DINOv3-L/16 primary, DINOv2-with-registers fallback.

No network and no weight downloads: the primary is built architecture-only from
the local spec, and the fallback branch is exercised by making the primary fail.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from track5.models import build_model
from track5.models.backbone import set_gradient_checkpointing
from track5.models.build import LOCAL_ARCH
from track5.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
DINOV3_L16 = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV2_REG_L = "facebook/dinov2-with-registers-large"

# DINOv3ViT landed in transformers 4.56; the dependency is unpinned in
# pyproject.toml, so say plainly when the environment is too old rather than
# failing with an opaque AttributeError.
try:
    import transformers  # noqa: F401
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    HAS_DINOV3 = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_DINOV3 = False

needs_dinov3 = pytest.mark.skipif(
    not HAS_DINOV3, reason="installed transformers has no DINOv3ViT (needs >=4.56)")


def test_local_arch_is_the_published_dinov3_l16():
    cls_name, kwargs = LOCAL_ARCH[DINOV3_L16]
    assert cls_name == "DINOv3ViTConfig"
    assert kwargs["hidden_size"] == 1024
    assert kwargs["num_hidden_layers"] == 24
    assert kwargs["num_attention_heads"] == 16
    assert kwargs["intermediate_size"] == 4096
    assert kwargs["patch_size"] == 16
    assert kwargs["num_register_tokens"] == 4


@needs_dinov3
def test_dinov3_l16_exact_parameter_count():
    """303.1M in PLAN D1, and comfortably under the 2B limit (C1)."""
    model = build_model({"model": {"backbone": DINOV3_L16, "pretrained": False,
                                   "stub": False}})
    assert model.backbone_name == DINOV3_L16
    total = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in model.backbone.parameters()) == 303_129_600
    assert total < 2_000_000_000
    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 1024 and model.head.out_features == 1


@needs_dinov3
@pytest.mark.parametrize("res", [512, 448, 384])
def test_patch16_resolutions_are_exact(res):
    """512 is a valid patch-16 size; TC2 §8 forbids substituting 504."""
    assert res % 16 == 0
    _, kwargs = LOCAL_ARCH[DINOV3_L16]
    tiny = DINOv3ViTModel(DINOv3ViTConfig(
        **{**kwargs, "hidden_size": 64, "intermediate_size": 128,
           "num_hidden_layers": 2, "num_attention_heads": 2}))
    with torch.no_grad():
        out = tiny(pixel_values=torch.randn(1, 3, res, res))
    # [CLS] + 4 registers + (res/16)^2 patches
    assert out.last_hidden_state.shape[1] == 1 + 4 + (res // 16) ** 2


def test_fallback_is_used_when_the_gated_primary_is_unavailable(monkeypatch, capsys):
    import track5.models.build as build

    class FakeBB(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 32})()

    def fake_load(name, pretrained):
        if name == DINOV3_L16:
            raise OSError("gated repo: access not granted")
        return FakeBB()

    monkeypatch.setattr(build, "_load_hf", fake_load)
    model = build.build_model({"model": {"backbone": DINOV3_L16,
                                         "fallback_backbone": DINOV2_REG_L,
                                         "pretrained": True}})
    assert model.backbone_name == DINOV2_REG_L      # recorded, not silently swapped
    assert "PLAN D1 fallback" in capsys.readouterr().err


def test_no_fallback_configured_means_the_error_propagates(monkeypatch):
    import track5.models.build as build

    monkeypatch.setattr(build, "_load_hf",
                        lambda n, p: (_ for _ in ()).throw(OSError("gated")))
    with pytest.raises(OSError):
        build.build_model({"model": {"backbone": DINOV3_L16, "pretrained": True}})


def test_primary_config_declares_the_pair_and_a_patch16_crop():
    cfg = load_config(REPO / "configs" / "dinov3l512.yaml")
    assert cfg["model"]["backbone"] == DINOV3_L16
    assert cfg["model"]["fallback_backbone"] == DINOV2_REG_L
    assert cfg["data"]["crop"] == 512 and cfg["data"]["crop"] % 16 == 0
    assert cfg["train"]["precision"] == "bf16"
    assert cfg["train"]["activation_checkpointing"] is True
    assert cfg["eval"]["worst_case_atoms"] == ["clean", "jpeg_30", "blur_20",
                                               "resize_025", "noise_010"]


def test_fallback_config_is_the_ungated_lane():
    cfg = load_config(REPO / "configs" / "dinov2l_reg448.yaml")
    assert cfg["model"]["backbone"] == DINOV2_REG_L
    assert "fallback_backbone" not in cfg["model"]   # it *is* the fallback
    assert cfg["data"]["crop"] % 14 != 0 or cfg["data"]["crop"] % 16 == 0


@needs_dinov3
def test_activation_checkpointing_toggle_reports_whether_it_applied():
    stub = build_model({"model": {"stub": True}})
    assert set_gradient_checkpointing(stub, True) is False  # stub has no HF hook

    model = build_model({"model": {"backbone": DINOV3_L16, "pretrained": False}})
    assert set_gradient_checkpointing(model, True) is True
    assert model.backbone.gradient_checkpointing
    assert set_gradient_checkpointing(model, False) is True
    assert not model.backbone.gradient_checkpointing
