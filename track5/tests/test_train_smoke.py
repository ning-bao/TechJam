import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from track5.train.ema import EMA
from track5.train.loop import Trainer, build_optimizer, cosine_warmup

CFG = {
    "seed": 17,
    "model": {"stub": True},
    "train": {"epochs": 1, "max_steps": 4, "lr_head": 1e-3, "lr_backbone": 1e-5,
              "weight_decay": 0.0, "warmup_frac": 0.5, "loss": "bce",
              "ema_decay": 0.9, "swa": False, "precision": "fp32", "grad_accum": 1},
    "eval": {"every_steps": 1},
}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4), nn.Flatten(),
                                      nn.LazyLinear(8))
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class SynthDS(Dataset):
    def __init__(self, n=32):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 16, 16, generator=g)
        self.y = torch.tensor([i % 2 for i in range(n)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"pixels": self.x[i], "label": self.y[i], "idx": i}


def make_trainer(tmp_path, eval_values):
    model = TinyModel()
    model(torch.randn(1, 3, 16, 16))  # materialize lazy layers
    it = iter(eval_values)

    def eval_fn(m):
        return {"worst_case_bacc": next(it)}

    loader = DataLoader(SynthDS(), batch_size=8)
    return Trainer(CFG, model, loader, eval_fn, tmp_path / "run"), model


def test_trainer_runs_and_checkpoints(tmp_path):
    trainer, _ = make_trainer(tmp_path, [0.5, 0.9, 0.7, 0.6])
    best = trainer.train()
    assert best == 0.9
    run = tmp_path / "run"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    ckpt = torch.load(run / "best.pt", weights_only=False)
    assert set(ckpt) == {"state_dict", "config", "calibration", "meta"}
    assert set(ckpt["calibration"]) == {"temperature", "alpha", "threshold"}
    assert ckpt["meta"]["metrics"]["worst_case_bacc"] == 0.9  # best kept at peak
    lines = (run / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[1])["worst_case_bacc"] == 0.9


def test_ema_roundtrip():
    m = TinyModel()
    m(torch.randn(1, 3, 16, 16))
    ema = EMA(m, decay=0.5)
    orig = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    ema.copy_to(m)
    shadowed = {n: p.detach().clone() for n, p in m.named_parameters()}
    ema.restore(m)
    for n, p in m.named_parameters():
        assert torch.allclose(p, orig[n] + 1.0)
        assert not torch.allclose(shadowed[n], p)  # shadow differs from live


def test_warmup_schedule():
    total, warm = 10, 5
    assert cosine_warmup(0, total, warm) < cosine_warmup(4, total, warm)
    assert cosine_warmup(4, total, warm) <= 1.0
    assert cosine_warmup(9, total, warm) < cosine_warmup(5, total, warm)


def test_two_param_groups_distinct_lrs():
    m = TinyModel()
    m(torch.randn(1, 3, 16, 16))
    opt = build_optimizer(m, {"lr_head": 1e-3, "lr_backbone": 1e-5})
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 1e-3]


def test_eval_fn_must_return_worst_case(tmp_path):
    model = TinyModel()
    model(torch.randn(1, 3, 16, 16))
    loader = DataLoader(SynthDS(), batch_size=8)
    trainer = Trainer(CFG, model, loader, lambda m: {"bacc": 0.5}, tmp_path / "r")
    with pytest.raises(ValueError, match="worst_case_bacc"):
        trainer.train()
