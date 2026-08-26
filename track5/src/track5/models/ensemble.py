import torch
import torch.nn as nn


class AverageEnsemble(nn.Module):
    """Plain probability averaging (PLAN D8). Each member's logits pass through
    its own calibration sigmoid((z+alpha)/T) first.

    NOTE: forward returns PROBABILITIES in [0,1], unlike single models which
    return logits."""

    def __init__(self, models, calibrations=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if calibrations is None:
            calibrations = [{"temperature": 1.0, "alpha": 0.0} for _ in models]
        assert len(calibrations) == len(models)
        self.calibrations = [
            {"temperature": float(c.get("temperature") or 1.0),
             "alpha": float(c.get("alpha") or 0.0)}
            for c in calibrations
        ]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        probs = []
        for model, cal in zip(self.models, self.calibrations):
            z = model(pixels)
            probs.append(torch.sigmoid((z + cal["alpha"]) / cal["temperature"]))
        return torch.stack(probs).mean(dim=0)
