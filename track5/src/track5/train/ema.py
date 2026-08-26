import torch


class EMA:
    """Shadow-parameter exponential moving average with copy_to/restore."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay,
                "shadow": {n: p.detach().cpu().clone() for n, p in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        for n, v in state["shadow"].items():
            if n in self.shadow:
                self.shadow[n].copy_(v.to(self.shadow[n].device))
