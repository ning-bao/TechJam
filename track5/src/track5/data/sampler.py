"""Resumable epoch ordering (TC2 guide §10.4) and worker seeding.

The order for an epoch is a pure function of (seed, epoch), so a resumed segment
replays the *remainder* of the interrupted epoch instead of a fresh random
fraction of it. `start_index` is in samples, not batches.

Everything here must survive being pickled to a Windows `spawn` worker: plain
attributes on the sampler, and a module-level `seed_worker` rather than a lambda
or a closure.
"""

import random

from torch.utils.data import Sampler

import numpy as np

from track5.utils.seed import item_seed


def seed_worker(worker_id: int) -> None:
    """Deterministic per-worker RNG seeding, derived from the loader generator.

    Module-level on purpose: DataLoader pickles `worker_init_fn` for spawn, and a
    lambda or nested function cannot be pickled.
    """
    import torch

    info = torch.utils.data.get_worker_info()
    base = 0 if info is None else int(info.seed) % (2**31)
    s = item_seed("worker", worker_id, base)
    random.seed(s)
    np.random.seed(s % (2**32))
    torch.manual_seed(s)


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(item_seed("epoch_order", seed, epoch)))
    return rng.permutation(n)


class EpochPermutationSampler(Sampler[int]):
    def __init__(self, n: int, seed: int, epoch: int = 0, start_index: int = 0):
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self):
        return iter(epoch_order(self.n, self.seed, self.epoch)[self.start_index:].tolist())

    def __len__(self) -> int:
        return max(0, self.n - self.start_index)
