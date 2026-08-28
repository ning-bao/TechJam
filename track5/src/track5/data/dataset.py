import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from track5.data.resolve import resolve_image_bytes
from track5.utils.seed import item_seed


class FolderDataset(Dataset):
    """Plain image-folder dataset, same order as ManifestDataset: bytes -> RGB ->
    (train only) distortion sampler -> crop -> tensor.

    Lives in the package rather than in a script so Windows `spawn` workers can
    import it by reference, and holds only picklable state: strings, ints and the
    sampler instance. Never store module objects, lambdas, nested functions or
    locally defined classes on a dataset - spawn pickles the whole instance and
    fails with `TypeError: cannot pickle 'module' object`.
    """

    def __init__(self, files, crop: int = 448, distortion_sampler=None,
                 seed: int = 17, train: bool = True):
        self.files = [str(f) for f in files]
        self.crop = int(crop)
        self.sampler = distortion_sampler
        self.seed = int(seed)
        self.train = bool(train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        from track5.models.preprocess import eval_crop, to_tensor, train_crop

        img = Image.open(self.files[i]).convert("RGB")
        rng = np.random.Generator(np.random.PCG64(item_seed("folder", i, self.seed)))
        if self.sampler is not None and self.train:
            img = self.sampler(img, rng)
        img = train_crop(img, rng, self.crop) if self.train else eval_crop(img, self.crop)
        return {"pixels": to_tensor(img), "label": i % 2, "idx": i}


class ManifestDataset(Dataset):
    """INTERFACES §10: bytes -> RGB -> (train) distortion sampler -> crop -> tensor."""

    def __init__(self, manifest_path, split: str | None = None, crop: int = 448,
                 distortion_sampler=None, seed: int = 17, data_root=None,
                 train: bool = True, normalize: bool = False):
        self.df = pd.read_parquet(manifest_path)
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        else:
            self.df = self.df[self.df["split"] != "denied"].reset_index(drop=True)
        repo = Path(manifest_path).resolve().parents[2]
        self.data_root = Path(data_root) if data_root else repo / "data" / "raw"
        self.crop = crop
        self.sampler = distortion_sampler
        self.seed = seed
        self.train = train
        self.normalize = bool(normalize)
        self.epoch = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        from track5.models.preprocess import eval_crop, to_tensor, train_crop

        row = self.df.iloc[idx]
        try:
            data = resolve_image_bytes(self.data_root, row["path"])
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception as e:
            print(f"[dataset] unreadable {row['path']}: {e}", file=sys.stderr)
            return self.__getitem__((idx + 1) % len(self))  # deterministic fallback
        if self.normalize:
            # PLAN D4: give both classes the same delivered-container statistics
            # before any augmentation, so "PNG => fake" is not learnable.
            from track5.data.normalize import normalize as normalize_container

            img = normalize_container(img, row["sha256"], row["format"], self.seed)
        # Epoch salt on the augmentation RNG only, so a second pass draws fresh
        # crops/distortions. normalize_container above keeps the unsalted seed:
        # the delivered container is data definition, not augmentation. epoch 0
        # reduces to self.seed, i.e. byte-identical to what is already trained.
        aug_seed = self.seed + 1_000_003 * self.epoch
        rng = np.random.Generator(np.random.PCG64(item_seed(row["sha256"], idx, aug_seed)))
        if self.sampler is not None and self.train:
            img = self.sampler(img, rng)
        img = train_crop(img, rng, self.crop) if self.train else eval_crop(img, self.crop)
        return {"pixels": to_tensor(img), "label": int(row["label"]), "idx": idx}
