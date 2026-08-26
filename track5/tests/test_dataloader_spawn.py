"""Windows `spawn` regression tests.

`DataLoader(num_workers>0)` pickles the dataset, its sampler and worker_init_fn
into fresh interpreters on Windows. A module object stored on the dataset (the
original `self._np = np`) crashes the whole run with

    TypeError: cannot pickle 'module' object

The spawn context is forced explicitly so this also fails on Linux CI, where the
default fork start method would hide the bug. workers=0 stays supported, but it
is tested as a fallback, not as a way to avoid the production path.
"""

import multiprocessing
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from track5.data.dataset import FolderDataset, ManifestDataset
from track5.data.sampler import EpochPermutationSampler, seed_worker
from track5.transforms.train_sampler import TrainDistortionSampler

REPO = Path(__file__).resolve().parents[1]
SPAWN = multiprocessing.get_context("spawn")


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("spawn_imgs")
    files = []
    for i in range(8):
        rng = np.random.Generator(np.random.PCG64(i))
        f = d / f"{i:02d}.png"
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(f)
        files.append(f)
    return files


def loader(ds, workers, batch_size=2):
    kwargs = {"multiprocessing_context": SPAWN} if workers else {}
    return DataLoader(ds, batch_size=batch_size, num_workers=workers,
                      drop_last=True, worker_init_fn=seed_worker if workers else None,
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(17), **kwargs)


# ------------------------------------------------------------- picklability

@pytest.mark.parametrize("train", [True, False])
def test_folder_dataset_is_picklable(images, train):
    ds = FolderDataset(images, crop=64,
                       distortion_sampler=TrainDistortionSampler() if train else None,
                       seed=17, train=train)
    clone = pickle.loads(pickle.dumps(ds))
    assert len(clone) == len(ds)
    assert torch.allclose(clone[0]["pixels"], ds[0]["pixels"])


def test_no_worker_visible_object_holds_a_module(images):
    """The precise failure mode: a module attribute anywhere on the dataset."""
    import types

    ds = FolderDataset(images, crop=64,
                       distortion_sampler=TrainDistortionSampler(), seed=17)
    for owner in (ds, ds.sampler):
        offenders = [k for k, v in vars(owner).items()
                     if isinstance(v, types.ModuleType)]
        assert not offenders, f"{type(owner).__name__} stores modules: {offenders}"


def test_sampler_and_worker_init_are_picklable():
    s = EpochPermutationSampler(10, seed=17, epoch=2, start_index=4)
    assert list(pickle.loads(pickle.dumps(s))) == list(s)
    assert pickle.loads(pickle.dumps(seed_worker)) is seed_worker


def test_distortion_sampler_is_picklable():
    s = TrainDistortionSampler()
    clone = pickle.loads(pickle.dumps(s))
    img = Image.new("RGB", (80, 80), (120, 90, 40))
    rng_a = np.random.Generator(np.random.PCG64(5))
    rng_b = np.random.Generator(np.random.PCG64(5))
    assert np.array_equal(np.asarray(s(img, rng_a)), np.asarray(clone(img, rng_b)))


# -------------------------------------------------------------- spawn loaders

@pytest.mark.parametrize("workers", [2, 0])
def test_folder_dataset_loads_under_spawn(images, workers):
    ds = FolderDataset(images, crop=64,
                       distortion_sampler=TrainDistortionSampler(), seed=17)
    batches = list(loader(ds, workers))
    assert len(batches) == 4
    for b in batches:
        assert b["pixels"].shape == (2, 3, 64, 64)
        assert b["label"].shape == (2,)


def test_manifest_dataset_loads_under_spawn(fixture_root):
    ds = ManifestDataset(fixture_root["manifest"], split="train", crop=64,
                         distortion_sampler=TrainDistortionSampler(), seed=17,
                         data_root=fixture_root["raw"])
    batches = list(loader(ds, workers=2, batch_size=4))
    assert batches and all(b["pixels"].shape == (4, 3, 64, 64) for b in batches)


def test_epoch_permutation_sampler_works_under_spawn(fixture_root):
    ds = ManifestDataset(fixture_root["manifest"], split="train", crop=64,
                         seed=17, data_root=fixture_root["raw"])
    dl = DataLoader(ds, batch_size=4, num_workers=2, drop_last=True,
                    sampler=EpochPermutationSampler(len(ds), seed=17, epoch=0),
                    worker_init_fn=seed_worker, multiprocessing_context=SPAWN)
    idx = torch.cat([b["idx"] for b in dl])
    assert idx.numel() == 16 and len(set(idx.tolist())) == 16


def test_spawn_workers_do_not_change_the_transform_result(images):
    """Per-item seeding is derived from the index, not from worker RNG state, so
    the same batch must come back regardless of worker count."""
    def run(workers):
        ds = FolderDataset(images, crop=64,
                           distortion_sampler=TrainDistortionSampler(), seed=17)
        return torch.cat([b["pixels"] for b in loader(ds, workers)])

    assert torch.allclose(run(0), run(2))


# --------------------------------------------------- the benchmark's own path

def test_benchmark_make_loader_survives_spawn(images, tmp_path):
    """The exact call the benchmark makes in end_to_end_dataloader mode."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_gpu_spawn", REPO / "scripts" / "benchmark_gpu.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    dl = bench.make_loader(images, crop=64, batch_size=2, workers=2, train=True,
                           seed=17, data_root=str(tmp_path),
                           multiprocessing_context=SPAWN)
    batch = next(iter(dl))
    assert batch["pixels"].shape == (2, 3, 64, 64)
    assert isinstance(dl.dataset, FolderDataset)   # from the package, not a script


@pytest.mark.skipif(sys.platform != "win32", reason="Windows default start method")
def test_windows_default_start_method_is_spawn():
    assert multiprocessing.get_start_method(allow_none=False) == "spawn"
