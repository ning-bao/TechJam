"""The training-set builder must reject parquet-row paths.

Reading "shard.parquet#row" decodes a whole ~844-image row group per image. It
is fine for one sequential pass and ruinous under shuffled training access, and
it comes back silently whenever a source manifest still carries it. Zip-member
paths look similar but are cheap, so the guard must not confuse the two.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_training_set.py"

COLS = ["path", "sha256", "phash", "label", "source", "generator_family",
        "width", "height", "format", "file_bytes", "jpeg_quality",
        "n_recompress", "split"]


def make_manifest(path: Path, paths, label: int, source: str, family: str = ""):
    rows = []
    for i, p in enumerate(paths):
        rows.append({
            "path": p, "sha256": f"{label}{source}{i:060d}"[:64],
            "phash": f"{i:016x}", "label": label, "source": source,
            "generator_family": family, "width": 512, "height": 512,
            "format": "png", "file_bytes": 1000 + i, "jpeg_quality": -1,
            "n_recompress": 0, "split": "unassigned"})
    pd.DataFrame(rows, columns=COLS).to_parquet(path, index=False)


def run_build(tmp_path, extra=()):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--crop", "448", "--out-dir", str(tmp_path),
         "--manifests", "r", "f", *extra],
        capture_output=True, text=True, cwd=str(REPO))


@pytest.fixture
def denylist_present():
    if not (REPO / "data/denylist/denylist.parquet").exists():
        pytest.skip("denylist not built")


def test_parquet_row_paths_are_rejected(tmp_path, denylist_present):
    make_manifest(tmp_path / "r.parquet",
                  [f"SID_Set/data/train-000{i:02d}.parquet#{i}" for i in range(40)],
                  0, "sid_set")
    make_manifest(tmp_path / "f.parquet",
                  [f"gen/{i}.png" for i in range(40)], 1, "wildfake", "sd")
    res = run_build(tmp_path)
    assert res.returncode == 3, res.stdout + res.stderr
    assert "parquet-row paths" in res.stderr
    assert "extract_sidset" in res.stderr


def test_zip_member_paths_are_accepted(tmp_path, denylist_present):
    """Zip members contain '#' too but cost one cheap read - never blocked."""
    make_manifest(tmp_path / "r.parquet",
                  [f"COCO/train2017.zip#train2017/{i:06d}.jpg" for i in range(40)],
                  0, "coco_train2017")
    make_manifest(tmp_path / "f.parquet",
                  [f"WildFake/Images/SD.zip#sd/{i}.png" for i in range(40)],
                  1, "wildfake", "sd")
    res = run_build(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "train.parquet").exists()


def test_override_flag_allows_parquet_paths(tmp_path, denylist_present):
    make_manifest(tmp_path / "r.parquet",
                  [f"SID_Set/data/train-000{i:02d}.parquet#{i}" for i in range(40)],
                  0, "sid_set")
    make_manifest(tmp_path / "f.parquet",
                  [f"gen/{i}.png" for i in range(40)], 1, "wildfake", "sd")
    res = run_build(tmp_path, extra=["--allow-parquet-paths"])
    assert res.returncode == 0, res.stdout + res.stderr
