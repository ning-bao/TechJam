import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from track5.data.denylist import apply_denylist, build_denylist_df
from track5.data.manifest import SCHEMA, build_manifest, row_from_bytes
from track5.data.probes import gate_passes, run_all_probes
from track5.data.resolve import resolve_image_bytes


def png_bytes(seed=0, w=48, h=32):
    rng = np.random.Generator(np.random.PCG64(seed))
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def jpg_bytes(seed=0, w=48, h=32, q=85):
    rng = np.random.Generator(np.random.PCG64(seed))
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "raw"
    # plain files: wildfake-style layout with Real dir + fake zip
    real_dir = root / "WildFake" / "Images" / "Real"
    real_dir.mkdir(parents=True)
    for i in range(4):
        (real_dir / f"r{i}.png").write_bytes(png_bytes(seed=i))
    fake_zip = root / "WildFake" / "Images" / "Diffusion_based" / "DDIM.zip"
    fake_zip.parent.mkdir(parents=True)
    with zipfile.ZipFile(fake_zip, "w") as zf:
        for i in range(4):
            zf.writestr(f"DDIM/f{i}.jpg", jpg_bytes(seed=100 + i))
    # COCO-style zip
    coco = root / "COCO"
    coco.mkdir()
    with zipfile.ZipFile(coco / "val2017.zip", "w") as zf:
        for i in range(3):
            zf.writestr(f"val2017/{i:012d}.jpg", jpg_bytes(seed=200 + i))
    # SID-style parquet with image struct + label
    sid = root / "SID_Set" / "data"
    sid.mkdir(parents=True)
    imgs = [{"bytes": png_bytes(seed=300 + i), "path": f"s{i}.png"} for i in range(4)]
    tbl = pa.table({
        "img_id": [f"s{i}" for i in range(4)],
        "image": imgs,
        "label": [0, 1, 2, 1],  # 2 = tampered, must be skipped
    })
    pq.write_table(tbl, sid / "train-00000-of-00001.parquet")
    return root


def test_resolve_grammars(data_root):
    plain = resolve_image_bytes(data_root, "WildFake/Images/Real/r0.png")
    assert plain == png_bytes(seed=0)
    zipped = resolve_image_bytes(data_root, "COCO/val2017.zip#val2017/000000000000.jpg")
    assert zipped == jpg_bytes(seed=200)
    parq = resolve_image_bytes(data_root, "SID_Set/data/train-00000-of-00001.parquet#1")
    assert parq == png_bytes(seed=301)


def test_build_manifest_wildfake(data_root, tmp_path):
    out = tmp_path / "wf.parquet"
    df = build_manifest("wildfake", data_root, out, workers=2)
    assert list(df.columns) == list(SCHEMA)
    for col, dtype in SCHEMA.items():
        assert str(df[col].dtype) == dtype, col
    assert (df[df["label"] == 0]["generator_family"] == "").all()
    assert set(df[df["label"] == 1]["generator_family"]) == {"ddpm"}
    assert len(df) == 8
    # zip members use the # grammar
    assert df["path"].str.contains("#").sum() == 4


def test_build_manifest_sid_skips_tampered(data_root, tmp_path):
    df = build_manifest("sid_set", data_root, tmp_path / "sid.parquet", workers=1)
    assert len(df) == 3  # label==2 row skipped
    assert set(df["label"]) == {0, 1}
    assert (df[df["label"] == 1]["generator_family"] == "flux").all()


def test_manifest_jpeg_quality_estimate(data_root):
    row = row_from_bytes("x.jpg", jpg_bytes(q=85), 0, "t", "")
    assert 75 <= row["jpeg_quality"] <= 95
    row_png = row_from_bytes("x.png", png_bytes(), 0, "t", "")
    assert row_png["jpeg_quality"] == -1


def test_denylist_exact_and_near(tmp_path):
    """The pHash gate catches near-duplicates of a protected image that arrive
    under a *different* source, which the family/source gates cannot see."""
    protected = row_from_bytes("a.png", png_bytes(seed=1), 0, "coco_val2017", "")
    h = int(protected["phash"], 16)
    flip4 = h ^ 0b1111          # hamming 4
    flip5 = h ^ 0b11111         # hamming 5
    candidate = {**protected, "source": "wildfake"}
    man = pd.DataFrame([
        {**candidate, "path": "exact", "split": "unassigned"},
        {**candidate, "path": "near4", "sha256": "0" * 64, "phash": f"{flip4:016x}",
         "split": "unassigned"},
        {**candidate, "path": "near5", "sha256": "1" * 64, "phash": f"{flip5:016x}",
         "split": "unassigned"},
    ])
    deny = build_denylist_df([pd.DataFrame([protected])])
    assert set(deny["reason"]) == {"coco_val2017"}
    out = apply_denylist(man, deny)
    assert list(out["split"]) == ["denied", "denied", "unassigned"]


def test_source_gate_denies_val2017_rows_without_any_hash_table():
    """A row that declares itself COCO val2017 is denied even with an empty
    content denylist - the source gate survives a renamed path."""
    row = row_from_bytes("whatever.png", png_bytes(seed=2), 0, "coco_val2017", "")
    man = pd.DataFrame([{**row, "split": "unassigned"}])
    out = apply_denylist(man, pd.DataFrame(columns=["sha256", "phash", "reason"]))
    assert list(out["split"]) == ["denied"]


def test_denylist_wildfake_dalle_reason():
    fake = row_from_bytes("d.png", png_bytes(seed=9), 1, "wildfake", "dalle")
    deny = build_denylist_df([pd.DataFrame([fake])])
    assert list(deny["reason"]) == ["wildfake_dalle"]


def _meta_df(n=400, planted_shortcut=False, seed=0):
    rng = np.random.Generator(np.random.PCG64(seed))
    y = np.array([0, 1] * (n // 2))
    if planted_shortcut:
        size = np.where(y == 0, 1000, 5000) + rng.integers(-10, 10, n)
    else:
        size = rng.integers(900, 5100, n)
    return pd.DataFrame({
        "label": y, "file_bytes": size,
        "width": rng.integers(200, 800, n), "height": rng.integers(200, 800, n),
        "jpeg_quality": rng.integers(60, 100, n),
    })


def test_probes_random_vs_planted():
    clean = run_all_probes(_meta_df(planted_shortcut=False))
    assert clean["file_size"] < 0.60
    assert gate_passes(clean)
    leaky = run_all_probes(_meta_df(planted_shortcut=True))
    assert leaky["file_size"] > 0.9
    assert not gate_passes(leaky)


def test_probe_gate_exit_codes(data_root, tmp_path, monkeypatch):
    import subprocess
    import sys as _sys
    df = _meta_df(planted_shortcut=True)
    df["split"] = "unassigned"
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    res = subprocess.run(
        [_sys.executable, "scripts/probe_gate.py", "--manifest", str(p)],
        capture_output=True, text=True, cwd="F:/Hackathon/track5")
    assert res.returncode == 1
    df2 = _meta_df(planted_shortcut=False)
    df2["split"] = "unassigned"
    p2 = tmp_path / "m2.parquet"
    df2.to_parquet(p2)
    res2 = subprocess.run(
        [_sys.executable, "scripts/probe_gate.py", "--manifest", str(p2)],
        capture_output=True, text=True, cwd="F:/Hackathon/track5")
    assert res2.returncode == 0, res2.stdout + res2.stderr


def test_manifest_dataset(data_root, tmp_path):
    torch = pytest.importorskip("torch")
    out = tmp_path / "wf.parquet"
    build_manifest("wildfake", data_root, out, workers=1)
    from track5.data.dataset import ManifestDataset

    ds = ManifestDataset(out, split=None, crop=64, seed=3, data_root=data_root)
    item = ds[0]
    assert item["pixels"].shape == (3, 64, 64)
    assert item["label"] in (0, 1)
    # deterministic given same idx/seed
    a, b = ds[2]["pixels"], ds[2]["pixels"]
    assert torch.equal(a, b)
