"""WildFake label-CSV manifest builder and the protected-set denylist (C2/D4).

The real archives are still downloading, so the builder must work from the CSVs,
resolve only what is present, and refuse the whole DALL·E family by path.
"""

import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from track5.data.denylist import apply_denylist, build_protected_paths
from track5.data.manifest import MissingImages, build_manifest
from track5.data.wildfake_csv import (ArchiveIndex, Row, coco_val2017_demo_ids,
                                      protected_keys, protected_reason,
                                      protected_summary, wildfake_key)

REPO = Path(__file__).resolve().parents[1]
HEADER = "Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num\n"


def png_bytes(i: int) -> bytes:
    rng = np.random.Generator(np.random.PCG64(i))
    buf = BytesIO()
    Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(
        buf, format="PNG")
    return buf.getvalue()


def write_zip(path: Path, members: dict[str, bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def write_csv(path: Path, rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{g},{a},{a},{a},{adv},{fake},{p},{i + 1}\n"
                   for i, (g, a, adv, fake, p) in enumerate(rows))
    path.write_text(HEADER + body, encoding="utf-8")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """A miniature WildFake with both archive layouts, one truncated archive, an
    undownloaded DALL·E family, and a COCO val2017 zip."""
    root = tmp_path_factory.mktemp("corpus")
    csvs = root / "WildFake" / "label_csv_files"
    images = root / "WildFake" / "Images"

    # layout A: the archive name is a CSV path component
    write_zip(images / "Diffusion_based" / "DDIM.zip",
              {f"DDIM/imgs/{i}.png": png_bytes(i) for i in range(4)})
    write_csv(csvs / "ddim.csv",
              [("Diffusion_based", "DDIM", 0, 1, f"./Diffusion_based/DDIM/imgs/{i}.png")
               for i in range(4)])

    # layout B: a split archive whose name is absent from the CSV path
    write_zip(images / "Diffusion_based" / "Midjourney" / "Typical" / "part_1.zip",
              {f"part_1/{i}.png": png_bytes(100 + i) for i in range(3)})
    write_csv(csvs / "mjv4.csv",
              [("Diffusion_based", "Midjourney", 0, 1,
                f"./Diffusion_based/Midjourney/Typical/{i}.png") for i in range(3)])

    # still downloading -> unusable, must be skipped not crashed on
    (images / "Diffusion_based" / "Imagen.zip").write_bytes(b"PK\x03\x04 truncated")
    write_csv(csvs / "imagen.csv",
              [("Diffusion_based", "Imagen", 0, 1,
                f"./Diffusion_based/Imagen/x/{i}.png") for i in range(2)])

    # protected: the DALL-E archives do not exist at all
    write_csv(csvs / "dalle3.csv",
              [("Diffusion_based", "DALLE", 1, 1,
                f"./Diffusion_based/DALLE/Advanced/DALLE3/{i}.jpg") for i in range(5)])
    write_csv(csvs / "dalle2.csv",
              [("Diffusion_based", "DALLE", 0, 1,
                f"./Diffusion_based/DALLE/Typical/DALLE2/{i}.png") for i in range(3)])

    # real COCO: train2017 is trainable, val2017 is protected
    real = {f"coco/coco2017/train2017/img{i:03d}.jpg": png_bytes(200 + i)
            for i in range(3)}
    real |= {f"coco/coco2017/val2017/img{i:03d}.jpg": png_bytes(300 + i)
             for i in range(2)}
    write_zip(images / "Real" / "coco.zip", real)
    write_csv(csvs / "real_coco.csv",
              [("Real", "coco", 0, 0, f"./Real/{k}") for k in real])

    return root


# ------------------------------------------------------------ key normalizing

@pytest.mark.parametrize("path,expected", [
    ("./Diffusion_based/DDIM/imgs/0.png", "diffusion_based/ddim/imgs/0.png"),
    ("WildFake/Images/Diffusion_based/DDIM.zip#DDIM/imgs/0.png",
     "diffusion_based/ddim/imgs/0.png"),
    ("WildFake/Images/Real/ffhq.zip#ffhq/images/x.jpg", "real/ffhq/images/x.jpg"),
    ("WildFake/Images/Other_based.zip#Other_based/Typical/VQVAE/x.png",
     "other_based/typical/vqvae/x.png"),
    ("WildFake/Images/Real/plain/x.jpg", "real/plain/x.jpg"),
    (r"WildFake\Images\Real\plain\x.jpg", "real/plain/x.jpg"),
])
def test_wildfake_key_normalizes_both_spellings(path, expected):
    assert wildfake_key(path) == expected


# ----------------------------------------------------------- archive indexing

def test_resolves_both_archive_layouts(corpus):
    idx = ArchiveIndex(corpus)
    a = idx.resolve("diffusion_based/ddim/imgs/2.png")
    assert a == "WildFake/Images/Diffusion_based/DDIM.zip#DDIM/imgs/2.png"
    b = idx.resolve("diffusion_based/midjourney/typical/1.png")
    assert b == ("WildFake/Images/Diffusion_based/Midjourney/Typical/"
                 "part_1.zip#part_1/1.png")


def test_truncated_archive_is_skipped_not_fatal(corpus):
    idx = ArchiveIndex(corpus)
    assert idx.resolve("diffusion_based/imagen/x/0.png") is None
    assert any("Imagen.zip" in z for z in idx.unavailable)


def test_undownloaded_family_resolves_to_nothing(corpus):
    idx = ArchiveIndex(corpus)
    assert idx.resolve("diffusion_based/dalle/advanced/dalle3/0.jpg") is None


def test_resolved_paths_load_through_the_manifest_resolver(corpus):
    from track5.data.resolve import resolve_image_bytes

    idx = ArchiveIndex(corpus)
    for key in ("diffusion_based/ddim/imgs/0.png",
                "diffusion_based/midjourney/typical/0.png"):
        data = resolve_image_bytes(corpus, idx.resolve(key))
        img = Image.open(BytesIO(data))
        img.load()
        assert img.size == (40, 40)


# ------------------------------------------------------------ protected paths

def test_protected_keys_cover_both_dalle_tiers_and_coco_val(corpus):
    keys = protected_keys(corpus)
    dalle = {k for k, v in keys.items() if v == "wildfake_dalle"}
    coco = {k for k, v in keys.items() if v == "coco_val2017"}
    assert len(dalle) == 8          # 5 Advanced + 3 Typical: the whole family
    assert all("dalle" in k for k in dalle)
    assert len(coco) == 2           # val2017 only
    assert all("/val2017/" in f"/{k}" for k in coco)
    assert not any("train2017" in k for k in keys)


def test_build_protected_paths_table(corpus):
    df = build_protected_paths(corpus)
    assert list(df.columns) == ["path_key", "reason"]
    assert set(df["reason"]) == {"wildfake_dalle", "coco_val2017"}
    assert len(df) == 10


def test_protected_summary_counts_both_tiers(corpus):
    s = protected_summary(corpus)
    assert s["wildfake_dalle_typical"] == 3
    assert s["wildfake_dalle_advanced"] == 5
    assert s["wildfake_dalle_total"] == 8
    assert s["coco_val2017_demo"] == 2


def wf_row(**over):
    base = dict(csv_name="mjv4.csv", key="diffusion_based/midjourney/typical/x.png",
                generator="Diffusion_based", architecture="Midjourney",
                category="Midjourney", is_advanced=False, is_fake=True)
    base.update(over)
    return Row(**base)


def test_clean_row_is_not_protected():
    assert protected_reason(wf_row()) == ""


@pytest.mark.parametrize("over", [
    {"csv_name": "dalle2.csv"},                       # source CSV
    {"csv_name": "dalle3.csv"},
    {"architecture": "DALLE"},                        # architecture column
    {"architecture": "dalle"},                        # case-insensitive
    {"category": "DALLE"},                            # category column
    {"key": "diffusion_based/dalle/advanced/x.jpg"},  # path
    {"key": "some/renamed/dall-e/x.jpg"},
])
def test_every_dalle_identifier_denies_independently(over):
    """Item 6: a renamed path must not bypass denial, and vice versa - each
    identifier alone is sufficient."""
    assert protected_reason(wf_row(**over)) == "wildfake_dalle"


def test_renamed_dalle_path_is_still_caught_by_its_architecture():
    sneaky = wf_row(csv_name="mjv4.csv", architecture="DALLE", category="misc",
                    key="diffusion_based/midjourney/typical/innocent.png")
    assert protected_reason(sneaky) == "wildfake_dalle"


def test_apply_denylist_denies_by_family_and_source_not_only_path():
    df = pd.DataFrame({
        "path": ["renamed/innocent_looking.png", "Real/coco/val/x.jpg", "ok/y.png"],
        "sha256": ["a", "b", "c"], "phash": ["0" * 16] * 3,
        "generator_family": ["dalle", "", "sd"],
        "source": ["wildfake", "coco_val2017", "wildfake"],
        "split": ["train"] * 3})
    out = apply_denylist(df, pd.DataFrame(columns=["sha256", "phash", "reason"]))
    assert list(out["split"]) == ["denied", "denied", "train"]


def test_coco_demo_subset_is_smaller_than_the_denylist(corpus):
    """Item 5: the benchmark subset and the denied archive are different sets."""
    demo = coco_val2017_demo_ids(corpus)
    assert demo == {"img000.jpg", "img001.jpg"}
    denied = {k for k, v in protected_keys(corpus).items() if v == "coco_val2017"}
    assert len(denied) == len(demo)  # this fixture's CSV lists only the subset


def test_coco_val_demo_manifest_excludes_archive_only_images(corpus, tmp_path):
    root = tmp_path / "raw"
    _copytree(corpus, root)
    (root / "COCO").mkdir(parents=True)
    # the archive holds the 2 benchmark images plus 2 extras
    write_zip(root / "COCO" / "val2017.zip",
              {"val2017/img000.jpg": _jpeg(0), "val2017/img001.jpg": _jpeg(1),
               "val2017/img900.jpg": _jpeg(9), "val2017/img901.jpg": _jpeg(8)})
    df = build_manifest("coco_val_demo", root, tmp_path / "demo.parquet", workers=2)
    names = {p.rsplit("/", 1)[-1] for p in df["path"]}
    assert names == {"img000.jpg", "img001.jpg"}     # benchmark subset only
    assert set(df["source"]) == {"coco_val2017"} and set(df["label"]) == {0}
    receipt = json.loads(
        (tmp_path / "demo.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["demo_expected"] == 2 and receipt["demo_found"] == 2
    assert receipt["outside_benchmark"] == 2         # denied but not evaluated
    assert receipt["complete"] is True


def test_apply_denylist_marks_protected_paths_denied(corpus):
    protected = build_protected_paths(corpus)
    df = pd.DataFrame({
        "path": ["WildFake/Images/Diffusion_based/DDIM.zip#DDIM/imgs/0.png",
                 "WildFake/Images/Diffusion_based/DALLE.zip#DALLE/Advanced/DALLE3/0.jpg",
                 "WildFake/Images/Real/coco.zip#coco/coco2017/val2017/img000.jpg",
                 "WildFake/Images/Real/coco.zip#coco/coco2017/train2017/img000.jpg"],
        "sha256": ["a", "b", "c", "d"], "phash": ["0" * 16] * 4,
        "split": ["train"] * 4})
    out = apply_denylist(df, pd.DataFrame(columns=["sha256", "phash", "reason"]),
                         protected_paths_df=protected)
    assert list(out["split"]) == ["train", "denied", "denied", "train"]


# ---------------------------------------------------------- manifest building

@pytest.fixture(scope="module")
def built(corpus, tmp_path_factory):
    """Development-mode build: the fixture corpus deliberately holds a truncated
    archive and an undownloaded family, which a strict build must reject."""
    out = tmp_path_factory.mktemp("man") / "wildfake.parquet"
    return build_manifest("wildfake_csv", corpus, out, workers=2, allow_missing=True)


def test_manifest_has_the_interfaces_schema(built):
    from track5.data.manifest import SCHEMA

    assert list(built.columns) == list(SCHEMA)
    assert set(built["split"]) == {"unassigned"}
    assert (built["file_bytes"] > 0).all()
    assert (built["width"] == 40).all()


def test_manifest_never_contains_the_protected_family(built, corpus):
    assert "dalle" not in set(built["generator_family"])
    keys = set(build_protected_paths(corpus)["path_key"])
    assert not any(wildfake_key(p) in keys for p in built["path"])
    assert not any("val2017" in p for p in built["path"])
    assert any("train2017" in p for p in built["path"])  # trainable COCO kept


def test_labels_and_families_come_from_the_csv(built):
    fam = dict(zip(built["path"], built["generator_family"]))
    ddim = [p for p in fam if "DDIM" in p]
    mj = [p for p in fam if "part_1" in p]
    assert ddim and all(fam[p] == "ddpm" for p in ddim)   # DDIM -> ddpm family
    assert mj and all(fam[p] == "mj" for p in mj)
    reals = built[built["label"] == 0]
    assert len(reals) == 3 and set(reals["generator_family"]) == {""}
    assert set(built["generator_family"]) <= {
        "", "sd", "mj", "adm", "ddpm", "vqdm", "gan", "glide", "dalle", "flux",
        "vae_sd15", "vae_sdxl", "other"}                  # INTERFACES §4 vocabulary


def test_undownloaded_and_truncated_rows_are_simply_absent(built):
    assert len(built) == 4 + 3 + 3   # ddim + midjourney + coco train2017
    assert not any("Imagen" in p for p in built["path"])


def test_per_family_limit_caps_each_generator(corpus, tmp_path):
    df = build_manifest("wildfake_csv", corpus, tmp_path / "cap.parquet",
                        workers=2, per_family_limit=2, allow_missing=True)
    assert df.groupby("generator_family").size().max() == 2


# ------------------------------------------------ strict vs development build

def test_final_build_fails_when_requested_images_are_missing(corpus, tmp_path):
    """Item 7: silently skipping 2.5M rows is only acceptable in dev mode."""
    out = tmp_path / "strict.parquet"
    with pytest.raises(MissingImages, match="requested images are not on disk"):
        build_manifest("wildfake_csv", corpus, out, workers=2)

    receipt = json.loads(
        (tmp_path / "strict.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is False
    assert receipt["allow_missing"] is False
    assert receipt["requested_unavailable"] == 2       # the truncated Imagen rows
    assert receipt["missing_examples"]
    assert any("Imagen" in z for z in receipt["unavailable_archives"])
    assert out.exists(), "the partial manifest is still written for inspection"


def test_scoping_to_available_csvs_lets_a_strict_build_pass(corpus, tmp_path):
    df = build_manifest("wildfake_csv", corpus, tmp_path / "scoped.parquet",
                        workers=2, csvs=["ddim.csv", "mjv4.csv"])
    assert len(df) == 7
    receipt = json.loads(
        (tmp_path / "scoped.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is True
    assert receipt["csvs"] == ["ddim.csv", "mjv4.csv"]
    assert receipt["requested_unavailable"] == 0


def test_development_build_is_labelled_as_such(corpus, tmp_path):
    build_manifest("wildfake_csv", corpus, tmp_path / "dev.parquet", workers=2,
                   allow_missing=True)
    receipt = json.loads(
        (tmp_path / "dev.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["allow_missing"] is True and receipt["complete"] is False


def test_unreadable_member_also_fails_a_strict_build(corpus, tmp_path):
    """An archive that opens but yields a corrupt member must not pass either."""
    root = tmp_path / "raw"
    _copytree(corpus, root)
    write_zip(root / "WildFake" / "Images" / "Diffusion_based" / "DDIM.zip",
              {f"DDIM/imgs/{i}.png": (png_bytes(i) if i else b"corrupt")
               for i in range(4)})
    with pytest.raises(MissingImages, match="could not be read"):
        build_manifest("wildfake_csv", root, tmp_path / "u.parquet", workers=2,
                       csvs=["ddim.csv"])
    receipt = json.loads(
        (tmp_path / "u.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["unreadable"] == 1
    assert "DDIM/imgs/0.png" in receipt["unreadable_examples"][0]["path"]


def test_cli_exit_codes_for_strict_and_dev_builds(corpus, tmp_path):
    def run(*extra):
        return subprocess.run(
            [sys.executable, "-u", "scripts/build_manifests.py",
             "--source", "wildfake_csv", "--data-root", str(corpus),
             "--out", str(tmp_path / "cli.parquet"), "--workers", "2", *extra],
            capture_output=True, text=True, cwd=str(REPO))

    strict = run()
    assert strict.returncode == 3
    assert "A final manifest must be built from a complete corpus" in strict.stderr

    dev = run("--allow-missing")
    assert dev.returncode == 0
    assert "DEVELOPMENT BUILD" in dev.stdout


# --------------------------------------------------------- denylist CLI (COCO)

def build_denylist(root: Path, out_dir: Path, *extra):
    return subprocess.run(
        [sys.executable, "-u", "scripts/build_denylist.py", "--coco-val",
         "--data-root", str(root), "--out", str(out_dir / "denylist.parquet"),
         "--out-paths", str(out_dir / "protected_paths.parquet"),
         "--workers", "2", *extra],
        capture_output=True, text=True, cwd=str(REPO))


def coco_root(corpus: Path, tmp_path: Path, members: dict) -> Path:
    root = tmp_path / "raw"
    (root / "COCO").mkdir(parents=True)
    write_zip(root / "COCO" / "val2017.zip", members)
    _copytree(corpus / "WildFake", root / "WildFake")
    return root


def test_build_denylist_hashes_coco_val2017(corpus, tmp_path):
    root = coco_root(corpus, tmp_path,
                     {f"val2017/{i:012d}.jpg": _jpeg(i) for i in range(6)})
    res = build_denylist(root, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr

    deny = pd.read_parquet(tmp_path / "denylist.parquet")
    assert list(deny.columns) == ["sha256", "phash", "reason"]
    assert len(deny) == 6 and set(deny["reason"]) == {"coco_val2017"}
    assert deny["sha256"].is_unique
    paths = pd.read_parquet(tmp_path / "protected_paths.parquet")
    assert set(paths["reason"]) == {"wildfake_dalle", "coco_val2017"}

    receipt = json.loads(
        (tmp_path / "denylist.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is True
    assert receipt["coverage"]["coco_val2017"] == {
        "source": str(root / "COCO" / "val2017.zip"), "present": True,
        "expected": 6, "hashed": 6, "failed": [], "complete": True,
        "attempted": True}


def test_partial_coco_archive_is_refused(corpus, tmp_path):
    """A half-hashed protected set is a silent C2 hole — the builder must say so."""
    members = {f"val2017/{i:012d}.jpg": _jpeg(i) for i in range(4)}
    members["val2017/000000000099.jpg"] = b"truncated jpeg bytes"
    root = coco_root(corpus, tmp_path, members)

    res = build_denylist(root, tmp_path)
    assert res.returncode == 2
    assert "INCOMPLETE" in res.stderr
    receipt = json.loads(
        (tmp_path / "denylist.parquet.receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is False
    assert receipt["coverage"]["coco_val2017"]["hashed"] == 4
    assert len(receipt["coverage"]["coco_val2017"]["failed"]) == 1

    ok = build_denylist(root, tmp_path, "--allow-partial")
    assert ok.returncode == 0
    assert "WARNING" in ok.stderr


def test_training_gate_refuses_an_incomplete_denylist(tmp_path):
    from src.train import assert_no_protected_overlap

    repo = tmp_path / "repo"
    (repo / "data" / "denylist").mkdir(parents=True)
    deny = repo / "data" / "denylist" / "denylist.parquet"
    pd.DataFrame({"sha256": ["a"], "phash": ["0" * 16],
                  "reason": ["coco_val2017"]}).to_parquet(deny, index=False)
    df = pd.DataFrame({"path": ["x.jpg"], "sha256": ["z"], "phash": ["f" * 16],
                       "split": ["train"]})

    deny.with_name(deny.name + ".receipt.json").write_text(
        json.dumps({"complete": False, "coverage": {"coco_val2017": {"hashed": 1}}}),
        encoding="utf-8")
    with pytest.raises(SystemExit, match="denylist is incomplete"):
        assert_no_protected_overlap(df, repo)

    deny.with_name(deny.name + ".receipt.json").write_text(
        json.dumps({"complete": True, "coverage": {}}), encoding="utf-8")
    assert assert_no_protected_overlap(df, repo)["denylist_complete"] is True


def test_training_gate_refuses_a_manifest_that_intersects(tmp_path):
    from src.train import assert_no_protected_overlap

    repo = tmp_path / "repo"
    (repo / "data" / "denylist").mkdir(parents=True)
    pd.DataFrame({"sha256": ["dead"], "phash": ["0" * 16],
                  "reason": ["coco_val2017"]}).to_parquet(
        repo / "data" / "denylist" / "denylist.parquet", index=False)
    df = pd.DataFrame({"path": ["x.jpg"], "sha256": ["dead"], "phash": ["f" * 16],
                       "split": ["train"]})
    with pytest.raises(SystemExit, match="intersects the protected set"):
        assert_no_protected_overlap(df, repo)


def test_training_gate_refuses_without_any_denylist(tmp_path):
    from src.train import assert_no_protected_overlap

    df = pd.DataFrame({"path": ["x.jpg"], "sha256": ["a"], "phash": ["0" * 16],
                       "split": ["train"]})
    with pytest.raises(SystemExit, match="no denylist table found"):
        assert_no_protected_overlap(df, tmp_path / "empty_repo")


def _jpeg(i: int) -> bytes:
    rng = np.random.Generator(np.random.PCG64(500 + i))
    buf = BytesIO()
    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
        buf, format="JPEG", quality=85)
    return buf.getvalue()


def _copytree(src: Path, dst: Path):
    import shutil

    shutil.copytree(src, dst)
