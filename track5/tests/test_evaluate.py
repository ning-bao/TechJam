"""src/evaluate.py — per-condition robustness evaluation (TC2 guide section 11)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import GROUPS, per_generator_metrics, resolve_conditions
from track5.transforms.eval_atoms import EVAL_15

REPO = Path(__file__).resolve().parents[1]
METRIC_KEYS = {"auroc", "ap", "bacc", "fpr_at_95tpr", "brier", "ece",
               "n_real", "n_fake"}


def run_eval(stub_config, stub_checkpoint, fixture_root, out, cond, *extra,
             denylist_dir=None):
    return subprocess.run(
        [sys.executable, "-u", "-m", "src.evaluate",
         "--config", str(stub_config["path"]),
         "--checkpoint", str(stub_checkpoint),
         "--manifest", str(fixture_root["manifest"]),
         "--condition", cond, "--output", str(out),
         "--data-root", str(fixture_root["raw"]),
         "--denylist-dir", str(denylist_dir or (out.parent / "no_denylist")),
         "--device", "cpu", *extra],
        capture_output=True, text=True, cwd=str(REPO))


# ------------------------------------------------------- condition resolution

def test_all_resolves_to_the_frozen_fifteen():
    assert resolve_conditions("all") == EVAL_15
    assert len(resolve_conditions("all")) == 15


def test_groups_partition_the_fifteen_exactly():
    seen = [c for g in ("clean_jpeg", "blur_resize", "noise_color_crop")
            for c in GROUPS[g]]
    assert sorted(seen) == sorted(EVAL_15)
    assert len(seen) == len(set(seen))          # no condition in two groups


@pytest.mark.parametrize("spec,expected", [
    ("jpeg_30", ["jpeg_30"]),
    ("jpeg_q30", ["jpeg_30"]),                  # TC2 spelling
    ("center_crop_0.80,blur_1.0", ["crop_80", "blur_10"]),
    ("group:blur_resize", GROUPS["blur_resize"]),
    ("noise_color_crop", GROUPS["noise_color_crop"]),
])
def test_condition_spec_forms(spec, expected):
    assert resolve_conditions(spec) == expected


def test_unknown_condition_and_group_are_rejected():
    with pytest.raises(KeyError):
        resolve_conditions("jpeg_q42")
    with pytest.raises(SystemExit):
        resolve_conditions("group:nope")


# ------------------------------------------------------------------ end-to-end

@pytest.fixture(scope="module")
def one_condition(tmp_path_factory, stub_config, stub_checkpoint, fixture_root):
    d = tmp_path_factory.mktemp("eval1")
    out = d / "jpeg_30.json"
    res = run_eval(stub_config, stub_checkpoint, fixture_root, out, "jpeg_30",
                   "--cache-dir", str(d / "cache"))
    assert res.returncode == 0, res.stdout + res.stderr
    return {"dir": d, "out": out, "res": res,
            "data": json.loads(out.read_text(encoding="utf-8"))}


def test_single_condition_writes_the_named_file(one_condition):
    assert one_condition["out"].exists()
    assert one_condition["data"]["condition"] == "jpeg_30"
    assert one_condition["data"]["n"] == 24


def test_all_required_metrics_are_reported(one_condition):
    m = one_condition["data"]["overall"]
    assert METRIC_KEYS <= set(m)
    assert 0.0 <= m["brier"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0
    assert m["n_real"] == 12 and m["n_fake"] == 12


def test_scores_are_calibrated_probabilities(one_condition):
    d = one_condition["data"]
    assert d["score_is_calibrated_probability"] is True
    assert d["calibration"]["temperature"] == 1.5   # from the checkpoint
    assert d["calibration"]["alpha"] == 0.25
    assert d["threshold"] == 0.6
    assert all(0.0 <= p["pred"] <= 1.0 for p in d["predictions"])
    assert len(d["predictions"]) == 24
    assert {p["image_id"] for p in d["predictions"]}.__len__() == 24


def test_per_generator_breakdown(one_condition):
    per = one_condition["data"]["per_generator"]
    assert set(per) == {"sd"}                       # the fixture's only family
    assert METRIC_KEYS <= set(per["sd"])
    assert per["sd"]["n_fake_family"] == 12
    assert per["sd"]["n_real"] == 12                # scored against all reals


def test_completion_marker_written_after_the_result(one_condition):
    marker = one_condition["dir"] / "jpeg_30.done.json"
    assert marker.exists()
    m = json.loads(marker.read_text(encoding="utf-8"))
    assert m["condition"] == "jpeg_30" and m["n"] == 24
    assert m["fingerprint"] == one_condition["data"]["receipt"]["fingerprint"]
    assert not list(one_condition["dir"].glob("*.tmp"))   # atomic writes


def test_receipt_identifies_the_run(one_condition):
    r = one_condition["data"]["receipt"]
    for key in ("checkpoint_sha256", "config_hash", "manifest_sha256",
                "atoms_version", "calibration", "crop", "seed", "backbone",
                "hostname", "torch", "device"):
        assert key in r, key


def test_all_conditions_write_one_file_each(tmp_path, stub_config, stub_checkpoint,
                                            fixture_root):
    out = tmp_path / "metrics"
    res = run_eval(stub_config, stub_checkpoint, fixture_root, out, "all",
                   "--cache-dir", str(tmp_path / "cache"))
    assert res.returncode == 0, res.stdout + res.stderr
    for cond in EVAL_15:
        assert (out / f"{cond}.json").exists(), cond
        assert (out / f"{cond}.done.json").exists(), cond
    assert len(list(out.glob("*.done.json"))) == 15


def test_group_selection_runs_only_that_group(tmp_path, stub_config,
                                              stub_checkpoint, fixture_root):
    out = tmp_path / "metrics"
    res = run_eval(stub_config, stub_checkpoint, fixture_root, out,
                   "group:blur_resize", "--cache-dir", str(tmp_path / "cache"))
    assert res.returncode == 0, res.stdout + res.stderr
    written = {p.stem for p in out.glob("*.json") if not p.name.endswith(".done.json")}
    assert written == set(GROUPS["blur_resize"])


# ----------------------------------------------------------------- resume

def test_resume_skips_completed_conditions(tmp_path, stub_config, stub_checkpoint,
                                           fixture_root):
    out = tmp_path / "metrics"
    cache = tmp_path / "cache"
    first = run_eval(stub_config, stub_checkpoint, fixture_root, out,
                     "group:clean_jpeg", "--cache-dir", str(cache))
    assert first.returncode == 0, first.stdout + first.stderr
    assert "5 evaluated, 0 already complete" in first.stdout

    second = run_eval(stub_config, stub_checkpoint, fixture_root, out,
                      "group:clean_jpeg", "--cache-dir", str(cache), "--resume")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "0 evaluated, 5 already complete" in second.stdout


def test_resume_reruns_when_the_marker_does_not_match(tmp_path, stub_config,
                                                      stub_checkpoint, fixture_root):
    out = tmp_path / "metrics"
    cache = tmp_path / "cache"
    assert run_eval(stub_config, stub_checkpoint, fixture_root, out, "clean",
                    "--cache-dir", str(cache)).returncode == 0

    marker = out / "clean.done.json"
    stale = json.loads(marker.read_text(encoding="utf-8"))
    stale["fingerprint"] = "0" * 16          # e.g. a different checkpoint
    marker.write_text(json.dumps(stale), encoding="utf-8")

    again = run_eval(stub_config, stub_checkpoint, fixture_root, out, "clean",
                     "--cache-dir", str(cache), "--resume")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "does not match this run, re-running" in again.stdout
    assert json.loads(marker.read_text(encoding="utf-8"))["fingerprint"] != "0" * 16


# ------------------------------------------------------------- protected runs

@pytest.fixture
def protected_denylist(tmp_path, fixture_root):
    """A denylist that marks the whole fixture manifest as protected."""
    d = tmp_path / "denylist"
    d.mkdir(parents=True)
    pd.DataFrame({"sha256": list(fixture_root["df"]["sha256"]),
                  "phash": list(fixture_root["df"]["phash"]),
                  "reason": ["coco_val2017"] * len(fixture_root["df"])}
                 ).to_parquet(d / "denylist.parquet", index=False)
    return d


def test_protected_manifest_refused_without_the_flag(tmp_path, stub_config,
                                                     stub_checkpoint, fixture_root,
                                                     protected_denylist):
    res = run_eval(stub_config, stub_checkpoint, fixture_root,
                   tmp_path / "m", "clean", "--cache-dir", str(tmp_path / "c"),
                   denylist_dir=protected_denylist)
    assert res.returncode == 2
    assert "single frozen-model event" in res.stderr


def test_protected_run_refuses_to_repeat_without_resume(tmp_path, stub_config,
                                                        stub_checkpoint,
                                                        fixture_root,
                                                        protected_denylist):
    out = tmp_path / "m"
    cache = tmp_path / "c"
    first = run_eval(stub_config, stub_checkpoint, fixture_root, out, "clean",
                     "--cache-dir", str(cache), "--protected-run",
                     denylist_dir=protected_denylist)
    assert first.returncode == 0, first.stdout + first.stderr
    data = json.loads((out / "clean.json").read_text(encoding="utf-8"))
    assert data["protected_run"] is True
    assert data["manifest_protection"]["hash_hits"] == 24

    repeat = run_eval(stub_config, stub_checkpoint, fixture_root, out, "clean",
                      "--cache-dir", str(cache), "--protected-run",
                      denylist_dir=protected_denylist)
    assert repeat.returncode == 3
    assert "Refusing to overwrite" in repeat.stderr

    resumed = run_eval(stub_config, stub_checkpoint, fixture_root, out, "clean",
                       "--cache-dir", str(cache), "--protected-run", "--resume",
                       denylist_dir=protected_denylist)
    assert resumed.returncode == 0


# ------------------------------------------------------------ transform reuse

def test_conditions_share_the_frozen_transform_cache(one_condition):
    cache = one_condition["dir"] / "cache" / "jpeg_30"
    assert cache.is_dir()
    files = list(cache.glob("*.jpg"))            # jpeg conditions cache as JPEG
    assert len(files) == 24
    assert all(f.stem for f in files)             # named by image sha256


def test_per_generator_helper_uses_all_reals():
    rows = pd.DataFrame({"label": [0, 0, 1, 1], "generator_family": ["", "", "sd", "mj"]})
    import numpy as np

    out = per_generator_metrics(rows, np.array([0.1, 0.2, 0.9, 0.8]), 0.5)
    assert set(out) == {"sd", "mj"}
    for fam in out.values():
        assert fam["n_real"] == 2 and fam["n_fake"] == 1
