"""src/predict.py — organiser record contract (TC2 guide §6):
one {"image_path", "pred"} record per input image, pred a calibrated probability.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run_predict(*extra):
    return subprocess.run([sys.executable, "-u", "-m", "src.predict", *extra],
                          capture_output=True, text=True, cwd=str(REPO))


def read_records(out: Path, fmt: str = "json"):
    text = out.read_text(encoding="utf-8")
    if fmt == "json":
        return json.loads(text)
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def test_default_output_is_a_valid_json_array(fixture_root, stub_checkpoint, tmp_path):
    """The organiser said only 'JSON'; an array parses with any JSON reader."""
    out = tmp_path / "preds.json"
    res = run_predict("--checkpoint", str(stub_checkpoint),
                      "--input", str(fixture_root["raw"] / "fixtures"),
                      "--output", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    parsed = json.loads(out.read_text(encoding="utf-8"))   # whole file, one call
    assert isinstance(parsed, list) and len(parsed) == 24
    assert all(set(r) == {"image_path", "pred"} for r in parsed)
    assert "JSON array" in res.stdout


def test_jsonl_is_still_available_explicitly(fixture_root, stub_checkpoint, tmp_path):
    out = tmp_path / "preds.jsonl"
    res = run_predict("--checkpoint", str(stub_checkpoint),
                      "--input", str(fixture_root["raw"] / "fixtures"),
                      "--output", str(out), "--format", "jsonl")
    assert res.returncode == 0, res.stdout + res.stderr
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 24
    assert all(set(json.loads(ln)) == {"image_path", "pred"} for ln in lines)
    assert "JSON Lines" in res.stdout


@pytest.mark.parametrize("fmt", ["jsonl", "json"])
def test_records_from_directory(fixture_root, stub_checkpoint, tmp_path, fmt):
    out = tmp_path / f"preds.{fmt}"
    res = run_predict("--checkpoint", str(stub_checkpoint),
                      "--input", str(fixture_root["raw"] / "fixtures"),
                      "--output", str(out), "--format", fmt, "--batch-size", "5")
    assert res.returncode == 0, res.stdout + res.stderr

    records = read_records(out, fmt)
    assert len(records) == 24
    for r in records:
        assert set(r) == {"image_path", "pred"}          # exactly the contract
        assert isinstance(r["pred"], float)
        assert 0.0 <= r["pred"] <= 1.0                   # a probability, not a logit
        assert not r["image_path"].startswith("/")       # relative to --input
    assert len({r["image_path"] for r in records}) == 24  # one unique row per image


def test_records_from_manifest(fixture_root, stub_checkpoint, tmp_path):
    out = tmp_path / "preds.json"
    res = run_predict("--checkpoint", str(stub_checkpoint),
                      "--manifest", str(fixture_root["manifest"]),
                      "--data-root", str(fixture_root["raw"]),
                      "--output", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    records = read_records(out)
    assert len(records) == 24
    manifest_paths = set(fixture_root["df"]["path"])
    assert {r["image_path"] for r in records} == manifest_paths


def test_calibration_changes_scores(fixture_root, stub_config, tmp_path):
    """pred must be sigmoid((z+alpha)/T), so T is observable in the output."""
    import torch

    from track5.models import build_model
    from track5.utils.config import config_hash

    cfg = stub_config["cfg"]
    scores = {}
    for T in (1.0, 6.0):
        torch.manual_seed(0)
        model = build_model(cfg)
        ck = tmp_path / f"ck_{T}.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "calibration": {"temperature": T, "alpha": 0.0, "threshold": 0.5},
                    "meta": {"config_hash": config_hash(cfg), "step": 0}}, ck)
        out = tmp_path / f"p_{T}.json"
        res = run_predict("--checkpoint", str(ck), "--input",
                          str(fixture_root["raw"] / "fixtures"), "--output", str(out))
        assert res.returncode == 0, res.stdout + res.stderr
        scores[T] = [r["pred"] for r in read_records(out)]
    assert scores[1.0] != scores[6.0]


def test_decode_failure_is_recorded_not_silent(fixture_root, stub_checkpoint, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    good = fixture_root["files"][0]
    (d / good.name).write_bytes(good.read_bytes())
    (d / "broken.png").write_bytes(b"not an image at all")

    out = tmp_path / "preds.json"
    res = run_predict("--checkpoint", str(stub_checkpoint), "--input", str(d),
                      "--output", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    records = read_records(out)
    assert len(records) == 2  # one row per expected image, none dropped

    errs = json.loads((tmp_path / "preds.json.errors.json").read_text(encoding="utf-8"))
    assert errs["n_errors"] == 1
    assert errs["errors"][0]["image_path"] == "broken.png"
    assert errs["errors"][0]["error"]
    assert "DECODE FAILURE" in res.stderr  # loud, never silent

    # and --strict turns it into a job failure
    res2 = run_predict("--checkpoint", str(stub_checkpoint), "--input", str(d),
                       "--output", str(tmp_path / "p2.json"), "--strict")
    assert res2.returncode != 0


def test_requires_exactly_one_input_source(stub_checkpoint, tmp_path):
    res = run_predict("--checkpoint", str(stub_checkpoint),
                      "--output", str(tmp_path / "p.json"))
    assert res.returncode != 0


# ------------------------------------------------------------ protected runs

def test_protected_run_implies_strict(fixture_root, stub_checkpoint, tmp_path):
    """Item 8: the pred=0.5 recovery mode must not survive into a final run."""
    d = tmp_path / "imgs"
    d.mkdir()
    good = fixture_root["files"][0]
    (d / good.name).write_bytes(good.read_bytes())
    (d / "broken.png").write_bytes(b"not an image")

    res = run_predict("--checkpoint", str(stub_checkpoint), "--input", str(d),
                      "--output", str(tmp_path / "p.json"), "--protected-run")
    assert res.returncode == 3
    assert "protected run" in res.stderr
    assert "implies --strict" in res.stdout
    assert not (tmp_path / "p.json.done.json").exists()   # no completion marker


def test_protected_run_writes_a_receipt_and_refuses_to_repeat(
        fixture_root, stub_checkpoint, tmp_path):
    out = tmp_path / "protected.json"
    res = run_predict("--checkpoint", str(stub_checkpoint),
                      "--input", str(fixture_root["raw"] / "fixtures"),
                      "--output", str(out), "--protected-run")
    assert res.returncode == 0, res.stdout + res.stderr

    done = json.loads(
        (tmp_path / "protected.json.done.json").read_text(encoding="utf-8"))
    assert done["n_records"] == 24 and done["n_errors"] == 0
    assert done["strict"] is True and done["format"] == "json"
    assert done["checkpoint_meta"]["model_hash"]
    assert done["calibration"]["temperature"] == 1.5

    errs = json.loads(
        (tmp_path / "protected.json.errors.json").read_text(encoding="utf-8"))
    assert errs["protected_run"] is True and errs["error_pred"] is None

    repeat = run_predict("--checkpoint", str(stub_checkpoint),
                         "--input", str(fixture_root["raw"] / "fixtures"),
                         "--output", str(out), "--protected-run")
    assert repeat.returncode == 5
    assert "already completed" in repeat.stderr


def test_exploratory_run_keeps_the_recovery_mode(fixture_root, stub_checkpoint,
                                                 tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    (d / fixture_root["files"][0].name).write_bytes(
        fixture_root["files"][0].read_bytes())
    (d / "broken.png").write_bytes(b"nope")
    out = tmp_path / "explore.json"
    res = run_predict("--checkpoint", str(stub_checkpoint), "--input", str(d),
                      "--output", str(out))
    assert res.returncode == 0
    recs = {r["image_path"]: r["pred"] for r in read_records(out)}
    assert recs["broken.png"] == 0.5
    errs = json.loads((tmp_path / "explore.json.errors.json").read_text(
        encoding="utf-8"))
    assert errs["error_pred"] == 0.5 and errs["strict"] is False
