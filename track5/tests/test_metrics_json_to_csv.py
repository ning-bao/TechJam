"""The metrics-JSON to long-CSV adapter (scripts/metrics_json_to_csv.py).

`evaluate.py` writes one JSON per condition; the R analysis in
`analysis/calibration_dca.R` reads one long CSV. These tests pin the failure
policy, because the dangerous outcome here is not a crash — it is a CSV that
looks complete and is not. The consumer's own thin-data guard fires on
`0 < n < 100`, so a zero-row table slips past it silently and renders empty
figures; and a stale CSV from an earlier successful run is indistinguishable
from a current one.

Fixtures are built inline rather than read from data/: the point is the policy,
not the corpus.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "metrics_json_to_csv.py"

EVAL_15 = ["clean", "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
           "blur_05", "blur_10", "blur_20", "resize_050", "resize_025",
           "noise_002", "noise_005", "noise_010", "jitter_pm20", "crop_80"]


def row(path, label, pred, family=""):
    return {"image_id": path, "path": path, "label": label,
            "generator_family": family, "pred": pred}


def write_condition(d: Path, cond, rows, threshold=0.7439, body_cond=None):
    """One condition result file, in the shape evaluate.py writes."""
    obj = {"condition": cond if body_cond is None else body_cond,
           "n": len(rows), "threshold": threshold,
           "calibration": {"temperature": 1.0, "alpha": 0.0,
                           "threshold": threshold},
           "overall": {"auroc": 0.9}, "predictions": rows}
    (d / f"{cond}.json").write_text(json.dumps(obj), encoding="utf-8")
    # evaluate.py drops a completion marker beside each result; it must be skipped
    (d / f"{cond}.done.json").write_text('{"done": true}', encoding="utf-8")


def run(metrics_dir: Path, out: Path, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics-dir", str(metrics_dir),
         "--out", str(out), *extra],
        capture_output=True, text=True)


def two_rows(cond):
    return [row(f"real/{cond}.jpg", 0, 0.1), row(f"fake/{cond}.jpg", 1, 0.9, "sd15")]


def full_set(d: Path, threshold=0.7439):
    for c in EVAL_15:
        write_condition(d, c, two_rows(c), threshold=threshold)


def test_happy_path_column_order_and_condition_order(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    full_set(m)
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode == 0, r.stderr
    lines = out.read_text(encoding="utf-8").splitlines()
    # the R script stopifnot()s on exactly these names
    assert lines[0] == "image_path,label,pred,condition,generator"
    assert len(lines) == 1 + 2 * len(EVAL_15)
    seen = []
    for ln in lines[1:]:
        cond = ln.split(",")[3]
        if cond not in seen:
            seen.append(cond)
    assert seen == EVAL_15


def test_generator_family_empty_becomes_real(tmp_path):
    # The R script and the reference tables use the literal `real` for label 0;
    # an empty generator_family must not reach the CSV as an empty field.
    m = tmp_path / "metrics"
    m.mkdir()
    full_set(m)
    out = tmp_path / "p.csv"
    assert run(m, out).returncode == 0
    body = out.read_text(encoding="utf-8").splitlines()[1:]
    for ln in body:
        image_path, label, _pred, _cond, generator = ln.split(",")
        assert (generator == "real") == (label == "0"), ln


@pytest.mark.parametrize("stem,body", [
    ("jpeg_q30", "jpeg_30"),      # TC2 spelling on the file, canonical in body
    ("blur_0.5", "blur_05"),
    ("resize_0.50", "resize_050"),  # not covered by eval_atoms.TC2_ALIASES
    ("jitter_20", "jitter_pm20"),   # ditto
])
def test_alias_spellings_resolve(tmp_path, stem, body):
    """--condition accepts either spelling, so output files may be named either
    way; both must land in the CSV under the canonical name."""
    m = tmp_path / "metrics"
    m.mkdir()
    obj = {"condition": body, "calibration": {"threshold": 0.7},
           "predictions": two_rows("x")}
    (m / f"{stem}.json").write_text(json.dumps(obj), encoding="utf-8")
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode == 0, r.stderr
    conds = {ln.split(",")[3] for ln in
             out.read_text(encoding="utf-8").splitlines()[1:]}
    assert conds == {body}


def test_genuine_condition_mismatch_is_fatal(tmp_path):
    # Alias tolerance must not soften a real disagreement.
    m = tmp_path / "metrics"
    m.mkdir()
    write_condition(m, "clean", two_rows("x"), body_cond="crop_80")
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode != 0
    assert "mismatch" in r.stderr
    assert not out.exists()


def test_zero_row_condition_is_missing_not_written(tmp_path):
    """A condition file with an empty predictions array must not become a
    header-only CSV: the consumer's guard is `n > 0 & n < 100`, so n == 0 passes
    it silently and the figures come out empty with no warning."""
    m = tmp_path / "metrics"
    m.mkdir()
    write_condition(m, "clean", [])
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode != 0
    assert "zero predictions" in r.stderr
    assert not out.exists()


def test_partial_set_warns_but_succeeds(tmp_path):
    """Conditions are evaluated incrementally, so a partial table is legitimate —
    it just must never look complete."""
    m = tmp_path / "metrics"
    m.mkdir()
    for c in ["clean", "jpeg_30"]:
        write_condition(m, c, two_rows(c))
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode == 0, r.stderr
    for c in EVAL_15:
        if c not in ("clean", "jpeg_30"):
            assert c in r.stderr
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1 + 4


def test_non_eval15_condition_excluded_then_admitted(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    full_set(m)
    write_condition(m, "crop_80_area", two_rows("extra"))
    out = tmp_path / "p.csv"

    r = run(m, out)
    assert r.returncode == 0, r.stderr
    assert "crop_80_area" in r.stderr
    assert "crop_80_area" not in out.read_text(encoding="utf-8")

    r = run(m, out, "--atoms", "crop_80_area")
    assert r.returncode == 0, r.stderr
    assert "crop_80_area" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad,needle", [
    ({"image_id": "a", "path": "x.jpg", "label": 0, "pred": 0.1}, "generator_family"),
    (row("x.jpg", 0, 1.5), "outside [0,1]"),
    (row("x.jpg", 0, float("nan")), "outside [0,1]"),
    (row("x.jpg", 2, 0.1), "label not in"),
    (row("x.jpg", True, 0.1), "label not in"),
])
def test_malformed_rows_hard_error(tmp_path, bad, needle):
    m = tmp_path / "metrics"
    m.mkdir()
    obj = {"condition": "clean", "calibration": {"threshold": 0.7},
           "predictions": [bad]}
    # NaN is not valid JSON but json.dumps emits it and json.load accepts it,
    # which is exactly how it would reach us from the producer.
    (m / "clean.json").write_text(json.dumps(obj), encoding="utf-8")
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode != 0
    assert needle in r.stderr
    assert not out.exists()


def test_missing_predictions_key_hard_errors(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    (m / "clean.json").write_text('{"condition": "clean", "n": 0}',
                                  encoding="utf-8")
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode != 0
    assert "missing predictions" in r.stderr


def test_duplicate_image_within_condition_hard_errors(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    write_condition(m, "clean", [row("x.jpg", 0, 0.1), row("x.jpg", 1, 0.9, "sd")])
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode != 0
    assert "duplicate" in r.stderr


def test_same_image_across_conditions_is_fine(tmp_path):
    # The long table is one row per (image, condition) by design.
    m = tmp_path / "metrics"
    m.mkdir()
    for c in ["clean", "jpeg_30"]:
        write_condition(m, c, [row("x.jpg", 0, 0.1), row("y.jpg", 1, 0.9, "sd")])
    out = tmp_path / "p.csv"
    assert run(m, out).returncode == 0


def test_disagreeing_thresholds_are_fatal_even_without_threshold_out(tmp_path):
    """One frozen operating point applies to every condition. A per-condition
    threshold means it was refit, which invalidates the whole robustness table —
    so this is fatal whether or not the threshold file was requested."""
    m = tmp_path / "metrics"
    m.mkdir()
    write_condition(m, "clean", two_rows("a"), threshold=0.7439)
    write_condition(m, "jpeg_30", two_rows("b"), threshold=0.5)
    out = tmp_path / "p.csv"
    r = run(m, out)
    assert r.returncode != 0
    assert "disagree on threshold" in r.stderr
    assert not out.exists()


def test_threshold_file_written_only_on_success(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    full_set(m, threshold=0.7439)
    out = tmp_path / "p.csv"
    tfile = tmp_path / "threshold.txt"
    r = run(m, out, "--threshold-out", str(tfile))
    assert r.returncode == 0, r.stderr
    assert abs(float(tfile.read_text()) - 0.7439) < 1e-9

    # a later bad condition must leave no threshold file behind
    bad = tmp_path / "metrics2"
    bad.mkdir()
    write_condition(bad, "clean", two_rows("a"), threshold=0.7439)
    write_condition(bad, "jpeg_30", two_rows("b"), threshold=0.5)
    t2 = tmp_path / "t2.txt"
    assert run(bad, tmp_path / "p2.csv", "--threshold-out", str(t2)).returncode != 0
    assert not t2.exists()


def test_failed_rerun_removes_stale_output(tmp_path):
    """The consumer cannot tell a stale CSV from a current one. A failing re-run
    must delete the previous table rather than leave it looking authoritative."""
    m = tmp_path / "metrics"
    m.mkdir()
    full_set(m)
    out = tmp_path / "p.csv"
    assert run(m, out).returncode == 0
    assert out.exists()

    write_condition(m, "jpeg_30", [row("bad.jpg", 0, 9.9)])
    r = run(m, out)
    assert r.returncode != 0
    assert not out.exists(), "stale CSV from the previous run was left in place"
    assert "stale" in r.stderr


def test_no_partial_file_left_behind(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    full_set(m)
    out = tmp_path / "p.csv"
    assert run(m, out).returncode == 0
    assert not (tmp_path / "p.csv.partial").exists()


def test_empty_metrics_dir_hard_errors(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    r = run(m, tmp_path / "p.csv")
    assert r.returncode != 0
    assert "no JSON files" in r.stderr


def test_done_markers_alone_are_not_results(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    (m / "clean.done.json").write_text('{"done": true}', encoding="utf-8")
    r = run(m, tmp_path / "p.csv")
    assert r.returncode != 0
    assert "no JSON files" in r.stderr
