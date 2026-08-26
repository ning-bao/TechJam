"""scripts/benchmark_gpu.py — the TC2 guide §8 qualification benchmark.

Runs on CPU with the stub backbone so CI exercises the harness itself: the
config ladder, the timing/reporting schema, the compute-vs-dataloader
distinction, and OOM containment. The bf16/fp16 rows need a GPU and are marked.
"""

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
COMPUTE_ONLY = "compute_only_random_tensor"
END_TO_END = "end_to_end_dataloader"


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "benchmark_gpu", REPO / "scripts" / "benchmark_gpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return load_bench()


def run_bench(tmp_path, *extra):
    out = tmp_path / "bench.json"
    res = subprocess.run(
        [sys.executable, "-u", "scripts/benchmark_gpu.py", "--stub",
         "--device", "cpu", "--out", str(out),
         "--synthetic-dir", str(tmp_path / "synth"), "--synthetic-images", "8",
         *extra],
        capture_output=True, text=True, cwd=str(REPO))
    return res, out


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    res, out = run_bench(
        tmp, "--precisions", "fp32", "--resolutions", "64,32",
        "--micro-batches", "2,1", "--modes", "train,inference",
        "--data", "compute,dataloader", "--checkpointing", "off",
        "--workers", "0", "--steps", "3", "--warmup-steps", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    return {"json": json.loads(out.read_text(encoding="utf-8")), "res": res}


# ----------------------------------------------------------------- the sweep

def test_config_ladder_is_ordered_largest_first(bench):
    args = Namespace(modes=["train", "inference"], data=[COMPUTE_ONLY],
                     resolutions=[384, 448, 512], precisions=["bf16", "fp16"],
                     micro_batches=[1, 2], checkpointing=[True, False],
                     workers=[4], grad_accum=1)
    cfgs = bench.build_configs(args)
    assert cfgs[0]["resolution"] == 512
    assert cfgs[-1]["resolution"] == 384
    # non-increasing memory demand, so an OOM is followed by smaller settings
    keys = [(c["resolution"], c["micro_batch"], c["mode"] == "train",
             not c["activation_checkpointing"]) for c in cfgs]
    assert keys == sorted(keys, reverse=True)
    # activation checkpointing is a train-only knob
    assert all(not c["activation_checkpointing"]
               for c in cfgs if c["mode"] == "inference")


def test_all_required_dimensions_are_covered(bench):
    args = Namespace(modes=["train", "inference"], data=[COMPUTE_ONLY, END_TO_END],
                     resolutions=[512, 448, 384], precisions=["bf16", "fp16"],
                     micro_batches=[1, 2], checkpointing=[True, False],
                     workers=[4], grad_accum=1)
    cfgs = bench.build_configs(args)
    assert {c["resolution"] for c in cfgs} == {512, 448, 384}
    assert {c["precision"] for c in cfgs} == {"bf16", "fp16"}
    assert {c["micro_batch"] for c in cfgs} == {1, 2}
    assert {c["mode"] for c in cfgs} == {"train", "inference"}
    assert {c["data_path"] for c in cfgs} == {COMPUTE_ONLY, END_TO_END}
    assert {c["activation_checkpointing"] for c in cfgs} == {True, False}


# ----------------------------------------------------------------- reporting

def test_report_records_environment_and_model(report):
    r = report["json"]
    for key in ("torch", "torch_cuda_runtime", "cudnn", "driver", "hostname",
                "cuda_available", "platform"):
        assert key in r["environment"], key
    assert r["model"]["requested"] and r["model"]["resolved"]
    assert r["model"]["params_total"] > 0
    assert r["model"]["params_under_2b"] is True
    assert r["config"]["model"]["backbone"]        # the build config is recorded
    assert r["config_hash"]
    assert r["settings"]["steps"] == 3 and r["settings"]["warmup_steps"] == 1


def test_every_ok_result_reports_the_required_metrics(report):
    ok = [r for r in report["json"]["results"] if r["status"] == "ok"]
    assert ok, report["res"].stdout
    for r in ok:
        for key in ("mode", "data_path", "resolution", "precision", "micro_batch",
                    "grad_accum", "effective_batch", "activation_checkpointing",
                    "workers", "steps", "warmup_steps"):
            assert key in r, key
        for key in ("median", "p95", "mean"):
            assert r["step_time_s"][key] > 0, key
            assert key in r["data_time_s"]
        assert r["step_time_s"]["p95"] >= r["step_time_s"]["median"]
        assert r["images_per_s"]["from_median_step"] > 0
        assert r["images_per_s"]["from_total_window"] > 0
        assert r["effective_batch"] == r["micro_batch"] * r["grad_accum"]


def test_compute_only_and_end_to_end_are_distinguished(report):
    ok = [r for r in report["json"]["results"] if r["status"] == "ok"]
    paths = {r["data_path"] for r in ok}
    assert paths == {COMPUTE_ONLY, END_TO_END}
    compute = [r for r in ok if r["data_path"] == COMPUTE_ONLY]
    loaded = [r for r in ok if r["data_path"] == END_TO_END]
    # the random-tensor path re-uses a device-resident batch: ~zero data time
    assert max(r["data_time_s"]["median"] for r in compute) < 1e-3
    # the end-to-end path decodes + augments + collates: measurably non-zero
    assert min(r["data_time_s"]["median"] for r in loaded) > 1e-4


def test_short_run_is_flagged_as_a_smoke_test(report):
    notes = " ".join(report["json"]["notes"])
    assert "below the TC2 section 8 minimum of 100 measured steps" in notes
    assert "synthetic" in notes  # the end-to-end image source is disclosed


def test_results_are_flushed_after_each_configuration(tmp_path):
    """A killed job must still leave the completed rows on disk."""
    res, out = run_bench(tmp_path, "--precisions", "fp32", "--resolutions", "32",
                         "--micro-batches", "1", "--modes", "inference",
                         "--data", "compute", "--checkpointing", "off",
                         "--steps", "2", "--warmup-steps", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    assert not list(tmp_path.glob("*.tmp"))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["results"]) == data["settings"]["n_configs"] == 1


# ---------------------------------------------------------------------- OOM

def test_oom_is_caught_per_configuration(bench, tmp_path):
    class Exploding(torch.nn.Module):
        backbone_name = "exploding"

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 1)

        def forward(self, x):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate")

    cfg = {"mode": "train", "data_path": COMPUTE_ONLY, "resolution": 32,
           "precision": "fp32", "micro_batch": 1, "grad_accum": 1,
           "effective_batch": 1, "activation_checkpointing": False, "workers": 0}
    args = Namespace(steps=2, warmup_steps=1, seed=17, data_root=str(tmp_path))
    result = bench.run_config(Exploding(), cfg, args, torch.device("cpu"), [])
    assert result["status"] == "oom"
    assert "out of memory" in result["reason"].lower()
    assert result["resolution"] == 32  # the failing configuration is identified


def test_error_row_carries_a_traceback_and_the_grid_continues(tmp_path):
    """Item 7: a broken configuration must not kill the sweep, must record what
    happened, and must make the job exit non-zero at the end."""
    res, out = run_bench(tmp_path, "--precisions", "fp32", "--resolutions", "32",
                         "--micro-batches", "1", "--modes", "inference",
                         "--data", "compute,dataloader", "--checkpointing", "off",
                         "--workers", "0", "--steps", "2", "--warmup-steps", "1",
                         "--manifest", str(tmp_path / "does_not_exist.parquet"))
    assert res.returncode == 6, res.stdout + res.stderr   # unexpected error

    data = json.loads(out.read_text(encoding="utf-8"))
    by_status = {r["status"]: r for r in data["results"]}
    assert set(by_status) == {"ok", "error"}              # the sweep continued
    err = by_status["error"]
    assert err["data_path"] == END_TO_END
    assert err["reason"]
    assert "Traceback" in err["traceback"]
    assert by_status["ok"]["data_path"] == COMPUTE_ONLY
    assert "ERROR" in res.stderr


def test_oom_alone_does_not_fail_the_job(bench, tmp_path):
    """OOM is an expected sweep outcome; only unexpected errors exit non-zero."""
    assert bench.DEFINITIVE == ("ok", "oom", "skipped")


def test_generic_runtime_error_is_recorded_not_raised(bench, tmp_path):
    class Broken(torch.nn.Module):
        backbone_name = "broken"

        def forward(self, x):
            raise RuntimeError("shape mismatch somewhere")

    cfg = {"mode": "inference", "data_path": COMPUTE_ONLY, "resolution": 32,
           "precision": "fp32", "micro_batch": 1, "grad_accum": 1,
           "effective_batch": 1, "activation_checkpointing": False, "workers": 0}
    args = Namespace(steps=2, warmup_steps=1, seed=17, data_root=str(tmp_path))
    result = bench.run_config(Broken(), cfg, args, torch.device("cpu"), [])
    assert result["status"] == "error"
    assert "shape mismatch" in result["reason"]
    assert "Traceback" in result["traceback"]


def test_non_runtime_exception_is_also_contained(bench, tmp_path):
    class Weird(torch.nn.Module):
        backbone_name = "weird"

        def forward(self, x):
            raise ValueError("not a RuntimeError at all")

    cfg = {"mode": "inference", "data_path": COMPUTE_ONLY, "resolution": 32,
           "precision": "fp32", "micro_batch": 1, "grad_accum": 1,
           "effective_batch": 1, "activation_checkpointing": False, "workers": 0}
    args = Namespace(steps=2, warmup_steps=1, seed=17, data_root=str(tmp_path),
                     min_measure_seconds=0.0)
    result = bench.run_config(Weird(), cfg, args, torch.device("cpu"), [])
    assert result["status"] == "error"
    assert "ValueError" in result["reason"]
    assert "Traceback" in result["traceback"]


# -------------------------------------------------------------------- resume

def test_resume_preserves_completed_rows(tmp_path):
    """Item 8: a partially completed grid keeps its finished rows."""
    common = ("--precisions", "fp32", "--resolutions", "32", "--modes",
              "inference", "--data", "compute", "--checkpointing", "off",
              "--steps", "2", "--warmup-steps", "1")
    first, out = run_bench(tmp_path, *common, "--micro-batches", "1")
    assert first.returncode == 0, first.stdout + first.stderr
    done = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert len(done) == 1
    original_median = done[0]["step_time_s"]["median"]

    second, _ = run_bench(tmp_path, *common, "--micro-batches", "2,1", "--resume")
    assert second.returncode == 0, second.stdout + second.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["results"]) == 2
    assert data["settings"]["resumed_rows"] == 1
    preserved = [r for r in data["results"] if r.get("resumed")]
    assert len(preserved) == 1
    assert preserved[0]["micro_batch"] == 1
    # byte-identical timings prove the row was reused, not re-measured
    assert preserved[0]["step_time_s"]["median"] == original_median
    assert "preserved from a previous run" in second.stdout


def test_resume_reruns_error_rows(tmp_path):
    common = ("--precisions", "fp32", "--resolutions", "32", "--micro-batches", "1",
              "--modes", "inference", "--data", "compute,dataloader",
              "--checkpointing", "off", "--workers", "0",
              "--steps", "2", "--warmup-steps", "1")
    first, out = run_bench(tmp_path, *common,
                           "--manifest", str(tmp_path / "missing.parquet"))
    assert first.returncode == 6

    second, _ = run_bench(tmp_path, *common, "--resume",
                          "--manifest", str(tmp_path / "missing.parquet"))
    assert second.returncode == 6                     # still broken, still reported
    data = json.loads(out.read_text(encoding="utf-8"))
    resumed = {r["data_path"]: r.get("resumed", False) for r in data["results"]}
    assert resumed[COMPUTE_ONLY] is True              # the good row was preserved
    assert resumed[END_TO_END] is False               # the error row was retried


def test_resume_discards_rows_from_a_different_grid(tmp_path):
    common = ("--precisions", "fp32", "--resolutions", "32", "--micro-batches", "1",
              "--modes", "inference", "--data", "compute", "--checkpointing", "off",
              "--warmup-steps", "1")
    first, out = run_bench(tmp_path, *common, "--steps", "2")
    assert first.returncode == 0
    second, _ = run_bench(tmp_path, *common, "--steps", "3", "--resume")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "starting fresh" in second.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["settings"]["resumed_rows"] == 0
    assert not any(r.get("resumed") for r in data["results"])


def test_mixed_precision_is_skipped_not_silently_fp32_on_cpu(tmp_path):
    res, out = run_bench(tmp_path, "--precisions", "bf16,fp16", "--resolutions", "32",
                         "--micro-batches", "1", "--modes", "inference",
                         "--data", "compute", "--checkpointing", "off",
                         "--steps", "2", "--warmup-steps", "1")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {r["status"] for r in data["results"]} == {"skipped"}
    assert all("CUDA" in r["reason"] for r in data["results"])
    assert res.returncode != 0  # nothing measured -> the job failed


# ------------------------------------------- L40S profile: auto-checkpointing

def auto_args(**over):
    base = dict(modes=["train", "inference"], data=[COMPUTE_ONLY],
                resolutions=[512], precisions=["bf16"], micro_batches=[1, 2, 4, 8],
                checkpointing=[False], workers=[8], grad_accum=1,
                checkpoint_auto=True)
    base.update(over)
    return Namespace(**base)


def test_auto_mode_plans_checkpointing_off_only(bench):
    cfgs = bench.build_configs(auto_args())
    assert len(cfgs) == 8                       # 2 modes x 4 micro-batches
    assert all(c["activation_checkpointing"] is False for c in cfgs)
    train = [c for c in cfgs if c["mode"] == "train"]
    infer = [c for c in cfgs if c["mode"] == "inference"]
    assert all(c["escalate_ckpt"] for c in train)
    assert not any(c["escalate_ckpt"] for c in infer)   # train-only knob
    # largest micro-batch first, so an OOM is followed by smaller batches
    assert [c["micro_batch"] for c in cfgs if c["mode"] == "train"] == [8, 4, 2, 1]


def test_explicit_mode_still_plans_both(bench):
    cfgs = bench.build_configs(auto_args(checkpointing=[True, False],
                                         checkpoint_auto=False))
    train = [c for c in cfgs if c["mode"] == "train"]
    assert {c["activation_checkpointing"] for c in train} == {True, False}
    assert not any(c["escalate_ckpt"] for c in cfgs)


@pytest.mark.parametrize("result,headroom,expected", [
    ({"status": "oom"}, 90.0, "OOM without activation checkpointing"),
    ({"status": "ok", "peak_allocated_pct": 94.2}, 90.0,
     "peak allocated 94.2% of total VRAM > 90.0% headroom"),
    ({"status": "ok", "peak_allocated_pct": 61.0}, 90.0, ""),
    ({"status": "ok"}, 90.0, ""),                       # CPU run: no VRAM figure
    ({"status": "skipped"}, 90.0, ""),
])
def test_escalation_is_need_based(bench, result, headroom, expected):
    assert bench.escalate_reason(result, headroom) == expected


def test_headroom_is_measured_against_available_not_total_vram(bench):
    """The observed 5070 Ti row: 6.119 GiB looks like 38% of 15.92 GiB total, but
    ~5.2 GiB is already held by the desktop, so the real figure is ~57%."""
    row = {"status": "ok", "peak_allocated_gib": 6.119, "vram_total_gib": 15.92,
           "peak_allocated_pct": 38.44, "peak_allocated_pct_of_available": 57.2,
           "device_used_by_others_gib": 5.2, "vram_usable_gib": 10.72}
    assert bench.headroom_pct_of(row) == 57.2          # not 38.44
    assert bench.escalate_reason(row, 90.0) == ""
    assert bench.escalate_reason(row, 50.0) == (
        "peak allocated 57.2% of available VRAM > 50.0% headroom")


def test_headroom_falls_back_to_total_when_availability_is_unknown(bench):
    assert bench.headroom_pct_of({"status": "ok", "peak_allocated_pct": 12.0}) == 12.0
    assert bench.headroom_pct_of({"status": "ok"}) is None


def test_auto_run_records_that_checkpointing_was_not_needed(tmp_path):
    """On a config that fits, the 'on' row must be skipped *and* say so."""
    res, out = run_bench(tmp_path, "--precisions", "fp32", "--resolutions", "32",
                         "--micro-batches", "1", "--modes", "train",
                         "--data", "compute", "--checkpointing", "auto",
                         "--steps", "2", "--warmup-steps", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["settings"]["checkpointing_mode"] == "auto"
    assert data["settings"]["vram_headroom_pct"] == 90.0
    assert len(data["results"]) == 1                    # no escalation row added
    # CPU has no VRAM figure, so the decision is recorded as not evaluated
    # rather than silently implying the config fit (escalate_reason covers the
    # GPU wording).
    assert data["results"][0]["escalated_to_checkpointing"] == (
        "not evaluated: no VRAM figure on this device")


def test_baseline_never_uses_torch_compile(report):
    assert report["json"]["settings"]["torch_compile"] is False
    ok = [r for r in report["json"]["results"] if r["status"] == "ok"]
    assert ok and all(r["torch_compile"] is False for r in ok)


# ------------------------------------------------------------ GPU utilization

def test_sampler_slices_only_the_measured_window(bench):
    s = bench.GpuSampler(interval_ms=250, uuid="GPU-abc")
    s.samples = [(10.0, 5, 1, 100), (20.0, 90, 40, 9000), (21.0, 80, 38, 9100),
                 (30.0, 3, 1, 120)]
    w = s.window(19.5, 25.0)
    assert w["available"] is True
    assert w["n_samples"] == 2
    assert w["gpu_util_pct"]["median"] == 85
    assert w["gpu_util_pct"]["max"] == 90
    assert w["memory_used_mib"]["max"] == 9100
    assert w["interval_ms"] == 250


def test_sampler_reports_absence_instead_of_failing(bench):
    s = bench.GpuSampler()
    assert s.window(0.0, 1.0)["available"] is False      # no samples collected
    s.error = "FileNotFoundError: nvidia-smi"
    assert s.window(0.0, 1.0) == {"available": False,
                                  "reason": "FileNotFoundError: nvidia-smi"}


def test_sampler_widens_when_the_window_is_shorter_than_the_interval(bench):
    """A 5-step 64px run finishes inside one sampling interval; report the
    nearest samples and flag it rather than dropping utilization entirely."""
    s = bench.GpuSampler(interval_ms=250)
    s.samples = [(10.0, 40, 10, 5000), (10.5, 90, 30, 9000)]
    w = s.window(10.20, 10.21)                  # 10 ms window, no sample inside
    assert w["available"] is True
    assert w["widened_to_nearest_samples"] is True
    assert w["n_samples"] == 2                  # one on each side
    assert w["window_s"] == 0.01

    inside = s.window(9.9, 10.6)
    assert inside["widened_to_nearest_samples"] is False


def test_sampler_takes_a_sample_before_the_loop(bench, monkeypatch):
    """Item 4: an initial datum exists before the caller's warm-up begins."""
    order = []
    s = bench.GpuSampler(interval_ms=250)
    monkeypatch.setattr(s, "sample_once", lambda: order.append("sample") or True)
    try:
        s.start()
    finally:
        s.stop()
    assert order and order[0] == "sample"


def test_min_measure_seconds_extends_a_short_run(tmp_path):
    """Item 5: duration-based, not step-count-based - a tiny config keeps
    stepping until the measured window is long enough to be sampled."""
    res, out = run_bench(tmp_path, "--precisions", "fp32", "--resolutions", "32",
                         "--micro-batches", "1", "--modes", "inference",
                         "--data", "compute", "--checkpointing", "off",
                         "--steps", "2", "--warmup-steps", "1",
                         "--min-measure-seconds", "0.5")
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    row = data["results"][0]
    assert data["settings"]["min_measure_seconds"] == 0.5
    assert row["total_window_s"] >= 0.5
    assert row["steps_measured"] > row["steps"] == 2
    assert row["images_per_s"]["from_total_window"] > 0


def test_gpu_sample_interval_is_configurable(tmp_path):
    res, out = run_bench(tmp_path, "--precisions", "fp32", "--resolutions", "32",
                         "--micro-batches", "1", "--modes", "inference",
                         "--data", "compute", "--checkpointing", "off",
                         "--steps", "2", "--warmup-steps", "1",
                         "--gpu-sample-ms", "40")
    assert res.returncode == 0, res.stdout + res.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["settings"]["gpu_sample_ms"] == 40


def test_sampler_parses_and_filters_by_uuid(bench):
    s = bench.GpuSampler(uuid="GPU-mine")

    class FakeProc:
        stdout = iter(["GPU-mine, 97, 45, 12345\n",
                       "GPU-other, 10, 2, 100\n",
                       "malformed line\n",
                       "GPU-mine, notanint, 1, 2\n",
                       "GPU-mine, 88, 40, 12000\n"])

    s.proc = FakeProc()
    s._read()
    assert [(v[1], v[2], v[3]) for v in s.samples] == [(97, 45, 12345), (88, 40, 12000)]


def test_cpu_run_reports_utilization_as_unavailable(report):
    ok = [r for r in report["json"]["results"] if r["status"] == "ok"]
    assert all(r["gpu_utilization"]["available"] is False for r in ok)
    assert all(r["gpu_utilization"]["reason"] for r in ok)


@pytest.mark.gpu
def test_gpu_utilization_is_measured_on_a_real_gpu(tmp_path, bench):
    """Duration-based (item 5): the window is held open long enough to be
    sampled instead of relying on a step count at a tiny resolution."""
    assert bench.device_uuid().startswith("GPU-")
    res, out = run_bench(tmp_path, "--precisions", "bf16", "--resolutions", "64",
                         "--micro-batches", "1", "--modes", "train",
                         "--data", "compute", "--checkpointing", "auto",
                         "--device", "cuda", "--steps", "20", "--warmup-steps", "5",
                         "--min-measure-seconds", "3", "--gpu-sample-ms", "100")
    assert res.returncode == 0, res.stdout + res.stderr
    ok = [r for r in json.loads(out.read_text(encoding="utf-8"))["results"]
          if r["status"] == "ok"]
    assert ok
    row = ok[0]
    assert row["total_window_s"] >= 3.0
    util = row["gpu_utilization"]
    assert util["available"] is True
    assert util["n_samples"] >= 5          # >= 3 s at 100 ms sampling
    assert util["widened_to_nearest_samples"] is False
    assert 0 <= util["gpu_util_pct"]["median"] <= 100
    assert util["memory_used_mib"]["max"] > 0


@pytest.mark.gpu
def test_real_device_memory_is_recorded(tmp_path):
    """Item 6: peak_allocated alone hides memory already held by other
    processes; mem_get_info makes the real headroom visible."""
    res, out = run_bench(tmp_path, "--precisions", "bf16", "--resolutions", "64",
                         "--micro-batches", "1", "--modes", "train",
                         "--data", "compute", "--checkpointing", "off",
                         "--device", "cuda", "--steps", "10", "--warmup-steps", "3")
    assert res.returncode == 0, res.stdout + res.stderr
    row = [r for r in json.loads(out.read_text(encoding="utf-8"))["results"]
           if r["status"] == "ok"][0]
    for key in ("vram_total_gib", "vram_usable_gib", "device_used_by_others_gib",
                "device_free_before_gib", "device_free_after_gib",
                "peak_allocated_pct", "peak_allocated_pct_of_available"):
        assert key in row, key
    assert row["device_used_by_others_gib"] >= 0
    assert row["vram_usable_gib"] <= row["vram_total_gib"]
    # available-based headroom is never the more optimistic of the two
    assert row["peak_allocated_pct_of_available"] >= row["peak_allocated_pct"]


@pytest.mark.gpu
def test_dataloader_mode_works_with_real_workers(tmp_path):
    """The configuration that crashed: end-to-end dataloader with workers > 0."""
    res, out = run_bench(tmp_path, "--precisions", "bf16", "--resolutions", "64",
                         "--micro-batches", "2", "--modes", "train",
                         "--data", "dataloader", "--checkpointing", "off",
                         "--workers", "2", "--device", "cuda",
                         "--steps", "10", "--warmup-steps", "3")
    assert res.returncode == 0, res.stdout + res.stderr
    rows = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert [r["status"] for r in rows] == ["ok"]
    assert rows[0]["workers"] == 2
    assert rows[0]["data_time_s"]["median"] > 0


@pytest.mark.gpu
def test_bf16_and_fp16_measure_vram_on_a_real_gpu(tmp_path):
    res, out = run_bench(tmp_path, "--precisions", "bf16,fp16",
                         "--resolutions", "64", "--micro-batches", "1",
                         "--modes", "train", "--data", "compute",
                         "--checkpointing", "off", "--device", "cuda",
                         "--steps", "5", "--warmup-steps", "2")
    assert res.returncode == 0, res.stdout + res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    ok = [r for r in data["results"] if r["status"] == "ok"]
    assert {r["precision"] for r in ok} <= {"bf16", "fp16"}
    for r in ok:
        assert r["peak_allocated_gib"] > 0
        assert r["peak_reserved_gib"] >= r["peak_allocated_gib"]
        assert 0 < r["peak_allocated_pct"] <= 100
    assert data["environment"]["gpu"]
