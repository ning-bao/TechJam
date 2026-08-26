"""src/cluster_probe.py and the two Slurm launchers (TC2 guide §7/§8).

The sbatch files must stay thin launchers over real entry points, carry the
directives the guide requires, and leak no credentials.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLUSTER = REPO / "cluster"
SBATCH = ["probe_gpu.sbatch", "benchmark_dinov3.sbatch", "train_dinov3l512.sbatch",
          "eval_array.sbatch"]

# Measured TC2 QoS caps (docs/cluster_profile.md): exceeding any of these makes
# the job unschedulable or breaches the aggregate per-user limit.
QOS_MAX_CPUS = 10
QOS_MAX_MEM_GB = 30
QOS_MAX_WALL_MIN = 6 * 60


def run_probe(*extra):
    return subprocess.run([sys.executable, "-u", "-m", "src.cluster_probe", *extra],
                          capture_output=True, text=True, cwd=str(REPO))


# ------------------------------------------------------------------- the probe

def test_probe_passes_on_cpu_with_the_escape_hatch(stub_config, tmp_path):
    out = tmp_path / "probe.json"
    res = run_probe("--config", str(stub_config["path"]), "--device", "cpu",
                    "--allow-cpu", "--output", str(out))
    assert res.returncode == 0, res.stdout + res.stderr

    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["status"] == "pass"
    assert rep["forward_backward_ok"] is True
    assert rep["params_under_2b"] is True
    assert rep["resolution"] == stub_config["cfg"]["data"]["crop"]
    for key in ("torch", "torch_cuda_runtime", "cudnn", "driver", "cuda_available",
                "hostname", "platform"):
        assert key in rep["environment"], key


def test_probe_fails_without_a_gpu_unless_allowed(stub_config):
    res = run_probe("--config", str(stub_config["path"]), "--device", "cpu")
    assert res.returncode == 2
    assert "cannot see an allocated GPU" in res.stderr


def test_probe_never_dumps_the_environment(stub_config, tmp_path):
    """TC2 §14: printing the environment wholesale can expose an HF token."""
    from src.cluster_probe import ENV_KEYS

    out = tmp_path / "p.json"
    run_probe("--config", str(stub_config["path"]), "--device", "cpu",
              "--allow-cpu", "--output", str(out))
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert set(rep["environment"]["env"]) <= set(ENV_KEYS)
    assert not any("TOKEN" in k.upper() for k in ENV_KEYS)


def test_probe_resolution_override(stub_config, tmp_path):
    out = tmp_path / "p.json"
    res = run_probe("--config", str(stub_config["path"]), "--device", "cpu",
                    "--allow-cpu", "--resolution", "32", "--output", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["resolution"] == 32


# ----------------------------------------------------------------- the sbatch

@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_exists_and_has_the_required_directives(name):
    text = (CLUSTER / name).read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash")
    for directive in ("--partition=", "--qos=", "--nodes=1", "--ntasks=1",
                      "--cpus-per-task=", "--gres=gpu:1", "--mem=", "--time=",
                      "--job-name=", "--output=logs/", "--error=logs/"):
        assert directive in text, f"{name} missing {directive}"
    assert "set -euo pipefail" in text
    assert "umask 077" in text
    assert "module purge" in text
    assert "conda activate" in text


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_requests_a_single_gpu(name):
    """TC2 §12: never request more than one GPU without an explicit grant."""
    text = (CLUSTER / name).read_text(encoding="utf-8")
    assert re.findall(r"--gres=gpu:(\d+)", text) == ["1"]


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_sets_hf_cache_before_python(name):
    text = (CLUSTER / name).read_text(encoding="utf-8")
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        assert f"export {var}=" in text, f"{name} missing {var}"
    assert text.index("export HF_HOME=") < text.index("python -u")


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_leaks_no_credentials(name):
    text = (CLUSTER / name).read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text), f"{name} embeds an HF token"
    for bad in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "hf auth login", "printenv"):
        assert bad not in text, f"{name} references {bad}"


def test_sbatch_entry_points_exist():
    probe = (CLUSTER / "probe_gpu.sbatch").read_text(encoding="utf-8")
    bench = (CLUSTER / "benchmark_dinov3.sbatch").read_text(encoding="utf-8")
    train = (CLUSTER / "train_dinov3l512.sbatch").read_text(encoding="utf-8")
    assert "-m src.cluster_probe" in probe
    assert (REPO / "src" / "cluster_probe.py").exists()
    assert "scripts/benchmark_gpu.py" in bench
    assert (REPO / "scripts" / "benchmark_gpu.py").exists()
    assert "-m src.train" in train
    assert (REPO / "src" / "train.py").exists()
    for text in (probe, bench, train):
        assert "configs/dinov3l512.yaml" in text or "dinov3-vitl16" in text


# --------------------------------------------------- L40S benchmark profile

def test_benchmark_sbatch_matches_the_l40s_spec():
    text = (CLUSTER / "benchmark_dinov3.sbatch").read_text(encoding="utf-8")
    assert "--resolutions 512" in text and "512,448" not in text  # 512 px only
    assert "--precisions bf16" in text and "fp16" not in text     # BF16 only
    assert "--micro-batches 8,4,2,1" in text                      # largest first
    assert "--modes train,inference" in text
    assert "--data compute,dataloader" in text
    assert "--checkpointing auto" in text                         # off, on if needed
    assert "--vram-headroom-pct 90" in text
    assert "--warmup-steps 20" in text
    assert "--steps 100" in text
    assert "--compile" not in text                                # baseline only
    assert "--workers 8" in text                                  # matches the 8 CPUs


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_stays_inside_the_measured_qos(name):
    text = (CLUSTER / name).read_text(encoding="utf-8")
    cpus = int(re.search(r"--cpus-per-task=(\d+)", text).group(1))
    mem_gb = int(re.search(r"--mem=(\d+)G", text).group(1))
    h, m, s = (int(x) for x in
               re.search(r"--time=(\d+):(\d+):(\d+)", text).groups())
    assert cpus <= QOS_MAX_CPUS, f"{name} asks for {cpus} CPUs"
    assert mem_gb <= QOS_MAX_MEM_GB, f"{name} asks for {mem_gb} GB"
    assert h * 60 + m + s / 60 <= QOS_MAX_WALL_MIN, f"{name} exceeds MaxWall"


# ----------------------------------------------------- production training

def test_train_sbatch_matches_the_production_request():
    text = (CLUSTER / "train_dinov3l512.sbatch").read_text(encoding="utf-8")
    assert "--cpus-per-task=8" in text
    assert "--mem=26G" in text
    assert "--time=05:50:00" in text
    assert "--signal=USR1@300" in text
    assert "--resume auto" in text
    assert "--max-wall-minutes 330" in text
    assert "#SBATCH --requeue" not in text   # not until TC2 confirms it works


def test_app_timer_leaves_room_before_the_hard_kill():
    """The application must stop itself well before Slurm kills the allocation."""
    text = (CLUSTER / "train_dinov3l512.sbatch").read_text(encoding="utf-8")
    h, m, _ = (int(x) for x in re.search(r"--time=(\d+):(\d+):(\d+)", text).groups())
    wall = h * 60 + m
    app = float(re.search(r"--max-wall-minutes (\d+)", text).group(1))
    assert wall == 350 and app == 330
    assert wall - app >= 15, "less than 15 min between the app timer and the kill"


def test_train_sbatch_runs_python_as_an_srun_step():
    """SIGUSR1 from --signal only reaches the process when it is a job step."""
    text = (CLUSTER / "train_dinov3l512.sbatch").read_text(encoding="utf-8")
    srun = text.index("srun --ntasks=1")
    assert srun < text.index("python -u -m src.train")
    assert text.index("--signal=USR1@300") < srun


# ------------------------------------------------------- discovered profile

DISCOVERED = {
    "partition": "MGPU-TC2", "qos": "normal", "gpu": "L40S",
    "vram": "46,068 MiB", "compute_capability": "8.9",
    "driver": "570.148.08", "cuda": "12.8",
    "max_gpus": "Max GPUs per user | 1", "max_cpus": "Max CPU cores | 10",
    "max_ram": "Max host RAM | 30 GB", "max_jobs": "MaxJobsPU | 2",
    "max_submit": "MaxSubmitPU | 2", "max_wall": "MaxWall | 6 h",
}


@pytest.mark.parametrize("key,value", sorted(DISCOVERED.items()))
def test_cluster_profile_records_the_discovered_values(key, value):
    text = (REPO / "docs" / "cluster_profile.md").read_text(encoding="utf-8")
    assert value in text, f"cluster_profile.md is missing {key}={value!r}"


def test_cluster_profile_separates_confirmed_from_unknown():
    text = (REPO / "docs" / "cluster_profile.md").read_text(encoding="utf-8")
    assert "TO CONFIRM" in text
    # the array policy gates eval_array.sbatch and must not be presented as known
    assert "how array tasks count against MaxSubmitPU=2" in text
    assert text.index("## TC2 — confirmed") < text.index("## TC2 — TO CONFIRM")


# ------------------------------------------------------- evaluation array

def eval_text():
    return (CLUSTER / "eval_array.sbatch").read_text(encoding="utf-8")


def test_eval_array_never_runs_more_than_one_task_at_a_time():
    """MaxJobsPU=2 and one GPU: 15 simultaneous jobs would be rejected."""
    text = eval_text()
    assert "#SBATCH --array=0-2%1" in text            # 3 groups, 1 concurrent
    assert re.search(r"--array=\d+-\d+%1", text)
    assert not re.search(r"#SBATCH --array=0-14(?!%)", text)
    assert "MaxSubmitPU" in text and "MaxJobsPU" in text
    assert "Never submit 15 independent jobs" in text


def test_eval_array_offers_a_no_array_fallback():
    text = eval_text()
    assert "ARRAY_MODE" in text
    assert "--condition all" in text                  # single resumable job
    assert "arrays are disallowed" in text


def test_eval_array_groups_match_the_code():
    """The sbatch task labels must be the groups src.evaluate actually knows."""
    from src.evaluate import GROUPS

    text = eval_text()
    listed = re.search(r"GROUPS=\(([^)]*)\)", text).group(1).split()
    assert sorted(listed) == sorted(GROUPS)


def test_eval_array_condition_list_is_the_frozen_fifteen():
    from track5.transforms.eval_atoms import EVAL_15

    conditions = re.search(r"CONDITIONS=\(\n(.*?)\n\)", eval_text(),
                           re.S).group(1).split()
    assert conditions == EVAL_15


def test_eval_array_is_resumable_and_leaves_the_protected_run_alone():
    text = eval_text()
    assert "-m src.evaluate" in text
    assert "--resume" in text
    assert "--protected-run" not in text.split("# The protected")[0]
    assert "happens exactly once" in text


def test_chain_depth_respects_maxsubmitpu():
    text = (CLUSTER / "train_dinov3l512.sbatch").read_text(encoding="utf-8")
    assert "MaxSubmitPU" in text
    assert "--dependency=afterok" in text
    assert text.count("sbatch --dependency=afterok") == 1  # a two-deep chain only
