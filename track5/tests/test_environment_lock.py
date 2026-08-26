"""The declared dependency range, the lock file, and the installed environment
must agree — a drifting transformers is what breaks DINOv3ViT (PLAN D1)."""

import importlib.metadata as md
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "environment.pip-freeze.txt"
VERIFIED_TRANSFORMERS = "5.15.1"


def declared() -> dict[str, str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    out = {}
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip()
        out[name.lower()] = spec
    return out


def locked() -> dict[str, str]:
    out = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        out[name.strip().lower()] = version.strip()
    return out


def test_transformers_is_pinned_to_a_tested_range():
    spec = declared()["transformers"]
    assert spec == "transformers>=5.15.1,<5.16", spec


def test_lock_exists_and_is_fully_pinned():
    entries = locked()
    assert len(entries) > 50
    assert all(v for v in entries.values()), "an entry is not pinned with =="
    assert not any(ln.startswith("-e ") for ln in
                   LOCK.read_text(encoding="utf-8").splitlines()), \
        "the lock must not carry a machine-specific editable path"


def test_lock_matches_the_declared_transformers_range():
    assert locked()["transformers"] == VERIFIED_TRANSFORMERS


def test_installed_environment_matches_the_lock():
    entries = locked()
    mismatched = []
    for name, want in entries.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            continue          # optional/platform-specific wheels
        if have != want:
            mismatched.append((name, want, have))
    assert not mismatched, f"environment drifted from the lock: {mismatched}"


def test_installed_transformers_is_inside_the_pin():
    have = md.version("transformers")
    assert have == VERIFIED_TRANSFORMERS
    major, minor = (int(x) for x in have.split(".")[:2])
    assert (major, minor) == (5, 15)


def test_every_declared_dependency_is_in_the_lock():
    entries = locked()
    missing = [n for n in declared() if n not in entries]
    assert not missing, f"declared but absent from the lock: {missing}"


@pytest.mark.parametrize("name", ["torch", "torchvision"])
def test_cuda_wheels_are_recorded_with_their_local_version(name):
    """The +cu128 tag is what makes the lock non-portable; keep it visible so it
    is not silently installed as a CPU build on the cluster."""
    assert "+cu" in locked()[name]
    assert "TC2 NOTE" in LOCK.read_text(encoding="utf-8")
