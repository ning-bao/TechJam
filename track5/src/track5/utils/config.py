import hashlib
import json
from pathlib import Path

import yaml


def load_config(path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a dict")
    return cfg


def config_hash(cfg: dict) -> str:
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]
