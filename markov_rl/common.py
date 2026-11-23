"""Shared helpers: paths, config loading, seeding, run metadata."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
CONFIGS = Path(__file__).resolve().parent / "configs"


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _coerce(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(path: str | None, overrides: list[str] | None = None, defaults: dict | None = None) -> dict:
    """defaults <- yaml file (with optional `base:` chain) <- `key=value` overrides."""
    cfg = dict(defaults or {})
    if path:
        p = Path(path)
        if not p.exists() and (CONFIGS / p.name).exists():
            p = CONFIGS / p.name
        raw = yaml.safe_load(p.read_text()) or {}
        if "base" in raw:
            cfg.update(load_config(str(p.parent / raw.pop("base"))))
        cfg.update(raw)
        cfg["config_path"] = str(p)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        cfg[k] = _coerce(v)
    return cfg


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT, stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unversioned"


def run_meta(cfg: dict) -> dict:
    meta = {
        "config": cfg,
        "git_hash": git_hash(),
        "argv": sys.argv,
        "python": platform.python_version(),
        "host": platform.node(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for mod in ("transformers", "peft", "vllm"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    return meta


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device() -> str:
    import torch

    if os.environ.get("MARKOV_DEVICE"):
        return os.environ["MARKOV_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
