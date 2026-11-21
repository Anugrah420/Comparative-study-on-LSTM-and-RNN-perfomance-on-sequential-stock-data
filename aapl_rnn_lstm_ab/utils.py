"""Small helpers: seeding, device selection, directories, JSON I/O."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a device string ('auto', 'cpu', 'cuda', 'mps') to a torch.device."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o):
    """Make numpy/Path objects JSON-serialisable."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj: dict, path: str | Path) -> Path:
    """Write a dict to a JSON file (creating parent dirs) and return the path."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=_json_default))
    return p


def load_json(path: str | Path) -> dict:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())