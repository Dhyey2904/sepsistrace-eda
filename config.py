"""Settings: loads config.yaml into a dict with attribute access (cfg.data.vitals)."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.yaml"


class Config(dict):
    """dict with attribute access, applied recursively (cfg.data.vitals)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def path(self, key: str) -> Path:
        """Absolute path for cfg.paths[key], resolved against the project root."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(quick: bool = False) -> Config:
    """Read config.yaml. quick=True applies its `quick_mode` overrides (2000-patient subset)."""
    raw = yaml.safe_load(CONFIG_FILE.read_text())
    overrides = raw.pop("quick_mode")
    return Config.wrap(_merge(raw, overrides) if quick else raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
