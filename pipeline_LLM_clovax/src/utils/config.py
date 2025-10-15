"""Utility helpers for loading YAML configs with environment variable expansion."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")



def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR:default} placeholders within the given value."""
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            default = match.group(2) if match.group(2) is not None else match.group(0)
            return os.getenv(key, default)

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, Mapping):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value



def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand environment variables."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)
