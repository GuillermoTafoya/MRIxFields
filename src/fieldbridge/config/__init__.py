"""Small YAML config helpers used by the CLI and smoke tests."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def load_yaml_config(path: str | Path) -> Config:
    """Load a YAML config file as a mutable dictionary."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of config: {config_path}")
    return dict(data)


def dump_yaml_config(config: Mapping[str, Any]) -> str:
    """Return a deterministic YAML rendering for terminal output."""

    return yaml.safe_dump(dict(config), sort_keys=False)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge two mapping-like configs."""

    merged: Config = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = deepcopy(value)
    return merged

