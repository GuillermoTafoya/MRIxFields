"""Checkpoint helpers with conservative size guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any], *, max_bytes: int = 10_000_000) -> Path:
    """Save a small checkpoint and reject unexpectedly large outputs."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    size = checkpoint_path.stat().st_size
    if size > max_bytes:
        checkpoint_path.unlink(missing_ok=True)
        raise ValueError(f"Checkpoint exceeded size guardrail: {size} bytes > {max_bytes} bytes.")
    return checkpoint_path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError("Expected checkpoint to contain a dictionary.")
    return state

