"""Checkpoint helpers with conservative size guardrails and run metadata."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from clbfield.utils.paths import project_root


def save_checkpoint(
    path: str | Path,
    state: dict[str, Any],
    *,
    max_bytes: int = 10_000_000,
    overwrite: bool = False,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    git_commit: str | None = None,
) -> Path:
    """Save a checkpoint with run metadata; reject silent overwrites and oversized outputs."""

    checkpoint_path = Path(path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint {checkpoint_path} already exists. Pass overwrite=True to replace it explicitly."
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(state)
    payload["_meta"] = {
        "seed": seed,
        "config": config,
        "git_commit": git_commit if git_commit is not None else resolve_git_commit(),
    }

    torch.save(payload, checkpoint_path)
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


def resolve_git_commit() -> str:
    """Best-effort git commit hash for the current checkout; "unknown" if git is unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def checkpoint_filename(stage: str, variant: str, step: int, *, timestamp: str | None = None) -> str:
    """Build a filename following the `{stage}_{variant}_{YYYYMMDD}_step{N}.pt` run-naming convention."""

    date = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{stage}_{variant}_{date}_step{int(step)}.pt"
