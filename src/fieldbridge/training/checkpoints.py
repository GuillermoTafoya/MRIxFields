"""Checkpoint helpers with conservative size guardrails and run metadata."""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fieldbridge.utils.paths import project_root


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
    """Atomically save a validated checkpoint in the destination directory.

    The temporary artifact is written, size-checked, loaded back, and fsynced before
    ``os.replace`` publishes it. An interrupted overwrite therefore leaves the previous
    checkpoint intact. Temporary artifacts are removed on every handled failure path.
    """

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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        size = temporary_path.stat().st_size
        if size <= 0:
            raise ValueError("Checkpoint serialization produced an empty artifact.")
        if size > max_bytes:
            raise ValueError(
                f"Checkpoint exceeded size guardrail: {size} bytes > {max_bytes} bytes."
            )
        validated = load_checkpoint(temporary_path, map_location="cpu")
        missing_keys = sorted(set(payload) - set(validated))
        if missing_keys or not isinstance(validated.get("_meta"), dict):
            raise ValueError(
                "Checkpoint validation failed before publication; "
                f"missing_keys={missing_keys}, metadata_valid="
                f"{isinstance(validated.get('_meta'), dict)}."
            )
        # Windows requires a writable descriptor for fsync.
        with temporary_path.open("ab") as handle:
            os.fsync(handle.fileno())
        if checkpoint_path.exists() and not overwrite:
            raise FileExistsError(
                f"Checkpoint {checkpoint_path} already exists. Pass overwrite=True "
                "to replace it explicitly."
            )
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path
    finally:
        temporary_path.unlink(missing_ok=True)


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
