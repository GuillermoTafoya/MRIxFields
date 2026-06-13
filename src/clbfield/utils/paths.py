"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def ensure_relative_to(path: str | Path, root: str | Path) -> Path:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{resolved_path} is not under {resolved_root}.") from exc
    return resolved_path

