"""Minimal no-dependency build backend for offline editable installs.

The project scaffold must support ``pip install -e ".[dev]"`` without
downloading build tooling. This backend emits pure-Python wheels using only the
standard library.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

NAME = "field-bridge"
NORMALIZED_NAME = "field_bridge"
VERSION = "0.1.0"
DIST_INFO = f"{NORMALIZED_NAME}-{VERSION}.dist-info"
TAG = "py3-none-any"
WHEEL_NAME = f"{NORMALIZED_NAME}-{VERSION}-{TAG}.whl"


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    del config_settings
    return []


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    del config_settings
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    del config_settings
    return _write_metadata_dir(Path(metadata_directory))


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    del config_settings
    return _write_metadata_dir(Path(metadata_directory))


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    files: dict[str, bytes] = {}
    for path in sorted((SRC_ROOT / "fieldbridge").rglob("*")):
        if path.is_file():
            archive_name = path.relative_to(SRC_ROOT).as_posix()
            files[archive_name] = path.read_bytes()
    _add_dist_info(files)
    return _write_wheel(Path(wheel_directory), files)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    files = {
        f"{NORMALIZED_NAME}.pth": f"{SRC_ROOT}{os.linesep}".encode("utf-8"),
    }
    _add_dist_info(files)
    return _write_wheel(Path(wheel_directory), files)


def _write_metadata_dir(metadata_directory: Path) -> str:
    dist_info = metadata_directory / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel(), encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(_entry_points(), encoding="utf-8")
    return DIST_INFO


def _add_dist_info(files: dict[str, bytes]) -> None:
    files[f"{DIST_INFO}/METADATA"] = _metadata().encode("utf-8")
    files[f"{DIST_INFO}/WHEEL"] = _wheel().encode("utf-8")
    files[f"{DIST_INFO}/entry_points.txt"] = _entry_points().encode("utf-8")


def _metadata() -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {NAME}",
        f"Version: {VERSION}",
        "Summary: Polymorphic PyTorch/MONAI-compatible framework for MRI field and contrast translation.",
        "Requires-Python: >=3.10",
        "License: MIT",
        "Requires-Dist: torch>=2.0",
        "Requires-Dist: PyYAML>=6.0",
        "Provides-Extra: dev",
        'Requires-Dist: pytest>=8.0; extra == "dev"',
        "Provides-Extra: quality",
        'Requires-Dist: mypy>=1.10; extra == "quality"',
        'Requires-Dist: ruff>=0.6; extra == "quality"',
        "Provides-Extra: monai",
        'Requires-Dist: monai>=1.3; extra == "monai"',
        "Provides-Extra: perceptual",
        'Requires-Dist: lpips>=0.1.4; extra == "perceptual"',
        "Provides-Extra: nifti",
        'Requires-Dist: nibabel>=5.0; extra == "nifti"',
        "",
    ]
    return "\n".join(lines)


def _wheel() -> str:
    lines = [
        "Wheel-Version: 1.0",
        "Generator: fieldbridge-build-backend 0.1",
        "Root-Is-Purelib: true",
        f"Tag: {TAG}",
        "",
    ]
    return "\n".join(lines)


def _entry_points() -> str:
    return "[console_scripts]\nfieldbridge = fieldbridge.cli:main\n"


def _write_wheel(wheel_directory: Path, files: dict[str, bytes]) -> str:
    wheel_directory.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_directory / WHEEL_NAME
    record_path = f"{DIST_INFO}/RECORD"
    record = _record(files, record_path)

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_name, content in files.items():
            wheel.writestr(archive_name, content)
        wheel.writestr(record_path, record.encode("utf-8"))

    return wheel_path.name


def _record(files: dict[str, bytes], record_path: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for archive_name, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow([archive_name, f"sha256={digest}", str(len(content))])
    writer.writerow([record_path, "", ""])
    return output.getvalue()
