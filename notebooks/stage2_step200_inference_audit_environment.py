"""Closed dependency bootstrap for the sealed A100 inference-only audit.

This file intentionally uses only the Python standard library until the pinned
preinstalled CUDA stack has been validated.  It never installs torch or
torchvision and invokes pip with ``--no-deps`` for a closed exact inventory.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


DEPENDENCY_LOCK_CONTRACT = "stage2-step200-inference-audit-dependency-lock-v1"
_LOCK_KEYS = {
    "authoritative_compatibility_sources",
    "contract_version",
    "installed_by_notebook",
    "preinstalled_cuda_stack",
}
_CUDA_STACK_KEYS = {"python", "torch", "torch_cuda", "torchvision"}
_INSTALL_KEYS = {
    "PyYAML",
    "lpips",
    "matplotlib",
    "nibabel",
    "numpy",
    "scikit-image",
    "scipy",
}


def load_dependency_lock(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _LOCK_KEYS:
        raise ValueError("Inference-audit dependency lock schema changed.")
    if payload.get("contract_version") != DEPENDENCY_LOCK_CONTRACT:
        raise ValueError("Inference-audit dependency lock version changed.")
    stack = payload.get("preinstalled_cuda_stack")
    installs = payload.get("installed_by_notebook")
    if not isinstance(stack, dict) or set(stack) != _CUDA_STACK_KEYS:
        raise ValueError("Pinned CUDA-stack inventory changed.")
    if not isinstance(installs, dict) or set(installs) != _INSTALL_KEYS:
        raise ValueError("Pinned install inventory changed.")
    for inventory in (stack, installs):
        if any(not isinstance(value, str) or not value for value in inventory.values()):
            raise ValueError("Dependency identities must be nonempty exact versions.")
    return payload


def validate_preinstalled_cuda_stack(lock: Mapping[str, Any]) -> dict[str, str]:
    expected = dict(lock["preinstalled_cuda_stack"])
    if platform.python_version() != expected["python"]:
        raise RuntimeError("Python version is outside the sealed inference-audit runtime.")
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("The sealed preinstalled CUDA stack is unavailable.") from exc
    observed = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "torch_cuda": str(torch.version.cuda),
    }
    if observed != expected:
        raise RuntimeError(
            "Preinstalled torch/torchvision/CUDA compatibility gate failed: "
            f"expected={expected!r}, observed={observed!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned PyTorch cannot see CUDA after the hardware gate.")
    return observed


def prepare_locked_environment(
    lock_path: str | Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    lock = load_dependency_lock(lock_path)
    cuda_stack = validate_preinstalled_cuda_stack(lock)
    exact = dict(lock["installed_by_notebook"])
    missing_or_changed: list[str] = []
    for distribution, expected in sorted(exact.items()):
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        if observed != expected:
            missing_or_changed.append(f"{distribution}=={expected}")
    pip_invoked = bool(missing_or_changed)
    pip_output = ""
    if missing_or_changed:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-deps",
            "--only-binary=:all:",
            "--progress-bar=off",
            *missing_or_changed,
        ]
        completed = run(command, text=True, capture_output=True, check=False)
        pip_output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            raise RuntimeError(
                "Closed inference dependency installation failed "
                f"with return code {completed.returncode}."
            )
    resolved_required: dict[str, str] = {}
    for distribution, expected in sorted(exact.items()):
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Locked dependency {distribution!r} is absent.") from exc
        if observed != expected:
            raise RuntimeError(
                f"Locked dependency {distribution!r} changed: {observed!r} != {expected!r}."
            )
        resolved_required[distribution] = observed
    complete_environment: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip().lower()
        if not name:
            continue
        value = str(distribution.version)
        previous = complete_environment.get(name)
        if previous is not None and previous != value:
            raise RuntimeError(f"Ambiguous installed distribution identity for {name!r}.")
        complete_environment[name] = value
    return {
        "contract_version": DEPENDENCY_LOCK_CONTRACT,
        "lock_file_sha256": _sha256_file(Path(lock_path)),
        "preinstalled_cuda_stack": cuda_stack,
        "locked_runtime_packages": resolved_required,
        "complete_resolved_environment": dict(sorted(complete_environment.items())),
        "pip_install_invoked": pip_invoked,
        "dependency_download_observed": "Downloading " in pip_output,
        "torch_or_torchvision_reinstalled": False,
        "pip_no_deps": True,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DEPENDENCY_LOCK_CONTRACT",
    "load_dependency_lock",
    "prepare_locked_environment",
    "validate_preinstalled_cuda_stack",
]
