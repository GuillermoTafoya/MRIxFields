"""Closed dependency bootstrap for the sealed A100 inference-only audit.

This file intentionally uses only the Python standard library until the
authenticated preinstalled CUDA stack has been validated. It never installs
torch, torchvision, or any preinstalled scientific package. The sole permitted
installation is the exact hash-pinned LPIPS wheel, with --no-deps.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


DEPENDENCY_LOCK_CONTRACT = "stage2-step200-inference-audit-dependency-lock-v2"
LPIPS_WHEEL_SHA256 = (
    "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e"
)
LPIPS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/9b/13/"
    "1df50c7925d9d2746702719f40e864f51ed66f307b20ad32392f1ad2bb87/"
    "lpips-0.1.4-py3-none-any.whl"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LOCK_KEYS = {
    "accepted_runtime_profile",
    "authoritative_compatibility_sources",
    "contract_version",
    "notebook_install",
    "preinstalled_runtime_packages",
    "provenance_only_unqualified_profiles",
}
_PROFILE_KEYS = {
    "gpu_name",
    "observed_driver",
    "observed_python_patch",
    "python_cache_tag",
    "python_implementation",
    "python_major_minor",
    "torch",
    "torch_cuda",
    "torchvision",
}
_PREINSTALLED_PACKAGE_KEYS = {
    "PyYAML",
    "matplotlib",
    "nibabel",
    "numpy",
    "scikit-image",
    "scipy",
}
_INSTALL_KEYS = {
    "artifact_filename",
    "artifact_sha256",
    "artifact_url",
    "distribution",
    "version",
}
_UNQUALIFIED_PROFILE_KEYS = {
    "accepted",
    "python",
    "reason",
    "torch",
    "torch_cuda",
    "torchvision",
}
_NO_ACTION_FIELDS = {
    "pip_install_invoked": False,
    "dependency_download_invoked": False,
    "model_weight_download_invoked": False,
    "drive_mount_invoked": False,
    "bank_accessed": False,
    "checkpoint_loaded": False,
    "private_data_accessed": False,
    "inference_invoked": False,
    "training_invoked": False,
}


class RuntimeProfileMismatch(RuntimeError):
    """Fail-closed runtime mismatch with a sanitized no-action receipt."""

    def __init__(self, reason: str, observed: Mapping[str, Any]) -> None:
        self.receipt = {
            "stage": "inference_audit_runtime_profile_preflight",
            "status": "fail",
            "reason": reason,
            "observed_compatibility": dict(observed),
            **_NO_ACTION_FIELDS,
        }
        super().__init__("Authenticated Colab inference runtime profile mismatch.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate dependency-lock key: {key!r}.")
        result[key] = value
    return result


def load_dependency_lock(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict) or set(payload) != _LOCK_KEYS:
        raise ValueError("Inference-audit dependency lock schema changed.")
    if payload.get("contract_version") != DEPENDENCY_LOCK_CONTRACT:
        raise ValueError("Inference-audit dependency lock version changed.")
    profile = payload.get("accepted_runtime_profile")
    packages = payload.get("preinstalled_runtime_packages")
    install = payload.get("notebook_install")
    unqualified = payload.get("provenance_only_unqualified_profiles")
    if not isinstance(profile, dict) or set(profile) != _PROFILE_KEYS:
        raise ValueError("Authenticated runtime-profile schema changed.")
    if (
        profile.get("python_implementation") != "CPython"
        or profile.get("python_cache_tag") != "cpython-313"
        or profile.get("python_major_minor") != [3, 13]
    ):
        raise ValueError("Authenticated Python ABI identity changed.")
    string_profile_keys = _PROFILE_KEYS - {"python_major_minor"}
    if any(
        not isinstance(profile.get(key), str) or not profile[key]
        for key in string_profile_keys
    ):
        raise ValueError("Authenticated runtime identities must be nonempty strings.")
    if not isinstance(packages, dict) or set(packages) != _PREINSTALLED_PACKAGE_KEYS:
        raise ValueError("Preinstalled runtime-package inventory changed.")
    if any(not isinstance(value, str) or not value for value in packages.values()):
        raise ValueError("Preinstalled package versions must be nonempty.")
    if not isinstance(install, dict) or set(install) != _INSTALL_KEYS:
        raise ValueError("Notebook-install artifact contract changed.")
    if (
        install.get("distribution") != "lpips"
        or install.get("version") != "0.1.4"
        or install.get("artifact_filename") != "lpips-0.1.4-py3-none-any.whl"
        or install.get("artifact_sha256") != LPIPS_WHEEL_SHA256
        or _SHA256_RE.fullmatch(str(install.get("artifact_sha256"))) is None
        or install.get("artifact_url") != LPIPS_WHEEL_URL
    ):
        raise ValueError("Pinned LPIPS distribution identity changed.")
    if not isinstance(unqualified, list) or len(unqualified) != 1:
        raise ValueError("Unqualified runtime provenance inventory changed.")
    old = unqualified[0]
    if (
        not isinstance(old, dict)
        or set(old) != _UNQUALIFIED_PROFILE_KEYS
        or old.get("accepted") is not False
    ):
        raise ValueError("Unqualified runtime profile must remain non-authorizing.")
    return payload


def _python_runtime_identity() -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_major_minor": [sys.version_info.major, sys.version_info.minor],
        "python_patch": platform.python_version(),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_preinstalled_cuda_stack(
    lock: Mapping[str, Any],
    *,
    hardware_gate: Mapping[str, Any],
) -> dict[str, Any]:
    expected = dict(lock["accepted_runtime_profile"])
    observed = _python_runtime_identity()
    observed.update(
        {
            "gpu_name": hardware_gate.get("gpu_name"),
            "nvidia_driver": hardware_gate.get("nvidia_driver"),
        }
    )
    if (
        hardware_gate.get("stage") != "standard_library_a100_80gb_gate"
        or hardware_gate.get("status") != "pass"
        or hardware_gate.get("gpu_name") != expected["gpu_name"]
    ):
        raise RuntimeProfileMismatch("hardware_gate_identity_changed", observed)
    python_contract = {
        "python_implementation": expected["python_implementation"],
        "python_cache_tag": expected["python_cache_tag"],
        "python_major_minor": expected["python_major_minor"],
    }
    if any(observed.get(key) != value for key, value in python_contract.items()):
        raise RuntimeProfileMismatch("python_abi_identity_changed", observed)
    observed["observed_profile_python_patch"] = expected["observed_python_patch"]
    observed["python_patch_matches_observed_profile"] = (
        observed["python_patch"] == expected["observed_python_patch"]
    )
    observed["observed_profile_driver"] = expected["observed_driver"]
    observed["driver_matches_observed_profile"] = (
        observed["nvidia_driver"] == expected["observed_driver"]
    )
    try:
        import torch
        import torchvision
    except ImportError:
        raise RuntimeProfileMismatch("preinstalled_cuda_stack_unavailable", observed)
    observed.update(
        {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    expected_stack = {
        "torch": expected["torch"],
        "torchvision": expected["torchvision"],
        "torch_cuda": expected["torch_cuda"],
    }
    if any(observed.get(key) != value for key, value in expected_stack.items()):
        raise RuntimeProfileMismatch("torch_torchvision_cuda_identity_changed", observed)
    if observed["cuda_available"] is not True:
        raise RuntimeProfileMismatch("cuda_visibility_changed_after_hardware_gate", observed)
    return observed


def prepare_locked_environment(
    lock_path: str | Path,
    *,
    hardware_gate: Mapping[str, Any],
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    lock = load_dependency_lock(lock_path)
    runtime_profile = validate_preinstalled_cuda_stack(
        lock, hardware_gate=hardware_gate
    )
    expected_preinstalled = dict(lock["preinstalled_runtime_packages"])
    observed_preinstalled = {
        name: _distribution_version(name) for name in sorted(expected_preinstalled)
    }
    if observed_preinstalled != dict(sorted(expected_preinstalled.items())):
        observed = dict(runtime_profile)
        observed["preinstalled_runtime_packages"] = observed_preinstalled
        raise RuntimeProfileMismatch("preinstalled_package_inventory_changed", observed)

    install = dict(lock["notebook_install"])
    distribution = install["distribution"]
    observed_lpips = _distribution_version(distribution)
    if observed_lpips not in (None, install["version"]):
        observed = dict(runtime_profile)
        observed["lpips"] = observed_lpips
        raise RuntimeProfileMismatch("installed_lpips_identity_changed", observed)

    pip_invoked = observed_lpips is None
    pip_output = ""
    if pip_invoked:
        requirement = (
            f"{distribution} @ {install['artifact_url']}"
            f"#sha256={install['artifact_sha256']}"
        )
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
            "--require-hashes",
            requirement,
        ]
        completed = run(command, text=True, capture_output=True, check=False)
        pip_output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            raise RuntimeError(
                "Hash-pinned LPIPS installation failed "
                f"with return code {completed.returncode}."
            )
    resolved_lpips = _distribution_version(distribution)
    if resolved_lpips != install["version"]:
        raise RuntimeError("Hash-pinned LPIPS installation identity changed.")

    complete_environment: dict[str, str] = {}
    for installed_distribution in importlib.metadata.distributions():
        name = str(installed_distribution.metadata.get("Name", "")).strip().lower()
        if not name:
            continue
        value = str(installed_distribution.version)
        previous = complete_environment.get(name)
        if previous is not None and previous != value:
            raise RuntimeError(f"Ambiguous installed distribution identity for {name!r}.")
        complete_environment[name] = value
    return {
        "contract_version": DEPENDENCY_LOCK_CONTRACT,
        "lock_file_sha256": _sha256_file(Path(lock_path)),
        "accepted_runtime_profile": dict(lock["accepted_runtime_profile"]),
        "observed_runtime_profile": runtime_profile,
        "observed_preinstalled_runtime_packages": observed_preinstalled,
        "locked_runtime_packages": {
            **dict(sorted(expected_preinstalled.items())),
            distribution: resolved_lpips,
        },
        "notebook_installed_packages": (
            {distribution: resolved_lpips} if pip_invoked else {}
        ),
        "lpips_distribution_artifact": install,
        "complete_resolved_environment": dict(sorted(complete_environment.items())),
        "pip_install_invoked": pip_invoked,
        "dependency_download_observed": "Downloading " in pip_output,
        "lpips_artifact_network_download_observed": "Downloading " in pip_output,
        "pip_no_deps": True,
        "pip_require_hashes": True,
        "torch_or_torchvision_reinstalled": False,
        "preinstalled_packages_mutated": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DEPENDENCY_LOCK_CONTRACT",
    "LPIPS_WHEEL_SHA256",
    "LPIPS_WHEEL_URL",
    "RuntimeProfileMismatch",
    "load_dependency_lock",
    "prepare_locked_environment",
    "validate_preinstalled_cuda_stack",
]
