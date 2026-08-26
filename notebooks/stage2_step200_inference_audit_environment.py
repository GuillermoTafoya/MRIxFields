"""Closed environment bootstrap for the sealed A100 inference-only audit.

The module validates the authenticated runtime and locked numerical dependency
closure before Drive access. Unrelated distribution metadata is retained as a
sanitized deterministic multimap. The sole permitted installation is the exact
hash-pinned LPIPS wheel, with a local transactional bootstrap receipt.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


DEPENDENCY_LOCK_CONTRACT = "stage2-step200-inference-audit-dependency-lock-v2"
ENVIRONMENT_PROVENANCE_CONTRACT = (
    "stage2-step200-inference-environment-provenance-v2"
)
LPIPS_BOOTSTRAP_RECEIPT_CONTRACT = (
    "stage2-step200-lpips-local-bootstrap-receipt-v1"
)
LPIPS_WHEEL_SHA256 = (
    "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e"
)
LPIPS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/9b/13/"
    "1df50c7925d9d2746702719f40e864f51ed66f307b20ad32392f1ad2bb87/"
    "lpips-0.1.4-py3-none-any.whl"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
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
_LOCKED_DISTRIBUTION_IMPORTS = {
    "PyYAML": "yaml",
    "lpips": "lpips",
    "matplotlib": "matplotlib",
    "nibabel": "nibabel",
    "numpy": "numpy",
    "scikit-image": "skimage",
    "scipy": "scipy",
    "torch": "torch",
    "torchvision": "torchvision",
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
_BOOTSTRAP_RECEIPT_KEYS = {
    "active_lpips_import",
    "contract_version",
    "cuda_stack",
    "dependency_lock_file_sha256",
    "implementation_commit",
    "installed_package_tree",
    "lpips_version",
    "python_abi",
    "receipt_sha256",
    "wheel",
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


def normalize_distribution_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Distribution name must be nonempty.")
    normalized = re.sub(r"[-_.]+", "-", name.strip()).lower()
    if not normalized or any(
        ord(char) < 32 or ord(char) == 127 for char in normalized
    ):
        raise ValueError("Distribution name contains invalid characters.")
    return normalized


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def load_dependency_lock(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path))
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
        raise RuntimeProfileMismatch(
            "cuda_visibility_changed_after_hardware_gate", observed
        )
    return observed


def _source_root(distribution: Any) -> Path:
    try:
        return Path(distribution.locate_file("")).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("Installed distribution source root is unavailable.") from exc


def _source_root_sha256(root: Path) -> str:
    label = os.path.normcase(str(root))
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _discover_distribution_records(
    distributions: Iterable[Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    source = importlib.metadata.distributions() if distributions is None else distributions
    for distribution in source:
        declared = str(distribution.metadata.get("Name", "")).strip()
        if not declared:
            continue
        normalized = normalize_distribution_name(declared)
        version = str(distribution.version)
        if not version or any(
            ord(char) < 32 or ord(char) == 127 for char in version
        ):
            raise RuntimeError("Installed distribution version is malformed.")
        root = _source_root(distribution)
        grouped.setdefault(normalized, []).append(
            {
                "_distribution": distribution,
                "_source_root": root,
                "normalized_name": normalized,
                "declared_metadata_name": declared,
                "version": version,
                "source_root_sha256": _source_root_sha256(root),
            }
        )
    for records in grouped.values():
        records.sort(
            key=lambda item: (
                item["declared_metadata_name"].casefold(),
                item["declared_metadata_name"],
                item["version"],
                item["source_root_sha256"],
            )
        )
    return dict(sorted(grouped.items()))


def build_complete_distribution_provenance(
    records: Mapping[str, list[dict[str, Any]]],
    *,
    locked_normalized_names: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    complete: dict[str, list[dict[str, Any]]] = {}
    ambiguities: list[dict[str, Any]] = []
    for normalized, entries in sorted(records.items()):
        count = len(entries)
        observations: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            observations.append(
                {
                    "normalized_name": normalized,
                    "declared_metadata_name": entry["declared_metadata_name"],
                    "version": entry["version"],
                    "distribution_index": index,
                    "distribution_count": count,
                    "source_root_sha256": entry["source_root_sha256"],
                }
            )
        complete[normalized] = observations
        if count > 1 and normalized not in locked_normalized_names:
            ambiguities.append(
                {
                    "normalized_name": normalized,
                    "distribution_count": count,
                    "versions": sorted({entry["version"] for entry in entries}),
                }
            )
    return complete, ambiguities


def _distribution_owns_module_file(record: Mapping[str, Any], module_file: Path) -> bool:
    distribution = record["_distribution"]
    files = distribution.files
    if files is None:
        raise RuntimeError("Installed distribution file inventory is unavailable.")
    for relative in files:
        try:
            candidate = Path(distribution.locate_file(relative)).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if candidate == module_file:
            return True
    return False


def _active_import_identity(
    distribution_name: str,
    import_name: str,
    expected_version: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        module = importlib.import_module(import_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Locked active import is unavailable for {distribution_name!r}."
        ) from exc
    module_label = getattr(module, "__file__", None)
    if not isinstance(module_label, str) or not module_label:
        raise RuntimeError(
            f"Locked active import has no file identity for {distribution_name!r}."
        )
    try:
        module_file = Path(module_label).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Locked active import file is unavailable for {distribution_name!r}."
        ) from exc
    owners = [record for record in records if _distribution_owns_module_file(record, module_file)]
    if not owners:
        raise RuntimeError(
            f"Locked active import is shadowed for {distribution_name!r}."
        )
    raw_active_version = getattr(module, "__version__", None)
    if raw_active_version is None:
        owner_versions = {record["version"] for record in owners}
        if len(owner_versions) != 1:
            raise RuntimeError(
                f"Locked active import ownership is ambiguous for {distribution_name!r}."
            )
        active_version = next(iter(owner_versions))
        version_source = "owning_distribution_metadata"
    else:
        active_version = str(raw_active_version)
        version_source = "active_import___version__"
    if active_version != expected_version:
        raise RuntimeError(
            f"Locked active import version changed for {distribution_name!r}."
        )
    public_owners = [
        {
            "declared_metadata_name": record["declared_metadata_name"],
            "source_root_sha256": record["source_root_sha256"],
            "version": record["version"],
        }
        for record in owners
    ]
    identity = {
        "distribution_name": distribution_name,
        "import_name": import_name,
        "active_version": active_version,
        "version_source": version_source,
        "module_file_sha256": _sha256_file(module_file),
        "owning_distribution_count": len(owners),
        "owning_source_root_sha256": sorted(
            record["source_root_sha256"] for record in owners
        ),
    }
    return identity, public_owners


def _validate_locked_distribution(
    distribution_name: str,
    expected_version: str,
    records: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = normalize_distribution_name(distribution_name)
    matches = list(records.get(normalized, []))
    if not matches:
        raise RuntimeError(
            f"Locked distribution is missing: {distribution_name!r}."
        )
    versions = sorted({record["version"] for record in matches})
    if versions != [expected_version]:
        raise RuntimeError(
            f"Locked distribution metadata version changed for {distribution_name!r}."
        )
    import_name = _LOCKED_DISTRIBUTION_IMPORTS[distribution_name]
    active, owners = _active_import_identity(
        distribution_name, import_name, expected_version, matches
    )
    observations = [
        {
            "declared_metadata_name": record["declared_metadata_name"],
            "source_root_sha256": record["source_root_sha256"],
            "version": record["version"],
        }
        for record in matches
    ]
    return {
        "normalized_name": normalized,
        "expected_version": expected_version,
        "metadata_entry_count": len(matches),
        "metadata_observations": observations,
        "active_import": active,
    }, owners


def _expected_locked_versions(lock: Mapping[str, Any]) -> dict[str, str]:
    profile = lock["accepted_runtime_profile"]
    versions = dict(lock["preinstalled_runtime_packages"])
    versions.update(
        {
            "torch": profile["torch"],
            "torchvision": profile["torchvision"],
            "lpips": lock["notebook_install"]["version"],
        }
    )
    return versions


def _validate_locked_closure(
    expected_versions: Mapping[str, str],
    records: Mapping[str, list[dict[str, Any]]],
    *,
    include_lpips: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    validated: dict[str, dict[str, Any]] = {}
    owners: dict[str, list[dict[str, Any]]] = {}
    for name, expected_version in sorted(expected_versions.items()):
        if name == "lpips" and not include_lpips:
            continue
        validated[name], owners[name] = _validate_locked_distribution(
            name, expected_version, records
        )
    return validated, owners


def _installed_lpips_package_tree(record: Mapping[str, Any]) -> dict[str, Any]:
    distribution = record["_distribution"]
    files = distribution.files
    if files is None:
        raise RuntimeError("LPIPS installed-file inventory is unavailable.")
    metadata_labels = [str(relative).replace(chr(92), "/") for relative in files]
    if not any(label == "lpips/__init__.py" for label in metadata_labels):
        raise RuntimeError("LPIPS installed-file inventory lacks its package root.")
    package_candidate = Path(distribution.locate_file("lpips"))
    if package_candidate.is_symlink():
        raise RuntimeError("LPIPS installed package root must not be a symlink.")
    try:
        package_root = package_candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("LPIPS installed package root is unavailable.") from exc
    if not package_root.is_dir():
        raise RuntimeError("LPIPS installed package root is not a directory.")
    observations: list[dict[str, Any]] = []
    pending = [package_root]
    while pending:
        directory = pending.pop()
        children = sorted(directory.iterdir(), key=lambda path: path.name)
        for candidate in children:
            if candidate.name == "__pycache__":
                continue
            if candidate.is_symlink():
                raise RuntimeError("LPIPS installed package contains a symlink.")
            if candidate.is_dir():
                pending.append(candidate)
                continue
            if not candidate.is_file():
                raise RuntimeError(
                    "LPIPS installed package member is not a regular file."
                )
            if candidate.suffix in {".pyc", ".pyo"}:
                continue
            relative = candidate.relative_to(package_root).as_posix()
            if not relative or any(
                part in {"", ".", ".."} for part in relative.split("/")
            ):
                raise RuntimeError("LPIPS installed package path is unsafe.")
            resolved = candidate.resolve(strict=True)
            observations.append(
                {
                    "relative_path": f"lpips/{relative}",
                    "sha256": _sha256_file(resolved),
                    "size_bytes": resolved.stat().st_size,
                }
            )
    observations.sort(key=lambda item: item["relative_path"])
    if not observations:
        raise RuntimeError("LPIPS installed package tree is empty.")
    labels = [item["relative_path"] for item in observations]
    if len(labels) != len(set(labels)):
        raise RuntimeError("LPIPS installed package tree contains duplicate labels.")
    tree_body = {"files": observations}
    return {
        "tree_sha256": _sha256_json(tree_body),
        "file_count": len(observations),
        "total_bytes": sum(item["size_bytes"] for item in observations),
    }


def _bootstrap_receipt_path(
    bootstrap_root: str | Path,
    implementation_commit: str,
) -> Path:
    if _GIT_COMMIT_RE.fullmatch(implementation_commit) is None:
        raise ValueError("Audit implementation commit must be a lowercase Git identity.")
    root = Path(bootstrap_root)
    if root.exists() and root.is_symlink():
        raise RuntimeError("LPIPS bootstrap root must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    namespace = root / f"implementation_{implementation_commit[:12]}"
    if namespace.exists() and namespace.is_symlink():
        raise RuntimeError("LPIPS bootstrap namespace must not be a symlink.")
    namespace.mkdir(exist_ok=True)
    return namespace / "lpips-bootstrap-receipt-v1.json"


def _expected_bootstrap_receipt_body(
    *,
    implementation_commit: str,
    dependency_lock_file_sha256: str,
    install: Mapping[str, Any],
    runtime: Mapping[str, Any],
    active_lpips_import: Mapping[str, Any],
    installed_package_tree: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": LPIPS_BOOTSTRAP_RECEIPT_CONTRACT,
        "implementation_commit": implementation_commit,
        "dependency_lock_file_sha256": dependency_lock_file_sha256,
        "wheel": {
            "filename": install["artifact_filename"],
            "url": install["artifact_url"],
            "sha256": install["artifact_sha256"],
        },
        "lpips_version": install["version"],
        "installed_package_tree": dict(installed_package_tree),
        "active_lpips_import": dict(active_lpips_import),
        "python_abi": {
            "implementation": runtime["python_implementation"],
            "cache_tag": runtime["python_cache_tag"],
            "major_minor": list(runtime["python_major_minor"]),
            "observed_patch": runtime["python_patch"],
        },
        "cuda_stack": {
            "torch": runtime["torch"],
            "torchvision": runtime["torchvision"],
            "torch_cuda": runtime["torch_cuda"],
        },
    }


def _load_and_verify_bootstrap_receipt(
    path: Path,
    expected_body: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("LPIPS bootstrap receipt is unavailable or unsafe.")
    payload = _load_json(path)
    if not isinstance(payload, dict) or set(payload) != _BOOTSTRAP_RECEIPT_KEYS:
        raise RuntimeError("LPIPS bootstrap receipt schema changed.")
    body = dict(payload)
    stored_hash = body.pop("receipt_sha256", None)
    if not isinstance(stored_hash, str) or _SHA256_RE.fullmatch(stored_hash) is None:
        raise RuntimeError("LPIPS bootstrap receipt identity is malformed.")
    if stored_hash != _sha256_json(body):
        raise RuntimeError("LPIPS bootstrap receipt self-hash changed.")
    if body != dict(expected_body):
        raise RuntimeError("LPIPS bootstrap receipt identity changed.")
    return payload


def _write_bootstrap_receipt_atomic(
    path: Path,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError("LPIPS bootstrap receipt already exists.")
    payload = dict(body)
    payload["receipt_sha256"] = _sha256_json(payload)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    descriptor, temporary_label = tempfile.mkstemp(
        prefix=".lpips-bootstrap-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_label)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _load_and_verify_bootstrap_receipt(path, body)


def _run_lpips_install(
    install: Mapping[str, Any],
    *,
    force_reinstall: bool,
    run: Callable[..., Any],
) -> None:
    requirement = (
        f"lpips @ {install['artifact_url']} "
        f"--hash=sha256:{install['artifact_sha256']}\n"
    )
    with tempfile.TemporaryDirectory(prefix="stage2-lpips-bootstrap-") as directory:
        requirements_path = Path(directory) / "requirements.txt"
        requirements_path.write_text(requirement, encoding="utf-8", newline="\n")
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
        ]
        if force_reinstall:
            command.append("--force-reinstall")
        command.extend(["-r", str(requirements_path)])
        run(command, check=True)


def _remove_lpips_imports() -> None:
    for name in list(sys.modules):
        if name == "lpips" or name.startswith("lpips."):
            del sys.modules[name]
    importlib.invalidate_caches()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_lpips_package_tree(
    records: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    matches = list(records.get("lpips", []))
    try:
        module_file = Path(importlib.import_module("lpips").__file__).resolve(
            strict=True
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("Active LPIPS import is unavailable.") from exc
    owners = [record for record in matches if _distribution_owns_module_file(record, module_file)]
    if not owners:
        raise RuntimeError("Active LPIPS import is not owned by locked metadata.")
    trees = [_installed_lpips_package_tree(record) for record in owners]
    first = trees[0]
    if any(tree != first for tree in trees[1:]):
        raise RuntimeError("Active LPIPS package tree identity is ambiguous.")
    return first


def prepare_locked_environment(
    lock_path: str | Path,
    *,
    hardware_gate: Mapping[str, Any],
    implementation_commit: str,
    bootstrap_root: str | Path,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Validate the locked runtime and transactionally establish LPIPS.

    The caller must invoke this only after the standard-library A100 gate. The
    function performs no Drive or private-artifact access.
    """

    lock_file = Path(lock_path)
    lock = load_dependency_lock(lock_file)
    dependency_lock_file_sha256 = _sha256_file(lock_file)
    if _GIT_COMMIT_RE.fullmatch(implementation_commit) is None:
        raise ValueError("Audit implementation commit must be a lowercase Git identity.")
    runtime = validate_preinstalled_cuda_stack(lock, hardware_gate=hardware_gate)
    receipt_path = _bootstrap_receipt_path(bootstrap_root, implementation_commit)
    install = lock["notebook_install"]
    expected_versions = _expected_locked_versions(lock)

    records = _discover_distribution_records()
    locked_provenance, _ = _validate_locked_closure(
        expected_versions, records, include_lpips=False
    )
    lpips_records = list(records.get("lpips", []))
    lpips_versions = sorted({record["version"] for record in lpips_records})
    if lpips_versions and lpips_versions != [install["version"]]:
        raise RuntimeProfileMismatch(
            "installed_lpips_identity_changed",
            {"lpips_metadata_versions": lpips_versions},
        )

    pip_install_invoked = False
    force_reinstall_invoked = False
    if not lpips_records:
        if receipt_path.exists():
            raise RuntimeError(
                "LPIPS bootstrap receipt exists but the distribution is absent."
            )
        bootstrap_state = "installed_from_absent"
        pip_install_invoked = True
        _run_lpips_install(install, force_reinstall=False, run=run)
    elif receipt_path.exists():
        bootstrap_state = "reused_verified"
    else:
        bootstrap_state = "interrupted_installation_reverified_by_forced_reinstall"
        pip_install_invoked = True
        force_reinstall_invoked = True
        _run_lpips_install(install, force_reinstall=True, run=run)

    if pip_install_invoked:
        _remove_lpips_imports()
        records = _discover_distribution_records()

    locked_provenance, _ = _validate_locked_closure(
        expected_versions, records, include_lpips=True
    )
    lpips_tree = _active_lpips_package_tree(records)
    active_lpips_import = locked_provenance["lpips"]["active_import"]
    expected_receipt_body = _expected_bootstrap_receipt_body(
        implementation_commit=implementation_commit,
        dependency_lock_file_sha256=dependency_lock_file_sha256,
        install=install,
        runtime=runtime,
        active_lpips_import=active_lpips_import,
        installed_package_tree=lpips_tree,
    )
    if receipt_path.exists():
        bootstrap_receipt = _load_and_verify_bootstrap_receipt(
            receipt_path, expected_receipt_body
        )
    else:
        bootstrap_receipt = _write_bootstrap_receipt_atomic(
            receipt_path, expected_receipt_body
        )

    locked_names = {
        normalize_distribution_name(name) for name in expected_versions
    }
    complete_environment, unlocked_ambiguities = (
        build_complete_distribution_provenance(
            records, locked_normalized_names=locked_names
        )
    )
    locked_runtime_packages = {
        name: expected_versions[name] for name in sorted(expected_versions)
    }
    return {
        "contract_version": ENVIRONMENT_PROVENANCE_CONTRACT,
        "dependency_lock_contract_version": DEPENDENCY_LOCK_CONTRACT,
        "dependency_lock_file_sha256": dependency_lock_file_sha256,
        "lock_file_sha256": dependency_lock_file_sha256,
        "runtime_profile": runtime,
        "observed_runtime_profile": runtime,
        "locked_runtime_packages": locked_runtime_packages,
        "locked_distribution_provenance": locked_provenance,
        "complete_installed_distributions": complete_environment,
        "unlocked_distribution_ambiguities": unlocked_ambiguities,
        "notebook_installed_packages": (
            {"lpips": install["version"]} if pip_install_invoked else {}
        ),
        "lpips_install_artifact": {
            "filename": install["artifact_filename"],
            "url": install["artifact_url"],
            "sha256": install["artifact_sha256"],
        },
        "lpips_distribution_artifact": {
            "filename": install["artifact_filename"],
            "url": install["artifact_url"],
            "sha256": install["artifact_sha256"],
        },
        "lpips_bootstrap_state": bootstrap_state,
        "lpips_force_reinstall_invoked": force_reinstall_invoked,
        "lpips_bootstrap_receipt_sha256": bootstrap_receipt["receipt_sha256"],
        "lpips_bootstrap_receipt_file_sha256": _sha256_file(receipt_path),
        "lpips_installed_package_tree": lpips_tree,
        "pip_install_invoked": pip_install_invoked,
        "pip_no_deps": True,
        "pip_require_hashes": True,
        "torch_or_torchvision_reinstalled": False,
        "preinstalled_packages_mutated": False,
        "dependency_download_invoked": pip_install_invoked,
        "dependency_download_observed": pip_install_invoked,
        "drive_mount_invoked": False,
        "private_data_accessed": False,
        "inference_invoked": False,
        "training_invoked": False,
    }


__all__ = [
    "DEPENDENCY_LOCK_CONTRACT",
    "ENVIRONMENT_PROVENANCE_CONTRACT",
    "LPIPS_BOOTSTRAP_RECEIPT_CONTRACT",
    "LPIPS_WHEEL_SHA256",
    "LPIPS_WHEEL_URL",
    "RuntimeProfileMismatch",
    "build_complete_distribution_provenance",
    "load_dependency_lock",
    "normalize_distribution_name",
    "prepare_locked_environment",
    "validate_preinstalled_cuda_stack",
]
