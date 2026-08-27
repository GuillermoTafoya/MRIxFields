from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from fieldbridge.evaluation import mrixfields2026_official as official
from fieldbridge.evaluation import stage2_step200_inference_audit as inference_audit
from fieldbridge.evaluation import stage2_step200_lpips_audit as sealed_lpips


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/stage2_step200_inference_audit_colab.ipynb"
LOCK = ROOT / "notebooks/stage2_step200_inference_audit_dependency_lock.json"
ENVIRONMENT = ROOT / "notebooks/stage2_step200_inference_audit_environment.py"


def _environment_module():
    spec = importlib.util.spec_from_file_location("audit_environment_test", ENVIRONMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(payload["cells"][0]["source"])


def test_wrong_gpu_stops_before_every_downstream_external_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []

    def probe(command, **kwargs):
        del kwargs
        calls.append(list(command))
        if command[0] != "nvidia-smi":
            raise AssertionError("a downstream command ran after the failed GPU gate")
        return subprocess.CompletedProcess(
            command, 0, "580.82.07, NVIDIA L4, 23034, 22000\n", ""
        )

    monkeypatch.setattr(subprocess, "run", probe)
    source = _notebook_source().replace(
        "__AUDIT_IMPLEMENTATION_COMMIT__", "a" * 40
    )
    prefix = source.split("A100_HARDWARE_GATE = standard_library_a100_gate()", 1)[0]
    prefix += "A100_HARDWARE_GATE = standard_library_a100_gate()\n"
    with pytest.raises(RuntimeError, match="NVIDIA A100 80 GB"):
        exec(compile(prefix, "sealed-notebook-gate", "exec"), {})
    assert calls == [
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ]
    receipt = json.loads(capsys.readouterr().out.strip())
    assert receipt["pip_install_invoked"] is False
    assert receipt["dependency_download_invoked"] is False
    assert receipt["model_weight_download_invoked"] is False
    assert receipt["git_clone_or_fetch_invoked"] is False
    assert receipt["drive_mount_invoked"] is False
    assert receipt["bank_accessed"] is False
    assert receipt["checkpoint_loaded"] is False
    assert receipt["private_data_accessed"] is False
    assert receipt["lpips_constructed"] is False
    assert receipt["inference_invoked"] is False
    assert receipt["training_invoked"] is False


def test_hardware_gate_records_authenticated_gpu_and_driver(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def probe(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            "580.82.07, NVIDIA A100-SXM4-80GB, 81920, 81153\n",
            "",
        )

    monkeypatch.setattr(subprocess, "run", probe)
    source = _notebook_source().replace(
        "__AUDIT_IMPLEMENTATION_COMMIT__", "a" * 40
    )
    prefix = source.split("A100_HARDWARE_GATE = standard_library_a100_gate()", 1)[0]
    prefix += "A100_HARDWARE_GATE = standard_library_a100_gate()\n"
    namespace: dict[str, object] = {}
    exec(compile(prefix, "sealed-notebook-gate", "exec"), namespace)
    receipt = json.loads(capsys.readouterr().out.strip())
    assert receipt["status"] == "pass"
    assert receipt["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert receipt["nvidia_driver"] == "580.82.07"
    assert receipt["gpu_total_memory_mib"] == 81920
    assert receipt["gpu_free_memory_mib"] == 81153
    assert receipt["pip_install_invoked"] is False
    assert receipt["git_clone_or_fetch_invoked"] is False


def _hardware_gate(driver: str = "580.82.07") -> dict[str, object]:
    return {
        "stage": "standard_library_a100_80gb_gate",
        "status": "pass",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "nvidia_driver": driver,
    }


def _install_authenticated_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    torch_version: str = "2.11.0+cu128",
    torchvision_version: str = "0.26.0+cu128",
    torch_cuda: str = "12.8",
    cuda_available: bool = True,
) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = torch_version
    fake_torch.version = SimpleNamespace(cuda=torch_cuda)
    fake_torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    fake_vision = ModuleType("torchvision")
    fake_vision.__version__ = torchvision_version
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_vision)


def _set_python_identity(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    implementation: str = "CPython",
    cache_tag: str = "cpython-313",
    major_minor: list[int] | None = None,
    patch: str = "3.13.15",
) -> None:
    monkeypatch.setattr(
        module,
        "_python_runtime_identity",
        lambda: {
            "python_implementation": implementation,
            "python_cache_tag": cache_tag,
            "python_major_minor": major_minor or [3, 13],
            "python_patch": patch,
        },
    )


def test_dependency_lock_v2_is_exact_and_only_lpips_is_installable() -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    assert lock["contract_version"].endswith("-v2")
    assert lock["accepted_runtime_profile"] == {
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "observed_driver": "580.82.07",
        "observed_python_patch": "3.13.15",
        "python_cache_tag": "cpython-313",
        "python_implementation": "CPython",
        "python_major_minor": [3, 13],
        "torch": "2.11.0+cu128",
        "torch_cuda": "12.8",
        "torchvision": "0.26.0+cu128",
    }
    assert lock["preinstalled_runtime_packages"] == {
        "PyYAML": "6.0.3",
        "matplotlib": "3.10.0",
        "nibabel": "5.4.2",
        "numpy": "2.1.3",
        "scikit-image": "0.25.2",
        "scipy": "1.16.3",
    }
    assert lock["notebook_install"] == {
        "artifact_filename": "lpips-0.1.4-py3-none-any.whl",
        "artifact_sha256": module.LPIPS_WHEEL_SHA256,
        "artifact_url": module.LPIPS_WHEEL_URL,
        "distribution": "lpips",
        "version": "0.1.4",
    }
    assert lock["provenance_only_unqualified_profiles"][0]["accepted"] is False
    assert lock["provenance_only_unqualified_profiles"][0]["python"] == "3.12.13"


@pytest.mark.parametrize("patch", ["3.13.15", "3.13.16"])
def test_authenticated_profile_accepts_python_patch_variation_and_records_it(
    patch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    _install_authenticated_modules(monkeypatch)
    _set_python_identity(monkeypatch, module, patch=patch)
    observed = module.validate_preinstalled_cuda_stack(
        lock, hardware_gate=_hardware_gate()
    )
    assert observed["python_patch"] == patch
    assert observed["python_patch_matches_observed_profile"] is (patch == "3.13.15")
    assert observed["nvidia_driver"] == "580.82.07"
    assert observed["torch_cuda"] == "12.8"


@pytest.mark.parametrize(
    ("implementation", "cache_tag", "major_minor"),
    [
        ("CPython", "cpython-312", [3, 12]),
        ("CPython", "cpython-314", [3, 14]),
        ("PyPy", "pypy313", [3, 13]),
        ("CPython", "cpython-313-custom", [3, 13]),
    ],
)
def test_other_python_abi_profiles_fail_closed(
    implementation: str,
    cache_tag: str,
    major_minor: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    _set_python_identity(
        monkeypatch,
        module,
        implementation=implementation,
        cache_tag=cache_tag,
        major_minor=major_minor,
    )
    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.validate_preinstalled_cuda_stack(
            lock, hardware_gate=_hardware_gate()
        )
    receipt = caught.value.receipt
    assert receipt["reason"] == "python_abi_identity_changed"
    assert all(receipt[key] is False for key in module._NO_ACTION_FIELDS)


@pytest.mark.parametrize(
    ("torch_version", "torchvision_version", "torch_cuda"),
    [
        ("2.11.1+cu128", "0.26.0+cu128", "12.8"),
        ("2.11.0+cu128", "0.26.1+cu128", "12.8"),
        ("2.11.0+cu128", "0.26.0+cu128", "12.9"),
    ],
)
def test_torch_torchvision_or_cuda_mismatch_fails_closed(
    torch_version: str,
    torchvision_version: str,
    torch_cuda: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    _set_python_identity(monkeypatch, module)
    _install_authenticated_modules(
        monkeypatch,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        torch_cuda=torch_cuda,
    )
    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.validate_preinstalled_cuda_stack(
            lock, hardware_gate=_hardware_gate()
        )
    assert caught.value.receipt["reason"] == (
        "torch_torchvision_cuda_identity_changed"
    )
    assert all(
        caught.value.receipt[key] is False for key in module._NO_ACTION_FIELDS
    )


def test_cuda_unavailable_after_hardware_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    _set_python_identity(monkeypatch, module)
    _install_authenticated_modules(monkeypatch, cuda_available=False)
    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.validate_preinstalled_cuda_stack(
            lock, hardware_gate=_hardware_gate()
        )
    assert caught.value.receipt["reason"] == (
        "cuda_visibility_changed_after_hardware_gate"
    )


def test_driver_variation_is_recorded_without_changing_the_accepted_cuda_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    _set_python_identity(monkeypatch, module)
    _install_authenticated_modules(monkeypatch)
    observed = module.validate_preinstalled_cuda_stack(
        lock, hardware_gate=_hardware_gate("580.90.01")
    )
    assert observed["nvidia_driver"] == "580.90.01"
    assert observed["observed_profile_driver"] == "580.82.07"
    assert observed["driver_matches_observed_profile"] is False


class _FakeDistribution:
    def __init__(self, name: str, version: str, root: Path, files: list[Path]) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._root = root
        self.files = files

    def locate_file(self, relative) -> Path:
        return self._root / Path(str(relative))


def _record(distribution: _FakeDistribution, module) -> dict[str, object]:
    root = distribution._root.resolve()
    return {
        "_distribution": distribution,
        "_source_root": root,
        "normalized_name": module.normalize_distribution_name(distribution.metadata["Name"]),
        "declared_metadata_name": distribution.metadata["Name"],
        "version": distribution.version,
        "source_root_sha256": module._source_root_sha256(root),
    }


def _make_distribution(
    tmp_path: Path,
    module,
    *,
    distribution_name: str,
    import_name: str,
    version: str,
    root_name: str,
) -> tuple[_FakeDistribution, ModuleType, dict[str, object]]:
    root = tmp_path / root_name
    package = root / import_name
    package.mkdir(parents=True)
    module_file = package / "__init__.py"
    module_file.write_text("# synthetic installed package\n", encoding="utf-8")
    installed_module = ModuleType(import_name)
    installed_module.__file__ = str(module_file)
    installed_module.__version__ = version
    distribution = _FakeDistribution(
        distribution_name, version, root, [Path(import_name) / "__init__.py"]
    )
    return distribution, installed_module, _record(distribution, module)


def test_distribution_name_normalization_is_pep503_style() -> None:
    module = _environment_module()
    assert {
        module.normalize_distribution_name(value)
        for value in ["Example-Pkg", "example_pkg", "EXAMPLE.PKG", "example...pkg"]
    } == {"example-pkg"}


@pytest.mark.parametrize("name", ["cryptography", "arbitrary_unused_package"])
def test_unlocked_conflicting_distributions_are_provenance_not_failure(name: str) -> None:
    module = _environment_module()
    normalized = module.normalize_distribution_name(name)
    records = {
        normalized: [
            {"declared_metadata_name": name, "version": "1.0", "source_root_sha256": "1" * 64},
            {"declared_metadata_name": name.upper(), "version": "2.0", "source_root_sha256": "2" * 64},
        ]
    }
    complete, ambiguities = module.build_complete_distribution_provenance(
        records, locked_normalized_names={"numpy"}
    )
    assert [item["version"] for item in complete[normalized]] == ["1.0", "2.0"]
    assert ambiguities == [{"normalized_name": normalized, "distribution_count": 2, "versions": ["1.0", "2.0"]}]


def test_complete_distribution_provenance_is_discovery_order_independent(tmp_path: Path) -> None:
    module = _environment_module()
    first = _FakeDistribution("Crypto_graphy", "2", tmp_path / "z", [])
    second = _FakeDistribution("cryptography", "1", tmp_path / "a", [])
    first._root.mkdir()
    second._root.mkdir()
    left = module._discover_distribution_records([first, second])
    right = module._discover_distribution_records([second, first])
    assert module.build_complete_distribution_provenance(left, locked_normalized_names=set()) == module.build_complete_distribution_provenance(right, locked_normalized_names=set())


def test_locked_conflicting_metadata_version_fails_before_import(tmp_path: Path) -> None:
    module = _environment_module()
    one, _, one_record = _make_distribution(tmp_path, module, distribution_name="numpy", import_name="numpy", version="2.1.3", root_name="one")
    two = _FakeDistribution("NumPy", "2.1.4", one._root, one.files)
    with pytest.raises(RuntimeError, match="metadata version changed"):
        module._validate_locked_distribution("numpy", "2.1.3", {"numpy": [one_record, _record(two, module)]})


def test_same_version_locked_duplicates_require_matching_active_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _environment_module()
    first, active, first_record = _make_distribution(tmp_path, module, distribution_name="PyYAML", import_name="yaml", version="6.0.3", root_name="yaml")
    duplicate = _FakeDistribution("py_yaml", "6.0.3", first._root, first.files)
    monkeypatch.setitem(sys.modules, "yaml", active)
    provenance, owners = module._validate_locked_distribution("PyYAML", "6.0.3", {"pyyaml": [first_record, _record(duplicate, module)]})
    assert provenance["metadata_entry_count"] == 2
    assert provenance["active_import"]["import_name"] == "yaml"
    assert len(owners) == 2


def test_shadowed_locked_active_import_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _environment_module()
    _, _, record = _make_distribution(tmp_path, module, distribution_name="scikit-image", import_name="skimage", version="0.25.2", root_name="metadata")
    shadow = tmp_path / "shadow" / "skimage" / "__init__.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("# shadow\n", encoding="utf-8")
    active = ModuleType("skimage")
    active.__file__ = str(shadow)
    active.__version__ = "0.25.2"
    monkeypatch.setitem(sys.modules, "skimage", active)
    with pytest.raises(RuntimeError, match="shadowed"):
        module._validate_locked_distribution("scikit-image", "0.25.2", {"scikit-image": [record]})


def test_same_version_metadata_with_changed_active_version_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _environment_module()
    _, active, record = _make_distribution(
        tmp_path,
        module,
        distribution_name="numpy",
        import_name="numpy",
        version="2.1.3",
        root_name="numpy",
    )
    active.__version__ = "2.1.4"
    monkeypatch.setitem(sys.modules, "numpy", active)
    with pytest.raises(RuntimeError, match="active import version changed"):
        module._validate_locked_distribution(
            "numpy", "2.1.3", {"numpy": [record]}
        )


def test_environment_implementation_has_no_unlocked_package_allowlist() -> None:
    assert "cryptography" not in ENVIRONMENT.read_text(encoding="utf-8").lower()


def test_installed_lpips_tree_hash_covers_unlisted_package_files(
    tmp_path: Path,
) -> None:
    module = _environment_module()
    distribution, _, record = _make_distribution(
        tmp_path,
        module,
        distribution_name="lpips",
        import_name="lpips",
        version="0.1.4",
        root_name="site-packages",
    )
    extra = distribution._root / "lpips" / "injected.py"
    extra.write_text("first\n", encoding="utf-8")
    before = module._installed_lpips_package_tree(record)
    assert before["file_count"] == 2
    extra.write_text("changed\n", encoding="utf-8")
    after = module._installed_lpips_package_tree(record)
    assert before["tree_sha256"] != after["tree_sha256"]


def _transaction_runtime(lock: dict[str, object]) -> dict[str, object]:
    profile = lock["accepted_runtime_profile"]
    return {
        "python_implementation": profile["python_implementation"],
        "python_cache_tag": profile["python_cache_tag"],
        "python_major_minor": profile["python_major_minor"],
        "python_patch": profile["observed_python_patch"],
        "torch": profile["torch"],
        "torchvision": profile["torchvision"],
        "torch_cuda": profile["torch_cuda"],
    }


def _configure_transaction_fixture(module, monkeypatch: pytest.MonkeyPatch, state: dict[str, dict[str, list[dict[str, object]]]], *, tree_sha256: str = "b" * 64) -> None:
    monkeypatch.setattr(module, "validate_preinstalled_cuda_stack", lambda value, hardware_gate: _transaction_runtime(value))
    monkeypatch.setattr(module, "_discover_distribution_records", lambda: state["records"])

    def validate(expected, records, include_lpips):
        del records
        result = {}
        for name, version in expected.items():
            if name == "lpips" and not include_lpips:
                continue
            result[name] = {
                "normalized_name": module.normalize_distribution_name(name),
                "expected_version": version,
                "metadata_entry_count": 1,
                "metadata_observations": [],
                "active_import": {
                    "distribution_name": name,
                    "import_name": module._LOCKED_DISTRIBUTION_IMPORTS[name],
                    "active_version": version,
                    "version_source": "synthetic",
                    "module_file_sha256": "a" * 64,
                    "owning_distribution_count": 1,
                    "owning_source_root_sha256": ["c" * 64],
                },
            }
        return result, {}

    monkeypatch.setattr(module, "_validate_locked_closure", validate)
    monkeypatch.setattr(module, "_active_lpips_package_tree", lambda records: {"tree_sha256": tree_sha256, "file_count": 12, "total_bytes": 3456})
    monkeypatch.setattr(module, "_remove_lpips_imports", lambda: None)


def _lpips_record(module, version: str = "0.1.4") -> dict[str, object]:
    return {"declared_metadata_name": "lpips", "normalized_name": "lpips", "version": version, "source_root_sha256": "c" * 64, "_distribution": object(), "_source_root": Path("synthetic")}


def _seed_valid_bootstrap(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    state: dict[str, object] = {"records": {}}
    _configure_transaction_fixture(module, monkeypatch, state)

    def run(command, **kwargs):
        assert kwargs == {"check": True}
        state["records"] = {"lpips": [_lpips_record(module)]}
        return subprocess.CompletedProcess(command, 0)

    arguments = {
        "hardware_gate": _hardware_gate(),
        "implementation_commit": "a" * 40,
        "bootstrap_root": tmp_path / "bootstrap",
    }
    module.prepare_locked_environment(LOCK, run=run, **arguments)
    receipt_path = (
        tmp_path
        / "bootstrap"
        / "implementation_aaaaaaaaaaaa"
        / "lpips-bootstrap-receipt-v1.json"
    )
    assert receipt_path.is_file()
    return state, arguments, receipt_path


def test_lpips_absent_installs_once_seals_receipt_then_reuses_without_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _environment_module()
    state = {"records": {}}
    _configure_transaction_fixture(module, monkeypatch, state)
    commands: list[list[str]] = []
    requirements: list[str] = []

    def run(command, **kwargs):
        assert kwargs == {"check": True}
        commands.append(list(command))
        requirements.append(Path(command[-1]).read_text(encoding="utf-8"))
        state["records"] = {"lpips": [_lpips_record(module)]}
        return subprocess.CompletedProcess(command, 0)

    arguments = {"hardware_gate": _hardware_gate(), "implementation_commit": "a" * 40, "bootstrap_root": tmp_path / "bootstrap"}
    first = module.prepare_locked_environment(LOCK, run=run, **arguments)
    assert first["lpips_bootstrap_state"] == "installed_from_absent"
    assert first["pip_install_invoked"] is True
    assert first["notebook_installed_packages"] == {"lpips": "0.1.4"}
    assert len(commands) == 1
    assert "--force-reinstall" not in commands[0]
    assert "--no-deps" in commands[0]
    assert "--only-binary=:all:" in commands[0]
    assert "--require-hashes" in commands[0]
    assert module.LPIPS_WHEEL_URL in requirements[0]
    assert module.LPIPS_WHEEL_SHA256 in requirements[0]
    second = module.prepare_locked_environment(LOCK, run=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError((args, kwargs))), **arguments)
    assert second["lpips_bootstrap_state"] == "reused_verified"
    assert second["pip_install_invoked"] is False
    assert first["lpips_bootstrap_receipt_sha256"] == second["lpips_bootstrap_receipt_sha256"]


def test_interrupted_exact_lpips_without_receipt_forces_one_verified_reinstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _environment_module()
    state = {"records": {"lpips": [_lpips_record(module)]}}
    _configure_transaction_fixture(module, monkeypatch, state)
    commands: list[list[str]] = []

    def run(command, **kwargs):
        assert kwargs == {"check": True}
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    receipt = module.prepare_locked_environment(LOCK, hardware_gate=_hardware_gate(), implementation_commit="d" * 40, bootstrap_root=tmp_path / "bootstrap", run=run)
    assert len(commands) == 1
    assert "--force-reinstall" in commands[0]
    assert receipt["lpips_force_reinstall_invoked"] is True
    assert receipt["lpips_bootstrap_state"].startswith("interrupted_installation")


@pytest.mark.parametrize(
    "field",
    ["implementation_commit", "dependency_lock_file_sha256", "wheel", "package_tree"],
)
def test_hash_valid_but_substituted_bootstrap_identity_fails_closed(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _environment_module()
    _, arguments, receipt_path = _seed_valid_bootstrap(
        module, monkeypatch, tmp_path
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if field == "implementation_commit":
        payload[field] = "b" * 40
    elif field == "dependency_lock_file_sha256":
        payload[field] = "0" * 64
    elif field == "wheel":
        payload[field]["sha256"] = "0" * 64
    else:
        payload["installed_package_tree"]["tree_sha256"] = "0" * 64
    body = dict(payload)
    body.pop("receipt_sha256")
    payload["receipt_sha256"] = module._sha256_json(body)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt identity changed"):
        module.prepare_locked_environment(
            LOCK,
            run=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError((args, kwargs))
            ),
            **arguments,
        )


def test_bootstrap_receipt_self_hash_substitution_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _environment_module()
    _, arguments, receipt_path = _seed_valid_bootstrap(
        module, monkeypatch, tmp_path
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["receipt_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="self-hash changed"):
        module.prepare_locked_environment(
            LOCK,
            run=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError((args, kwargs))
            ),
            **arguments,
        )


def test_changed_active_lpips_tree_fails_against_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _environment_module()
    _, arguments, _ = _seed_valid_bootstrap(module, monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "_active_lpips_package_tree",
        lambda records: {
            "tree_sha256": "9" * 64,
            "file_count": 12,
            "total_bytes": 3456,
        },
    )
    with pytest.raises(RuntimeError, match="receipt identity changed"):
        module.prepare_locked_environment(
            LOCK,
            run=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError((args, kwargs))
            ),
            **arguments,
        )


def test_changed_lock_file_bytes_fail_against_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _environment_module()
    _, arguments, _ = _seed_valid_bootstrap(module, monkeypatch, tmp_path)
    alternate_lock = tmp_path / "same-contract-different-bytes.json"
    alternate_lock.write_text(
        json.dumps(json.loads(LOCK.read_text(encoding="utf-8")), indent=4) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="receipt identity changed"):
        module.prepare_locked_environment(
            alternate_lock,
            run=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError((args, kwargs))
            ),
            **arguments,
        )


def test_runtime_receipt_normalization_preserves_exact_resume_across_bootstrap_state() -> None:
    first = {
        "dependency_lock_file_sha256": "1" * 64,
        "lpips_bootstrap_receipt_sha256": "2" * 64,
        "lpips_installed_package_tree": {"tree_sha256": "3" * 64},
        "lpips_bootstrap_state": "installed_from_absent",
        "lpips_force_reinstall_invoked": False,
        "notebook_installed_packages": {"lpips": "0.1.4"},
        "dependency_download_invoked": True,
        "dependency_download_observed": True,
        "pip_install_invoked": True,
    }
    resumed = {
        **first,
        "lpips_bootstrap_state": "reused_verified",
        "notebook_installed_packages": {},
        "dependency_download_invoked": False,
        "dependency_download_observed": False,
        "pip_install_invoked": False,
    }
    assert inference_audit._stable_runtime_provenance(first) == (
        inference_audit._stable_runtime_provenance(resumed)
    )
    substituted = {**resumed, "dependency_lock_file_sha256": "9" * 64}
    assert inference_audit._stable_runtime_provenance(first) != (
        inference_audit._stable_runtime_provenance(substituted)
    )


def test_runtime_receipt_reuse_returns_original_bytes_after_bootstrap_reuse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dependency-runtime-receipt.json"
    first = {
        "dependency_lock_file_sha256": "1" * 64,
        "lpips_bootstrap_receipt_sha256": "2" * 64,
        "lpips_bootstrap_state": "interrupted_installation_reverified_by_forced_reinstall",
        "lpips_force_reinstall_invoked": True,
        "notebook_installed_packages": {"lpips": "0.1.4"},
        "dependency_download_invoked": True,
        "dependency_download_observed": True,
        "pip_install_invoked": True,
    }
    stored = inference_audit._seal_or_reuse_runtime_receipt(
        path, first, label="dependency_environment"
    )
    bytes_before = path.read_bytes()
    resumed = {
        **first,
        "lpips_bootstrap_state": "reused_verified",
        "lpips_force_reinstall_invoked": False,
        "notebook_installed_packages": {},
        "dependency_download_invoked": False,
        "dependency_download_observed": False,
        "pip_install_invoked": False,
    }
    reused = inference_audit._seal_or_reuse_runtime_receipt(
        path, resumed, label="dependency_environment"
    )
    assert reused == stored
    assert path.read_bytes() == bytes_before


def test_wrong_installed_lpips_is_not_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _environment_module()
    state = {"records": {"lpips": [_lpips_record(module, "0.1.3")]}}
    _configure_transaction_fixture(module, monkeypatch, state)
    calls = 0

    def forbidden_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError((args, kwargs))

    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.prepare_locked_environment(LOCK, hardware_gate=_hardware_gate(), implementation_commit="e" * 40, bootstrap_root=tmp_path / "bootstrap", run=forbidden_run)
    assert calls == 0
    assert caught.value.receipt["reason"] == "installed_lpips_identity_changed"


def test_wrong_runtime_profile_invokes_no_pip_or_downstream_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _environment_module()
    _set_python_identity(monkeypatch, module, cache_tag="cpython-312", major_minor=[3, 12])
    calls = 0

    def forbidden_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError((args, kwargs))

    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.prepare_locked_environment(LOCK, hardware_gate=_hardware_gate(), implementation_commit="f" * 40, bootstrap_root=tmp_path / "bootstrap", run=forbidden_run)
    assert calls == 0
    assert all(caught.value.receipt[key] is False for key in module._NO_ACTION_FIELDS)
    assert caught.value.receipt["observed_compatibility"]["python_cache_tag"] == "cpython-312"


def test_lpips_artifact_hash_substitution_fails_closed(
    tmp_path: Path,
) -> None:
    module = _environment_module()
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload["notebook_install"]["artifact_sha256"] = "0" * 64
    changed = tmp_path / "changed-lock.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="LPIPS distribution identity"):
        module.load_dependency_lock(changed)


class _FakeLPIPSNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return ((left - right) ** 2).mean().reshape(1, 1, 1, 1) * self.scale


def test_cached_lpips_is_constructed_once_and_matches_legacy_numerics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = 0

    class FakeLPIPSModule(ModuleType):
        def LPIPS(self, **kwargs):
            nonlocal constructions
            assert kwargs.get("net") == "alex"
            constructions += 1
            return _FakeLPIPSNetwork()

    monkeypatch.setitem(sys.modules, "lpips", FakeLPIPSModule("lpips"))
    prediction = np.linspace(0.0, 1.0, 8 * 8 * 3).reshape(8, 8, 3)
    target = np.flip(prediction, axis=0).copy()
    legacy = official.official_task3_lpips(prediction, target, device="cpu")
    with pytest.raises(RuntimeError, match="deterministic evaluation mode"):
        official.OfficialTask3LPIPSEvaluator(_FakeLPIPSNetwork(), device="cpu")
    cached = official.OfficialTask3LPIPSEvaluator(
        _FakeLPIPSNetwork().eval(), device="cpu"
    )
    values = [cached(prediction, target) for _ in range(14)]
    assert max(abs(value - legacy) for value in values) == 0.0
    assert constructions == 1
    assert cached.network.training is False
    assert not any(parameter.requires_grad for parameter in cached.network.parameters())


def test_sealed_initializer_constructs_one_network_and_records_weight_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alex = tmp_path / "alexnet-owt-7be5be79.pth"
    linear = tmp_path / "alex.pth"
    alex.write_bytes(b"alexnet")
    linear.write_bytes(b"linear")
    constructions = 0

    class FakeLPIPSModule(ModuleType):
        def LPIPS(self, **kwargs):
            nonlocal constructions
            constructions += 1
            assert kwargs == {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": False,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "eval_mode": True,
                "verbose": False,
            }
            return _FakeLPIPSNetwork().eval()

    weight = SimpleNamespace(url=sealed_lpips.ALEXNET_WEIGHT_URL)

    class FakeWeights:
        IMAGENET1K_V1 = weight
        DEFAULT = weight

    torchvision = ModuleType("torchvision")
    models = ModuleType("torchvision.models")
    models.AlexNet_Weights = FakeWeights
    monkeypatch.setitem(sys.modules, "lpips", FakeLPIPSModule("lpips"))
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setattr(sealed_lpips, "_alexnet_cache_path", lambda: alex)
    monkeypatch.setattr(sealed_lpips, "_lpips_linear_weight_path", lambda: linear)
    monkeypatch.setattr(
        sealed_lpips,
        "_sha256_file",
        lambda path: (
            sealed_lpips.ALEXNET_WEIGHT_SHA256
            if path == alex
            else sealed_lpips.LPIPS_LINEAR_WEIGHT_SHA256
        ),
    )
    sealed = sealed_lpips.initialize_sealed_official_lpips(device="cpu")
    reused = sealed_lpips.initialize_sealed_official_lpips(device="cpu")
    assert constructions == 1
    assert reused is sealed
    assert sealed.provenance["lpips_construction_count"] == 1
    assert sealed.provenance["torchvision_alexnet_weight_enum"] == (
        "AlexNet_Weights.IMAGENET1K_V1"
    )
    assert sealed.provenance["alexnet_weight_downloaded"] is False
    assert sealed.verify_unchanged()["unchanged"] is True


def test_canonical_lpips_state_hash_detects_substitution() -> None:
    network = _FakeLPIPSNetwork()
    before = sealed_lpips.canonical_tensor_state_sha256(network)
    network.scale.add_(1.0)
    after = sealed_lpips.canonical_tensor_state_sha256(network)
    assert before != after


def test_existing_alexnet_weight_substitution_fails_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "alexnet-owt-7be5be79.pth"
    path.write_bytes(b"substituted")
    monkeypatch.setattr(
        sealed_lpips.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network must not repair a substituted cache entry")
        ),
    )
    with pytest.raises(RuntimeError, match="substituted or changed"):
        sealed_lpips._ensure_alexnet_weight(path)


def test_case_loop_network_guard_fails_closed() -> None:
    with sealed_lpips.forbid_network_access():
        with pytest.raises(RuntimeError, match="Network access is forbidden"):
            __import__("socket").create_connection(("example.invalid", 443))
