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


def test_observed_environment_installs_only_hash_pinned_lpips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    installed: dict[str, str] = dict(lock["preinstalled_runtime_packages"])
    monkeypatch.setattr(
        module,
        "validate_preinstalled_cuda_stack",
        lambda value, hardware_gate: dict(value["accepted_runtime_profile"]),
    )
    monkeypatch.setattr(module, "_distribution_version", installed.get)
    monkeypatch.setattr(module.importlib.metadata, "distributions", lambda: [])
    commands: list[list[str]] = []

    def run(command, **kwargs):
        del kwargs
        commands.append(list(command))
        installed["lpips"] = "0.1.4"
        return subprocess.CompletedProcess(command, 0, "Downloading lpips wheel\n", "")

    receipt = module.prepare_locked_environment(
        LOCK, hardware_gate=_hardware_gate(), run=run
    )
    assert len(commands) == 1
    command = commands[0]
    assert "--no-deps" in command
    assert "--only-binary=:all:" in command
    assert "--require-hashes" in command
    assert "--no-input" in command
    assert not any(item.startswith("torch==") for item in command)
    assert not any(item.startswith("torchvision==") for item in command)
    assert sum("files.pythonhosted.org" in item for item in command) == 1
    assert any(
        f"#sha256={module.LPIPS_WHEEL_SHA256}" in item for item in command
    )
    for name in lock["preinstalled_runtime_packages"]:
        assert not any(name in item for item in command)
    assert receipt["notebook_installed_packages"] == {"lpips": "0.1.4"}
    assert receipt["pip_install_invoked"] is True
    assert receipt["pip_require_hashes"] is True
    assert receipt["torch_or_torchvision_reinstalled"] is False
    assert receipt["preinstalled_packages_mutated"] is False


def test_exact_preinstalled_lpips_is_verified_without_pip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    versions = {**lock["preinstalled_runtime_packages"], "lpips": "0.1.4"}
    monkeypatch.setattr(
        module,
        "validate_preinstalled_cuda_stack",
        lambda value, hardware_gate: dict(value["accepted_runtime_profile"]),
    )
    monkeypatch.setattr(module, "_distribution_version", versions.get)
    monkeypatch.setattr(module.importlib.metadata, "distributions", lambda: [])
    receipt = module.prepare_locked_environment(
        LOCK,
        hardware_gate=_hardware_gate(),
        run=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError((args, kwargs))
        ),
    )
    assert receipt["pip_install_invoked"] is False
    assert receipt["dependency_download_observed"] is False
    assert receipt["notebook_installed_packages"] == {}
    assert receipt["locked_runtime_packages"]["lpips"] == "0.1.4"


@pytest.mark.parametrize(
    ("distribution", "observed"),
    [("numpy", "2.1.4"), ("PyYAML", None)],
)
def test_preinstalled_package_change_fails_without_pip(
    distribution: str,
    observed: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    versions: dict[str, str | None] = dict(lock["preinstalled_runtime_packages"])
    versions[distribution] = observed
    versions["lpips"] = None
    monkeypatch.setattr(
        module,
        "validate_preinstalled_cuda_stack",
        lambda value, hardware_gate: dict(value["accepted_runtime_profile"]),
    )
    monkeypatch.setattr(module, "_distribution_version", versions.get)
    calls = 0

    def forbidden_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError((args, kwargs))

    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.prepare_locked_environment(
            LOCK, hardware_gate=_hardware_gate(), run=forbidden_run
        )
    assert calls == 0
    assert caught.value.receipt["reason"] == (
        "preinstalled_package_inventory_changed"
    )


def test_wrong_installed_lpips_is_not_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    versions = {**lock["preinstalled_runtime_packages"], "lpips": "0.1.3"}
    monkeypatch.setattr(
        module,
        "validate_preinstalled_cuda_stack",
        lambda value, hardware_gate: dict(value["accepted_runtime_profile"]),
    )
    monkeypatch.setattr(module, "_distribution_version", versions.get)
    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.prepare_locked_environment(
            LOCK,
            hardware_gate=_hardware_gate(),
            run=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError((args, kwargs))
            ),
        )
    assert caught.value.receipt["reason"] == "installed_lpips_identity_changed"


def test_wrong_runtime_profile_invokes_no_pip_or_downstream_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    _set_python_identity(
        monkeypatch, module, cache_tag="cpython-312", major_minor=[3, 12]
    )
    calls = 0

    def forbidden_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError((args, kwargs))

    with pytest.raises(module.RuntimeProfileMismatch) as caught:
        module.prepare_locked_environment(
            LOCK, hardware_gate=_hardware_gate(), run=forbidden_run
        )
    assert calls == 0
    assert all(
        caught.value.receipt[key] is False for key in module._NO_ACTION_FIELDS
    )
    assert caught.value.receipt["observed_compatibility"]["python_cache_tag"] == (
        "cpython-312"
    )


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
