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
        return subprocess.CompletedProcess(command, 0, "NVIDIA L4, 23034, 22000\n", "")

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
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ]
    receipt = json.loads(capsys.readouterr().out.strip())
    assert receipt["pip_install_invoked"] is False
    assert receipt["dependency_download_invoked"] is False
    assert receipt["model_weight_download_invoked"] is False
    assert receipt["drive_mount_invoked"] is False
    assert receipt["bank_accessed"] is False
    assert receipt["checkpoint_loaded"] is False
    assert receipt["private_data_accessed"] is False
    assert receipt["inference_invoked"] is False
    assert receipt["training_invoked"] is False


def test_dependency_lock_is_exact_closed_and_excludes_cuda_stack_install() -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    assert lock["preinstalled_cuda_stack"] == {
        "python": "3.12.13",
        "torch": "2.8.0+cu126",
        "torch_cuda": "12.6",
        "torchvision": "0.23.0+cu126",
    }
    assert set(lock["installed_by_notebook"]) == module._INSTALL_KEYS
    assert all(value and "*" not in value for value in lock["installed_by_notebook"].values())
    assert "torch" not in lock["installed_by_notebook"]
    assert "torchvision" not in lock["installed_by_notebook"]


def test_torch_torchvision_cuda_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.8.0+cu126"
    fake_torch.version = SimpleNamespace(cuda="12.6")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_vision = ModuleType("torchvision")
    fake_vision.__version__ = "0.23.0+cu126"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_vision)
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.13")
    assert module.validate_preinstalled_cuda_stack(lock)["torch_cuda"] == "12.6"
    fake_vision.__version__ = "0.24.0+cu126"
    with pytest.raises(RuntimeError, match="compatibility gate failed"):
        module.validate_preinstalled_cuda_stack(lock)
    fake_vision.__version__ = "0.23.0+cu126"
    fake_torch.__version__ = "2.9.0+cu126"
    with pytest.raises(RuntimeError, match="compatibility gate failed"):
        module.validate_preinstalled_cuda_stack(lock)
    fake_torch.__version__ = "2.8.0+cu126"
    fake_torch.version.cuda = "12.8"
    with pytest.raises(RuntimeError, match="compatibility gate failed"):
        module.validate_preinstalled_cuda_stack(lock)


def test_closed_installer_uses_only_exact_no_deps_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _environment_module()
    lock = module.load_dependency_lock(LOCK)
    installed: dict[str, str] = {}
    monkeypatch.setattr(
        module,
        "validate_preinstalled_cuda_stack",
        lambda value: dict(value["preinstalled_cuda_stack"]),
    )
    monkeypatch.setattr(
        module.importlib.metadata,
        "version",
        lambda name: installed.get(name, "changed"),
    )
    monkeypatch.setattr(module.importlib.metadata, "distributions", lambda: [])
    commands: list[list[str]] = []

    def run(command, **kwargs):
        del kwargs
        commands.append(list(command))
        installed.update(lock["installed_by_notebook"])
        return subprocess.CompletedProcess(command, 0, "", "")

    receipt = module.prepare_locked_environment(LOCK, run=run)
    assert len(commands) == 1
    command = commands[0]
    assert "--no-deps" in command
    assert "--only-binary=:all:" in command
    assert not any(item.startswith("torch==") for item in command)
    assert not any(item.startswith("torchvision==") for item in command)
    assert set(item for item in command if "==" in item) == {
        f"{name}=={version}"
        for name, version in lock["installed_by_notebook"].items()
    }
    assert receipt["torch_or_torchvision_reinstalled"] is False


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
