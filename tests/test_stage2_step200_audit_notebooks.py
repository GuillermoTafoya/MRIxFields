from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPU_NOTEBOOK = ROOT / "notebooks/stage2_step200_pilot_audit_colab.ipynb"
GPU_NOTEBOOK = ROOT / "notebooks/stage2_step200_inference_audit_colab.ipynb"
CPU_OPERATOR = ROOT / "notebooks/stage2_step200_pilot_audit_operator.py"
GPU_OPERATOR = ROOT / "notebooks/stage2_step200_inference_audit_operator.py"
TRAINING_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"


def _source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert len(payload["cells"]) == 1
    cell = payload["cells"][0]
    assert cell["cell_type"] == "code"
    assert cell["execution_count"] is None
    assert cell["outputs"] == []
    return "".join(cell["source"])


def test_both_notebooks_are_one_cell_unexecuted_fail_closed_templates():
    for path in (CPU_NOTEBOOK, GPU_NOTEBOOK):
        source = _source(path)
        assert TRAINING_COMMIT in source
        assert "__AUDIT_IMPLEMENTATION_COMMIT__" in source or re.search(
            r"AUDIT_IMPLEMENTATION_COMMIT = '[0-9a-f]{40}'", source
        )
        assert "detached" in source
        assert "checkout" in source
        assert "--detach" in source
        assert "status', '--porcelain" in source
        assert "merge-base', '--is-ancestor" in source
        assert "Protected training/model/data/config/package objects changed" in source
        assert "git', 'reset" not in source
        assert "git', 'clean" not in source
        assert "shutil.rmtree" not in source


def test_cpu_notebook_is_metadata_only_and_installs_nothing():
    notebook = _source(CPU_NOTEBOOK)
    operator = CPU_OPERATOR.read_text(encoding="utf-8")
    combined = notebook + operator
    assert "CPU High-RAM" in combined
    assert "packages_installed_or_downloaded" in combined
    assert "pip" not in combined
    assert "torch.cuda" not in operator
    assert "PhotometryFactoredLatentBankIndex" not in operator
    assert "Gate01InputManifest" not in operator
    assert "run_step200_pilot_evidence_audit" in operator
    assert "private_arrays_opened" in operator
    assert "training_invoked" in operator
    assert "inference_invoked" in operator


def test_inference_notebook_is_a100_streaming_recovery_only():
    notebook = _source(GPU_NOTEBOOK)
    operator = GPU_OPERATOR.read_text(encoding="utf-8")
    assert "NVIDIA A100 80 GB" in notebook + operator
    assert "iter_gate01_p0006_evaluation_cases" not in operator
    assert "run_step200_p0006_inference_audit" in operator
    assert "load_gate01_p0006_evaluation_protocol" not in operator
    assert "train-stage2-unified" not in operator
    assert "run_stage2_unified_train" not in operator
    assert "STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION" in operator
    assert "training_invoked" in operator
    assert "gradients_enabled" in operator
    assert "optimizer_loaded" in operator
    assert "long_run_training_authorized" in operator
    assert "/content/stage2_gate01_recovery_v8_scratch" in operator
    assert "P0007" not in operator
    assert "P0009" not in operator


def test_a100_gate_precedes_large_bank_copy_and_private_streaming():
    operator = GPU_OPERATOR.read_text(encoding="utf-8")
    gpu_gate = operator.index("torch.cuda.get_device_properties")
    bank_copy = operator.index("copy_verified_stage2_bank_tar_to_local(")
    runtime_load = operator.index("load_unified_step200_inference_runtime(")
    audit_run = operator.index("run_step200_p0006_inference_audit(")
    assert gpu_gate < bank_copy < runtime_load < audit_run
    assert operator.index("preflight_gate01_p0006_archive(") < bank_copy


def test_audit_namespaces_are_new_and_identity_keyed():
    cpu = CPU_OPERATOR.read_text(encoding="utf-8")
    gpu = GPU_OPERATOR.read_text(encoding="utf-8")
    for source in (cpu, gpu):
        assert "step200_audit_training_82633d66e5ea_" in source
        assert "checkpoint_09b157d7d9b2_" in source
        assert "AUDIT_IMPLEMENTATION_COMMIT[:12]" in source
    assert "protocol_2cd8e1720717_" in gpu
    assert "implementation_82633d66e5ea" in cpu
    assert "implementation_82633d66e5ea" in gpu


def test_operator_sources_contain_no_training_launch_or_long_run_authorization():
    combined = CPU_OPERATOR.read_text(encoding="utf-8") + GPU_OPERATOR.read_text(
        encoding="utf-8"
    )
    forbidden = (
        "train-stage2-unified",
        "run_stage2_unified_train",
        "long_run_training_authorized\": True",
        "population_or_generalization_claims_authorized\": True",
        "P0006_training_or_model_selection_use\": True",
    )
    assert all(item not in combined for item in forbidden)
