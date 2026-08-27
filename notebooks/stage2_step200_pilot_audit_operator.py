"""CPU-only body for the sealed Stage-2 step-200 pilot evidence audit notebook."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
RECOVERY_IMPLEMENTATION_COMMIT = "d3949c3591dfb5d1b5270b92af78360bf73e18aa"
DRIVE_ROOT = Path("/content/drive/MyDrive/MRIxFields2026")
OUTPUT_ROOT = DRIVE_ROOT / "UnifiedStage2_1ca2b4a_01"
STAGE2_ROOT = OUTPUT_ROOT / "stage2_unified_v7"
BANK_NAMESPACE = STAGE2_ROOT / "bank_8081ce89a0ea"
TRAINING_NAMESPACE = BANK_NAMESPACE / "implementation_82633d66e5ea"
PILOT_ROOT = TRAINING_NAMESPACE / "unified_full_objective_pilot_200"
PILOT_ATTEMPT = PILOT_ROOT / "scientific_attempts" / "attempt-0001"
CHECKPOINT_ROOT = PILOT_ATTEMPT / "checkpoints"
CHECKPOINT = CHECKPOINT_ROOT / "stage2_unified_full_step000000200.pt"
HISTORY = PILOT_ATTEMPT / "history.jsonl"
RESOLVED_CONFIG = PILOT_ROOT / "resolved_config.json"
SELECTION_RECEIPT = CHECKPOINT_ROOT / "stage2_unified_full_selection_step000000200.json"
VALIDATION_PLAN = CHECKPOINT_ROOT / "stage2_unified_validation_plan_v2.json"
A100_QUALIFICATION = (
    TRAINING_NAMESPACE
    / "a100_full_objective_gate_1"
    / "scientific_attempts"
    / "attempt-0001"
    / "checkpoints"
    / "stage2_unified_a100_qualification_only_receipt_v1.json"
)
RECOVERY_NAMESPACE = BANK_NAMESPACE / (
    "recovery_training_82633d66e5ea_operator_" + RECOVERY_IMPLEMENTATION_COMMIT[:12]
)
RECOVERY_RECEIPT = RECOVERY_NAMESPACE / "stage2_gate01_recovery_receipt_v2.json"
AUDIT_NAMESPACE = BANK_NAMESPACE / (
    "step200_audit_training_82633d66e5ea_"
    "checkpoint_09b157d7d9b2_"
    "implementation_" + AUDIT_IMPLEMENTATION_COMMIT[:12]
)
AUDIT_OUTPUT = AUDIT_NAMESPACE / "cpu_pilot_evidence"


source_dir = str(REPO_DIR / "src")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, source_dir)
for module_name in tuple(sys.modules):
    if module_name == "fieldbridge" or module_name.startswith("fieldbridge."):
        del sys.modules[module_name]
importlib.invalidate_caches()
dependency_versions = {}
for dependency_name in ("numpy", "torch", "matplotlib"):
    dependency = importlib.import_module(dependency_name)
    dependency_versions[dependency_name] = str(
        getattr(dependency, "__version__", "available")
    )
import fieldbridge

if REPO_DIR not in Path(fieldbridge.__file__).resolve().parents:
    raise RuntimeError("FieldBridge import escaped the detached audit checkout.")
if git_text("status", "--porcelain=v1", "--untracked-files=all"):
    raise RuntimeError("Audit checkout changed before Drive mount.")
print(
    json.dumps(
        {
            "stage": "cpu_audit_dependency_preflight",
            "status": "pass",
            "runtime_recommendation": "CPU High-RAM",
            "gpu_required": False,
            "gpu_used": False,
            "packages_installed_or_downloaded": False,
            "dependencies": dependency_versions,
        },
        sort_keys=True,
    ),
    flush=True,
)


from google.colab import drive

drive.mount("/content/drive")

from fieldbridge.evaluation.stage2_step200_pilot_audit import (
    Step200PilotAuditInputs,
    run_step200_pilot_evidence_audit,
)


inputs = Step200PilotAuditInputs(
    checkpoint=CHECKPOINT,
    history_jsonl=HISTORY,
    resolved_config=RESOLVED_CONFIG,
    selection_receipt=SELECTION_RECEIPT,
    validation_plan=VALIDATION_PLAN,
    a100_qualification_receipt=A100_QUALIFICATION,
    recovery_receipt=RECOVERY_RECEIPT,
)
result = run_step200_pilot_evidence_audit(
    inputs,
    output_dir=AUDIT_OUTPUT,
    audit_implementation_commit=AUDIT_IMPLEMENTATION_COMMIT,
    step20_evidence_independent=True,
)
print(
    json.dumps(
        {
            "stage": "stage2_step200_cpu_pilot_evidence_audit",
            "status": "pass",
            "training_row_count": result["training_row_count"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "run_fingerprint": result["run_fingerprint"],
            "convergence_declared": False,
            "training_invoked": False,
            "inference_invoked": False,
            "bank_opened": False,
            "private_arrays_opened": False,
            "gpu_used": False,
            "long_run_training_authorized": False,
            "next_action": "REVIEW_SANITIZED_STEP200_PILOT_EVIDENCE",
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
