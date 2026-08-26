"""A100 inference-only body for the sealed Stage-2 step-200 audit notebook."""

from __future__ import annotations

import gc
import importlib
import json
import os
import sys
from pathlib import Path


TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
RECOVERY_IMPLEMENTATION_COMMIT = "d3949c3591dfb5d1b5270b92af78360bf73e18aa"
CHECKPOINT_SHA256 = "09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021"
SELECTION_RECEIPT_FILE_SHA256 = "c8d73fec48815224fcb87333dfd093c15738cc41dce89c4fb8ccf2cd874ef828"
VALIDATION_PLAN_SHA256 = "3afca2bab6a440529f88e7c8d9a9294fed9ecbf07eea1e308ed0910e2ba16421"
SELECTION_RULE_SHA256 = "fd15be634185a29d5ddedec3f2d7a24527bf5e59a49731f101f62cafcf1b06d6"
A100_QUALIFICATION_RUN_FINGERPRINT = (
    "502a0989591cad3d09841d7deec841d41ac000d4c9e4ff314b53a8d4067ba5d7"
)
PROTOCOL_FILE_SHA256 = "3c11092a4a5e5342726947d705eca8fd8c52a70b82b96892529fe564c5f5f809"
PROTOCOL_SHA256 = "2cd8e17207175f6a8f1f11f8afd748beca11ae0f47a08a0b2e1529a2272274e4"
READINESS_FILE_SHA256 = "dc6695d3a9d9f69749af1421e92a3b008f240e147a59619957cd4888af71d7d2"
BANK_ARCHIVE_SHA256 = "78d323c02ceccdfcb054307da3c9e14575210869d22cade6c5ecd4afa4baf8d5"
BANK_TREE_SHA256 = "f9cb09bfa177a3e389f87f087b0d756a2709e2054559a39c85e8272d5e1cfaa3"
BANK_ARTIFACT_SHA256 = "8081ce89a0eac1522b4fb28cd7919de4a4ecf1d5af72552d141a0ee9b9944194"
DRIVE_ROOT = Path("/content/drive/MyDrive/MRIxFields2026")
GATE01_ROOT = DRIVE_ROOT / "Gate01Private_8012a3f"
OUTPUT_ROOT = DRIVE_ROOT / "UnifiedStage2_1ca2b4a_01"
STAGE2_ROOT = OUTPUT_ROOT / "stage2_unified_v7"
BANK_NAMESPACE = STAGE2_ROOT / "bank_8081ce89a0ea"
TRAINING_NAMESPACE = BANK_NAMESPACE / "implementation_82633d66e5ea"
PILOT_ROOT = TRAINING_NAMESPACE / "unified_full_objective_pilot_200"
PILOT_ATTEMPT = PILOT_ROOT / "scientific_attempts" / "attempt-0001"
CHECKPOINT_ROOT = PILOT_ATTEMPT / "checkpoints"
CHECKPOINT = CHECKPOINT_ROOT / "stage2_unified_full_step000000200.pt"
RESOLVED_CONFIG = PILOT_ROOT / "resolved_config.json"
SELECTION_RECEIPT = CHECKPOINT_ROOT / "stage2_unified_full_selection_step000000200.json"
RECOVERY_NAMESPACE = BANK_NAMESPACE / (
    "recovery_training_82633d66e5ea_operator_" + RECOVERY_IMPLEMENTATION_COMMIT[:12]
)
P0006_PROTOCOL = (
    RECOVERY_NAMESPACE
    / "stage2_gate01_p0006_development_validation_evaluation_only_protocol_v4.json"
)
EVALUATION_READINESS = RECOVERY_NAMESPACE / "stage2_long_run_evaluation_readiness_v4.json"
BANK_ARCHIVE = OUTPUT_ROOT / "photometry_factored_latent_bank_v2.tar"
PHOTOMETRY_ARTIFACT = OUTPUT_ROOT / "stage2_photometry_factorization_v1.json"
VAE_CHECKPOINT = DRIVE_ROOT / "vae_kl_vae_best.pt"
VAE_CONFIG = GATE01_ROOT / "stage1-run-c.yaml"
LOCAL_SCRATCH = Path("/content/stage2_gate01_recovery_v8_scratch")
LOCAL_BANK_ARCHIVE = LOCAL_SCRATCH / "photometry_factored_latent_bank_v2.tar"
LOCAL_BANK_ROOT = LOCAL_SCRATCH / "photometry_factored_latent_bank_v2"
AUDIT_NAMESPACE = BANK_NAMESPACE / (
    "step200_audit_training_82633d66e5ea_"
    "checkpoint_09b157d7d9b2_"
    "protocol_2cd8e1720717_"
    "implementation_" + AUDIT_IMPLEMENTATION_COMMIT[:12]
)
AUDIT_OUTPUT = AUDIT_NAMESPACE / "a100_p0006_inference_only"


source_dir = str(REPO_DIR / "src")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, source_dir)
for module_name in tuple(sys.modules):
    if module_name == "fieldbridge" or module_name.startswith("fieldbridge."):
        del sys.modules[module_name]
importlib.invalidate_caches()
if not isinstance(AUDIT_DEPENDENCY_PROVENANCE, dict):
    raise RuntimeError("Sealed dependency provenance is missing before Drive mount.")
if (
    AUDIT_DEPENDENCY_PROVENANCE.get("contract_version")
    != "stage2-step200-inference-environment-provenance-v2"
    or AUDIT_DEPENDENCY_PROVENANCE.get("dependency_lock_contract_version")
    != "stage2-step200-inference-audit-dependency-lock-v2"
    or AUDIT_DEPENDENCY_PROVENANCE.get("torch_or_torchvision_reinstalled") is not False
    or AUDIT_DEPENDENCY_PROVENANCE.get("preinstalled_packages_mutated") is not False
    or AUDIT_DEPENDENCY_PROVENANCE.get("pip_no_deps") is not True
    or AUDIT_DEPENDENCY_PROVENANCE.get("pip_require_hashes") is not True
    or set(AUDIT_DEPENDENCY_PROVENANCE.get("notebook_installed_packages", {}))
    - {"lpips"}
):
    raise RuntimeError("Sealed dependency provenance contract changed.")
for dependency_name in ("numpy", "scipy", "torch", "yaml", "matplotlib", "nibabel", "skimage", "lpips"):
    importlib.import_module(dependency_name)
import fieldbridge

if REPO_DIR not in Path(fieldbridge.__file__).resolve().parents:
    raise RuntimeError("FieldBridge import escaped the detached audit checkout.")
if git_text("status", "--porcelain=v1", "--untracked-files=all"):
    raise RuntimeError("Audit checkout changed before Drive mount.")
print(
    json.dumps(
        {
            "stage": "inference_audit_dependency_preflight",
            "status": "pass",
            "runtime_requirement": "NVIDIA A100-SXM4-80GB",
            "training_authorized": False,
            "observed_runtime_profile": AUDIT_DEPENDENCY_PROVENANCE[
                "observed_runtime_profile"
            ],
            "notebook_installed_packages": AUDIT_DEPENDENCY_PROVENANCE[
                "notebook_installed_packages"
            ],
            "lpips_bootstrap_state": AUDIT_DEPENDENCY_PROVENANCE[
                "lpips_bootstrap_state"
            ],
            "lpips_bootstrap_receipt_file_sha256": AUDIT_DEPENDENCY_PROVENANCE[
                "lpips_bootstrap_receipt_file_sha256"
            ],
            "unlocked_distribution_ambiguity_count": len(
                AUDIT_DEPENDENCY_PROVENANCE[
                    "unlocked_distribution_ambiguities"
                ]
            ),
            "lpips_distribution_artifact": AUDIT_DEPENDENCY_PROVENANCE[
                "lpips_distribution_artifact"
            ],
            "dependency_lock_file_sha256": AUDIT_DEPENDENCY_PROVENANCE[
                "lock_file_sha256"
            ],
            "pip_install_invoked": AUDIT_DEPENDENCY_PROVENANCE[
                "pip_install_invoked"
            ],
            "dependency_download_observed": AUDIT_DEPENDENCY_PROVENANCE[
                "dependency_download_observed"
            ],
        },
        sort_keys=True,
    ),
    flush=True,
)


import torch

if not torch.cuda.is_available():
    raise RuntimeError("Attach the required NVIDIA A100 80 GB runtime before this audit.")
gpu = torch.cuda.get_device_properties(0)
if str(gpu.name) != "NVIDIA A100-SXM4-80GB" or int(gpu.total_memory) < 79 * 1024**3:
    raise RuntimeError("The first owner inference qualification requires NVIDIA A100 80 GB.")
print(
    json.dumps(
        {
            "stage": "a100_identity_preflight",
            "status": "pass",
            "gpu_name": str(gpu.name),
            "gpu_total_memory_bytes": int(gpu.total_memory),
            "training_authorized": False,
        },
        sort_keys=True,
    ),
    flush=True,
)

from fieldbridge.evaluation.stage2_step200_lpips_audit import (
    cached_official_gate01_metric_fn,
    initialize_sealed_official_lpips,
)

SEALED_LPIPS = initialize_sealed_official_lpips(device="cuda")
LPIPS_METRIC_FN = cached_official_gate01_metric_fn(SEALED_LPIPS)
print(
    json.dumps(
        {
            "stage": "lpips_weight_and_state_preflight",
            "status": "pass",
            "lpips_construction_count": 1,
            "lpips_initialization_seconds": SEALED_LPIPS.provenance[
                "initialization_seconds"
            ],
            "alexnet_weight_downloaded": SEALED_LPIPS.provenance[
                "alexnet_weight_downloaded"
            ],
            "lpips_linear_weight_downloaded": False,
            "alexnet_weight_file_sha256": SEALED_LPIPS.provenance[
                "alexnet_weight_file_sha256"
            ],
            "lpips_linear_weight_file_sha256": SEALED_LPIPS.provenance[
                "lpips_linear_weight_file_sha256"
            ],
            "canonical_tensor_state_sha256": SEALED_LPIPS.provenance[
                "canonical_tensor_state_sha256"
            ],
            "private_data_accessed": False,
            "training_invoked": False,
        },
        sort_keys=True,
    ),
    flush=True,
)

from google.colab import drive

drive.mount("/content/drive")

from fieldbridge.data.photometry_factorization import sha256_file
from fieldbridge.evaluation.stage2_step200_inference_audit import (
    load_unified_step200_inference_runtime,
    preflight_frozen_stage1_run_c_config,
    preflight_reviewed_photometry_namespace_artifact,
    run_step200_p0006_inference_audit,
    verify_frozen_stage1_vae_bank_provenance,
    verify_reviewed_photometry_bank_provenance,
)
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    copy_verified_stage2_bank_tar_to_local,
    preflight_gate01_p0006_archive,
    preflight_stage2_local_disk_capacity,
    restore_verified_stage2_bank_tar,
    verify_completed_stage2_pilot_evidence,
)


def count_progress(payload):
    allowed = {
        "stage",
        "status",
        "mode",
        "bytes_processed",
        "total_bytes",
        "files_processed",
        "total_files",
        "case_count",
        "expected_case_count",
        "acquisition_node_count",
        "case_receipt_reused",
    }
    if set(payload) - allowed:
        raise ValueError("Inference audit progress attempted to expose forbidden fields.")
    for key, value in payload.items():
        if key in {"stage", "status", "mode"}:
            continue
        if isinstance(value, bool):
            if key != "case_receipt_reused":
                raise ValueError("Inference progress boolean appeared in a count field.")
        elif not isinstance(value, int) or value < 0:
            raise ValueError("Inference progress counts must be nonnegative integers.")
    print(json.dumps(payload, sort_keys=True), flush=True)


vae_config_preflight = preflight_frozen_stage1_run_c_config(VAE_CONFIG)
print(
    json.dumps(
        {
            "stage": "frozen_stage1_run_c_config_preflight",
            "status": "pass",
            **vae_config_preflight.sanitized_provenance(),
            "bank_copy_invoked": False,
            "bank_extraction_invoked": False,
            "checkpoint_deserialized": False,
            "model_constructed": False,
            "private_array_opened": False,
            "inference_invoked": False,
        },
        sort_keys=True,
    ),
    flush=True,
)


photometry_preflight = preflight_reviewed_photometry_namespace_artifact(
    PHOTOMETRY_ARTIFACT
)
print(
    json.dumps(
        {
            "stage": "frozen_photometry_namespace_preflight",
            "status": "pass",
            **photometry_preflight.sanitized_provenance(),
            "bank_copy_invoked": False,
            "bank_extraction_invoked": False,
            "checkpoint_deserialized": False,
            "model_constructed": False,
            "private_array_opened": False,
            "inference_invoked": False,
        },
        sort_keys=True,
    ),
    flush=True,
)


if sha256_file(P0006_PROTOCOL) != PROTOCOL_FILE_SHA256:
    raise RuntimeError("Pinned P:0006 protocol file identity changed.")
if sha256_file(EVALUATION_READINESS) != READINESS_FILE_SHA256:
    raise RuntimeError("Pinned evaluation-readiness file identity changed.")
metadata_preflight = preflight_gate01_p0006_archive(
    GATE01_ROOT,
    expected_gate01_result_sha256=(
        "454747cd3e4b1376855915244a7c40fe281b758150e86f584fbea96f94d531f5"
    ),
)
if (
    metadata_preflight.get("status") != "pass"
    or metadata_preflight.get("private_array_payloads_opened") != 0
    or metadata_preflight.get("expected_gate01_result_match") is not True
):
    raise RuntimeError("Gate 0.1 metadata-only preflight did not pass.")
print(
    json.dumps(
        {
            "stage": "gate01_metadata_only_preflight",
            "status": "pass",
            "inventory_entry_count": metadata_preflight[
                "normalized_inventory_entry_count"
            ],
            "private_arrays_opened": False,
        },
        sort_keys=True,
    ),
    flush=True,
)

capacity = preflight_stage2_local_disk_capacity(
    BANK_ARCHIVE,
    LOCAL_SCRATCH,
    local_archive_path=LOCAL_BANK_ARCHIVE,
    local_bank_root=LOCAL_BANK_ROOT,
    expected_extracted_bytes=12873486620,
)
print(json.dumps({"stage": "local_capacity", "status": "pass", **capacity.to_dict()}, sort_keys=True), flush=True)
local_archive = copy_verified_stage2_bank_tar_to_local(
    BANK_ARCHIVE,
    LOCAL_BANK_ARCHIVE,
    expected_archive_sha256=BANK_ARCHIVE_SHA256,
    progress_callback=count_progress,
)
bank_restore = restore_verified_stage2_bank_tar(
    local_archive,
    LOCAL_BANK_ROOT,
    expected_archive_sha256=BANK_ARCHIVE_SHA256,
    expected_tree_sha256=BANK_TREE_SHA256,
    expected_bank_artifact_sha256=BANK_ARTIFACT_SHA256,
    expected_file_count=3312,
    expected_total_bytes=12873486620,
    progress_callback=count_progress,
)
print(
    json.dumps(
        {
            "stage": "reviewed_bank_reuse_or_restore",
            "status": "pass",
            "archive_copied": local_archive.copied_from_source,
            "bank_extracted": bank_restore.restored_from_tar,
            "local_bank_reused": not bank_restore.restored_from_tar,
            "file_count": bank_restore.file_count,
            "total_bytes": bank_restore.total_bytes,
        },
        sort_keys=True,
    ),
    flush=True,
)

verified_photometry_provenance = verify_reviewed_photometry_bank_provenance(
    photometry_preflight,
    bank_dir=LOCAL_BANK_ROOT,
)
print(
    json.dumps(
        {
            "stage": "frozen_photometry_bank_provenance",
            "status": "pass",
            **verified_photometry_provenance.sanitized_provenance(),
            "checkpoint_deserialized": False,
            "model_constructed": False,
            "private_array_opened": False,
            "inference_invoked": False,
        },
        sort_keys=True,
    ),
    flush=True,
)

verified_vae_provenance = verify_frozen_stage1_vae_bank_provenance(
    vae_config_preflight,
    vae_checkpoint_path=VAE_CHECKPOINT,
    bank_dir=LOCAL_BANK_ROOT,
)
print(
    json.dumps(
        {
            "stage": "frozen_stage1_run_c_bank_provenance",
            "status": "pass",
            **verified_vae_provenance.sanitized_provenance(),
            "checkpoint_deserialized": False,
            "model_constructed": False,
            "private_array_opened": False,
            "inference_invoked": False,
        },
        sort_keys=True,
    ),
    flush=True,
)

completed = verify_completed_stage2_pilot_evidence(
    TRAINING_NAMESPACE,
    bank_dir=LOCAL_BANK_ROOT,
    expected_selection_receipt_path=SELECTION_RECEIPT,
    training_evidence_commit=TRAINING_EVIDENCE_COMMIT,
    expected_selection_receipt_file_sha256=SELECTION_RECEIPT_FILE_SHA256,
    expected_validation_plan_sha256=VALIDATION_PLAN_SHA256,
    expected_selection_rule_sha256=SELECTION_RULE_SHA256,
)
if (
    completed.get("status") != "pass"
    or completed.get("training_reused") is not True
    or completed.get("training_invoked") is not False
    or completed.get("checkpoint_file_sha256") != CHECKPOINT_SHA256
    or not isinstance(completed.get("qualification"), dict)
    or completed["qualification"].get("run_fingerprint")
    != A100_QUALIFICATION_RUN_FINGERPRINT
):
    raise RuntimeError("Completed step-200 evidence is incompatible; inference is forbidden.")
del completed
gc.collect()

runtime = load_unified_step200_inference_runtime(
    checkpoint_path=CHECKPOINT,
    resolved_config_path=RESOLVED_CONFIG,
    vae_config_path=VAE_CONFIG,
    vae_checkpoint_path=VAE_CHECKPOINT,
    photometry_artifact_path=PHOTOMETRY_ARTIFACT,
    bank_dir=LOCAL_BANK_ROOT,
    device="cuda",
    verified_vae_provenance=verified_vae_provenance,
    verified_photometry_provenance=verified_photometry_provenance,
)
result = run_step200_p0006_inference_audit(
    protocol_path=P0006_PROTOCOL,
    evaluation_readiness_path=EVALUATION_READINESS,
    runtime=runtime,
    output_dir=AUDIT_OUTPUT,
    audit_implementation_commit=AUDIT_IMPLEMENTATION_COMMIT,
    require_a100=True,
    progress_callback=count_progress,
    metric_fn=LPIPS_METRIC_FN,
    dependency_provenance=AUDIT_DEPENDENCY_PROVENANCE,
    lpips_provenance=SEALED_LPIPS.provenance,
    lpips_integrity_verifier=SEALED_LPIPS.verify_unchanged,
)
print(
    json.dumps(
        {
            "stage": "stage2_step200_p0006_inference_audit",
            "status": "pass",
            "case_count": result["case_count"],
            "training_invoked": False,
            "gradients_enabled": False,
            "optimizer_loaded": False,
            "P0006_training_or_model_selection_use": False,
            "population_or_generalization_claims_authorized": False,
            "long_run_training_authorized": False,
            "dependency_download_observed": result[
                "dependency_download_observed"
            ],
            "alexnet_weight_downloaded": result["alexnet_weight_downloaded"],
            "lpips_initialization_seconds": result[
                "lpips_initialization_seconds"
            ],
            "one_case_inference_seconds": result["one_case_inference_seconds"],
            "projected_60_case_inference_seconds": result[
                "projected_60_case_inference_seconds"
            ],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
            "final_stop": "STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION",
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
print("STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION", flush=True)
