"""Recovery-only body for the unexecuted Stage-2 Gate 0.1 Colab notebook.

The bootstrap notebook defines REPO_DIR, OPERATOR_IMPLEMENTATION_COMMIT,
TRAINING_EVIDENCE_COMMIT, CLI_ENV, and git_text before executing this file.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_GATE01_RESULT_SHA256 = (
    "454747cd3e4b1376855915244a7c40fe281b758150e86f584fbea96f94d531f5"
)
EXPECTED_SELECTION_RECEIPT_FILE_SHA256 = (
    "c8d73fec48815224fcb87333dfd093c15738cc41dce89c4fb8ccf2cd874ef828"
)
EXPECTED_VALIDATION_PLAN_SHA256 = (
    "3afca2bab6a440529f88e7c8d9a9294fed9ecbf07eea1e308ed0910e2ba16421"
)
EXPECTED_SELECTION_RULE_SHA256 = (
    "fd15be634185a29d5ddedec3f2d7a24527bf5e59a49731f101f62cafcf1b06d6"
)
DRIVE_ROOT = Path("/content/drive/MyDrive/MRIxFields2026")
GATE01_PRIVATE_ARCHIVE_ROOT = DRIVE_ROOT / "Gate01Private_8012a3f"
STAGE2_V7_ROOT = DRIVE_ROOT / "stage2_unified_v7"
BANK_NAMESPACE = STAGE2_V7_ROOT / "bank_8081ce89a0ea"
TRAINING_NAMESPACE = BANK_NAMESPACE / "implementation_82633d66e5ea"
SELECTION_RECEIPT = (
    TRAINING_NAMESPACE
    / "unified_full_objective_pilot_200"
    / "scientific_attempts"
    / "attempt-0001"
    / "checkpoints"
    / "stage2_unified_full_selection_step000000200.json"
)
RECOVERY_NAMESPACE = BANK_NAMESPACE / (
    "recovery_training_"
    + TRAINING_EVIDENCE_COMMIT[:12]
    + "_operator_"
    + OPERATOR_IMPLEMENTATION_COMMIT[:12]
)
LOCAL_SCRATCH_ROOT = Path("/content/stage2_gate01_recovery_v8_scratch")
LOCAL_BANK_ROOT = LOCAL_SCRATCH_ROOT / "photometry_factored_latent_bank_v2"
LOCAL_LOG_ROOT = LOCAL_SCRATCH_ROOT / "logs"


# Use the scientific stack already supplied by Colab; do not install or download packages.
source_dir = str(REPO_DIR / "src")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONPATH"] = source_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, source_dir)
for module_name in tuple(sys.modules):
    if module_name == "fieldbridge" or module_name.startswith("fieldbridge."):
        del sys.modules[module_name]
importlib.invalidate_caches()
import fieldbridge

if REPO_DIR not in Path(fieldbridge.__file__).resolve().parents:
    raise RuntimeError("FieldBridge import escaped the detached operator checkout.")
CLI_ENV["PYTHONPATH"] = source_dir + os.pathsep + CLI_ENV.get("PYTHONPATH", "")
if git_text("status", "--porcelain"):
    raise RuntimeError("Operator checkout changed before Drive mount.")


from google.colab import drive

drive.mount("/content/drive")


def drive_retry(label, operation, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (OSError, IOError) as error:
            last_error = error
            if attempt == attempts:
                break
            print(
                {"drive_retry": label, "attempt": attempt, "error": repr(error)},
                flush=True,
            )
            try:
                drive.mount("/content/drive", force_remount=True)
            except Exception as remount_error:
                print({"drive_remount_warning": repr(remount_error)}, flush=True)
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(
        f"Drive operation failed after {attempts} attempts: {label}"
    ) from last_error

from fieldbridge.data.photometry_factorization import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    GATE01_P0006_EVALUATION_PROTOCOL,
    P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
    P0006_EVIDENCE_LIMITATION,
    P0009_CONFIRMATION_STATUS,
    load_gate01_p0006_evaluation_protocol,
    preflight_gate01_p0006_archive,
    verify_completed_stage2_pilot_evidence,
)


# First Drive artifact check: metadata-only, before bank/checkpoint/private-array work.
gate01_preflight = drive_retry(
    "early-gate01-metadata-preflight",
    lambda: preflight_gate01_p0006_archive(
        GATE01_PRIVATE_ARCHIVE_ROOT,
        expected_gate01_result_sha256=EXPECTED_GATE01_RESULT_SHA256,
    ),
)
if (
    gate01_preflight.get("status") != "pass"
    or gate01_preflight.get("private_array_payloads_opened") != 0
    or gate01_preflight.get("expected_gate01_result_match") is not True
):
    raise RuntimeError("Gate 0.1 early metadata preflight did not pass.")
print(
    json.dumps(
        {
            "stage": "early_gate01_metadata_preflight",
            "layout": gate01_preflight["archive_layout_contract"],
            "inventory_format": gate01_preflight["inventory_format"],
            "inventory_sha256": gate01_preflight["inventory_file_sha256"],
            "entry_count": gate01_preflight["normalized_inventory_entry_count"],
            "expected_result_match": True,
            "status": "pass",
        },
        sort_keys=True,
    ),
    flush=True,
)


LOCAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
for environment_name in ("TMPDIR", "TEMP", "TMP", "TORCH_HOME"):
    CLI_ENV[environment_name] = str(LOCAL_SCRATCH_ROOT / environment_name.lower())
    Path(CLI_ENV[environment_name]).mkdir(parents=True, exist_ok=True)


def tree_identity(root):
    root = Path(root).resolve()
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    ]
    if not rows:
        raise RuntimeError(f"Artifact tree is empty: {root}")
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": sha256_json(rows),
    }


def unique_existing(candidates, label):
    matches = [Path(path).resolve() for path in candidates if Path(path).is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {len(matches)}.")
    return matches[0]


def unique_existing_directory(candidates, label):
    matches = [Path(path).resolve() for path in candidates if Path(path).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {len(matches)}.")
    return matches[0]


def restore_bank_archive_to_scratch(bank_archive_dir):
    archive_identity = drive_retry(
        "hash-bank-archive", lambda: tree_identity(bank_archive_dir)
    )
    if LOCAL_BANK_ROOT.exists():
        local_identity = tree_identity(LOCAL_BANK_ROOT)
    else:
        last_error = None
        for attempt in range(1, 4):
            staging = LOCAL_SCRATCH_ROOT / f"bank_restore_attempt-{attempt:04d}.partial"
            if staging.exists():
                continue
            try:
                shutil.copytree(bank_archive_dir, staging)
                local_identity = tree_identity(staging)
                if local_identity != archive_identity:
                    raise RuntimeError("Drive-to-local bank staging hash mismatch.")
                os.rename(staging, LOCAL_BANK_ROOT)
                break
            except (OSError, IOError) as error:
                last_error = error
                try:
                    drive.mount("/content/drive", force_remount=True)
                except Exception as remount_error:
                    print({"drive_remount_warning": repr(remount_error)}, flush=True)
        else:
            raise RuntimeError(
                "Drive-to-local bank staging failed; partial attempts were preserved."
            ) from last_error
    if local_identity != archive_identity:
        raise RuntimeError("Hash-verified Drive-to-local bank staging failed closed.")
    return archive_identity


def run_logged(command, operation):
    attempt_root = RECOVERY_NAMESPACE / "operation_attempts" / operation
    drive_retry("create-operation-attempt-root", lambda: attempt_root.mkdir(parents=True, exist_ok=True))
    attempt = 1
    while (
        (attempt_root / f"attempt-{attempt:04d}.log").exists()
        or (attempt_root / f"attempt-{attempt:04d}.receipt.json").exists()
        or (LOCAL_LOG_ROOT / f"{operation}.attempt-{attempt:04d}.log").exists()
    ):
        attempt += 1
    local_log = LOCAL_LOG_ROOT / f"{operation}.attempt-{attempt:04d}.log"
    with local_log.open("x", encoding="utf-8", buffering=1) as log:
        log.write(
            json.dumps(
                {
                    "operation": operation,
                    "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
                    "operator_implementation_commit": OPERATOR_IMPLEMENTATION_COMMIT,
                    "command": command,
                },
                sort_keys=True,
            )
            + chr(10)
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=CLI_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write(json.dumps({"return_code": return_code}) + chr(10))
        log.flush()
        os.fsync(log.fileno())
    local_sha = sha256_file(local_log)
    archived_log = attempt_root / f"attempt-{attempt:04d}.log"
    partial_log = attempt_root / f"attempt-{attempt:04d}.log.partial"
    if partial_log.exists() or archived_log.exists():
        raise FileExistsError("Recovery log publication is no-clobber.")
    with local_log.open("rb") as source, partial_log.open("xb") as destination:
        shutil.copyfileobj(source, destination)
        destination.flush()
        os.fsync(destination.fileno())
    if sha256_file(partial_log) != local_sha:
        raise RuntimeError("Archived recovery log hash mismatch.")
    os.rename(partial_log, archived_log)
    receipt = {
        "contract": "stage2-gate01-recovery-operation-attempt-v1",
        "attempt": attempt,
        "operation": operation,
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "operator_implementation_commit": OPERATOR_IMPLEMENTATION_COMMIT,
        "command": command,
        "return_code": return_code,
        "archived_log": str(archived_log),
        "archived_log_sha256": sha256_file(archived_log),
        "drive_file_held_open_during_process": False,
        "archive_no_clobber": True,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    drive_retry(
        "publish-operation-receipt",
        lambda: write_json_atomic(
            attempt_root / f"attempt-{attempt:04d}.receipt.json",
            receipt,
            refuse_existing=True,
        ),
    )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return receipt


def load_self_hashed_json(path, hash_key):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    body = dict(payload)
    stored = body.pop(hash_key, None)
    if stored != sha256_json(body):
        raise ValueError(f"Self-hash mismatch: {path}")
    return payload


def seal_or_verify_exact_json(path, payload, hash_key):
    path = Path(path)
    if path.exists():
        existing = load_self_hashed_json(path, hash_key)
        if existing != payload:
            raise RuntimeError(f"Existing recovery receipt changed: {path}")
        return existing
    drive_retry(
        "publish-exact-recovery-json",
        lambda: write_json_atomic(path, payload, refuse_existing=True),
    )
    return payload


# No alternate bank may be built. One of these exact historical archive locations must exist.
BANK_ARCHIVE_DIR = unique_existing_directory(
    (
        BANK_NAMESPACE / "photometry_factored_latent_bank_v2",
        DRIVE_ROOT / "photometry_factored_latent_bank_v2",
    ),
    "immutable factored-latent bank archive",
)
bank_archive_identity = restore_bank_archive_to_scratch(BANK_ARCHIVE_DIR)


# This loads sealed model checkpoints on CPU only and reconstructs run fingerprints.
completed_evidence = verify_completed_stage2_pilot_evidence(
    TRAINING_NAMESPACE,
    bank_dir=LOCAL_BANK_ROOT,
    expected_selection_receipt_path=SELECTION_RECEIPT,
    training_evidence_commit=TRAINING_EVIDENCE_COMMIT,
    expected_selection_receipt_file_sha256=EXPECTED_SELECTION_RECEIPT_FILE_SHA256,
    expected_validation_plan_sha256=EXPECTED_VALIDATION_PLAN_SHA256,
    expected_selection_rule_sha256=EXPECTED_SELECTION_RULE_SHA256,
)
if (
    completed_evidence.get("status") != "pass"
    or completed_evidence.get("training_reused") is not True
    or completed_evidence.get("training_invoked") is not False
    or completed_evidence.get("latest_completed_step") != 200
):
    raise RuntimeError(
        "Compatible completed pilot evidence is absent; recovery refuses to retrain."
    )
print(
    json.dumps(
        {
            "stage": "completed_evidence_reuse",
            "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
            "selection_receipt_sha256": completed_evidence[
                "selection_receipt_file_sha256"
            ],
            "checkpoint_sha256": completed_evidence["checkpoint_file_sha256"],
            "run_fingerprint": completed_evidence["run_fingerprint"],
            "latest_step": 200,
            "status": "pass",
            "training_invoked": False,
        },
        sort_keys=True,
    ),
    flush=True,
)


PAIR_FEASIBILITY = unique_existing(
    (
        TRAINING_NAMESPACE / "stage2_retrospective_pair_feasibility_v2.json",
        BANK_NAMESPACE / "stage2_retrospective_pair_feasibility_v2.json",
        DRIVE_ROOT / "stage2_retrospective_pair_feasibility_v2.json",
    ),
    "sealed retrospective paired-feasibility receipt",
)
pair_feasibility = load_self_hashed_json(PAIR_FEASIBILITY, "result_sha256")
if (
    pair_feasibility.get("complete_inventory_no_selection") is not True
    or pair_feasibility.get("paired_evaluation_possible") is not False
):
    raise RuntimeError(
        "P:0006 recovery requires sealed proof that genuine paired R validation is unavailable."
    )


drive_retry(
    "create-recovery-namespace",
    lambda: RECOVERY_NAMESPACE.mkdir(parents=True, exist_ok=True),
)
P0006_EVALUATION_PROTOCOL = (
    RECOVERY_NAMESPACE
    / "stage2_gate01_p0006_development_validation_evaluation_only_protocol_v3.json"
)
LONG_RUN_EVALUATION_READINESS = (
    RECOVERY_NAMESPACE / "stage2_long_run_evaluation_readiness_v3.json"
)
FROZEN_VALIDATION_PLAN = Path(
    completed_evidence["pilot_200"]["selection_receipt"]
).parent / "stage2_unified_validation_plan_v2.json"

if P0006_EVALUATION_PROTOCOL.exists():
    protocol, cases, _baselines = load_gate01_p0006_evaluation_protocol(
        P0006_EVALUATION_PROTOCOL
    )
    if len(cases) != 60:
        raise RuntimeError("Existing P:0006 recovery protocol is incomplete.")
else:
    run_logged(
        [
            sys.executable,
            "-m",
            "fieldbridge.cli",
            "import-stage2-gate01-p0006-evaluation",
            "--archive-root",
            str(GATE01_PRIVATE_ARCHIVE_ROOT),
            "--expected-gate01-result-sha256",
            EXPECTED_GATE01_RESULT_SHA256,
            "--bank-dir",
            str(LOCAL_BANK_ROOT),
            "--validation-plan",
            str(FROZEN_VALIDATION_PLAN),
            "--out",
            str(P0006_EVALUATION_PROTOCOL),
        ],
        "import-P0006-development-validation-evaluation-only",
    )
    protocol, cases, _baselines = load_gate01_p0006_evaluation_protocol(
        P0006_EVALUATION_PROTOCOL
    )
if (
    protocol.get("contract_version") != GATE01_P0006_EVALUATION_PROTOCOL
    or protocol.get("data_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
    or protocol.get("evidence_interpretation") != P0006_EVIDENCE_LIMITATION
    or protocol.get("population_or_generalization_claims_authorized") is not False
    or protocol.get("training_or_model_selection_use") is not False
    or protocol.get("P0009_confirmation_status") != P0009_CONFIRMATION_STATUS
    or protocol.get("P0009_executed") is not False
):
    raise RuntimeError("Recovered P:0006 protocol changed its scientific role.")

if LONG_RUN_EVALUATION_READINESS.exists():
    readiness = load_self_hashed_json(
        LONG_RUN_EVALUATION_READINESS, "readiness_sha256"
    )
else:
    run_logged(
        [
            sys.executable,
            "-m",
            "fieldbridge.cli",
            "seal-stage2-long-run-evaluation-readiness",
            "--feasibility",
            str(PAIR_FEASIBILITY),
            "--p0006-evaluation-protocol",
            str(P0006_EVALUATION_PROTOCOL),
            "--out",
            str(LONG_RUN_EVALUATION_READINESS),
        ],
        "seal-long-run-evaluation-readiness-P0006",
    )
    readiness = load_self_hashed_json(
        LONG_RUN_EVALUATION_READINESS, "readiness_sha256"
    )
if (
    readiness.get("long_run_authorized_by_evaluation_path") is not True
    or readiness.get("evaluation_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
    or readiness.get("population_or_generalization_claims_authorized") is not False
    or readiness.get("prospective_training_or_model_selection_use") is not False
    or readiness.get("P0009_confirmation_status") != P0009_CONFIRMATION_STATUS
    or readiness.get("P0009_executed") is not False
):
    raise RuntimeError("Long-run evaluation-readiness scientific semantics changed.")


AUTHORIZE_100K_TRAINING = False
AUTHORIZE_LONG_FULL_MODEL = False
AUTHORIZE_BACKWARD_ABLATIONS_AFTER_FULL_REVIEW = False
if any(
    (
        AUTHORIZE_100K_TRAINING,
        AUTHORIZE_LONG_FULL_MODEL,
        AUTHORIZE_BACKWARD_ABLATIONS_AFTER_FULL_REVIEW,
    )
):
    raise RuntimeError("Recovery notebook cannot authorize long-run training.")

recovery_receipt = {
    "contract": "stage2-gate01-legacy-inventory-recovery-v1",
    "status": "evaluation_readiness_sealed_stop_before_long_run",
    "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
    "operator_implementation_commit": OPERATOR_IMPLEMENTATION_COMMIT,
    "training_namespace": str(TRAINING_NAMESPACE),
    "training_namespace_read_only": True,
    "recovery_namespace": str(RECOVERY_NAMESPACE),
    "bank_archive": str(BANK_ARCHIVE_DIR),
    "bank_archive_tree": bank_archive_identity,
    "gate01_preflight_sha256": gate01_preflight["preflight_sha256"],
    "gate01_layout_contract": gate01_preflight["archive_layout_contract"],
    "gate01_inventory_file_sha256": gate01_preflight["inventory_file_sha256"],
    "completed_evidence_reuse_sha256": completed_evidence[
        "reuse_verification_sha256"
    ],
    "selection_receipt_file_sha256": EXPECTED_SELECTION_RECEIPT_FILE_SHA256,
    "checkpoint_file_sha256": completed_evidence["checkpoint_file_sha256"],
    "run_fingerprint": completed_evidence["run_fingerprint"],
    "validation_plan_sha256": EXPECTED_VALIDATION_PLAN_SHA256,
    "selection_rule_sha256": EXPECTED_SELECTION_RULE_SHA256,
    "p0006_protocol_path": str(P0006_EVALUATION_PROTOCOL),
    "p0006_protocol_file_sha256": sha256_file(P0006_EVALUATION_PROTOCOL),
    "evaluation_readiness_path": str(LONG_RUN_EVALUATION_READINESS),
    "evaluation_readiness_file_sha256": sha256_file(
        LONG_RUN_EVALUATION_READINESS
    ),
    "training_reused": True,
    "training_invoked": False,
    "private_training_or_checkpoint_selection_use": False,
    "population_or_generalization_claims_authorized": False,
    "P0009_confirmation_status": P0009_CONFIRMATION_STATUS,
    "P0009_executed": False,
    "descriptor_coupling": False,
    "learned_disentanglement_claim": False,
    "StarGAN_control_claim": False,
    "long_run_training_authorized": False,
}
recovery_receipt["receipt_sha256"] = sha256_json(recovery_receipt)
RECOVERY_RECEIPT = RECOVERY_NAMESPACE / "stage2_gate01_recovery_receipt_v1.json"
seal_or_verify_exact_json(RECOVERY_RECEIPT, recovery_receipt, "receipt_sha256")

print(
    json.dumps(
        {
            "status": recovery_receipt["status"],
            "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
            "operator_implementation_commit": OPERATOR_IMPLEMENTATION_COMMIT,
            "gate01_layout": recovery_receipt["gate01_layout_contract"],
            "selection_receipt_sha256": EXPECTED_SELECTION_RECEIPT_FILE_SHA256,
            "checkpoint_sha256": completed_evidence["checkpoint_file_sha256"],
            "run_fingerprint": completed_evidence["run_fingerprint"],
            "p0006_protocol_sha256": recovery_receipt[
                "p0006_protocol_file_sha256"
            ],
            "evaluation_readiness_sha256": recovery_receipt[
                "evaluation_readiness_file_sha256"
            ],
            "training_reused": True,
            "training_invoked": False,
            "long_run_training_authorized": False,
            "next_action": "STOP_FOR_RESOURCE_BOUNDED_TRAINING_DESIGN_REVIEW",
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
