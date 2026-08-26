"""Sanitized, metadata-only audit of the completed unified Stage-2 step-200 pilot.

The audit reads persisted JSON/JSONL evidence and the checkpoint container on CPU.  It
never opens a bank tensor or patient array, performs inference, or authorizes training.
Only a fixed numeric/domain allowlist can enter reportable artifacts.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import sha256_json
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.stage2_unified import (
    DEFAULT_UNIFIED_WEIGHTS,
    UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
    UNIFIED_A100_GATE_CONTRACT,
    UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES,
    UNIFIED_ANATOMY_MEMORY_CONTRACT,
    UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
    UNIFIED_HISTORY_CONTRACT,
    UNIFIED_RESUME_CONTRACT,
    UNIFIED_SELECTION_RULE_SHA256,
    UNIFIED_TERM_GRADIENT_QUALIFICATION_CONTRACT,
    UNIFIED_VALIDATION_PLAN_CONTRACT,
)


PILOT_AUDIT_CONTRACT = "stage2-step200-pilot-evidence-audit-v1"
PILOT_AUDIT_MANIFEST_CONTRACT = "stage2-step200-pilot-audit-artifact-manifest-v1"
PILOT_AUDIT_SANITIZATION_CONTRACT = "stage2-step200-history-domain-numeric-only-v1"
TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
STEP200_CHECKPOINT_SHA256 = (
    "09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021"
)
STEP200_RUN_FINGERPRINT = (
    "c814c948a5b85bd3a694db7c8e074894e97c16a96a36acbfa6f370faf2dac0aa"
)
STEP200_SELECTION_RECEIPT_FILE_SHA256 = (
    "c8d73fec48815224fcb87333dfd093c15738cc41dce89c4fb8ccf2cd874ef828"
)
STEP200_SELECTION_RULE_SHA256 = (
    "fd15be634185a29d5ddedec3f2d7a24527bf5e59a49731f101f62cafcf1b06d6"
)
STEP200_VALIDATION_PLAN_SHA256 = (
    "3afca2bab6a440529f88e7c8d9a9294fed9ecbf07eea1e308ed0910e2ba16421"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_EXPECTED_STEPS = tuple(range(1, 201))
_TERMS = tuple(DEFAULT_UNIFIED_WEIGHTS)
_GRADIENT_EXPLOSION_MAX_NORM = 100.0
_A100_QUALIFICATION_RECEIPT_KEYS = frozenset(
    {
        "contract_version",
        "status",
        "step",
        "gate",
        "anatomy_memory_qualification",
        "run_fingerprint",
        "validation_plan_sha256",
        "complete_validation_executed",
        "checkpoint_written",
        "generator_optimizer_updates",
        "generator_gradient_accumulation_contract",
        "term_gradient_qualification_contract",
        "frozen_decoder_state_sha256",
        "decoder_activation_checkpoint_sha256",
        "receipt_sha256",
    }
)
_A100_GATE_KEYS = frozenset(
    {
        "contract_version",
        "status",
        "required_gpu",
        "gpu_name",
        "gpu_identity_matches",
        "peak_allocated_limit_bytes",
        "peak_allocated_bytes",
        "within_allocated_limit",
        "before_pilot_steps",
        "full_objective",
        "batch_size",
        "precision",
        "integration_steps",
        "integration_solver",
    }
)
_ANATOMY_QUALIFICATION_KEYS = frozenset(
    {
        "contract_version",
        "status",
        "full_volume",
        "source_decode_checkpointed",
        "generated_decode_checkpoint_mode",
        "group_norm_scope",
        "spatial_crop_or_tile",
        "allocator_fallback",
        "step",
        "decoder_state_sha256_before",
        "decoder_state_sha256_after",
        "decoder_state_unchanged",
        "checkpoint_evidence_sha256",
        "checkpoint_evidence",
    }
)
_DECODER_CHECKPOINT_EVIDENCE_KEYS = frozenset(
    {
        "contract_version",
        "mode",
        "spatial_dims",
        "full_volume",
        "group_norm_scope",
        "upsample_regions",
        "residual_branch_regions",
        "residual_skip_checkpointed",
        "residual_skip_evaluation_order",
        "outer_full_decoder_checkpoint",
        "checkpoint_use_reentrant",
        "checkpoint_preserve_rng_state",
        "source_no_grad_decode_checkpointed",
        "state_dict_schema_changed",
    }
)

_CORE_NUMERIC_FIELDS = (
    *(f"raw/{term}" for term in _TERMS),
    *(f"weighted/{term}" for term in _TERMS),
    "weighted/generator_total",
    "weighted/auxiliary_total",
    "weighted/aux_to_flow_ratio",
    "gradient/generator_norm",
    "gradient/critic_norm",
    *(f"gradient/term_{term}" for term in _TERMS),
    "critic/real_score_mean",
    "critic/fake_score_mean",
    "critic/score_separation",
    "critic/score_saturation_fraction",
    "critic/real_domain_accuracy",
    "critic/generated_domain_accuracy",
    "generator_lr",
    "critic_lr",
    "step_seconds",
    "examples_per_second",
    "peak_cuda_bytes",
    "peak_cuda_reserved_bytes",
)


@dataclass(frozen=True, slots=True)
class Step200PilotAuditInputs:
    checkpoint: Path
    history_jsonl: Path
    resolved_config: Path
    selection_receipt: Path
    validation_plan: Path
    a100_qualification_receipt: Path
    recovery_receipt: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("step200_checkpoint", self.checkpoint),
            ("history_jsonl", self.history_jsonl),
            ("resolved_config", self.resolved_config),
            ("selection_receipt", self.selection_receipt),
            ("validation_plan", self.validation_plan),
            ("a100_qualification_receipt", self.a100_qualification_receipt),
            ("recovery_receipt", self.recovery_receipt),
        )


@dataclass(frozen=True, slots=True)
class VerifiedStep200PilotEvidence:
    training_rows: tuple[dict[str, Any], ...]
    pilot_report: dict[str, Any]
    validation_event: dict[str, Any]
    input_manifest: dict[str, Any]
    resolved_config: dict[str, Any]
    selection_receipt: dict[str, Any]
    validation_plan: dict[str, Any]
    a100_qualification_receipt: dict[str, Any]
    recovery_receipt: dict[str, Any]


def sha256_file_streaming(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one regular file without reading it into memory."""

    source = _regular_file(Path(path), "audit input")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_step200_pilot_evidence(
    inputs: Step200PilotAuditInputs,
    *,
    audit_implementation_commit: str,
) -> VerifiedStep200PilotEvidence:
    """Verify immutable step-200 evidence while retaining no bank or image tensors."""

    _require_git(audit_implementation_commit, "audit implementation commit")
    input_files: dict[str, dict[str, Any]] = {}
    for label, path in inputs.items():
        source = _regular_file(path, label)
        input_files[label] = {
            "file_sha256": sha256_file_streaming(source),
            "size_bytes": source.stat().st_size,
        }
    if input_files["step200_checkpoint"]["file_sha256"] != STEP200_CHECKPOINT_SHA256:
        raise ValueError("Step-200 checkpoint file SHA-256 mismatch.")
    if (
        input_files["selection_receipt"]["file_sha256"]
        != STEP200_SELECTION_RECEIPT_FILE_SHA256
    ):
        raise ValueError("Step-200 selection-receipt file SHA-256 mismatch.")

    selection = _load_self_hashed(inputs.selection_receipt, "selection_sha256")
    if (
        selection.get("latest_step") != 200
        or selection.get("terminal_step") != 200
        or selection.get("receipt_step") != 200
        or selection.get("run_complete") is not True
        or selection.get("variant") != "full"
        or selection.get("selection_rule_sha256") != STEP200_SELECTION_RULE_SHA256
        or selection.get("selection_rule_sha256") != UNIFIED_SELECTION_RULE_SHA256
        or selection.get("paired_targets_used") is not False
        or selection.get("complete_r_validation_inventory") is not True
    ):
        raise ValueError("Step-200 selection receipt is incomplete or scientifically incompatible.")
    hashes = selection.get("checkpoint_hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("Step-200 selection receipt lacks checkpoint identities.")
    for role in ("latest", "best", "final"):
        item = hashes.get(role)
        if (
            not isinstance(item, Mapping)
            or Path(str(item.get("path", ""))).name != inputs.checkpoint.name
            or item.get("file_sha256") != STEP200_CHECKPOINT_SHA256
        ):
            raise ValueError(f"Step-200 {role} checkpoint identity changed.")

    validation_plan = _load_self_hashed(inputs.validation_plan, "validation_plan_sha256")
    if (
        validation_plan.get("contract_version") != UNIFIED_VALIDATION_PLAN_CONTRACT
        or validation_plan.get("validation_plan_sha256") != STEP200_VALIDATION_PLAN_SHA256
        or selection.get("validation_plan_sha256") != STEP200_VALIDATION_PLAN_SHA256
        or validation_plan.get("scope") != "complete_R_validation_inventory"
        or validation_plan.get("required_directed_domain_cell_count") != 60
    ):
        raise ValueError("Frozen step-200 validation plan identity or scope changed.")
    entries = validation_plan.get("entries")
    if not isinstance(entries, list) or any(
        str(entry.get(role, "")).startswith("P:")
        for entry in entries
        if isinstance(entry, Mapping)
        for role in ("source_subject_group_identity", "target_subject_group_identity")
    ):
        raise ValueError("Frozen validation plan is malformed or contains a P endpoint.")

    resolved = _load_json(inputs.resolved_config)
    training = resolved.get("training")
    if not isinstance(training, Mapping) or training.get("device") != "auto":
        raise ValueError("Resolved step-200 configuration must retain declared device='auto'.")
    weights = training.get("loss_weights")
    if not isinstance(weights, Mapping) or dict(weights) != DEFAULT_UNIFIED_WEIGHTS:
        raise ValueError("Resolved step-200 configuration changed the full six-term objective.")
    if (
        training.get("batch_size") != 1
        or training.get("precision") != "bf16"
        or training.get("integration_steps") != 4
        or training.get("integration_solver") != "heun"
    ):
        raise ValueError("Resolved step-200 batch/precision/four-step-Heun contract changed.")

    checkpoint = load_checkpoint(inputs.checkpoint, map_location="cpu")
    try:
        if checkpoint.get("contract_version") != UNIFIED_RESUME_CONTRACT:
            raise ValueError("Step-200 checkpoint resume contract changed.")
        meta = checkpoint.get("_meta")
        if not isinstance(meta, Mapping):
            raise ValueError("Step-200 checkpoint metadata is malformed.")
        _require_git(str(meta.get("git_commit", "")), "checkpoint training commit")
        if meta.get("git_commit") != TRAINING_EVIDENCE_COMMIT:
            raise ValueError("Step-200 checkpoint training commit changed.")
        if checkpoint.get("training_cursor") != 200:
            raise ValueError("Step-200 checkpoint training cursor changed.")
        if checkpoint.get("run_fingerprint") != STEP200_RUN_FINGERPRINT:
            raise ValueError("Step-200 checkpoint run fingerprint changed.")
        if checkpoint.get("validation_plan_sha256") != STEP200_VALIDATION_PLAN_SHA256:
            raise ValueError("Step-200 checkpoint validation-plan identity changed.")
        if checkpoint.get("selection_rule_sha256") != STEP200_SELECTION_RULE_SHA256:
            raise ValueError("Step-200 checkpoint selection-rule identity changed.")
        checkpoint_selection = checkpoint.get("validation_selection")
        if not isinstance(checkpoint_selection, Mapping) or any(
            checkpoint_selection.get(key) != selection.get(key)
            for key in (
                "validation_plan_sha256",
                "selection_rule_sha256",
                "paired_targets_used",
                "complete_r_validation_inventory",
                "latest_step",
                "best_step",
                "best_score",
            )
        ):
            raise ValueError("Checkpoint selection metadata and selection receipt disagree.")
        pilot_report = checkpoint.get("pilot_report")
        if not isinstance(pilot_report, Mapping):
            raise ValueError("Checkpoint lacks its embedded step-200 pilot report.")
        pilot = dict(pilot_report)
    finally:
        del checkpoint

    history_rows, validation_event, history_pilot = _load_history(
        inputs.history_jsonl,
        run_fingerprint=STEP200_RUN_FINGERPRINT,
        validation_plan_sha256=STEP200_VALIDATION_PLAN_SHA256,
    )
    if history_pilot != pilot:
        raise ValueError("History and checkpoint-contained pilot reports disagree.")
    _validate_pilot_report(pilot)

    qualification = _load_self_hashed(inputs.a100_qualification_receipt, "receipt_sha256")
    _validate_a100_qualification_receipt(qualification, pilot=pilot)

    recovery = _load_self_hashed(inputs.recovery_receipt, "receipt_sha256")
    if (
        recovery.get("training_evidence_commit") != TRAINING_EVIDENCE_COMMIT
        or recovery.get("selection_receipt_file_sha256")
        != STEP200_SELECTION_RECEIPT_FILE_SHA256
        or recovery.get("checkpoint_file_sha256") != STEP200_CHECKPOINT_SHA256
        or recovery.get("run_fingerprint") != STEP200_RUN_FINGERPRINT
        or recovery.get("training_reused") is not True
        or recovery.get("training_invoked") is not False
        or recovery.get("long_run_training_authorized") is not False
        or recovery.get("population_or_generalization_claims_authorized") is not False
        or recovery.get("P0009_executed") is not False
    ):
        raise ValueError("Recovery receipt does not seal the reviewed step-200 evidence.")

    input_manifest = {
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "audit_implementation_commit": audit_implementation_commit,
        "step200_checkpoint_sha256": STEP200_CHECKPOINT_SHA256,
        "step200_run_fingerprint": STEP200_RUN_FINGERPRINT,
        "pilot_report_container": "step200_checkpoint",
        "pilot_report_sha256": sha256_json(pilot),
        "inputs": input_files,
    }
    input_manifest["input_manifest_sha256"] = sha256_json(input_manifest)
    return VerifiedStep200PilotEvidence(
        training_rows=tuple(history_rows),
        pilot_report=pilot,
        validation_event=validation_event,
        input_manifest=input_manifest,
        resolved_config=resolved,
        selection_receipt=selection,
        validation_plan=validation_plan,
        a100_qualification_receipt=qualification,
        recovery_receipt=recovery,
    )


def run_step200_pilot_evidence_audit(
    inputs: Step200PilotAuditInputs,
    *,
    output_dir: str | Path,
    audit_implementation_commit: str,
    step20_evidence_independent: bool = True,
) -> dict[str, Any]:
    """Verify, sanitize, render, and atomically seal the CPU-only pilot dashboard."""

    verified = verify_step200_pilot_evidence(
        inputs, audit_implementation_commit=audit_implementation_commit
    )
    root = Path(output_dir)
    manifest_path = root / "artifact_manifest.json"
    if root.exists() and any(root.iterdir()):
        return _verify_existing_audit(root, verified.input_manifest)
    root.mkdir(parents=True, exist_ok=True)

    sanitized = sanitize_step200_history(verified.training_rows)
    csv_path = root / "stage2_step200_history_sanitized.csv"
    _write_sanitized_csv(csv_path, sanitized)
    summary = build_step200_pilot_summary(
        verified,
        step20_evidence_independent=step20_evidence_independent,
    )
    summary_path = root / "stage2_step200_summary.json"
    _write_json_atomic(summary_path, summary)

    png_path = root / "stage2_step200_training_curves.png"
    svg_path = root / "stage2_step200_training_curves.svg"
    pdf_path = root / "stage2_step200_pilot_audit.pdf"
    rendering = _render_training_dashboard(sanitized, summary, png_path, svg_path, pdf_path)
    html_path = root / "stage2_step200_pilot_audit.html"
    _render_self_contained_html(html_path, summary, png_path)

    outputs = {}
    for path in (csv_path, summary_path, png_path, svg_path, html_path, pdf_path):
        outputs[path.name] = {
            "file_sha256": sha256_file_streaming(path),
            "size_bytes": path.stat().st_size,
        }
    manifest: dict[str, Any] = {
        "contract_version": PILOT_AUDIT_MANIFEST_CONTRACT,
        "audit_contract_version": PILOT_AUDIT_CONTRACT,
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "audit_implementation_commit": audit_implementation_commit,
        "checkpoint_sha256": STEP200_CHECKPOINT_SHA256,
        "run_fingerprint": STEP200_RUN_FINGERPRINT,
        "input_manifest": verified.input_manifest,
        "training_row_count": 200,
        "sanitization_contract": {
            "contract_version": PILOT_AUDIT_SANITIZATION_CONTRACT,
            "record_identifiers_exported": False,
            "filesystem_paths_exported": False,
            "patient_or_case_identifiers_exported": False,
            "domain_labels_exported": True,
            "allowed_numeric_fields": list(_CORE_NUMERIC_FIELDS),
        },
        "rendering": rendering,
        "outputs": outputs,
        "training_invoked": False,
        "inference_invoked": False,
        "bank_opened": False,
        "private_arrays_opened": False,
        "gpu_used": False,
        "long_run_training_authorized": False,
        "population_or_generalization_claims_authorized": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    _write_json_atomic(manifest_path, manifest)
    return {**summary, "artifact_manifest_file_sha256": sha256_file_streaming(manifest_path)}


def sanitize_step200_history(
    rows: Sequence[Mapping[str, Any]], *, rolling_window: int = 20
) -> list[dict[str, Any]]:
    """Return only predeclared numeric fields and canonical domain transitions."""

    if len(rows) != 200 or [row.get("step") for row in rows] != list(_EXPECTED_STEPS):
        raise ValueError("Sanitized history requires exact training steps 1..200.")
    if type(rolling_window) is not int or rolling_window <= 0:
        raise ValueError("Rolling window must be a positive integer.")
    sanitized: list[dict[str, Any]] = []
    numeric_series: dict[str, list[float]] = {field: [] for field in _CORE_NUMERIC_FIELDS}
    for row in rows:
        transition = _canonical_transition(str(row.get("transition", "")))
        item: dict[str, Any] = {"step": int(row["step"]), "transition": transition}
        for field in _CORE_NUMERIC_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"History field {field!r} is missing or nonnumeric.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"History field {field!r} contains a nonfinite value.")
            item[field] = number
            numeric_series[field].append(number)
            start = max(0, len(numeric_series[field]) - rolling_window)
            item[f"rolling20/{field}"] = float(
                np.mean(numeric_series[field][start:])
            )
        sanitized.append(item)
    forbidden_tokens = ("record", "subject", "patient", "case_id", "path", "filename")
    if any(any(token in key.lower() for token in forbidden_tokens) for key in sanitized[0]):
        raise RuntimeError("Sanitization allowlist unexpectedly contains a sensitive field.")
    return sanitized


def build_step200_pilot_summary(
    verified: VerifiedStep200PilotEvidence,
    *,
    step20_evidence_independent: bool,
) -> dict[str, Any]:
    rows = sanitize_step200_history(verified.training_rows)
    pilot = verified.pilot_report
    failures = {str(value) for value in pilot.get("failures", [])}
    gradients = {
        term: [float(row[f"gradient/term_{term}"]) for row in rows] for term in _TERMS
    }
    generator_gradients = [float(row["gradient/generator_norm"]) for row in rows]
    critic_gradients = [float(row["gradient/critic_norm"]) for row in rows]
    saturation = [float(row["critic/score_saturation_fraction"]) for row in rows]
    aux_ratio = [float(row["weighted/aux_to_flow_ratio"]) for row in rows]
    step_seconds = [float(row["step_seconds"]) for row in rows]
    validation = dict(verified.validation_event["validation"])
    assessment = {
        "nonfinite_values": {"status": "pass", "observed_count": 0},
        "uncontrolled_loss_growth": {
            "status": "fail" if "uncontrolled_smoothed_loss_growth" in failures else "pass",
            "source": "persisted pilot threshold/result",
        },
        "missing_gradients": {
            "status": (
                "pass"
                if all(max(values) > 0.0 for values in gradients.values())
                and max(generator_gradients) > 0.0
                and max(critic_gradients) > 0.0
                else "fail"
            ),
            "all_six_terms_enabled": True,
        },
        "gradient_explosion": {
            "status": (
                "pass"
                if max((*generator_gradients, *critic_gradients))
                <= _GRADIENT_EXPLOSION_MAX_NORM
                else "fail"
            ),
            "predeclared_max_norm": _GRADIENT_EXPLOSION_MAX_NORM,
        },
        "critic_saturation": {
            "status": "fail" if "critic_saturation" in failures else "pass",
            "observed_maximum": max(saturation),
        },
        "auxiliary_dominance": {
            "status": (
                "fail" if "auxiliary_objectives_dominate_flow" in failures else "pass"
            ),
            "observed_maximum": max(aux_ratio),
        },
        "runtime_viability": {
            "status": "pass" if all(value > 0 and math.isfinite(value) for value in step_seconds) else "fail",
            "mean_step_seconds": float(np.mean(step_seconds)),
        },
        "available_validation_evidence": {
            "status": "terminal_complete_unpaired_R_validation_only",
            "complete_inventory_used": validation.get("complete_inventory_used") is True,
            "paired_targets_used": False,
            "validation_trend_inferable": False,
            "early_stopping_behavior_inferable": False,
        },
    }
    transitions = Counter(str(row["transition"]) for row in rows)
    numeric_summary = {
        field: {
            "first": float(rows[0][field]),
            "last": float(rows[-1][field]),
            "minimum": float(min(float(row[field]) for row in rows)),
            "maximum": float(max(float(row[field]) for row in rows)),
            "mean": float(np.mean([float(row[field]) for row in rows])),
        }
        for field in _CORE_NUMERIC_FIELDS
    }
    body: dict[str, Any] = {
        "contract_version": PILOT_AUDIT_CONTRACT,
        "scientific_scope": "engineering audit of a full-objective 200-step pilot",
        "model_status": "pilot_not_converged_model",
        "convergence_declared": False,
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "checkpoint_sha256": STEP200_CHECKPOINT_SHA256,
        "run_fingerprint": STEP200_RUN_FINGERPRINT,
        "selection_receipt_file_sha256": STEP200_SELECTION_RECEIPT_FILE_SHA256,
        "selection_rule_sha256": STEP200_SELECTION_RULE_SHA256,
        "validation_plan_sha256": STEP200_VALIDATION_PLAN_SHA256,
        "training_row_count": 200,
        "steps": [1, 200],
        "objective_terms": list(_TERMS),
        "objective_weights": dict(DEFAULT_UNIFIED_WEIGHTS),
        "full_objective": True,
        "step20_and_step200_relationship": (
            "separate_pilots_no_continuous_trajectory_claim"
            if step20_evidence_independent
            else "prefix_equivalence_must_be_proven_externally"
        ),
        "transition_counts": dict(sorted(transitions.items())),
        "numeric_summary": numeric_summary,
        "selection_and_validation": {
            "best_step": verified.selection_receipt.get("best_step"),
            "latest_step": verified.selection_receipt.get("latest_step"),
            "best_score": verified.selection_receipt.get("best_score"),
            "complete_r_validation_inventory": True,
            "paired_targets_used": False,
            "P_records_used_for_training_or_selection": 0,
        },
        "engineering_assessment": assessment,
        "limitations": [
            "The checkpoint is a 200-step full-objective pilot, not a converged model.",
            "Only terminal complete unpaired R-validation evidence is available; a validation trend and early-stopping behavior cannot be inferred.",
            "No P record entered training or checkpoint selection.",
            "Population or generalization claims are not authorized.",
        ],
        "training_invoked": False,
        "inference_invoked": False,
        "private_arrays_opened": False,
        "bank_opened": False,
        "gpu_used": False,
        "long_run_training_authorized": False,
        "population_or_generalization_claims_authorized": False,
    }
    body["summary_sha256"] = sha256_json(body)
    return body


def _load_history(
    path: Path, *, run_fingerprint: str, validation_plan_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        _regular_file(path, "history JSONL").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        raw = _json_loads_strict(line, f"history row {line_number}")
        if not isinstance(raw, Mapping):
            raise ValueError(f"History row {line_number} is not an object.")
        row = dict(raw)
        if row.get("contract_version") != UNIFIED_HISTORY_CONTRACT:
            raise ValueError("History contract changed.")
        if row.get("run_fingerprint") != run_fingerprint:
            raise ValueError("History run fingerprint changed.")
        _reject_nonfinite_recursive(row, f"history row {line_number}")
        rows.append(row)
    if any(row.get("event") == "oom_hard_stop" for row in rows):
        raise ValueError("History contains an OOM hard stop.")
    training_rows = [row for row in rows if "event" not in row]
    if [row.get("step") for row in training_rows] != list(_EXPECTED_STEPS):
        raise ValueError("History training rows must be exactly ordered steps 1..200.")
    validations = [
        row
        for row in rows
        if row.get("event") == "unpaired_validation" and row.get("step") == 200
    ]
    pilots = [
        row for row in rows if row.get("event") == "full_objective_pilot" and row.get("step") == 200
    ]
    if len(validations) != 1 or len(pilots) != 1:
        raise ValueError("History must contain one terminal validation and pilot report.")
    if pilots[0].get("validation_plan_sha256") != validation_plan_sha256:
        raise ValueError("History pilot validation-plan identity changed.")
    validation = validations[0].get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("complete_inventory_used") is not True
        or validation.get("validation_plan_sha256") != validation_plan_sha256
    ):
        raise ValueError("History terminal validation evidence is incomplete.")
    pilot = pilots[0].get("pilot")
    if not isinstance(pilot, Mapping):
        raise ValueError("History terminal pilot report is malformed.")
    return training_rows, validations[0], dict(pilot)


def _validate_a100_qualification_receipt(
    qualification: Mapping[str, Any],
    *,
    pilot: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        qualification,
        _A100_QUALIFICATION_RECEIPT_KEYS,
        "A100 qualification-only receipt",
    )
    run_fingerprint = qualification.get("run_fingerprint")
    if (
        qualification.get("contract_version") != UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT
        or qualification.get("status") != "pass"
        or type(qualification.get("step")) is not int
        or qualification.get("step") != 1
        or qualification.get("complete_validation_executed") is not False
        or qualification.get("checkpoint_written") is not False
        or type(qualification.get("generator_optimizer_updates")) is not int
        or qualification.get("generator_optimizer_updates") != 1
        or qualification.get("validation_plan_sha256") != STEP200_VALIDATION_PLAN_SHA256
        or not isinstance(run_fingerprint, str)
        or _SHA256_RE.fullmatch(run_fingerprint) is None
        or run_fingerprint == STEP200_RUN_FINGERPRINT
        or qualification.get("generator_gradient_accumulation_contract")
        != UNIFIED_GENERATOR_ACCUMULATION_CONTRACT
        or qualification.get("term_gradient_qualification_contract")
        != UNIFIED_TERM_GRADIENT_QUALIFICATION_CONTRACT
        or _SHA256_RE.fullmatch(str(qualification.get("receipt_sha256"))) is None
    ):
        raise ValueError("A100 qualification-only receipt is incomplete or changed.")

    gate = qualification.get("gate")
    _validate_a100_gate(gate)
    anatomy = qualification.get("anatomy_memory_qualification")
    _validate_anatomy_qualification(anatomy, label="A100 qualification")
    assert isinstance(anatomy, Mapping)
    checkpoint_evidence = anatomy["checkpoint_evidence"]
    assert isinstance(checkpoint_evidence, Mapping)

    frozen_decoder_sha = qualification.get("frozen_decoder_state_sha256")
    checkpoint_evidence_sha = qualification.get("decoder_activation_checkpoint_sha256")
    if (
        not isinstance(frozen_decoder_sha, str)
        or _SHA256_RE.fullmatch(frozen_decoder_sha) is None
        or frozen_decoder_sha != anatomy.get("decoder_state_sha256_before")
        or not isinstance(checkpoint_evidence_sha, str)
        or _SHA256_RE.fullmatch(checkpoint_evidence_sha) is None
        or checkpoint_evidence_sha != sha256_json(checkpoint_evidence)
        or checkpoint_evidence_sha != anatomy.get("checkpoint_evidence_sha256")
    ):
        raise ValueError("A100 qualification decoder identities changed.")

    pilot_anatomy = pilot.get("one_step_anatomy_memory_qualification")
    _validate_anatomy_qualification(pilot_anatomy, label="step-200 pilot")
    assert isinstance(pilot_anatomy, Mapping)
    pilot_checkpoint_evidence = pilot_anatomy["checkpoint_evidence"]
    assert isinstance(pilot_checkpoint_evidence, Mapping)
    if (
        frozen_decoder_sha != pilot_anatomy.get("decoder_state_sha256_before")
        or checkpoint_evidence_sha != sha256_json(pilot_checkpoint_evidence)
    ):
        raise ValueError(
            "A100 qualification and step-200 pilot used different decoder identities."
        )


def _validate_a100_gate(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("A100 qualification gate is malformed.")
    _require_exact_keys(value, _A100_GATE_KEYS, "A100 qualification gate")
    peak = value.get("peak_allocated_bytes")
    gpu_name = value.get("gpu_name")
    if (
        value.get("contract_version") != UNIFIED_A100_GATE_CONTRACT
        or value.get("status") != "pass"
        or value.get("required_gpu") != "NVIDIA A100"
        or not isinstance(gpu_name, str)
        or not gpu_name
        or "A100" not in gpu_name.upper()
        or value.get("gpu_identity_matches") is not True
        or value.get("peak_allocated_limit_bytes")
        != UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
        or type(peak) is not int
        or not 0 < peak <= UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
        or value.get("within_allocated_limit") is not True
        or value.get("before_pilot_steps") != [20, 200]
        or value.get("full_objective") is not True
        or type(value.get("batch_size")) is not int
        or value.get("batch_size") != 1
        or value.get("precision") != "bf16"
        or type(value.get("integration_steps")) is not int
        or value.get("integration_steps") != 4
        or value.get("integration_solver") != "heun"
    ):
        raise ValueError("A100 qualification gate did not pass exactly.")


def _validate_anatomy_qualification(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} anatomy-memory qualification is malformed.")
    _require_exact_keys(value, _ANATOMY_QUALIFICATION_KEYS, f"{label} anatomy qualification")
    before = value.get("decoder_state_sha256_before")
    checkpoint = value.get("checkpoint_evidence")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{label} decoder checkpoint evidence is malformed.")
    _require_exact_keys(
        checkpoint,
        _DECODER_CHECKPOINT_EVIDENCE_KEYS,
        f"{label} decoder checkpoint evidence",
    )
    residual_regions = checkpoint.get("residual_branch_regions")
    if (
        value.get("contract_version") != UNIFIED_ANATOMY_MEMORY_CONTRACT
        or value.get("status") != "pass"
        or value.get("full_volume") is not True
        or value.get("source_decode_checkpointed") is not False
        or value.get("generated_decode_checkpoint_mode")
        != "fine_grained_full_volume_v1"
        or value.get("group_norm_scope") != "complete_spatial_volume"
        or value.get("spatial_crop_or_tile") is not False
        or value.get("allocator_fallback") is not False
        or type(value.get("step")) is not int
        or value.get("step") != 1
        or not isinstance(before, str)
        or _SHA256_RE.fullmatch(before) is None
        or value.get("decoder_state_sha256_after") != before
        or value.get("decoder_state_unchanged") is not True
        or checkpoint.get("contract_version")
        != "klvae3d-full-volume-fine-grained-activation-checkpoint-v1"
        or checkpoint.get("mode") != "fine_grained_full_volume_v1"
        or type(checkpoint.get("spatial_dims")) is not int
        or checkpoint.get("spatial_dims") != 3
        or checkpoint.get("full_volume") is not True
        or checkpoint.get("group_norm_scope") != "complete_spatial_volume"
        or checkpoint.get("upsample_regions") != ["up1", "up2"]
        or not isinstance(residual_regions, list)
        or not residual_regions
        or len(residual_regions) != len(set(residual_regions))
        or any(not isinstance(region, str) or not region for region in residual_regions)
        or checkpoint.get("residual_skip_checkpointed") is not False
        or checkpoint.get("residual_skip_evaluation_order")
        != "after_branch2_before_add"
        or checkpoint.get("outer_full_decoder_checkpoint") is not False
        or checkpoint.get("checkpoint_use_reentrant") is not False
        or checkpoint.get("checkpoint_preserve_rng_state") is not True
        or checkpoint.get("source_no_grad_decode_checkpointed") is not False
        or checkpoint.get("state_dict_schema_changed") is not False
        or value.get("checkpoint_evidence_sha256") != sha256_json(checkpoint)
    ):
        raise ValueError(f"{label} anatomy-memory qualification changed.")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} key inventory changed; missing={missing}, unexpected={unexpected}."
        )


def _validate_pilot_report(pilot: Mapping[str, Any]) -> None:
    if (
        pilot.get("status") != "pass"
        or pilot.get("failures") != []
        or pilot.get("steps") != 200
        or pilot.get("full_objective") is not True
    ):
        raise ValueError("Checkpoint-contained step-200 pilot report did not pass.")
    gradients = pilot.get("term_gradient_norms")
    if not isinstance(gradients, Mapping) or set(gradients) != set(_TERMS):
        raise ValueError("Pilot report lacks all six term-gradient summaries.")
    for term in _TERMS:
        item = gradients[term]
        if (
            not isinstance(item, Mapping)
            or item.get("enabled") is not True
            or any(
                isinstance(item.get(key), bool)
                or not isinstance(item.get(key), (int, float))
                or not math.isfinite(float(item[key]))
                or float(item[key]) <= 0
                for key in ("mean", "minimum", "maximum")
            )
        ):
            raise ValueError(f"Pilot report term-gradient evidence changed for {term}.")


def _render_training_dashboard(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    png_path: Path,
    svg_path: Path,
    pdf_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(6, 2, figsize=(16, 24), constrained_layout=True)
    flat = list(axes.flat)
    colors = plt.get_cmap("tab10")

    for index, term in enumerate(_TERMS):
        flat[0].plot(steps, [row[f"raw/{term}"] for row in rows], color=colors(index), alpha=0.35)
        flat[0].plot(
            steps,
            [row[f"rolling20/raw/{term}"] for row in rows],
            color=colors(index),
            label=f"{term} (20-step mean)",
        )
    flat[0].set_title("Raw six-term losses: exact + 20-step rolling mean")
    flat[0].legend(fontsize=7, ncol=2)

    for index, term in enumerate(_TERMS):
        flat[1].plot(
            steps,
            [row[f"weighted/{term}"] for row in rows],
            color=colors(index),
            alpha=0.35,
            label=term,
        )
        flat[1].plot(
            steps,
            [row[f"rolling20/weighted/{term}"] for row in rows],
            color=colors(index),
            linestyle="--",
            label=f"{term} (20-step mean)",
        )
    flat[1].plot(steps, [row["weighted/generator_total"] for row in rows], color="black", label="generator total")
    flat[1].plot(
        steps,
        [row["rolling20/weighted/generator_total"] for row in rows],
        color="black",
        linestyle="--",
        label="generator total (20-step mean)",
    )
    flat[1].set_title("Weighted losses: exact + 20-step rolling mean")
    flat[1].legend(fontsize=7, ncol=2)

    flat[2].plot(steps, [row["gradient/generator_norm"] for row in rows], label="generator")
    flat[2].plot(steps, [row["gradient/critic_norm"] for row in rows], label="critic")
    flat[2].set_title("Generator and critic gradient norms")
    flat[2].legend()

    for index, term in enumerate(_TERMS):
        flat[3].plot(steps, [row[f"gradient/term_{term}"] for row in rows], color=colors(index), label=term)
    flat[3].set_title("Qualified per-term translator gradient norms")
    flat[3].legend(fontsize=7, ncol=2)

    flat[4].plot(steps, [row["critic/real_score_mean"] for row in rows], label="real")
    flat[4].plot(steps, [row["critic/fake_score_mean"] for row in rows], label="fake")
    flat[4].plot(steps, [row["critic/score_separation"] for row in rows], label="separation")
    flat[4].plot(steps, [row["critic/score_saturation_fraction"] for row in rows], label="saturation fraction")
    flat[4].set_title("Critic scores, separation, and saturation")
    flat[4].legend(fontsize=8)

    flat[5].plot(steps, [row["critic/real_domain_accuracy"] for row in rows], label="real")
    flat[5].plot(steps, [row["critic/generated_domain_accuracy"] for row in rows], label="generated")
    flat[5].set_ylim(-0.02, 1.02)
    flat[5].set_title("Real/generated domain accuracy")
    flat[5].legend()

    flat[6].plot(steps, [row["weighted/aux_to_flow_ratio"] for row in rows])
    flat[6].set_title("Auxiliary-to-flow ratio")
    flat[7].plot(steps, [row["generator_lr"] for row in rows], label="generator")
    flat[7].plot(steps, [row["critic_lr"] for row in rows], label="critic")
    flat[7].set_title("Learning rates")
    flat[7].legend()

    flat[8].plot(steps, [row["step_seconds"] for row in rows], label="seconds/step")
    rate_axis = flat[8].twinx()
    rate_axis.plot(steps, [row["examples_per_second"] for row in rows], color="tab:orange", label="examples/s")
    flat[8].set_title("Step duration and examples per second")
    flat[9].plot(steps, [row["peak_cuda_bytes"] / 1024**3 for row in rows], label="allocated GiB")
    flat[9].plot(steps, [row["peak_cuda_reserved_bytes"] / 1024**3 for row in rows], label="reserved GiB")
    flat[9].set_title("Persisted CUDA memory summaries")
    flat[9].legend()

    labels = [Domain(field, contrast).label for contrast in CONTRASTS for field in FIELD_STRENGTHS_T]
    index = {label: item for item, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in rows:
        source, target = str(row["transition"]).split("->")
        matrix[index[source], index[target]] += 1
    image = flat[10].imshow(matrix, cmap="viridis", interpolation="none")
    flat[10].set_title("Domain-transition counts (domain labels only)")
    flat[10].set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
    flat[10].set_yticks(range(len(labels)), labels, fontsize=5)
    fig.colorbar(image, ax=flat[10], fraction=0.03)

    selection = summary["selection_and_validation"]
    text = (
        f"Selection score: {selection['best_score']}\n"
        "Selection: critic-independent frozen rule\n"
        "Validation: complete unpaired R inventory at terminal step\n"
        "Trend / early stopping: not inferable\n"
        "Status: 200-step pilot; convergence not declared"
    )
    flat[11].axis("off")
    flat[11].text(0.02, 0.98, text, va="top", family="monospace")
    flat[11].set_title("Selection and validation summary")
    for axis in flat[:10]:
        axis.set_xlabel("training step")
        axis.grid(alpha=0.2)

    _save_figure_no_clobber(fig, png_path, format="png", dpi=150)
    _save_figure_no_clobber(fig, svg_path, format="svg")
    _save_figure_no_clobber(fig, pdf_path, format="pdf")
    plt.close(fig)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "backend": "Agg",
        "rolling_window_steps": 20,
        "plot_count": 12,
    }


def _render_self_contained_html(path: Path, summary: Mapping[str, Any], png_path: Path) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    summary_text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    escaped = (
        summary_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage-2 step-200 pilot audit</title>
<style>:root{{color-scheme:light dark}}body{{font:16px system-ui;max-width:1400px;margin:auto;padding:2rem;background:Canvas;color:CanvasText}}img{{width:100%;height:auto;border:1px solid GrayText}}pre{{white-space:pre-wrap;background:color-mix(in srgb,CanvasText 8%,Canvas);padding:1rem;overflow:auto}}.warning{{border-left:.4rem solid #c77d00;padding:1rem}}</style></head>
<body><h1>Stage-2 step-200 pilot evidence audit</h1>
<p class="warning">Engineering evidence only. This 200-step checkpoint is not a converged model. No long-run training or population/generalization claim is authorized.</p>
<img alt="Sanitized Stage-2 training dashboard" src="data:image/png;base64,{encoded}">
<h2>Machine-readable summary</h2><pre>{escaped}</pre></body></html>"""
    _write_bytes_atomic(path, html.encode("utf-8"))


def _write_sanitized_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty sanitized history.")
    buffer = io.StringIO(newline="")
    fieldnames = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_bytes_atomic(path, buffer.getvalue().encode("utf-8"))


def _verify_existing_audit(root: Path, expected_inputs: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_self_hashed(root / "artifact_manifest.json", "manifest_sha256")
    if manifest.get("input_manifest") != dict(expected_inputs):
        raise ValueError("Existing pilot audit belongs to different immutable inputs.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Existing pilot audit output manifest is malformed.")
    for filename, item in outputs.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(item, Mapping)
        ):
            raise ValueError("Existing pilot audit output identity is malformed.")
        target = _regular_file(root / filename, "existing audit output")
        if (
            sha256_file_streaming(target) != item.get("file_sha256")
            or target.stat().st_size != item.get("size_bytes")
        ):
            raise ValueError("Existing pilot audit output changed.")
    return _load_self_hashed(root / "stage2_step200_summary.json", "summary_sha256")


def _load_self_hashed(path: Path, hash_key: str) -> dict[str, Any]:
    payload = _load_json(path)
    body = dict(payload)
    stored = body.pop(hash_key, None)
    if stored != sha256_json(body):
        raise ValueError(f"Self-hash mismatch for {path.name}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "JSON input")
    payload = _json_loads_strict(source.read_bytes().decode("utf-8-sig"), source.name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {source.name}")
    _reject_nonfinite_recursive(payload, source.name)
    return dict(payload)


def _json_loads_strict(text: str, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {label}: {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON constant in {label}: {value}.")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {label}.") from exc


def _reject_nonfinite_recursive(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite numeric field in {label}.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite_recursive(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite_recursive(item, label)


def _canonical_transition(value: str) -> str:
    parts = value.split("->")
    if len(parts) != 2:
        raise ValueError("History transition is malformed.")
    allowed = {
        Domain(field, contrast).label for contrast in CONTRASTS for field in FIELD_STRENGTHS_T
    }
    if parts[0] not in allowed or parts[1] not in allowed:
        raise ValueError("History transition contains a noncanonical domain label.")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink.")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {label}: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file.")
    return resolved


def _require_git(value: str, label: str) -> None:
    if _GIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git identity.")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite audit artifact: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Audit temporary path already exists: {temporary.name}")
    try:
        temporary.write_bytes(payload)
        with temporary.open("ab") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure_no_clobber(fig: Any, path: Path, **kwargs: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite audit figure: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PILOT_AUDIT_CONTRACT",
    "PILOT_AUDIT_MANIFEST_CONTRACT",
    "PILOT_AUDIT_SANITIZATION_CONTRACT",
    "STEP200_CHECKPOINT_SHA256",
    "STEP200_RUN_FINGERPRINT",
    "Step200PilotAuditInputs",
    "VerifiedStep200PilotEvidence",
    "build_step200_pilot_summary",
    "run_step200_pilot_evidence_audit",
    "sanitize_step200_history",
    "sha256_file_streaming",
    "verify_step200_pilot_evidence",
]
