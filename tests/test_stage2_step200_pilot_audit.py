from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from fieldbridge.data.photometry_factorization import sha256_json
from fieldbridge.evaluation import stage2_step200_pilot_audit as audit
from fieldbridge.training.stage2_unified import (
    DEFAULT_UNIFIED_WEIGHTS,
    UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
    UNIFIED_HISTORY_CONTRACT,
    UNIFIED_RESUME_CONTRACT,
    UNIFIED_SELECTION_RULE_SHA256,
    UNIFIED_VALIDATION_PLAN_CONTRACT,
)


AUDIT_COMMIT = "a" * 40
TRAINING_COMMIT = audit.TRAINING_EVIDENCE_COMMIT
RUN_FINGERPRINT = "b" * 64
PLAN_SHA = "c" * 64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_self(path: Path, body: dict[str, Any], key: str) -> dict[str, Any]:
    payload = {**body, key: sha256_json(body)}
    _write_json(path, payload)
    return payload


def _pilot() -> dict[str, Any]:
    return {
        "status": "pass",
        "failures": [],
        "steps": 200,
        "full_objective": True,
        "term_gradient_norms": {
            term: {"enabled": True, "mean": 0.2, "minimum": 0.1, "maximum": 0.3}
            for term in DEFAULT_UNIFIED_WEIGHTS
        },
    }


def _training_row(step: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "contract_version": UNIFIED_HISTORY_CONTRACT,
        "run_fingerprint": RUN_FINGERPRINT,
        "step": step,
        "transition": "0.1T/T1w->1.5T/T1w",
        "source_records": [f"PRIVATE-SOURCE-{step}"],
        "target_records": [f"PRIVATE-TARGET-{step}"],
        "weighted/generator_total": 1.0 / step,
        "weighted/auxiliary_total": 0.5 / step,
        "weighted/aux_to_flow_ratio": 0.5,
        "gradient/generator_norm": 0.4,
        "gradient/critic_norm": 0.3,
        "critic/real_score_mean": 0.6,
        "critic/fake_score_mean": -0.2,
        "critic/score_separation": 0.8,
        "critic/score_saturation_fraction": 0.01,
        "critic/real_domain_accuracy": 0.7,
        "critic/generated_domain_accuracy": 0.6,
        "generator_lr": 1.0e-4,
        "critic_lr": 1.0e-4,
        "step_seconds": 2.0,
        "examples_per_second": 0.5,
        "peak_cuda_bytes": 1024,
        "peak_cuda_reserved_bytes": 2048,
    }
    for term, weight in DEFAULT_UNIFIED_WEIGHTS.items():
        row[f"raw/{term}"] = float(step) / 100.0
        row[f"weighted/{term}"] = row[f"raw/{term}"] * weight
        row[f"gradient/term_{term}"] = 0.1 + step / 10000.0
    return row


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> audit.Step200PilotAuditInputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan_path = tmp_path / "stage2_unified_validation_plan_v2.json"
    plan_body = {
        "contract_version": UNIFIED_VALIDATION_PLAN_CONTRACT,
        "scope": "complete_R_validation_inventory",
        "required_directed_domain_cell_count": 60,
        "entries": [],
    }
    plan_sha = sha256_json(plan_body)
    _write_json(plan_path, {**plan_body, "validation_plan_sha256": plan_sha})
    monkeypatch.setattr(audit, "STEP200_VALIDATION_PLAN_SHA256", plan_sha)

    checkpoint = tmp_path / "stage2_unified_full_step000000200.pt"
    pilot = _pilot()
    state = {
        "contract_version": UNIFIED_RESUME_CONTRACT,
        "_meta": {"git_commit": TRAINING_COMMIT, "config": {"device": "cuda"}},
        "training_cursor": 200,
        "run_fingerprint": RUN_FINGERPRINT,
        "validation_plan_sha256": plan_sha,
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "validation_selection": {
            "validation_plan_sha256": plan_sha,
            "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
            "paired_targets_used": False,
            "complete_r_validation_inventory": True,
            "latest_step": 200,
            "best_step": 200,
            "best_score": 0.25,
        },
        "pilot_report": pilot,
    }
    torch.save(state, checkpoint)
    checkpoint_sha = audit.sha256_file_streaming(checkpoint)
    monkeypatch.setattr(audit, "STEP200_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(audit, "STEP200_RUN_FINGERPRINT", RUN_FINGERPRINT)
    monkeypatch.setattr(audit, "STEP200_SELECTION_RULE_SHA256", UNIFIED_SELECTION_RULE_SHA256)

    selection_path = tmp_path / "stage2_unified_full_selection_step000000200.json"
    selection_body = {
        "receipt_contract_version": "stage2-unified-selection-receipt-v3",
        "contract_version": "stage2-unified-unpaired-validation-selection-v3",
        "validation_plan_sha256": plan_sha,
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "paired_targets_used": False,
        "complete_r_validation_inventory": True,
        "latest_step": 200,
        "latest_checkpoint": str(checkpoint),
        "best_step": 200,
        "best_checkpoint": str(checkpoint),
        "best_score": 0.25,
        "variant": "full",
        "receipt_step": 200,
        "terminal_step": 200,
        "run_complete": True,
        "checkpoint_hashes": {
            role: {"path": str(checkpoint), "file_sha256": checkpoint_sha}
            for role in ("latest", "best", "final")
        },
    }
    _write_self(selection_path, selection_body, "selection_sha256")
    selection_file_sha = audit.sha256_file_streaming(selection_path)
    monkeypatch.setattr(audit, "STEP200_SELECTION_RECEIPT_FILE_SHA256", selection_file_sha)

    config_path = tmp_path / "resolved_config.json"
    _write_json(
        config_path,
        {
            "training": {
                "device": "auto",
                "batch_size": 1,
                "precision": "bf16",
                "integration_steps": 4,
                "integration_solver": "heun",
                "loss_weights": DEFAULT_UNIFIED_WEIGHTS,
            }
        },
    )
    history_path = tmp_path / "history.jsonl"
    rows = [_training_row(step) for step in range(1, 201)]
    rows.extend(
        [
            {
                "contract_version": UNIFIED_HISTORY_CONTRACT,
                "run_fingerprint": RUN_FINGERPRINT,
                "event": "unpaired_validation",
                "step": 200,
                "validation": {
                    "complete_inventory_used": True,
                    "validation_plan_sha256": plan_sha,
                },
            },
            {
                "contract_version": UNIFIED_HISTORY_CONTRACT,
                "run_fingerprint": RUN_FINGERPRINT,
                "event": "full_objective_pilot",
                "step": 200,
                "pilot": pilot,
                "validation_plan_sha256": plan_sha,
            },
        ]
    )
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    qualification_path = tmp_path / "stage2_unified_a100_qualification_only_receipt_v1.json"
    _write_self(
        qualification_path,
        {
            "contract_version": UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
            "status": "pass",
            "step": 1,
            "complete_validation_executed": False,
            "checkpoint_written": False,
            "generator_optimizer_updates": 1,
            "validation_plan_sha256": plan_sha,
            "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
            "run_fingerprint": "d" * 64,
            "gate": {
                "status": "pass",
                "gpu_identity_matches": True,
                "within_allocated_limit": True,
                "full_objective": True,
            },
        },
        "receipt_sha256",
    )
    recovery_path = tmp_path / "stage2_gate01_recovery_receipt_v2.json"
    _write_self(
        recovery_path,
        {
            "training_evidence_commit": TRAINING_COMMIT,
            "selection_receipt_file_sha256": selection_file_sha,
            "checkpoint_file_sha256": checkpoint_sha,
            "run_fingerprint": RUN_FINGERPRINT,
            "training_reused": True,
            "training_invoked": False,
            "long_run_training_authorized": False,
            "population_or_generalization_claims_authorized": False,
            "P0009_executed": False,
        },
        "receipt_sha256",
    )
    return audit.Step200PilotAuditInputs(
        checkpoint=checkpoint,
        history_jsonl=history_path,
        resolved_config=config_path,
        selection_receipt=selection_path,
        validation_plan=plan_path,
        a100_qualification_receipt=qualification_path,
        recovery_receipt=recovery_path,
    )


def _rewrite_history(path: Path, mutate: Any) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_exact_200_row_evidence_verifies_and_sanitizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    verified = audit.verify_step200_pilot_evidence(
        inputs, audit_implementation_commit=AUDIT_COMMIT
    )
    assert len(verified.training_rows) == 200
    sanitized = audit.sanitize_step200_history(verified.training_rows)
    assert [row["step"] for row in sanitized] == list(range(1, 201))
    serialized = json.dumps(sanitized)
    assert "PRIVATE-SOURCE" not in serialized
    assert "PRIVATE-TARGET" not in serialized
    assert "source_records" not in serialized
    assert "target_records" not in serialized
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows: rows.pop(50), "steps 1..200"),
        (lambda rows: rows.insert(50, dict(rows[49])), "steps 1..200"),
        (lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]), "steps 1..200"),
    ],
)
def test_missing_duplicate_or_out_of_order_history_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    _rewrite_history(inputs.history_jsonl, mutation)
    with pytest.raises(ValueError, match=message):
        audit.verify_step200_pilot_evidence(inputs, audit_implementation_commit=AUDIT_COMMIT)


def test_run_fingerprint_mismatch_and_nonfinite_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    _rewrite_history(
        inputs.history_jsonl,
        lambda rows: rows[0].__setitem__("run_fingerprint", "e" * 64),
    )
    with pytest.raises(ValueError, match="run fingerprint"):
        audit.verify_step200_pilot_evidence(inputs, audit_implementation_commit=AUDIT_COMMIT)

    inputs = _fixture(tmp_path / "nonfinite", monkeypatch)
    text = inputs.history_jsonl.read_text(encoding="utf-8")
    inputs.history_jsonl.write_text(text.replace('"raw/sb": 0.01', '"raw/sb": NaN', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.verify_step200_pilot_evidence(inputs, audit_implementation_commit=AUDIT_COMMIT)


def test_summary_is_deterministic_and_labels_separate_pilots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    verified = audit.verify_step200_pilot_evidence(
        inputs, audit_implementation_commit=AUDIT_COMMIT
    )
    first = audit.build_step200_pilot_summary(verified, step20_evidence_independent=True)
    second = audit.build_step200_pilot_summary(verified, step20_evidence_independent=True)
    assert first == second
    assert first["step20_and_step200_relationship"] == "separate_pilots_no_continuous_trajectory_claim"
    assert first["convergence_declared"] is False
    assert first["engineering_assessment"]["available_validation_evidence"]["validation_trend_inferable"] is False


def test_full_cpu_audit_renders_and_exactly_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    inputs = _fixture(tmp_path / "inputs", monkeypatch)
    out = tmp_path / "audit"
    first = audit.run_step200_pilot_evidence_audit(
        inputs, output_dir=out, audit_implementation_commit=AUDIT_COMMIT
    )
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.iterdir()
        if path.is_file()
    }
    second = audit.run_step200_pilot_evidence_audit(
        inputs, output_dir=out, audit_implementation_commit=AUDIT_COMMIT
    )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.iterdir()
        if path.is_file()
    }
    assert first["summary_sha256"] == second["summary_sha256"]
    assert before == after
    expected = {
        "stage2_step200_history_sanitized.csv",
        "stage2_step200_summary.json",
        "stage2_step200_training_curves.svg",
        "stage2_step200_training_curves.png",
        "stage2_step200_pilot_audit.html",
        "stage2_step200_pilot_audit.pdf",
        "artifact_manifest.json",
    }
    assert {path.name for path in out.iterdir()} == expected
    with (out / "stage2_step200_history_sanitized.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert not any("record" in key.lower() or "path" in key.lower() for key in rows[0])
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bank_opened"] is False
    assert manifest["private_arrays_opened"] is False
    assert manifest["gpu_used"] is False
    assert manifest["rendering"]["plot_count"] == 12
    assert "filename" not in json.dumps(manifest).lower()


def test_cpu_audit_source_has_no_bank_array_or_gpu_access() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "PhotometryFactoredLatentBankIndex" not in source
    assert "torch.cuda" not in source
    assert "load_gate01" not in source
    assert ".source_records" not in source
    assert ".target_records" not in source
