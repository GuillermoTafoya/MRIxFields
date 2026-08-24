from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_file


class _Index:
    def __init__(self, split: str) -> None:
        self.split = split
        self.artifact_sha256 = ("a" if split == "train" else "b") * 64
        self.records = [
            SimpleNamespace(sidecar={"cohort": "R"}, resume_key=f"{split}-record")
        ]
        self.manifest = {"vae": {"checkpoint_sha256": "c" * 64}}


class _Stats:
    artifact_sha256 = "d" * 64

    @classmethod
    def from_bank(cls, bank_dir: str | Path) -> "_Stats":
        del bank_dir
        return cls()


def _selection_path(root: Path, steps: int) -> Path:
    path = (
        root
        / f"unified_full_objective_pilot_{steps}"
        / "scientific_attempts"
        / "attempt-0001"
        / "checkpoints"
        / f"stage2_unified_full_selection_step{steps:09d}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(f"synthetic-selection:{steps}" + chr(10), encoding="utf-8")
    return path


def _pilot_summary(steps: int) -> dict[str, object]:
    return {
        "status": "pass",
        "steps": steps,
        "latest_checkpoint_file_sha256": "e" * 64,
        "run_fingerprint": ("f" if steps == 200 else "1") * 64,
        "resolved_config_sha256": "2" * 64,
        "bank_artifact_sha256": "a" * 64,
        "validation_bank_artifact_sha256": "b" * 64,
        "frozen_decoder_state_sha256": "3" * 64,
        "decoder_activation_checkpoint_sha256": "4" * 64,
        "complete_r_validation_inventory": True,
        "paired_targets_used": False,
    }


def _qualification_summary() -> dict[str, object]:
    return {
        "status": "pass",
        "frozen_decoder_state_sha256": "3" * 64,
        "decoder_activation_checkpoint_sha256": "4" * 64,
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_compatible_step200_evidence_is_reused_without_training_or_namespace_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_commit = "9" * 64
    root = tmp_path / f"implementation_{training_commit[:12]}"
    full_receipt = _selection_path(root, 200)
    _selection_path(root, 20)
    before = _tree_bytes(root)
    calls: list[int] = []

    monkeypatch.setattr(
        p0006,
        "PhotometryFactoredLatentBankIndex",
        lambda bank_dir, split: _Index(split),
    )
    monkeypatch.setattr(p0006, "FactoredLatentStats", _Stats)

    def verify_pilot(run_dir, *, steps, **kwargs):
        del run_dir, kwargs
        calls.append(steps)
        return _pilot_summary(steps)

    monkeypatch.setattr(p0006, "_verify_completed_pilot_attempt", verify_pilot)
    monkeypatch.setattr(
        p0006,
        "_verify_completed_a100_qualification",
        lambda *args, **kwargs: _qualification_summary(),
    )
    result = p0006.verify_completed_stage2_pilot_evidence(
        root,
        bank_dir=tmp_path / "bank",
        expected_selection_receipt_path=full_receipt,
        training_evidence_commit=training_commit,
        expected_selection_receipt_file_sha256=sha256_file(full_receipt),
        expected_validation_plan_sha256="5" * 64,
        expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
    )
    assert calls == [200, 20]
    assert result["status"] == "pass"
    assert result["training_reused"] is True
    assert result["training_invoked"] is False
    assert result["latest_completed_step"] == 200
    assert result["complete_r_validation_inventory"] is True
    assert result["paired_targets_used"] is False
    assert _tree_bytes(root) == before


def test_incompatible_completed_evidence_fails_without_retraining_or_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_commit = "8" * 64
    root = tmp_path / f"implementation_{training_commit[:12]}"
    full_receipt = _selection_path(root, 200)
    _selection_path(root, 20)
    before = _tree_bytes(root)
    monkeypatch.setattr(
        p0006,
        "PhotometryFactoredLatentBankIndex",
        lambda bank_dir, split: _Index(split),
    )
    monkeypatch.setattr(p0006, "FactoredLatentStats", _Stats)
    monkeypatch.setattr(
        p0006,
        "_verify_completed_pilot_attempt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("persisted history is incomplete through step 200")
        ),
    )
    with pytest.raises(ValueError, match="incomplete through step 200"):
        p0006.verify_completed_stage2_pilot_evidence(
            root,
            bank_dir=tmp_path / "bank",
            expected_selection_receipt_path=full_receipt,
            training_evidence_commit=training_commit,
            expected_selection_receipt_file_sha256=sha256_file(full_receipt),
            expected_validation_plan_sha256="5" * 64,
            expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
        )
    assert _tree_bytes(root) == before


def test_completed_history_requires_exact_rows_through_step200(tmp_path: Path) -> None:
    run_fingerprint = "a" * 64
    plan_sha = "b" * 64
    pilot = {"status": "pass", "steps": 200}
    rows = []
    for step in range(1, 201):
        row = {
            "contract_version": p0006.UNIFIED_HISTORY_CONTRACT,
            "run_fingerprint": run_fingerprint,
            "step": step,
        }
        for term in p0006.DEFAULT_UNIFIED_WEIGHTS:
            row[f"raw/{term}"] = 1.0
            row[f"weighted/{term}"] = 0.1
            row[f"gradient/term_{term}"] = 0.01
        rows.append(row)
    rows.extend(
        [
            {
                "contract_version": p0006.UNIFIED_HISTORY_CONTRACT,
                "run_fingerprint": run_fingerprint,
                "event": "unpaired_validation",
                "step": 200,
                "validation": {
                    "complete_inventory_used": True,
                    "validation_plan_sha256": plan_sha,
                },
            },
            {
                "contract_version": p0006.UNIFIED_HISTORY_CONTRACT,
                "run_fingerprint": run_fingerprint,
                "event": "full_objective_pilot",
                "step": 200,
                "pilot": pilot,
                "validation_plan_sha256": plan_sha,
            },
        ]
    )
    history = tmp_path / "history.jsonl"
    history.write_text(
        "".join(json.dumps(row, sort_keys=True) + chr(10) for row in rows),
        encoding="utf-8",
    )
    result = p0006._verify_completed_history(
        history,
        steps=200,
        run_fingerprint=run_fingerprint,
        validation_plan_sha256=plan_sha,
        pilot=pilot,
    )
    assert result["training_row_count"] == 200
    history.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + chr(10)
            for row in rows
            if row.get("step") != 199
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly complete"):
        p0006._verify_completed_history(
            history,
            steps=200,
            run_fingerprint=run_fingerprint,
            validation_plan_sha256=plan_sha,
            pilot=pilot,
        )
