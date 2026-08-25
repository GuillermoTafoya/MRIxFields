from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_file


AUTHENTIC_TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
CONFIGURATION_OVERRIDE_PROVENANCE = {
    "contract_version": (
        p0006.STAGE2_COMPLETED_EVIDENCE_CONFIG_RECONSTRUCTION_CONTRACT
    ),
    "declared_device": "auto",
    "effective_device": "cuda",
    "override_source": p0006.STAGE2_COMPLETED_EVIDENCE_DEVICE_OVERRIDE_SOURCE,
    "normalized_changed_fields": ["device"],
    "no_other_normalized_configuration_field_changed": True,
}


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
        "resolved_config_file_sha256": "6" * 64,
        "declared_normalized_config_sha256": "7" * 64,
        "effective_normalized_config_sha256": "2" * 64,
        "bank_artifact_sha256": "a" * 64,
        "validation_bank_artifact_sha256": "b" * 64,
        "frozen_decoder_state_sha256": "3" * 64,
        "decoder_activation_checkpoint_sha256": "4" * 64,
        "configuration_override_provenance": CONFIGURATION_OVERRIDE_PROVENANCE,
        "complete_r_validation_inventory": True,
        "paired_targets_used": False,
    }


def _qualification_summary() -> dict[str, object]:
    return {
        "status": "pass",
        "frozen_decoder_state_sha256": "3" * 64,
        "decoder_activation_checkpoint_sha256": "4" * 64,
        "configuration_override_provenance": CONFIGURATION_OVERRIDE_PROVENANCE,
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
    training_commit = AUTHENTIC_TRAINING_EVIDENCE_COMMIT
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
    assert result["training_evidence_commit"] == AUTHENTIC_TRAINING_EVIDENCE_COMMIT
    assert Path(result["training_namespace"]).name == (
        "implementation_" + AUTHENTIC_TRAINING_EVIDENCE_COMMIT[:12]
    )
    serialized = json.dumps(result, sort_keys=True)
    assert json.loads(serialized)["training_evidence_commit"] == (
        AUTHENTIC_TRAINING_EVIDENCE_COMMIT
    )
    assert _tree_bytes(root) == before


@pytest.mark.parametrize(
    "training_commit",
    [
        "a" * 39,
        "a" * 41,
        "a" * 64,
        "A" * 40,
        "g" * 40,
    ],
)
def test_training_evidence_commit_rejects_non_git_identities(
    tmp_path: Path, training_commit: str
) -> None:
    with pytest.raises(ValueError, match="lowercase Git commit identity"):
        p0006.verify_completed_stage2_pilot_evidence(
            tmp_path / "missing",
            bank_dir=tmp_path / "bank",
            training_evidence_commit=training_commit,
        )


@pytest.mark.parametrize(
    "training_commit",
    [AUTHENTIC_TRAINING_EVIDENCE_COMMIT, "a" * 40],
)
def test_valid_lowercase_git_commit_identity_reaches_evidence_resolution(
    tmp_path: Path, training_commit: str
) -> None:
    missing = tmp_path / f"implementation_{training_commit[:12]}"
    with pytest.raises(FileNotFoundError, match="Completed training namespace is missing"):
        p0006.verify_completed_stage2_pilot_evidence(
            missing,
            bank_dir=tmp_path / "bank",
            training_evidence_commit=training_commit,
        )


@pytest.mark.parametrize(
    ("parameter", "valid_sha256"),
    [
        ("expected_selection_receipt_file_sha256", "a" * 64),
        ("expected_validation_plan_sha256", "b" * 64),
        ("expected_selection_rule_sha256", p0006.UNIFIED_SELECTION_RULE_SHA256),
    ],
)
def test_sha256_parameters_accept_64_character_lowercase_identities(
    tmp_path: Path, parameter: str, valid_sha256: str
) -> None:
    kwargs = {
        "expected_selection_receipt_file_sha256": "a" * 64,
        "expected_validation_plan_sha256": "b" * 64,
        "expected_selection_rule_sha256": p0006.UNIFIED_SELECTION_RULE_SHA256,
        parameter: valid_sha256,
    }
    with pytest.raises(FileNotFoundError, match="Completed training namespace is missing"):
        p0006.verify_completed_stage2_pilot_evidence(
            tmp_path / f"implementation_{AUTHENTIC_TRAINING_EVIDENCE_COMMIT[:12]}",
            bank_dir=tmp_path / "bank",
            training_evidence_commit=AUTHENTIC_TRAINING_EVIDENCE_COMMIT,
            **kwargs,
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "expected_selection_receipt_file_sha256",
        "expected_validation_plan_sha256",
        "expected_selection_rule_sha256",
    ],
)
def test_sha256_parameters_reject_40_character_git_identity(
    tmp_path: Path, parameter: str
) -> None:
    kwargs = {
        "expected_selection_receipt_file_sha256": "a" * 64,
        "expected_validation_plan_sha256": "b" * 64,
        "expected_selection_rule_sha256": p0006.UNIFIED_SELECTION_RULE_SHA256,
        parameter: AUTHENTIC_TRAINING_EVIDENCE_COMMIT,
    }
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        p0006.verify_completed_stage2_pilot_evidence(
            tmp_path / f"implementation_{AUTHENTIC_TRAINING_EVIDENCE_COMMIT[:12]}",
            bank_dir=tmp_path / "bank",
            training_evidence_commit=AUTHENTIC_TRAINING_EVIDENCE_COMMIT,
            **kwargs,
        )


def test_completed_evidence_uses_distinct_git_and_sha_identity_contracts() -> None:
    source = Path(p0006.__file__).read_text(encoding="utf-8")
    verifier = source[
        source.index("def verify_completed_stage2_pilot_evidence(") : source.index(
            "def _verify_completed_pilot_attempt(")
    ]
    assert "_GIT_COMMIT_RE.fullmatch(training_evidence_commit)" in verifier
    assert "_SHA256_RE.fullmatch(training_evidence_commit)" not in verifier


def test_incompatible_completed_evidence_fails_without_retraining_or_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_commit = "8" * 40
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
