from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json


TRAINING_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
DECODER_SHA256 = "d" * 64


class _Index:
    def __init__(self, split: str) -> None:
        self.split = split
        self.artifact_sha256 = ("a" if split == "train" else "b") * 64
        self.records = [
            SimpleNamespace(sidecar={"cohort": "R"}, resume_key=f"{split}-record")
        ]
        self.manifest = {"vae": {"checkpoint_sha256": "c" * 64}}


class _Stats:
    artifact_sha256 = "e" * 64

    @classmethod
    def from_bank(cls, _bank_dir: str | Path) -> "_Stats":
        return cls()


def _resolved_payload(*, steps: int, pilot_steps: int, device: Any = "auto") -> dict:
    return {
        "training": {
            "steps": steps,
            "device": device,
            "pilot": {"steps": pilot_steps},
            "variant": "full",
            "loss_weights": dict(p0006.DEFAULT_UNIFIED_WEIGHTS),
        }
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_self_hashed(path: Path, payload: dict, key: str) -> dict:
    sealed = dict(payload)
    sealed[key] = sha256_json(payload)
    _write_json(path, sealed)
    return sealed


def _checkpoint_evidence() -> dict[str, Any]:
    return {
        "mode": "fine_grained_full_volume_v1",
        "full_volume": True,
        "group_norm_scope": "complete_spatial_volume",
        "upsample_regions": ["up1", "up2"],
        "residual_branch_regions": ["synthetic-residual-branch"],
        "outer_full_decoder_checkpoint": False,
    }


def _anatomy() -> dict[str, Any]:
    checkpoint = _checkpoint_evidence()
    return {
        "contract_version": p0006.UNIFIED_ANATOMY_MEMORY_CONTRACT,
        "status": "pass",
        "decoder_state_unchanged": True,
        "decoder_state_sha256_before": DECODER_SHA256,
        "decoder_state_sha256_after": DECODER_SHA256,
        "source_decode_checkpointed": False,
        "spatial_crop_or_tile": False,
        "allocator_fallback": False,
        "checkpoint_evidence": checkpoint,
        "checkpoint_evidence_sha256": sha256_json(checkpoint),
    }


def _gate() -> dict[str, Any]:
    return {
        "contract_version": p0006.UNIFIED_A100_GATE_CONTRACT,
        "status": "pass",
        "gpu_identity_matches": True,
        "within_allocated_limit": True,
        "peak_allocated_limit_bytes": p0006.UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES,
        "peak_allocated_bytes": 1,
        "full_objective": True,
    }


def _pilot_report(steps: int) -> dict[str, Any]:
    accumulation = p0006._expected_accumulation_identity()
    accumulation["joint_six_term_gradient_probe"] = False
    values = {term: 1.0 for term in p0006.DEFAULT_UNIFIED_WEIGHTS}
    gradients = {
        term: {"enabled": True, "mean": 1.0, "minimum": 1.0, "maximum": 1.0}
        for term in p0006.DEFAULT_UNIFIED_WEIGHTS
    }
    return {
        "status": "pass",
        "failures": [],
        "steps": steps,
        "full_objective": True,
        "term_gradient_norms": gradients,
        "raw_term_means": values,
        "weighted_term_means": values,
        "decoder_activation_checkpoint": {
            "contract_version": (
                "klvae3d-full-volume-fine-grained-activation-checkpoint-v1"
            ),
            "mode": "fine_grained_full_volume_v1",
            "outer_full_decoder_checkpoint": False,
        },
        "generator_gradient_accumulation": accumulation,
        "one_step_a100_memory_gate": _gate(),
        "one_step_anatomy_memory_qualification": _anatomy(),
    }


def _validation_plan(validation_index: _Index) -> dict[str, Any]:
    body = {
        "contract_version": p0006.UNIFIED_VALIDATION_PLAN_CONTRACT,
        "scope": "complete_R_validation_inventory",
        "required_directed_domain_cell_count": 60,
        "bank_artifact_sha256": validation_index.artifact_sha256,
        "validation_seed": 20_260_818,
        "entries": [],
    }
    return {**body, "validation_plan_sha256": sha256_json(body)}


def _history_bytes(
    *, steps: int, run_fingerprint: str, plan_sha256: str, pilot: dict[str, Any]
) -> bytes:
    rows: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        row: dict[str, Any] = {
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
                "step": steps,
                "validation": {
                    "complete_inventory_used": True,
                    "validation_plan_sha256": plan_sha256,
                },
            },
            {
                "contract_version": p0006.UNIFIED_HISTORY_CONTRACT,
                "run_fingerprint": run_fingerprint,
                "event": "full_objective_pilot",
                "step": steps,
                "pilot": pilot,
                "validation_plan_sha256": plan_sha256,
            },
        ]
    )
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _selection_fields(
    *, steps: int, checkpoint_path: Path, plan_sha256: str
) -> dict[str, Any]:
    return {
        "contract_version": p0006.UNIFIED_SELECTION_CONTRACT,
        "validation_plan_sha256": plan_sha256,
        "selection_rule_sha256": p0006.UNIFIED_SELECTION_RULE_SHA256,
        "paired_targets_used": False,
        "complete_r_validation_inventory": True,
        "latest_step": steps,
        "latest_checkpoint": str(checkpoint_path),
        "best_step": steps,
        "best_checkpoint": str(checkpoint_path),
        "best_score": 0.0,
    }


def _build_pilot_attempt(
    run_dir: Path,
    *,
    steps: int,
    train_index: _Index,
    validation_index: _Index,
    stats: _Stats,
    plan: dict[str, Any],
    checkpoint_device: str = "cuda",
    checkpoint_other_change: bool = False,
    state_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    _write_json(
        run_dir / "resolved_config.json",
        _resolved_payload(steps=steps, pilot_steps=steps),
    )
    checkpoint_dir = run_dir / "scientific_attempts" / "attempt-0001" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    plan_path = checkpoint_dir / "stage2_unified_validation_plan_v2.json"
    _write_json(plan_path, plan)
    reconstruction = p0006._load_completed_stage2_config(run_dir)
    effective_config = dict(reconstruction.effective_config)
    checkpoint_config = dict(effective_config)
    checkpoint_config["device"] = checkpoint_device
    if checkpoint_other_change:
        checkpoint_config["seed"] = int(checkpoint_config["seed"]) + 1
    pilot = _pilot_report(steps)
    run_fingerprint = p0006._rebuild_stage2_run_fingerprint(
        effective_config,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        validation_plan=plan,
        frozen_decoder_state_sha256=DECODER_SHA256,
        decoder_checkpoint_evidence=_checkpoint_evidence(),
        git_commit=TRAINING_COMMIT,
        execution_mode="train_with_frozen_validation_and_selection",
        accumulation=pilot["generator_gradient_accumulation"],
    )
    attempt_root = checkpoint_dir.parent
    history_path = attempt_root / "history.jsonl"
    history = _history_bytes(
        steps=steps,
        run_fingerprint=run_fingerprint,
        plan_sha256=plan["validation_plan_sha256"],
        pilot=pilot,
    )
    history_path.write_bytes(history)
    checkpoint_path = checkpoint_dir / f"stage2_unified_full_step{steps:09d}.pt"
    selection = _selection_fields(
        steps=steps,
        checkpoint_path=checkpoint_path,
        plan_sha256=plan["validation_plan_sha256"],
    )
    state: dict[str, Any] = {
        "contract_version": p0006.UNIFIED_RESUME_CONTRACT,
        "_meta": {
            "seed": effective_config["seed"],
            "config": checkpoint_config,
            "git_commit": TRAINING_COMMIT,
        },
        "validation_plan_sha256": plan["validation_plan_sha256"],
        "selection_rule_sha256": p0006.UNIFIED_SELECTION_RULE_SHA256,
        "run_fingerprint": run_fingerprint,
        "training_cursor": steps,
        "pilot_report": pilot,
        "validation_selection": selection,
        "history_prefix_bytes": len(history),
        "history_prefix_sha256": hashlib.sha256(history).hexdigest(),
    }
    if state_mutator is not None:
        state_mutator(state)
    torch.save(state, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    hashes = {
        role: {"path": str(checkpoint_path), "file_sha256": checkpoint_sha256}
        for role in ("latest", "best", "final")
    }
    receipt_body = {
        "receipt_contract_version": p0006.UNIFIED_SELECTION_RECEIPT_CONTRACT,
        **selection,
        "variant": "full",
        "receipt_step": steps,
        "terminal_step": steps,
        "run_complete": True,
        "checkpoint_hashes": hashes,
    }
    receipt_path = (
        checkpoint_dir / f"stage2_unified_full_selection_step{steps:09d}.json"
    )
    _write_self_hashed(receipt_path, receipt_body, "selection_sha256")
    return receipt_path


def _build_qualification(
    run_dir: Path,
    *,
    train_index: _Index,
    validation_index: _Index,
    stats: _Stats,
    plan: dict[str, Any],
) -> Path:
    _write_json(
        run_dir / "resolved_config.json",
        _resolved_payload(steps=1, pilot_steps=0),
    )
    checkpoint_dir = run_dir / "scientific_attempts" / "attempt-0001" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    _write_json(checkpoint_dir / "stage2_unified_validation_plan_v2.json", plan)
    reconstruction = p0006._load_completed_stage2_config(run_dir)
    anatomy = _anatomy()
    run_fingerprint = p0006._rebuild_stage2_run_fingerprint(
        reconstruction.effective_config,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        validation_plan=plan,
        frozen_decoder_state_sha256=DECODER_SHA256,
        decoder_checkpoint_evidence=anatomy["checkpoint_evidence"],
        git_commit=TRAINING_COMMIT,
        execution_mode=p0006.UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
        accumulation=None,
    )
    body = {
        "contract_version": p0006.UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
        "status": "pass",
        "step": 1,
        "complete_validation_executed": False,
        "checkpoint_written": False,
        "generator_optimizer_updates": 1,
        "validation_plan_sha256": plan["validation_plan_sha256"],
        "selection_rule_sha256": p0006.UNIFIED_SELECTION_RULE_SHA256,
        "gate": _gate(),
        "anatomy_memory_qualification": anatomy,
        "frozen_decoder_state_sha256": DECODER_SHA256,
        "decoder_activation_checkpoint_sha256": sha256_json(
            anatomy["checkpoint_evidence"]
        ),
        "run_fingerprint": run_fingerprint,
    }
    path = checkpoint_dir / p0006.UNIFIED_A100_QUALIFICATION_RECEIPT_FILENAME
    _write_self_hashed(path, body, "receipt_sha256")
    return path


@pytest.mark.parametrize("steps", [20, 200])
def test_direct_pilot_checkpoint_and_fingerprint_use_effective_cuda_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, steps: int
) -> None:
    train_index = _Index("train")
    validation_index = _Index("validation")
    stats = _Stats()
    plan = _validation_plan(validation_index)
    monkeypatch.setattr(p0006, "build_unified_validation_plan", lambda *_a, **_k: plan)
    run_dir = tmp_path / f"unified_full_objective_pilot_{steps}"
    receipt = _build_pilot_attempt(
        run_dir,
        steps=steps,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
    )

    result = p0006._verify_completed_pilot_attempt(
        run_dir,
        steps=steps,
        selection_receipt_path=receipt,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        training_evidence_commit=TRAINING_COMMIT,
        expected_validation_plan_sha256=plan["validation_plan_sha256"],
        expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
    )

    assert result["status"] == "pass"
    assert result["configuration_override_provenance"]["declared_device"] == "auto"
    assert result["configuration_override_provenance"]["effective_device"] == "cuda"
    assert result["declared_normalized_config_sha256"] != result[
        "effective_normalized_config_sha256"
    ]
    assert result["resolved_config_sha256"] == result[
        "effective_normalized_config_sha256"
    ]


def test_resolved_auto_reconstructs_only_effective_cuda(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "resolved_config.json",
        _resolved_payload(steps=20, pilot_steps=20),
    )
    reconstruction = p0006._load_completed_stage2_config(run_dir)
    differing = sorted(
        key
        for key in reconstruction.declared_config
        if reconstruction.declared_config[key] != reconstruction.effective_config[key]
    )
    assert reconstruction.declared_config["device"] == "auto"
    assert reconstruction.effective_config["device"] == "cuda"
    assert differing == ["device"]
    assert reconstruction.override_provenance == {
        "contract_version": (
            p0006.STAGE2_COMPLETED_EVIDENCE_CONFIG_RECONSTRUCTION_CONTRACT
        ),
        "declared_device": "auto",
        "effective_device": "cuda",
        "override_source": p0006.STAGE2_COMPLETED_EVIDENCE_DEVICE_OVERRIDE_SOURCE,
        "normalized_changed_fields": ["device"],
        "no_other_normalized_configuration_field_changed": True,
    }


def test_historical_device_replay_is_pinned_and_runtime_independent() -> None:
    source = Path(p0006.__file__).read_text(encoding="utf-8")
    loader = source[
        source.index("def _load_completed_stage2_config(") : source.index(
            "def _rebuild_stage2_run_fingerprint("
        )
    ]
    assert 'STAGE2_COMPLETED_EVIDENCE_DECLARED_DEVICE = "auto"' in source
    assert 'STAGE2_COMPLETED_EVIDENCE_EFFECTIVE_DEVICE = "cuda"' in source
    assert "torch.cuda.is_available" not in loader
    assert "os.environ" not in loader


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", None, {"bad": True}])
def test_resolved_non_auto_device_fails_closed(tmp_path: Path, device: Any) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "resolved_config.json",
        _resolved_payload(steps=20, pilot_steps=20, device=device),
    )
    with pytest.raises(ValueError, match="must declare device='auto'"):
        p0006._load_completed_stage2_config(run_dir)


def test_resolved_missing_device_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    payload = _resolved_payload(steps=20, pilot_steps=20)
    del payload["training"]["device"]
    _write_json(run_dir / "resolved_config.json", payload)
    with pytest.raises(ValueError, match="must declare device='auto'"):
        p0006._load_completed_stage2_config(run_dir)


@pytest.mark.parametrize("checkpoint_device", ["auto", "cpu", "cuda:0"])
def test_checkpoint_requires_exact_effective_cuda_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_device: str
) -> None:
    train_index = _Index("train")
    validation_index = _Index("validation")
    stats = _Stats()
    plan = _validation_plan(validation_index)
    monkeypatch.setattr(p0006, "build_unified_validation_plan", lambda *_a, **_k: plan)
    run_dir = tmp_path / "run"
    receipt = _build_pilot_attempt(
        run_dir,
        steps=20,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
        checkpoint_device=checkpoint_device,
    )
    with pytest.raises(ValueError, match="effective normalized configuration"):
        p0006._verify_completed_pilot_attempt(
            run_dir,
            steps=20,
            selection_receipt_path=receipt,
            train_index=train_index,
            validation_index=validation_index,
            stats=stats,
            training_evidence_commit=TRAINING_COMMIT,
            expected_validation_plan_sha256=plan["validation_plan_sha256"],
            expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
        )


def test_checkpoint_rejects_any_other_effective_config_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_index = _Index("train")
    validation_index = _Index("validation")
    stats = _Stats()
    plan = _validation_plan(validation_index)
    monkeypatch.setattr(p0006, "build_unified_validation_plan", lambda *_a, **_k: plan)
    run_dir = tmp_path / "run"
    receipt = _build_pilot_attempt(
        run_dir,
        steps=20,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
        checkpoint_other_change=True,
    )
    with pytest.raises(ValueError, match="effective normalized configuration"):
        p0006._verify_completed_pilot_attempt(
            run_dir,
            steps=20,
            selection_receipt_path=receipt,
            train_index=train_index,
            validation_index=validation_index,
            stats=stats,
            training_evidence_commit=TRAINING_COMMIT,
            expected_validation_plan_sha256=plan["validation_plan_sha256"],
            expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda state: state.update(contract_version="changed"), "resume contract"),
        (lambda state: state.update(_meta=[]), "_meta mapping"),
        (
            lambda state: state["_meta"].update(git_commit="0" * 40),
            "Git commit",
        ),
        (
            lambda state: state.update(validation_plan_sha256="0" * 64),
            "validation-plan SHA",
        ),
        (
            lambda state: state.update(selection_rule_sha256="0" * 64),
            "selection-rule SHA",
        ),
    ],
)
def test_checkpoint_identity_failures_are_field_specific(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    train_index = _Index("train")
    validation_index = _Index("validation")
    stats = _Stats()
    plan = _validation_plan(validation_index)
    monkeypatch.setattr(p0006, "build_unified_validation_plan", lambda *_a, **_k: plan)
    run_dir = tmp_path / "run"
    receipt = _build_pilot_attempt(
        run_dir,
        steps=20,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
        state_mutator=mutator,
    )
    with pytest.raises(ValueError, match=message):
        p0006._verify_completed_pilot_attempt(
            run_dir,
            steps=20,
            selection_receipt_path=receipt,
            train_index=train_index,
            validation_index=validation_index,
            stats=stats,
            training_evidence_commit=TRAINING_COMMIT,
            expected_validation_plan_sha256=plan["validation_plan_sha256"],
            expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
        )


def test_run_fingerprint_rejects_declared_auto_and_accepts_effective_cuda(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "resolved_config.json",
        _resolved_payload(steps=20, pilot_steps=20),
    )
    reconstruction = p0006._load_completed_stage2_config(run_dir)
    train_index = _Index("train")
    validation_index = _Index("validation")
    plan = _validation_plan(validation_index)
    kwargs = {
        "train_index": train_index,
        "validation_index": validation_index,
        "stats": _Stats(),
        "validation_plan": plan,
        "frozen_decoder_state_sha256": DECODER_SHA256,
        "decoder_checkpoint_evidence": _checkpoint_evidence(),
        "git_commit": TRAINING_COMMIT,
        "execution_mode": "train_with_frozen_validation_and_selection",
        "accumulation": {
            **p0006._expected_accumulation_identity(),
            "joint_six_term_gradient_probe": False,
        },
    }
    assert len(
        p0006._rebuild_stage2_run_fingerprint(
            reconstruction.effective_config, **kwargs
        )
    ) == 64
    with pytest.raises(ValueError, match="pinned effective device='cuda'"):
        p0006._rebuild_stage2_run_fingerprint(
            reconstruction.declared_config, **kwargs
        )


def test_a100_qualification_uses_same_effective_config_reconstruction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_index = _Index("train")
    validation_index = _Index("validation")
    stats = _Stats()
    plan = _validation_plan(validation_index)
    monkeypatch.setattr(p0006, "build_unified_validation_plan", lambda *_a, **_k: plan)
    run_dir = tmp_path / "a100_full_objective_gate_1"
    _build_qualification(
        run_dir,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
    )
    result = p0006._verify_completed_a100_qualification(
        run_dir,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        training_evidence_commit=TRAINING_COMMIT,
        expected_validation_plan_sha256=plan["validation_plan_sha256"],
        expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
    )
    assert result["status"] == "pass"
    assert result["configuration_override_provenance"]["declared_device"] == "auto"
    assert result["configuration_override_provenance"]["effective_device"] == "cuda"


def test_completed_evidence_integration_reaches_real_checkpoint_and_fingerprint_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / f"implementation_{TRAINING_COMMIT[:12]}"
    train_index = _Index("train")
    validation_index = _Index("validation")
    stats = _Stats()
    plan = _validation_plan(validation_index)
    monkeypatch.setattr(p0006, "build_unified_validation_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        p0006,
        "PhotometryFactoredLatentBankIndex",
        lambda _bank_dir, split: train_index if split == "train" else validation_index,
    )
    monkeypatch.setattr(p0006, "FactoredLatentStats", _Stats)
    full_receipt = _build_pilot_attempt(
        root / "unified_full_objective_pilot_200",
        steps=200,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
    )
    _build_pilot_attempt(
        root / "unified_full_objective_pilot_20",
        steps=20,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
    )
    _build_qualification(
        root / "a100_full_objective_gate_1",
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        plan=plan,
    )

    result = p0006.verify_completed_stage2_pilot_evidence(
        root,
        bank_dir=tmp_path / "bank",
        expected_selection_receipt_path=full_receipt,
        training_evidence_commit=TRAINING_COMMIT,
        expected_selection_receipt_file_sha256=sha256_file(full_receipt),
        expected_validation_plan_sha256=plan["validation_plan_sha256"],
        expected_selection_rule_sha256=p0006.UNIFIED_SELECTION_RULE_SHA256,
    )

    assert result["status"] == "pass"
    assert result["pilot_20"]["status"] == "pass"
    assert result["pilot_200"]["status"] == "pass"
    assert result["qualification"]["status"] == "pass"
    assert result["configuration_override_provenance"] == result["pilot_20"][
        "configuration_override_provenance"
    ]
    assert result["configuration_override_provenance"] == result["qualification"][
        "configuration_override_provenance"
    ]
    assert result["training_invoked"] is False
