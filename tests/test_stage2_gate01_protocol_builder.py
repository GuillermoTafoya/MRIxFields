from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import fieldbridge.evaluation.stage2_gate01_builder as builder
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.evaluation.stage2_gate01 import (
    canonical_loaded_array_sha256,
    fixed_montage_specifications,
    frozen_artifact_provenance,
    gate01_selection_fingerprint,
    load_gate01_input_manifest,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    GATE01_CALIBRATOR_SOURCE_MODULES,
    RESPLIT_FINGERPRINT,
    TrainingTemplateVolume,
    fit_posthoc_target_calibrator,
)
from fieldbridge.evaluation.stage2_gate01_protocol import (
    GATE01_SCIENTIFIC_MODULES,
    Gate01ProtocolLock,
    frozen_protocol_artifact_provenance,
)


def _fit_calibrator(tmp_path: Path):
    commit = "synthetic-reviewed-commit"
    volumes = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            values = torch.linspace(0.0, 0.9, 27).reshape(3, 3, 3)
            values *= 0.5 + 0.02 * contrast_index + 0.03 * field_index
            volumes.append(
                TrainingTemplateVolume(
                    volume=values,
                    domain=Domain(field, contrast),
                    record_identity=f"training-{contrast.value}-{field:g}",
                )
            )
    calibrator = fit_posthoc_target_calibrator(
        volumes,
        split_fingerprint=RESPLIT_FINGERPRINT,
        training_cohort_identity="synthetic-retrospective-training",
        code_commit=commit,
        code_provenance={
            "git_head": commit,
            "checkout_clean": True,
            "module_sha256": {
                name: f"{index + 1:x}" * 64
                for index, name in enumerate(GATE01_CALIBRATOR_SOURCE_MODULES)
            },
        },
        num_quantiles=5,
    )
    path = calibrator.save(tmp_path / "calibrator.json")
    return calibrator, path


def _array(value: float) -> np.ndarray:
    result = np.full((2, 2, 2), value, dtype=np.float32)
    result[0, 0, 0] = 0.0
    return result


def _write_array(path: Path, value: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {
        "path": path.as_posix(),
        "expected_sha256": canonical_loaded_array_sha256(value),
    }


def _private_bundle(tmp_path: Path) -> dict[str, object]:
    calibrator, calibrator_path = _fit_calibrator(tmp_path)
    data = tmp_path / "arrays"
    acquisition: dict[tuple[str, float], dict[str, str]] = {}
    stage1: dict[tuple[str, float], dict[str, str]] = {}
    support: dict[tuple[str, float], dict[str, str]] = {}
    predictions: dict[tuple[str, float, float], dict[str, str]] = {}

    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            key = (contrast.value, float(field))
            acq = _array(0.1 + 0.1 * contrast_index + 0.01 * field_index)
            recon = _array(0.2 + 0.1 * contrast_index + 0.01 * field_index)
            mask = acq != 0.0
            stem = f"{contrast.value}-{field:g}T"
            acquisition[key] = _write_array(data / f"acq-{stem}.npy", acq)
            stage1[key] = _write_array(data / f"stage1-{stem}.npy", recon)
            support[key] = _write_array(data / f"support-{stem}.npy", mask)

        for source_index, source in enumerate(FIELD_STRENGTHS_T):
            for target_index, target in enumerate(FIELD_STRENGTHS_T):
                if source == target:
                    continue
                value = 0.3 + 0.1 * contrast_index + 0.01 * source_index + 0.001 * target_index
                predictions[(contrast.value, float(source), float(target))] = _write_array(
                    data / f"sb-{contrast.value}-{source:g}T-to-{target:g}T.npy",
                    _array(value),
                )

    # Plan paths are external and portable relative to the plan location.
    def portable(reference: dict[str, str]) -> dict[str, str]:
        return {
            "path": Path(reference["path"]).relative_to(tmp_path).as_posix(),
            "expected_sha256": reference["expected_sha256"],
        }

    traveller = hashlib.sha256(b"synthetic-private-traveller").hexdigest()
    cases = []
    descriptors = []
    for contrast in CONTRASTS:
        for source in FIELD_STRENGTHS_T:
            for target in FIELD_STRENGTHS_T:
                if source == target:
                    continue
                case_id = f"private-{contrast.value}-{source:g}T-to-{target:g}T"
                cases.append(
                    {
                        "case_id": case_id,
                        "traveller_identity_sha256": traveller,
                        "source_domain": Domain(source, contrast).to_dict(),
                        "target_domain": Domain(target, contrast).to_dict(),
                        "source_image": portable(acquisition[(contrast.value, float(source))]),
                        "source_support_mask": portable(support[(contrast.value, float(source))]),
                        "target": portable(acquisition[(contrast.value, float(target))]),
                        "raw_identity": portable(stage1[(contrast.value, float(source))]),
                        "raw_sb_v2": portable(
                            predictions[(contrast.value, float(source), float(target))]
                        ),
                        "stage1_reconstruction_ceiling": portable(
                            stage1[(contrast.value, float(target))]
                        ),
                        "wrong_target_sb_v2": {
                            f"{wrong:g}T": portable(
                                predictions[(contrast.value, float(source), float(wrong))]
                            )
                            for wrong in FIELD_STRENGTHS_T
                            if wrong != source and wrong != target
                        },
                    }
                )
                descriptors.append(
                    {
                        "case_identity_sha256": hashlib.sha256(
                            case_id.encode("utf-8")
                        ).hexdigest(),
                        "traveller_identity_sha256": traveller,
                        "contrast": contrast.value,
                        "source_field_t": float(source),
                        "target_field_t": float(target),
                    }
                )

    selection = gate01_selection_fingerprint(descriptors)
    lock_spec = {
        "traveller_identity_sha256": traveller,
        "selection_fingerprint_sha256": selection,
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "support_threshold": 0.0,
        "calibrator_artifact_sha256": calibrator.artifact_sha256,
        "calibrator_template_sha256": calibrator.template_sha256,
        "evaluation_git_commit": "e" * 40,
        "evaluation_module_sha256": {
            name: f"{index + 1:064x}"
            for index, name in enumerate(GATE01_SCIENTIFIC_MODULES)
        },
        "artifact_provenance": frozen_protocol_artifact_provenance(),
        "official_metrics": ["nrmse", "ssim", "lpips"],
        "montage_specification": fixed_montage_specifications(),
    }
    spec_path = tmp_path / "protocol-spec.json"
    spec_path.write_text(json.dumps(lock_spec), encoding="utf-8")
    lock_path = tmp_path / "protocol-lock.json"
    lock = builder.write_gate01_protocol_lock(spec_path, lock_path)

    source_identities = {
        Domain(field, contrast).label: {
            "canonical_loaded_array_sha256": acquisition[
                (contrast.value, float(field))
            ]["expected_sha256"],
            "shape": [1, 1, 2, 2, 2],
        }
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    producer_spec = {
        "contract_version": builder.GATE01_PRIVATE_PRODUCER_SPEC_VERSION,
        "selection_artifact_sha256": "1" * 64,
        "selection_fingerprint_sha256": selection,
        "traveller_identity_sha256": traveller,
        "split_file_sha256": "2" * 64,
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "selected_source_acquisitions": source_identities,
        "selected_payload_identity_set_sha256": "3" * 64,
        "selected_payload_count": 15,
        "latent_bank": {
            "artifact_sha256": "4" * 64,
            "manifest_sha256": "5" * 64,
            "stats_sha256": "6" * 64,
            "record_count": 15,
            "build_git_commit": builder.FULL_LATENT_BANK_BUILD_COMMIT,
            "vae_checkpoint_sha256": builder.STAGE1_RUN_C_CHECKPOINT_SHA256,
            "encode_provenance": {"strategy": "full", "path_used": ["full"]},
        },
        "stage1_config_sha256": builder.STAGE1_RUN_C_CONFIG_SHA256,
        "stage1_checkpoint_sha256": builder.STAGE1_RUN_C_CHECKPOINT_SHA256,
        "sb_v2_config_sha256": builder.SB_V2_CONFIG_SHA256,
        "sb_v2_checkpoint_sha256": builder.SB_V2_CHECKPOINT_SHA256,
        "sampler": {"solver": "heun", "n_steps": 20},
        "decode": {
            "strategy": "full",
            "block_size": [4, 4, 4],
            "halo": [0, 0, 0],
            "precision": "float32",
        },
        "deterministic_seed": 13,
        "protocol_lock_artifact_sha256": lock.artifact_sha256,
    }
    producer_spec["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            producer_spec, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    producer_spec_path = tmp_path / "producer-spec.json"
    producer_spec_path.write_text(json.dumps(producer_spec), encoding="utf-8")
    producer_receipt = builder.sealed_gate01_producer_receipt(producer_spec)

    plan = {
        "contract_version": builder.GATE01_PRIVATE_BUILD_PLAN_VERSION,
        "execution_mode": "scientific",
        "selection_fingerprint_sha256": selection,
        "evidence_scope": {
            "role": "private Gate 0.1 frozen traveller",
            "evidence_kind": "private",
            "traveller_identity_sha256": traveller,
            "private_data_run": True,
        },
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "artifact_provenance": frozen_artifact_provenance(),
        "source_support_contract": {
            "derivation": "abs(source_image)>threshold",
            "threshold": 0.0,
        },
        "producer_receipt": producer_receipt,
        "cases": cases,
    }
    plan_path = tmp_path / "private-build-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    unique_references: dict[tuple[str, str, bool], None] = {}
    for case in cases:
        for role in (
            "source_image",
            "source_support_mask",
            "target",
            "raw_identity",
            "raw_sb_v2",
            "stage1_reconstruction_ceiling",
        ):
            reference = case[role]
            unique_references[
                (
                    reference["path"],
                    reference["expected_sha256"],
                    role == "source_support_mask",
                )
            ] = None
        for reference in case["wrong_target_sb_v2"].values():
            unique_references[
                (reference["path"], reference["expected_sha256"], False)
            ] = None
    assert len(unique_references) == 105
    producer_state = {
        "contract_version": builder.GATE01_PRIVATE_PRODUCER_STATE_VERSION,
        "producer_spec_artifact_sha256": producer_spec["artifact_sha256"],
        "protocol_lock_artifact_sha256": lock.artifact_sha256,
        "producer_provenance": {"decode_strategy": "full", "path_used": ["full"]},
        "status": "complete",
        "completed": {
            f"array:{index}": {
                "relative_path": path,
                "canonical_loaded_array_sha256": digest,
                "mask": mask,
            }
            for index, (path, digest, mask) in enumerate(unique_references)
        },
        "pending": {},
        "counts": {"stage1_inference_count": 15, "sb_v2_inference_count": 60},
        "build_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "acquisition_count": 15,
        "direction_count": 60,
        "wrong_target_reference_count": 180,
        "producer_receipt": producer_receipt,
    }
    producer_state_path = tmp_path / "producer-state.json"
    producer_state_path.write_text(json.dumps(producer_state), encoding="utf-8")
    return {
        "plan": plan,
        "plan_path": plan_path,
        "producer_spec": producer_spec,
        "producer_spec_path": producer_spec_path,
        "producer_state": producer_state,
        "producer_state_path": producer_state_path,
        "lock": lock,
        "lock_path": lock_path,
        "calibrator": calibrator,
        "calibrator_path": calibrator_path,
        "output": tmp_path / "built" / "manifest.json",
        "state": tmp_path / "built" / "state.json",
        "first_acquisition": data / "acq-T1w-0.1T.npy",
    }


def _build(bundle: dict[str, object], *, resume: bool):
    return builder.build_gate01_private_manifest(
        bundle["plan_path"],
        bundle["producer_spec_path"],
        bundle["producer_state_path"],
        bundle["lock_path"],
        bundle["calibrator_path"],
        bundle["output"],
        bundle["state"],
        resume=resume,
    )


def test_private_builder_is_deterministic_resumable_and_protocol_validated(tmp_path) -> None:
    bundle = _private_bundle(tmp_path)
    result = _build(bundle, resume=False)
    output = bundle["output"]
    first_bytes = output.read_bytes()
    assert result["case_count"] == 60
    assert result["status"] == "complete"
    assert result["producer_receipt"]["producer_receipt"]["decode_strategy"] == "full"
    assert result["producer_receipt"]["producer_receipt"]["path_used"] == ["full"]

    repeated = _build(bundle, resume=True)
    assert repeated["manifest_sha256"] == result["manifest_sha256"]
    assert output.read_bytes() == first_bytes
    assert not list(output.parent.glob("*.tmp"))

    with pytest.raises(ValueError, match="requires an independent protocol lock"):
        load_gate01_input_manifest(output)
    streamed, metadata = load_gate01_input_manifest(
        output,
        protocol_lock=bundle["lock"],
        calibrator=bundle["calibrator"],
    )
    assert len(streamed.case_specs) == 60
    assert metadata["scientific_hash_graph"]["wrong_target_comparison_count"] == 180
    assert metadata["producer_receipt"]["producer_state_file_sha256"] == hashlib.sha256(
        bundle["producer_state_path"].read_bytes()
    ).hexdigest()


def test_private_builder_resumes_after_interruption(tmp_path, monkeypatch) -> None:
    bundle = _private_bundle(tmp_path)
    original = builder._verify_and_build_case
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(builder, "_verify_and_build_case", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _build(bundle, resume=False)
    state = json.loads(bundle["state"].read_text(encoding="utf-8"))
    assert len(state["completed_cases"]) == 2

    monkeypatch.setattr(builder, "_verify_and_build_case", original)
    result = _build(bundle, resume=True)
    assert result["case_count"] == 60


def test_private_builder_rejects_input_mutation_and_stale_resume(tmp_path) -> None:
    bundle = _private_bundle(tmp_path)
    _build(bundle, resume=False)
    mutated = np.load(bundle["first_acquisition"], allow_pickle=False)
    mutated[1, 1, 1] += 0.25
    np.save(bundle["first_acquisition"], mutated, allow_pickle=False)
    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        _build(bundle, resume=True)

    bundle = _private_bundle(tmp_path / "stale")
    _build(bundle, resume=False)
    plan = dict(bundle["plan"])
    plan["evidence_scope"] = {**plan["evidence_scope"], "role": "changed role"}
    bundle["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="state/plan/spec provenance"):
        _build(bundle, resume=True)


def test_private_builder_rejects_f28056e_v1_plan_without_producer_provenance(
    tmp_path,
) -> None:
    bundle = _private_bundle(tmp_path)
    plan = dict(bundle["plan"])
    plan["contract_version"] = "stage2-gate01-private-build-plan-v1"
    plan.pop("producer_receipt")
    bundle["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch|contract is incompatible"):
        _build(bundle, resume=False)


@pytest.mark.parametrize("kind", ["spec", "state"])
def test_private_builder_rejects_mutated_producer_spec_or_state(tmp_path, kind) -> None:
    bundle = _private_bundle(tmp_path)
    if kind == "spec":
        payload = dict(bundle["producer_spec"])
        payload["deterministic_seed"] = 99
        payload.pop("artifact_sha256")
        payload["artifact_sha256"] = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        bundle["producer_spec_path"].write_text(json.dumps(payload), encoding="utf-8")
        message = "state/plan/spec provenance"
    else:
        payload = dict(bundle["producer_state"])
        payload["producer_provenance"] = {
            "decode_strategy": "full",
            "path_used": ["tiled"],
        }
        bundle["producer_state_path"].write_text(json.dumps(payload), encoding="utf-8")
        message = "state/plan/spec provenance"
    with pytest.raises(ValueError, match=message):
        _build(bundle, resume=False)


def test_private_builder_refuses_repository_outputs(tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    spec = tmp_path / "invalid-is-not-read.json"
    with pytest.raises(ValueError, match="outside the Git repository"):
        builder.write_gate01_protocol_lock(
            spec, fake_repo / "protocol-lock.json", repo_root=fake_repo
        )


def test_protocol_lock_rejects_calibrator_identity_mismatch(tmp_path) -> None:
    bundle = _private_bundle(tmp_path)
    payload = bundle["lock"].to_dict()
    payload["calibrator_artifact_sha256"] = "0" * 64
    payload.pop("artifact_sha256")
    payload.pop("montage_specification_sha256")
    payload.pop("contract_version")
    wrong = Gate01ProtocolLock.from_spec(payload)
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        wrong.assert_calibrator(bundle["calibrator"])
