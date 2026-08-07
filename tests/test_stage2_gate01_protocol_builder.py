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
        "cases": cases,
    }
    plan_path = tmp_path / "private-build-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return {
        "plan": plan,
        "plan_path": plan_path,
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
    with pytest.raises(ValueError, match="build state is stale"):
        _build(bundle, resume=True)


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
