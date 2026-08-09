from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SOURCE_MODULES,
    PhotometryFitVolume,
    canonical_tensor_sha256,
    fit_frozen_photometry,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    FIXED_MAP_METHOD,
    GATE01_POSTHOC_METHOD,
    RAW_IDENTITY_METHOD,
    STAGE1_CEILING_METHOD,
    ContinuityReference,
    build_continuity_reference_from_gate01,
)
from fieldbridge.evaluation.stage2_gate01 import GATE01_CONTRACT_VERSION
from fieldbridge.evaluation.stage2_photometry_protocol import (
    PAIRED_EVALUATION_ROLE,
    evaluate_paired_variant_a,
    load_paired_evaluation_manifest,
    seal_paired_evaluation_manifest,
)


def _volume(field: float, contrast_index: int, *, bias: float = 0.0) -> torch.Tensor:
    values = torch.linspace(0.05, 0.95, 63)
    scale = 0.45 + 0.04 * FIELD_STRENGTHS_T.index(field) + 0.03 * contrast_index
    return torch.cat([torch.zeros(1), (values * scale + bias).clamp(0.0, 1.0)]).reshape(
        4, 4, 4
    )


def _artifact():
    items = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field in FIELD_STRENGTHS_T:
            identity = f"R_train_{contrast.value}_{field:g}"
            items.append(
                PhotometryFitVolume(
                    volume=_volume(field, contrast_index),
                    domain=Domain(field, contrast),
                    record_identity=identity,
                    subject_identity=identity,
                    metadata_prefix="R",
                    source_path_identity=f"external/{identity}.nii.gz",
                    source_file_sha256=sha256_text(identity),
                )
            )
    commit = "synthetic-commit"
    return fit_frozen_photometry(
        items,
        source_split_file_sha256="a" * 64,
        source_membership_fingerprint="membership-v1",
        source_recovery_fingerprint="recovery-v1",
        code_commit=commit,
        code_provenance={
            "git_head": commit,
            "checkout_clean": True,
            "module_sha256": {
                name: f"{index + 1:x}" * 64
                for index, name in enumerate(PHOTOMETRY_SOURCE_MODULES)
            },
        },
        resolved_config={"contract": "stage2-photometry-variant-a-config-v1"},
    )


def _continuity() -> ContinuityReference:
    return build_continuity_reference_from_gate01(
        {
            "contract_version": GATE01_CONTRACT_VERSION,
            "overall": {
                "methods": {
                    "calibrated_identity": {
                        "nrmse": 0.3,
                        "ssim": 0.9,
                        "lpips": 0.09,
                    },
                    "raw_identity": {"nrmse": 0.5, "ssim": 0.8, "lpips": 0.1},
                    "stage1_reconstruction_ceiling": {
                        "nrmse": 0.1,
                        "ssim": 0.96,
                        "lpips": 0.03,
                    },
                }
            },
        },
        source_result_sha256="f" * 64,
        evaluation_identity="synthetic-paired-selection",
    )


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    denominator = max(float(torch.linalg.vector_norm(target)), 1e-12)
    mae = float(error.abs().mean())
    return {
        "nrmse": float(torch.linalg.vector_norm(error)) / denominator,
        "ssim": 1.0 - mae,
        "lpips": mae,
    }


def _array_spec(path: Path, tensor: torch.Tensor) -> dict:
    np.save(path, tensor.numpy(), allow_pickle=False)
    return {
        "content_identity": f"synthetic:{path.name}",
        "array_path": str(path),
        "file_sha256": sha256_file(path),
        "loaded_array_sha256": canonical_tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _endpoint(
    path: Path,
    tensor: torch.Tensor,
    *,
    case_id: str,
    subject_id: str,
    field: float,
    contrast: str,
) -> dict:
    return {
        "case_id": case_id,
        "subject_id": subject_id,
        "metadata_prefix": "P",
        "cohort": "P",
        "domain": Domain(field, contrast).to_dict(),
        **_array_spec(path, tensor),
    }


def _manifest(tmp_path: Path, artifact) -> tuple[Path, dict]:
    cases = []
    definitions = (
        ("case-a", "0100", "T1w", 0.1, 7.0, 0.35, -0.02),
        ("case-b", "0101", "T2w", 1.5, 5.0, 0.0, 0.02),
    )
    for index, (case, subject, contrast, source_field, target_field, sbias, tbias) in enumerate(
        definitions
    ):
        contrast_index = ("T1w", "T2w", "T2-FLAIR").index(contrast)
        source = _volume(source_field, contrast_index, bias=sbias)
        target = _volume(target_field, contrast_index, bias=tbias)
        stage1 = target * 0.98
        cases.append(
            {
                "case_identity": case,
                "genuinely_paired": True,
                "source": _endpoint(
                    tmp_path / f"source-{index}.npy",
                    source,
                    case_id=f"P_{subject}_{contrast}_{source_field:g}",
                    subject_id=subject,
                    field=source_field,
                    contrast=contrast,
                ),
                "target": _endpoint(
                    tmp_path / f"target-{index}.npy",
                    target,
                    case_id=f"P_{subject}_{contrast}_{target_field:g}",
                    subject_id=subject,
                    field=target_field,
                    contrast=contrast,
                ),
                "stage1_reconstruction": _array_spec(
                    tmp_path / f"stage1-{index}.npy", stage1
                ),
            }
        )
    payload = seal_paired_evaluation_manifest(
        {
            "contract_version": "stage2-photometry-paired-evaluation-manifest-v1",
            "evaluation_identity": "synthetic-paired-selection",
            "data_role": PAIRED_EVALUATION_ROLE,
            "metrics": ["nrmse", "ssim", "lpips"],
            "raw_identity_catastrophic_boundary": 1.0,
            "photometry_artifact_sha256": artifact.artifact_sha256,
            "photometry_config_sha256": artifact.provenance["resolved_config_sha256"],
            "split_provenance": {
                "role": "synthetic-paired",
                "file_sha256": "e" * 64,
                "source_membership_fingerprint": "c" * 64,
                "source_recovery_fingerprint": "d" * 64,
            },
            "cases": cases,
            "provenance": {"authorization_reference": "synthetic-test-only"},
        }
    )
    path = write_json_atomic(tmp_path / "paired-manifest.json", payload)
    return path, payload


def _evaluate(tmp_path: Path, *, resume: bool = False, metric_function=_metrics):
    artifact = _artifact()
    manifest_path, _ = _manifest(tmp_path, artifact)
    return evaluate_paired_variant_a(
        artifact,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        continuity=_continuity(),
        metric_function=metric_function,
        metric_runtime_provenance={"metrics": ["nrmse", "ssim", "lpips"], "cpu": True},
        evaluation_code_provenance={"git_head": "synthetic-commit", "module_sha256": {}},
        resume=resume,
    )


def test_paired_protocol_emits_same_case_reductions_strata_controls_and_provenance(
    tmp_path: Path,
) -> None:
    result = _evaluate(tmp_path)
    assert result["contract_version"] == "stage2-photometry-dual-baseline-result-v2"
    assert len(result["cases"]) == 2
    assert set(result["reductions"]["per_contrast_equal_domain"]) == {"T1w", "T2w"}
    assert result["reductions"]["macro_equal_domain"]["domain_count"] == 2
    assert result["strata"]["ordinary_case_count"] + result["strata"][
        "catastrophic_case_count"
    ] == 2
    assert result["external_continuity_track"]["included_in_same_case_reductions"] is False
    assert GATE01_POSTHOC_METHOD not in result["same_case_methods"]
    for row in result["cases"]:
        assert set(row["methods"]) == {
            RAW_IDENTITY_METHOD,
            FIXED_MAP_METHOD,
            STAGE1_CEILING_METHOD,
        }
        assert set(row["methods"][FIXED_MAP_METHOD]["metrics"]) == {
            "nrmse",
            "ssim",
            "lpips",
        }
        assert row["support"]["source_only"] is True
        assert len(row["support"]["canonical_byte_sha256"]) == 64
        assert row["methods"][FIXED_MAP_METHOD]["controls"]
        assert row["source"]["loaded_array_sha256"]
        assert row["target"]["loaded_array_sha256"]
    assert (tmp_path / "out" / "run_contract.json").is_file()
    assert len(list((tmp_path / "out" / "case_shards").glob("*.json"))) == 2


def test_paired_protocol_resume_is_deterministic_and_final_is_no_clobber(tmp_path: Path) -> None:
    first = _evaluate(tmp_path)
    first_bytes = (tmp_path / "out" / "result.json").read_bytes()
    artifact = _artifact()
    manifest_path = tmp_path / "paired-manifest.json"
    resumed = evaluate_paired_variant_a(
        artifact,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        continuity=_continuity(),
        metric_function=_metrics,
        metric_runtime_provenance={"metrics": ["nrmse", "ssim", "lpips"], "cpu": True},
        evaluation_code_provenance={"git_head": "synthetic-commit", "module_sha256": {}},
        resume=True,
    )
    assert resumed == first
    assert (tmp_path / "out" / "result.json").read_bytes() == first_bytes
    with pytest.raises(FileExistsError, match="nonempty"):
        evaluate_paired_variant_a(
            artifact,
            manifest_path=manifest_path,
            output_dir=tmp_path / "out",
            continuity=_continuity(),
            metric_function=_metrics,
            metric_runtime_provenance={"metrics": ["nrmse", "ssim", "lpips"]},
            evaluation_code_provenance={"git_head": "synthetic-commit"},
            resume=False,
        )


def test_paired_protocol_refuses_uncontracted_resume_directory(tmp_path: Path) -> None:
    artifact = _artifact()
    manifest_path, _ = _manifest(tmp_path, artifact)
    output = tmp_path / "uncontracted"
    output.mkdir()
    (output / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="uncontracted"):
        evaluate_paired_variant_a(
            artifact,
            manifest_path=manifest_path,
            output_dir=output,
            continuity=_continuity(),
            metric_function=_metrics,
            metric_runtime_provenance={"metrics": ["nrmse", "ssim", "lpips"]},
            evaluation_code_provenance={"git_head": "synthetic-commit"},
            resume=True,
        )
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_paired_manifest_rejects_unrelated_subjects_and_conflicting_prefix(tmp_path: Path) -> None:
    artifact = _artifact()
    path, payload = _manifest(tmp_path, artifact)
    unrelated = copy.deepcopy(payload)
    unrelated["cases"][0]["target"]["subject_id"] = "different"
    unrelated["split_provenance"].pop("evaluation_membership_fingerprint")
    unrelated = seal_paired_evaluation_manifest(unrelated)
    path.write_text(json.dumps(unrelated, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="same subject group"):
        load_paired_evaluation_manifest(path, artifact=artifact)

    conflicting = copy.deepcopy(payload)
    conflicting["cases"][0]["source"]["metadata_prefix"] = "R"
    conflicting["split_provenance"].pop("evaluation_membership_fingerprint")
    conflicting = seal_paired_evaluation_manifest(conflicting)
    path.write_text(json.dumps(conflicting, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="identity conflict"):
        load_paired_evaluation_manifest(path, artifact=artifact)


@pytest.mark.parametrize(
    "metrics",
    [
        ["nrmse", "ssim"],
        ["nrmse", "ssim", "lpips", "psnr"],
        ["nrmse", "ssim", "lpips", "lpips"],
    ],
)
def test_paired_manifest_requires_exact_metric_names(tmp_path: Path, metrics: list[str]) -> None:
    artifact = _artifact()
    path, payload = _manifest(tmp_path, artifact)
    payload["metrics"] = metrics
    payload = seal_paired_evaluation_manifest(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly nrmse, ssim, and lpips"):
        load_paired_evaluation_manifest(path, artifact=artifact)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_membership_fingerprint", "source split membership"),
        ("source_recovery_fingerprint", "source split recovery"),
    ],
)
def test_paired_manifest_requires_source_split_fingerprints(
    tmp_path: Path, field: str, message: str
) -> None:
    artifact = _artifact()
    path, payload = _manifest(tmp_path, artifact)
    payload["split_provenance"].pop(field)
    payload = seal_paired_evaluation_manifest(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_paired_evaluation_manifest(path, artifact=artifact)


def test_paired_manifest_rejects_declared_shape_or_dtype_mismatch(tmp_path: Path) -> None:
    artifact = _artifact()
    path, payload = _manifest(tmp_path, artifact)
    payload["cases"][0]["source"]["shape"] = [2, 2, 2]
    payload["split_provenance"].pop("evaluation_membership_fingerprint")
    payload = seal_paired_evaluation_manifest(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="loaded shape mismatch"):
        load_paired_evaluation_manifest(path, artifact=artifact)


def test_paired_runner_rejects_missing_or_extra_metric_results(tmp_path: Path) -> None:
    def missing(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        del prediction, target
        return {"nrmse": 0.0, "ssim": 1.0}

    with pytest.raises(ValueError, match="exactly nrmse, ssim, and lpips"):
        _evaluate(tmp_path, metric_function=missing)

    def nonfinite(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        del prediction, target
        return {"nrmse": float("nan"), "ssim": 1.0, "lpips": 0.0}

    (tmp_path / "nonfinite").mkdir()
    with pytest.raises(ValueError, match="finite"):
        _evaluate(tmp_path / "nonfinite", metric_function=nonfinite)


def test_paired_manifest_rejects_loaded_array_and_file_mutation(tmp_path: Path) -> None:
    artifact = _artifact()
    path, payload = _manifest(tmp_path, artifact)
    target_path = Path(payload["cases"][0]["target"]["array_path"])
    target = torch.from_numpy(np.load(target_path, allow_pickle=False))
    np.save(target_path, (target + 0.1).numpy(), allow_pickle=False)
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_paired_evaluation_manifest(path, artifact=artifact)

    second = tmp_path / "loaded-hash"
    second.mkdir()
    path, payload = _manifest(second, artifact)
    loaded_hash_mutation = copy.deepcopy(payload)
    loaded_hash_mutation["cases"][0]["source"]["loaded_array_sha256"] = "0" * 64
    loaded_hash_mutation["split_provenance"].pop("evaluation_membership_fingerprint")
    loaded_hash_mutation = seal_paired_evaluation_manifest(loaded_hash_mutation)
    path.write_text(json.dumps(loaded_hash_mutation, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="loaded-array SHA-256 mismatch"):
        load_paired_evaluation_manifest(path, artifact=artifact)


def test_paired_resume_rejects_mutated_case_shard(tmp_path: Path) -> None:
    _evaluate(tmp_path)
    (tmp_path / "out" / "result.json").unlink()
    shard = next((tmp_path / "out" / "case_shards").glob("*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["case"]["methods"][RAW_IDENTITY_METHOD]["metrics"]["nrmse"] += 1.0
    shard.write_text(json.dumps(payload), encoding="utf-8")
    artifact = _artifact()
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_paired_variant_a(
            artifact,
            manifest_path=tmp_path / "paired-manifest.json",
            output_dir=tmp_path / "out",
            continuity=_continuity(),
            metric_function=_metrics,
            metric_runtime_provenance={"metrics": ["nrmse", "ssim", "lpips"], "cpu": True},
            evaluation_code_provenance={"git_head": "synthetic-commit", "module_sha256": {}},
            resume=True,
        )
