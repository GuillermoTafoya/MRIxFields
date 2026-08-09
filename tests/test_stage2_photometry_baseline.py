from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fieldbridge.cli import (
    _load_variant_a_config,
    _require_variant_a_retrospective_record,
    _select_variant_a_retrospective_records,
    build_parser,
)
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SOURCE_MODULES,
    FrozenPhotometryArtifact,
    PhotometryFitVolume,
    fit_frozen_photometry,
    interpolate_fixed_grid,
    sha256_file,
    sha256_json,
    sha256_text,
    tensor_sha256,
    write_json_atomic,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    CONTINUITY_METHODS,
    FIXED_MAP_METHOD,
    GATE01_POSTHOC_METHOD,
    RAW_IDENTITY_METHOD,
    STAGE1_CEILING_METHOD,
    BaselineEvaluationCase,
    ContinuityReference,
    QualificationVolume,
    VariantAQualificationThresholds,
    build_continuity_reference_from_gate01,
    evaluate_factorized_identity_case,
    qualify_variant_a,
)
from fieldbridge.evaluation.stage2_gate01 import GATE01_CONTRACT_VERSION


def _volume(field: float, contrast_index: int, *, offset: float = 0.0) -> torch.Tensor:
    values = torch.linspace(0.05, 0.95, 63, dtype=torch.float32)
    scale = 0.45 + 0.04 * FIELD_STRENGTHS_T.index(field) + 0.04 * contrast_index
    return torch.cat([torch.zeros(1), values * scale + offset]).reshape(4, 4, 4)


def _artifact(*, excluded=(), num_quantiles: int = 17):
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
    provenance = {
        "git_head": "synthetic-commit",
        "checkout_clean": True,
        "module_sha256": {
            name: f"{index + 1:x}" * 64
            for index, name in enumerate(PHOTOMETRY_SOURCE_MODULES)
        },
    }
    return fit_frozen_photometry(
        items,
        source_split_file_sha256="a" * 64,
        source_membership_fingerprint="membership-v1",
        source_recovery_fingerprint="recovery-v3",
        code_commit="synthetic-commit",
        code_provenance=provenance,
        resolved_config={"contract": "stage2-photometry-variant-a-config-v1"},
        num_quantiles=num_quantiles,
        excluded_prospective_records=excluded,
    )


def _artifact_from_mutated_payload(payload: dict) -> FrozenPhotometryArtifact:
    payload.pop("artifact_sha256", None)
    payload["artifact_sha256"] = sha256_json(payload)
    return FrozenPhotometryArtifact.from_dict(payload)


def _set_template_quantiles(template: dict, quantiles: list[float]) -> None:
    template["quantiles"] = quantiles
    template["template_sha256"] = tensor_sha256(
        torch.tensor(quantiles, dtype=torch.float64)
    )


def _artifact_with_endpoint_policy(*, extreme: bool) -> FrozenPhotometryArtifact:
    payload = _artifact(num_quantiles=101).to_dict()
    for template in payload["domain_templates"].values():
        quantiles = list(template["quantiles"])
        quantiles[0] = 0.0 if extreme else quantiles[1]
        quantiles[-1] = 1.0 if extreme else quantiles[-2]
        _set_template_quantiles(template, quantiles)
    for template in payload["canonical_templates"].values():
        quantiles = list(template["quantiles"])
        quantiles[0] = 0.0 if extreme else quantiles[1]
        quantiles[-1] = 1.0 if extreme else quantiles[-2]
        _set_template_quantiles(template, quantiles)
    return _artifact_from_mutated_payload(payload)


def _artifact_with_degenerate_domain(domain: Domain) -> FrozenPhotometryArtifact:
    payload = _artifact(num_quantiles=101).to_dict()
    domain_template = payload["domain_templates"][domain.label]
    _set_template_quantiles(
        domain_template,
        [0.25] * len(domain_template["quantiles"]),
    )
    canonical = payload["canonical_templates"][domain.contrast.value]
    field_quantiles = [
        payload["domain_templates"][label]["quantiles"]
        for label in canonical["domain_labels"]
    ]
    recomputed = [
        sum(values) / len(values) for values in zip(*field_quantiles)
    ]
    _set_template_quantiles(canonical, recomputed)
    return _artifact_from_mutated_payload(payload)


def _qualification_volumes() -> list[QualificationVolume]:
    records = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field in FIELD_STRENGTHS_T:
            identity = f"R_validation_{contrast.value}_{field:g}"
            records.append(
                QualificationVolume(
                    volume=_volume(field, contrast_index),
                    domain=Domain(field, contrast),
                    record_identity=identity,
                    subject_identity=identity,
                    metadata_prefix="R",
                    source_path_identity=f"external/{identity}.nii.gz",
                    source_file_sha256=sha256_text(identity),
                )
            )
    return records


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.detach().cpu().to(torch.float64)
    tgt = target.detach().cpu().to(torch.float64)
    error = float(
        torch.linalg.vector_norm(pred - tgt)
        / torch.linalg.vector_norm(tgt).clamp_min(1e-12)
    )
    if error < 1e-6:
        error = 0.0
    return {
        "nrmse": error,
        "ssim": max(-1.0, 1.0 - error),
        "lpips": error,
    }


def _permissive_thresholds() -> VariantAQualificationThresholds:
    return replace(
        VariantAQualificationThresholds(),
        roundtrip_macro_nrmse_max=1.0,
        roundtrip_worst_domain_nrmse_max=1.0,
        roundtrip_p99_range_fraction_max=1.0,
        histogram_macro_distance_max=1.0,
        histogram_worst_field_distance_max=1.0,
        spearman_min=0.0,
        contrast_macro_f1_drop_max=1.0,
        contrast_recall_drop_max=1.0,
        scaling_histogram_distance_max=1.0,
        scaling_ssim_min=0.0,
    )


def _qualification_kwargs():
    return {
        "resolved_config": {"contract": "stage2-photometry-variant-a-config-v1"},
        "source_split_file_sha256": "a" * 64,
        "source_membership_fingerprint": "membership-v1",
        "source_recovery_fingerprint": "recovery-v3",
        "vae_provenance": {
            "checkpoint_sha256": "c" * 64,
            "config_file_sha256": "d" * 64,
            "encoder_statistic": "posterior_mean",
            "encode_strategy": "full",
            "decode_strategy": "full",
        },
        "continuity": _continuity("f" * 64),
        "metric_function": _metrics,
    }


def test_checked_in_config_exposes_exact_versioned_proposed_thresholds() -> None:
    path = __import__("pathlib").Path(
        "configs/experiment/stage2_photometry_factorization_a_v1.yaml"
    )
    config = _load_variant_a_config(path)
    thresholds = VariantAQualificationThresholds.from_mapping(config)
    assert thresholds.roundtrip_macro_nrmse_max == 0.01
    assert thresholds.histogram_worst_field_distance_max == 0.03
    assert thresholds.scaling_factors == (0.9, 1.1)
    assert thresholds.vae_domain_lpips_increase_max == 0.008
    assert config["qualification"]["threshold_status"].startswith("versioned_proposed")
    assert config["evaluation"]["endpoint_cohort"] == "R"
    assert config["qualification"]["monotonicity_audit"]["tolerance_scale"] == (
        "positive_robust_target_range_q99_minus_q01"
    )


def test_variant_a_v1_config_rejects_weakened_thresholds(monkeypatch) -> None:
    path = __import__("pathlib").Path(
        "configs/experiment/stage2_photometry_factorization_a_v1.yaml"
    )
    config = copy.deepcopy(_load_variant_a_config(path))
    config["qualification"]["thresholds"]["roundtrip_macro_nrmse_max"] = 1.0
    monkeypatch.setattr("fieldbridge.cli.load_yaml_config", lambda ignored: config)
    with pytest.raises(ValueError, match="reviewed proposed defaults"):
        _load_variant_a_config(path)


def test_variant_a_v1_config_rejects_dormant_prospective_pair_support(monkeypatch) -> None:
    path = Path("configs/experiment/stage2_photometry_factorization_a_v1.yaml")
    config = copy.deepcopy(_load_variant_a_config(path))
    config["evaluation"]["endpoint_cohort"] = "P"
    monkeypatch.setattr("fieldbridge.cli.load_yaml_config", lambda ignored: config)
    with pytest.raises(ValueError, match="retrospective-only"):
        _load_variant_a_config(path)


def test_qualification_reports_macro_worst_domain_scaling_and_equal_counts() -> None:
    artifact = _artifact()
    result = qualify_variant_a(
        artifact,
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=lambda volume, domain: volume,
        **_qualification_kwargs(),
    )
    assert result["contract_version"] == "stage2-photometry-variant-a-qualification-v1"
    assert result["record_count"] == 15
    assert set(result["aggregate"]["per_domain"]) == {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    assert all(item["count"] == 1 for item in result["aggregate"]["per_domain"].values())
    assert result["aggregate"]["macro"]
    assert result["aggregate"]["worst_domain"]
    assert all(len(row["scaling_sensitivity"]) == 2 for row in result["records"])
    assert result["failure_classification"] == []
    assert result["canonical_latent_bank_authorized"] is True
    assert result["gate01_substitution_forbidden"] is True
    assert result["interpolation_qualification"]["map_count"] == 30
    assert result["interpolation_qualification"]["dense_grid_points"] == 4097
    assert len(result["contrast_preservation_control"]["folds"]) == 15
    for fold in result["contrast_preservation_control"]["folds"]:
        assert fold["validation_group_identity"] not in fold["training_group_identities"]
        assert fold["training_domain_counts"]
        assert fold["validation_domain_counts"]
    assert all(
        item["finite"] and item["nondecreasing_within_tolerance"]
        for item in result["interpolation_qualification"]["maps"]
    )
    assert {
        item["direction"].split(":", 1)[0]
        for item in result["interpolation_qualification"]["maps"]
    } == {"N_d", "P_d"}
    for item in result["interpolation_qualification"]["maps"]:
        field_text, contrast_text = item["domain"].split("/", 1)
        domain = Domain(float(field_text.removesuffix("T")), contrast_text)
        target_grid = (
            artifact.canonical_templates[contrast_text].quantiles
            if item["direction"].startswith("N_d:")
            else artifact.domain_templates[domain.label].quantiles
        )
        expected_q01, expected_q99 = (
            float(value)
            for value in interpolate_fixed_grid(
                torch.tensor([0.01, 0.99], dtype=torch.float64),
                artifact.probabilities,
                target_grid,
            )
        )
        expected_range = expected_q99 - expected_q01
        assert item["target_q01"] == pytest.approx(expected_q01)
        assert item["target_q99"] == pytest.approx(expected_q99)
        assert item["robust_target_range"] == pytest.approx(expected_range)
        assert item["target_q99"] > item["target_q01"]
        assert item["absolute_tolerance"] == pytest.approx(
            1e-7 * item["robust_target_range"], rel=0.0, abs=1e-15
        )


def _split_record(case_id: str, prefix: str, subject: str = "0100") -> VolumeRecord:
    return VolumeRecord(
        case_id=case_id,
        image_path=f"external/{case_id}.nii.gz",
        domain=Domain(0.1, "T1w"),
        subject_id=subject,
        metadata={"prefix": prefix},
    )


def test_mixed_split_excludes_labelled_p_before_r_only_conversion_and_seals_it() -> None:
    retrospective = _split_record("R_0042_T1w", "R", "0042")
    prospective = _split_record("P_0100_T1w", "P")
    selected, excluded = _select_variant_a_retrospective_records(
        (prospective, retrospective), split="train"
    )
    assert selected == (retrospective,)
    assert [item["record_identity"] for item in excluded] == ["P_0100_T1w"]
    assert excluded[0]["reason"] == "prospective-cohort-excluded-before-array-load"
    with pytest.raises(ValueError, match="rejects every P record"):
        _require_variant_a_retrospective_record(prospective)

    artifact = _artifact(excluded=excluded)
    assert artifact.provenance["excluded_prospective_records"] == list(excluded)
    assert artifact.provenance["eligibility_proof"]["prospective_excluded_count"] == 1


@pytest.mark.parametrize(
    ("case_id", "prefix"),
    [("P_0100_T1w", "R"), ("R_0042_T1w", "P")],
)
def test_mixed_split_raises_on_inconsistent_identity_metadata(
    case_id: str, prefix: str
) -> None:
    with pytest.raises(ValueError, match="identity conflict"):
        _select_variant_a_retrospective_records(
            (_split_record(case_id, prefix),), split="validation"
        )


def test_qualification_seals_validation_p_exclusions() -> None:
    _, excluded = _select_variant_a_retrospective_records(
        (_split_record("P_0100_T1w", "P"),), split="validation"
    )
    result = qualify_variant_a(
        _artifact(),
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=lambda volume, domain: volume,
        excluded_prospective_records=excluded,
        **_qualification_kwargs(),
    )
    assert result["eligibility_proof"]["prospective_excluded_count"] == 1
    assert result["excluded_prospective_records"] == list(excluded)


def test_qualification_endpoint_outliers_use_only_sealed_q01_q99_tolerance() -> None:
    outlier_artifact = _artifact_with_endpoint_policy(extreme=True)
    trimmed_artifact = _artifact_with_endpoint_policy(extreme=False)
    outlier_result = qualify_variant_a(
        outlier_artifact,
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=lambda volume, domain: volume,
        **_qualification_kwargs(),
    )
    trimmed_result = qualify_variant_a(
        trimmed_artifact,
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=lambda volume, domain: volume,
        **_qualification_kwargs(),
    )
    outlier_rows = {
        (row["domain"], row["direction"]): row
        for row in outlier_result["interpolation_qualification"]["maps"]
    }
    trimmed_rows = {
        (row["domain"], row["direction"]): row
        for row in trimmed_result["interpolation_qualification"]["maps"]
    }
    assert set(outlier_rows) == set(trimmed_rows)
    assert {direction.split(":", 1)[0] for _, direction in outlier_rows} == {
        "N_d",
        "P_d",
    }

    query = torch.tensor([0.01, 0.99], dtype=torch.float64)
    for key, outlier_row in outlier_rows.items():
        domain_label, direction = key
        _, contrast = domain_label.split("/", 1)
        outlier_target = (
            outlier_artifact.canonical_templates[contrast].quantiles
            if direction.startswith("N_d:")
            else outlier_artifact.domain_templates[domain_label].quantiles
        )
        trimmed_target = (
            trimmed_artifact.canonical_templates[contrast].quantiles
            if direction.startswith("N_d:")
            else trimmed_artifact.domain_templates[domain_label].quantiles
        )
        expected = interpolate_fixed_grid(
            query, outlier_artifact.probabilities, outlier_target
        )
        expected_q01, expected_q99 = float(expected[0]), float(expected[1])
        expected_range = expected_q99 - expected_q01
        trimmed_row = trimmed_rows[key]

        assert outlier_row["target_q01"] == expected_q01
        assert outlier_row["target_q99"] == expected_q99
        assert outlier_row["robust_target_range"] == expected_range
        assert outlier_row["absolute_tolerance"] == 1e-7 * expected_range
        assert trimmed_row["target_q01"] == expected_q01
        assert trimmed_row["target_q99"] == expected_q99
        assert trimmed_row["robust_target_range"] == expected_range
        assert trimmed_row["absolute_tolerance"] == outlier_row["absolute_tolerance"]

        outlier_full_range = float(outlier_target[-1] - outlier_target[0])
        trimmed_full_range = float(trimmed_target[-1] - trimmed_target[0])
        assert outlier_full_range > trimmed_full_range
        assert outlier_row["absolute_tolerance"] != 1e-7 * outlier_full_range


@pytest.mark.parametrize("direction", ["N_d", "P_d"])
def test_qualification_classifies_interior_realized_map_decrease(
    monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    artifact = _artifact_with_endpoint_policy(extreme=True)
    domain = Domain(0.1, "T1w")
    domain_grid = artifact.domain_templates[domain.label].quantiles
    canonical_grid = artifact.canonical_templates[domain.contrast.value].quantiles
    audited_source = domain_grid if direction == "N_d" else canonical_grid
    audited_target = canonical_grid if direction == "N_d" else domain_grid
    mutation_evidence: dict[str, bool] = {"injected": False, "endpoints_unchanged": False}

    def interpolate_with_interior_decrease(
        values: torch.Tensor,
        source_grid: torch.Tensor,
        target_grid: torch.Tensor,
    ) -> torch.Tensor:
        # Mutation-test the audit's observed realized map.  The public qualification
        # entry point, production interpolation qualification, aggregation, and
        # failure classification remain unpatched.
        realized = interpolate_fixed_grid(values, source_grid, target_grid)
        is_selected_dense_audit = (
            values.numel() > 4097
            and source_grid.data_ptr() == audited_source.data_ptr()
            and target_grid.data_ptr() == audited_target.data_ptr()
        )
        if not is_selected_dense_audit:
            return realized
        mutated = realized.clone()
        midpoint = mutated.numel() // 2
        mutated[midpoint] = mutated[midpoint - 1] - 1e-3
        mutation_evidence["injected"] = True
        mutation_evidence["endpoints_unchanged"] = bool(
            mutated[0] == realized[0] and mutated[-1] == realized[-1]
        )
        return mutated

    monkeypatch.setattr(
        "fieldbridge.evaluation.stage2_photometry_baseline.interpolate_fixed_grid",
        interpolate_with_interior_decrease,
    )
    result = qualify_variant_a(
        artifact,
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=lambda volume, domain: volume,
        **_qualification_kwargs(),
    )
    row = next(
        row
        for row in result["interpolation_qualification"]["maps"]
        if row["domain"] == domain.label and row["direction"].startswith(f"{direction}:")
    )
    assert mutation_evidence == {"injected": True, "endpoints_unchanged": True}
    assert row["minimum_increment"] < -row["absolute_tolerance"]
    assert row["nondecreasing_within_tolerance"] is False
    assert result["interpolation_qualification"]["pass"] is False
    assert result["aggregate"]["photometry_checks"][
        "interpolation_finite_monotonic"
    ] is False
    assert "photometry_factorization_failure" in result["failure_classification"]
    assert result["canonical_latent_bank_authorized"] is False


def test_qualification_rejects_degenerate_p_target_robust_range() -> None:
    domain = Domain(0.1, "T1w")
    artifact = _artifact_with_degenerate_domain(domain)
    vae_calls = 0

    def vae_roundtrip(volume: torch.Tensor, ignored_domain: Domain) -> torch.Tensor:
        nonlocal vae_calls
        del ignored_domain
        vae_calls += 1
        return volume

    with pytest.raises(
        ValueError,
        match=r"0\.1T/T1w P_d:canonical_to_domain target robust interval must be positive",
    ):
        qualify_variant_a(
            artifact,
            _qualification_volumes(),
            thresholds=_permissive_thresholds(),
            vae_roundtrip=vae_roundtrip,
            **_qualification_kwargs(),
        )
    assert vae_calls == 2 * len(_qualification_volumes())


def test_qualification_masks_both_vae_paths_and_reports_raw_decoder_leakage() -> None:
    calls: dict[str, int] = {}

    def leaky_roundtrip(volume: torch.Tensor, domain: Domain) -> torch.Tensor:
        calls[domain.label] = calls.get(domain.label, 0) + 1
        result = volume.clone()
        if calls[domain.label] % 2 == 1:
            result.reshape(-1)[0] = 0.75
        return result

    def support_checked_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        assert prediction.reshape(-1)[0].item() == 0.0
        assert target.reshape(-1)[0].item() == 0.0
        return _metrics(prediction, target)

    kwargs = _qualification_kwargs()
    kwargs["metric_function"] = support_checked_metrics
    result = qualify_variant_a(
        _artifact(),
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=leaky_roundtrip,
        **kwargs,
    )
    for row in result["records"]:
        assert row["support"]["dtype"] == "bool"
        assert row["support"]["voxel_count"] == 63
        assert len(row["support"]["canonical_byte_sha256"]) == 64
        assert row["vae"]["raw_pre_mask_decoder_leakage"]["nonzero_voxel_count"] == 1
    assert result["fit_weighting_evidence"]["weighting"]["per_field_weight"] == 0.2
    ceiling = result["stage1_reconstruction_ceiling_continuity"]
    assert ceiling["method_identity"] == STAGE1_CEILING_METHOD
    assert "not a held-out-retrospective observation" in ceiling["interpretation"]


def test_qualification_classifies_photometry_failure_and_blocks_bank() -> None:
    thresholds = replace(
        _permissive_thresholds(),
        scaling_histogram_distance_max=0.0,
        scaling_ssim_min=1.0,
    )
    result = qualify_variant_a(
        _artifact(),
        _qualification_volumes(),
        thresholds=thresholds,
        vae_roundtrip=lambda volume, domain: volume,
        **_qualification_kwargs(),
    )
    assert "photometry_factorization_failure" in result["failure_classification"]
    assert result["canonical_latent_bank_authorized"] is False


def test_qualification_classifies_field_specific_canonical_vae_shift() -> None:
    calls: dict[str, int] = {}

    def roundtrip(volume: torch.Tensor, domain: Domain) -> torch.Tensor:
        calls[domain.label] = calls.get(domain.label, 0) + 1
        if domain.field_strength_t == 7.0 and calls[domain.label] % 2 == 0:
            return torch.zeros_like(volume)
        return volume

    result = qualify_variant_a(
        _artifact(),
        _qualification_volumes(),
        thresholds=_permissive_thresholds(),
        vae_roundtrip=roundtrip,
        **_qualification_kwargs(),
    )
    assert "canonical_vae_distribution_shift_failure" in result["failure_classification"]
    assert result["aggregate"]["canonical_vae_compatibility_pass"] is False
    assert result["canonical_latent_bank_authorized"] is False
    worst = result["aggregate"]["worst_domain"]["vae_domain_deltas"]
    assert worst["nrmse_absolute_increase"] > 0.03


@pytest.mark.parametrize(
    ("cohort", "split", "identity", "subject", "message"),
    [
        ("P", "validation", "P_0010_T1w", "0010", "rejects every P record"),
        ("P", "validation", "P_0006_T1w", "0006", "traveller 0006"),
        ("P", "validation", "P_0007_T1w", "0007", "traveller 0007"),
        ("P", "validation", "P_0009_T1w", "0009", "traveller 0009"),
        ("R", "train", "R_train_T1w", "R-1", "split=validation"),
    ],
)
def test_qualification_rejects_nonvalidation_prospective_and_travellers(
    cohort: str, split: str, identity: str, subject: str, message: str
) -> None:
    records = _qualification_volumes()
    item = records[0]
    records[0] = QualificationVolume(
        volume=item.volume,
        domain=item.domain,
        record_identity=identity,
        subject_identity=subject,
        metadata_prefix=cohort,
        source_path_identity=item.source_path_identity,
        source_file_sha256=item.source_file_sha256,
        cohort=cohort,
        split=split,
    )
    with pytest.raises(ValueError, match=message):
        qualify_variant_a(
            _artifact(),
            records,
            thresholds=_permissive_thresholds(),
            vae_roundtrip=lambda volume, domain: volume,
            **_qualification_kwargs(),
        )


@pytest.mark.parametrize(
    ("identity", "metadata_prefix", "cohort", "message"),
    [
        ("P_0010_T1w", "R", "R", "identity conflict"),
        ("R_0010_T1w", "P", "R", "identity conflict"),
        ("R_0010_T1w", None, "R", "metadata prefix"),
    ],
)
def test_qualification_rejects_mislabeled_conflicting_and_missing_prefixes(
    identity: str, metadata_prefix: str | None, cohort: str, message: str
) -> None:
    records = _qualification_volumes()
    records[0] = replace(
        records[0],
        record_identity=identity,
        subject_identity="0010",
        metadata_prefix=metadata_prefix,
        cohort=cohort,
    )
    with pytest.raises(ValueError, match=message):
        qualify_variant_a(
            _artifact(),
            records,
            thresholds=_permissive_thresholds(),
            vae_roundtrip=lambda volume, domain: volume,
            **_qualification_kwargs(),
        )


def test_qualification_rejects_split_hash_and_content_mutation() -> None:
    kwargs = _qualification_kwargs()
    kwargs["source_split_file_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="split-file SHA-256"):
        qualify_variant_a(
            _artifact(),
            _qualification_volumes(),
            thresholds=_permissive_thresholds(),
            vae_roundtrip=lambda volume, domain: volume,
            **kwargs,
        )

    records = _qualification_volumes()
    item = records[0]
    records[0] = replace(item, source_file_sha256="invalid")
    with pytest.raises(ValueError, match="source file SHA-256"):
        qualify_variant_a(
            _artifact(),
            records,
            thresholds=_permissive_thresholds(),
            vae_roundtrip=lambda volume, domain: volume,
            **_qualification_kwargs(),
        )


def _continuity(source_hash: str) -> ContinuityReference:
    return build_continuity_reference_from_gate01(
        _gate01_result(),
        source_result_sha256=source_hash,
        evaluation_identity="synthetic-paired-direction",
    )


def test_continuity_reference_is_source_hash_verified(tmp_path) -> None:
    source = tmp_path / "gate01.json"
    source.write_text('{"frozen": true}\n', encoding="utf-8")
    reference = _continuity(sha256_file(source))
    reference_path = write_json_atomic(
        tmp_path / "continuity.json", reference.to_dict(), refuse_existing=True
    )
    loaded = ContinuityReference.load(reference_path, source_result_path=source)
    assert loaded.artifact_sha256 == reference.artifact_sha256

    source.write_text('{"frozen": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source-result SHA-256 mismatch"):
        ContinuityReference.load(reference_path, source_result_path=source)


def _gate01_result() -> dict:
    return {
        "contract_version": GATE01_CONTRACT_VERSION,
        "overall": {
            "methods": {
                "calibrated_identity": {
                    "nrmse": 0.323585,
                    "ssim": 0.899374,
                    "lpips": 0.091186,
                },
                "raw_identity": {
                    "nrmse": 0.595189,
                    "ssim": 0.875809,
                    "lpips": 0.096859,
                },
                "stage1_reconstruction_ceiling": {
                    "nrmse": 0.131321,
                    "ssim": 0.965984,
                    "lpips": 0.036060,
                },
            }
        },
    }


def test_continuity_builder_extracts_exact_methods_and_round_trips(tmp_path) -> None:
    source = tmp_path / "gate01.json"
    source.write_text(json.dumps(_gate01_result(), sort_keys=True) + "\n", encoding="utf-8")
    reference = build_continuity_reference_from_gate01(
        _gate01_result(),
        source_result_sha256=sha256_file(source),
        evaluation_identity="paired-selection-v1",
    )
    path = write_json_atomic(tmp_path / "continuity.json", reference.to_dict())
    assert ContinuityReference.load(path, source_result_path=source) == reference
    assert reference.provenance["source_contract_version"] == GATE01_CONTRACT_VERSION


def test_documented_continuity_example_is_byte_exact_and_loadable(tmp_path) -> None:
    runbook = (
        Path(__file__).parents[1] / "docs" / "stage2_photometry_variant_a_runbook.md"
    ).read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", runbook, flags=re.DOTALL)
    source_text = next(
        block
        for block in blocks
        if '"contract_version":"stage2-gate01-equal-photometry-v2"' in block
    )
    documented_reference = json.loads(
        next(
            block
            for block in blocks
            if '"contract_version": "stage2-photometry-continuity-reference-v2"'
            in block
        )
    )
    source = tmp_path / "documented-gate01.json"
    source.write_text(source_text + "\n", encoding="utf-8", newline="\n")
    assert sha256_file(source) == documented_reference["source_result_sha256"]
    built = build_continuity_reference_from_gate01(
        json.loads(source_text),
        source_result_sha256=sha256_file(source),
        evaluation_identity="documented-synthetic-selection",
    )
    assert built.to_dict() == documented_reference
    path = write_json_atomic(tmp_path / "documented-reference.json", documented_reference)
    assert ContinuityReference.load(path, source_result_path=source) == built


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_continuity_builder_rejects_missing_or_extra_metrics(mutation: str) -> None:
    payload = _gate01_result()
    metrics = payload["overall"]["methods"]["raw_identity"]
    if mutation == "missing":
        metrics.pop("lpips")
    else:
        metrics["psnr"] = 1.0
    with pytest.raises(ValueError, match="exactly nrmse, ssim, and lpips"):
        build_continuity_reference_from_gate01(
            payload,
            source_result_sha256="f" * 64,
            evaluation_identity="selection",
        )


def test_continuity_reference_rejects_missing_provenance_and_hash_mutation() -> None:
    payload = _continuity("f" * 64).to_dict()
    missing = copy.deepcopy(payload)
    missing["provenance"] = {}
    missing["artifact_sha256"] = sha256_text("irrelevant")
    with pytest.raises(ValueError, match="source provenance"):
        ContinuityReference.from_dict(missing)
    mutated = copy.deepcopy(payload)
    mutated["methods"][RAW_IDENTITY_METHOD]["nrmse"] += 0.01
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        ContinuityReference.from_dict(mutated)


def test_dual_baseline_result_keeps_all_method_identities_separate() -> None:
    artifact = _artifact()
    source = _volume(0.1, 0)
    target = _volume(7.0, 0)
    continuity = _continuity("f" * 64)
    result = evaluate_factorized_identity_case(
        artifact,
        BaselineEvaluationCase(
            case_identity="synthetic-direction",
            selection_identity=continuity.evaluation_identity,
            source=source,
            target=target,
            source_domain=Domain(0.1, "T1w"),
            target_domain=Domain(7.0, "T1w"),
        ),
        continuity=continuity,
        metric_function=_metrics,
    )
    assert set(result["methods"]) == {FIXED_MAP_METHOD, RAW_IDENTITY_METHOD}
    assert result["external_continuity_track"]["methods"][GATE01_POSTHOC_METHOD][
        "nrmse"
    ] == 0.323585
    assert result["external_continuity_track"]["included_in_same_case_reductions"] is False
    assert "no runtime prediction CDF" in result["methods"][FIXED_MAP_METHOD]["semantics"]
    assert "must not be averaged" in result["method_identity_invariant"]


def test_evaluation_target_never_changes_fixed_transform() -> None:
    artifact = _artifact()
    source = _volume(0.1, 0)
    continuity = _continuity("f" * 64)
    captured: list[torch.Tensor] = []

    def capture(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        del target
        captured.append(prediction.clone())
        return {"nrmse": 0.0, "ssim": 1.0, "lpips": 0.0}

    for target in (torch.zeros_like(source), torch.ones_like(source)):
        evaluate_factorized_identity_case(
            artifact,
            BaselineEvaluationCase(
                case_identity="synthetic-direction",
                selection_identity=continuity.evaluation_identity,
                source=source,
                target=target,
                source_domain=Domain(0.1, "T1w"),
                target_domain=Domain(7.0, "T1w"),
            ),
            continuity=continuity,
            metric_function=capture,
        )
    # Each diagnostic evaluates raw then fixed-map; compare only fixed-map predictions.
    torch.testing.assert_close(captured[1], captured[3], rtol=0.0, atol=0.0)


def test_cli_registers_only_the_four_variant_a_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "fit-stage2-photometry",
        "audit-stage2-photometry",
        "build-stage2-photometry-continuity-reference",
        "eval-stage2-photometry-baseline",
    ):
        assert command in help_text
    assert "build-photometry-factored-latent-bank" not in help_text
    assert "train-stage2-field-graph" not in help_text


def test_continuity_reference_rejects_merged_or_missing_method_labels() -> None:
    with pytest.raises(ValueError, match="separate methods"):
        ContinuityReference(
            evaluation_identity="selection",
            source_result_sha256="f" * 64,
            methods={"calibrated_identity": {"nrmse": 0.3}},
            provenance={"source": "synthetic"},
        )
