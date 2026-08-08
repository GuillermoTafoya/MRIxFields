from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fieldbridge.cli import _load_variant_a_config, build_parser
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SOURCE_MODULES,
    PhotometryFitVolume,
    fit_frozen_photometry,
    sha256_file,
    sha256_text,
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
    evaluate_factorized_identity_case,
    qualify_variant_a,
)


def _volume(field: float, contrast_index: int, *, offset: float = 0.0) -> torch.Tensor:
    values = torch.linspace(0.05, 0.95, 63, dtype=torch.float32)
    scale = 0.45 + 0.04 * FIELD_STRENGTHS_T.index(field) + 0.04 * contrast_index
    return torch.cat([torch.zeros(1), values * scale + offset]).reshape(4, 4, 4)


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
        num_quantiles=17,
    )


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


def test_variant_a_v1_config_rejects_weakened_thresholds(monkeypatch) -> None:
    path = __import__("pathlib").Path(
        "configs/experiment/stage2_photometry_factorization_a_v1.yaml"
    )
    config = copy.deepcopy(_load_variant_a_config(path))
    config["qualification"]["thresholds"]["roundtrip_macro_nrmse_max"] = 1.0
    monkeypatch.setattr("fieldbridge.cli.load_yaml_config", lambda ignored: config)
    with pytest.raises(ValueError, match="reviewed proposed defaults"):
        _load_variant_a_config(path)


def test_qualification_reports_macro_worst_domain_scaling_and_equal_counts() -> None:
    result = qualify_variant_a(
        _artifact(),
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
    assert result["interpolation_qualification"]["grid_count"] == 18
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
        ("P", "validation", "P_0010_T1w", "0010", "cohort R only"),
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
    return ContinuityReference(
        evaluation_identity="selection-0006-direction",
        source_result_sha256=source_hash,
        methods={
            GATE01_POSTHOC_METHOD: {"nrmse": 0.323585, "ssim": 0.899374, "lpips": 0.091186},
            RAW_IDENTITY_METHOD: {"nrmse": 0.595189, "ssim": 0.875809, "lpips": 0.096859},
            STAGE1_CEILING_METHOD: {"nrmse": 0.131321, "ssim": 0.965984, "lpips": 0.036060},
        },
        provenance={"source_contract": "stage2-gate01-result-v3", "observation": "external"},
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
    assert set(result["methods"]) == {FIXED_MAP_METHOD, *CONTINUITY_METHODS}
    assert result["methods"][GATE01_POSTHOC_METHOD]["metrics"]["nrmse"] == 0.323585
    assert "prediction-CDF" in result["methods"][GATE01_POSTHOC_METHOD]["semantics"]
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
    torch.testing.assert_close(captured[0], captured[1], rtol=0.0, atol=0.0)


def test_cli_registers_only_the_three_variant_a_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "fit-stage2-photometry",
        "audit-stage2-photometry",
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
