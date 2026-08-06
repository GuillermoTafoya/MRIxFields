from __future__ import annotations

import copy
import gc
import inspect
import weakref

import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.evaluation.intensity_baselines import (
    ImageIntensityBaseline,
    fit_image_intensity_baselines,
)
from fieldbridge.evaluation.stage2_gate01 import (
    frozen_artifact_provenance,
    validate_frozen_artifact_provenance,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    GATE01_CALIBRATION_SEMANTICS,
    PosthocTargetCalibrator,
    RESPLIT_FINGERPRINT,
    TrainingTemplateVolume,
    all_domain_labels,
    fit_posthoc_target_calibrator,
    reject_target_derived_calibration_fields,
)


def _training_volume(field: float, contrast_index: int, offset: int) -> torch.Tensor:
    values = torch.linspace(0.05, 0.95, 63, dtype=torch.float32)
    scale = 0.35 + 0.05 * contrast_index + 0.04 * FIELD_STRENGTHS_T.index(field)
    foreground = (values * scale + 0.01 * offset).clamp_max(1.0)
    return torch.cat([torch.zeros(1), foreground]).reshape(4, 4, 4)


def _fit_calibrator() -> PosthocTargetCalibrator:
    records: list[TrainingTemplateVolume] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            # Deliberately unequal counts: templates remain independent and each volume
            # contributes equally within its own domain.
            for offset in range(1 + (field_index % 3)):
                records.append(
                    TrainingTemplateVolume(
                        volume=_training_volume(field, contrast_index, offset),
                        domain=Domain(field, contrast),
                        record_identity=f"synthetic-{contrast.value}-{field:g}-{offset}",
                    )
                )
    return fit_posthoc_target_calibrator(
        records,
        split_fingerprint=RESPLIT_FINGERPRINT,
        training_cohort_identity="synthetic-retrospective-training",
        code_commit="synthetic-code-commit",
        num_quantiles=17,
    )


@pytest.fixture(scope="module")
def calibrator() -> PosthocTargetCalibrator:
    return _fit_calibrator()


def test_gate01_fits_all_15_domains_with_documented_unequal_count_balancing(
    calibrator: PosthocTargetCalibrator,
) -> None:
    assert set(calibrator.templates) == set(all_domain_labels())
    assert len(calibrator.templates) == 15
    assert set(calibrator.provenance["domain_volume_counts"].values()) == {1, 2, 3}
    balancing = calibrator.provenance["balancing"]
    assert "equal weight per training volume" in balancing["within_domain"]
    assert "independent template per domain" in balancing["across_domains"]
    assert "retained and disclosed" in balancing["unequal_domain_counts_policy"]


def test_calibration_uses_prediction_and_frozen_target_template_only(
    calibrator: PosthocTargetCalibrator,
) -> None:
    prediction = torch.tensor(
        [[[0.0, 0.1], [0.3, 0.5]], [[0.7, 0.9], [0.2, 0.4]]],
        dtype=torch.float32,
    )
    requested_domain = Domain(7.0, "T1w")
    paired_target_a = torch.zeros_like(prediction)
    paired_target_b = torch.full_like(prediction, 0.987)

    support = prediction != 0
    first = calibrator.apply(prediction, requested_domain, support_mask=support)
    # Paired targets are deliberately changed but are not accepted by or visible to apply().
    paired_target_a.copy_(paired_target_b)
    second = calibrator.apply(prediction, requested_domain, support_mask=support)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert calibrator.to_dict()["semantics"] == GATE01_CALIBRATION_SEMANTICS
    assert calibrator.to_dict()["target_independence"]["calibration_inputs"] == [
        "method_prediction",
        "requested_target_domain",
        "frozen_source_support_mask",
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        "target_mask",
        "target_histogram",
        "target_quantiles",
        "target_statistics",
        "calibration_target",
    ],
)
def test_target_derived_masks_and_statistics_are_rejected(forbidden: str) -> None:
    with pytest.raises(ValueError, match="forbids target-derived"):
        reject_target_derived_calibration_fields(
            {"cases": [{"calibration": {forbidden: [1, 2, 3]}}]}
        )


def test_calibration_is_deterministic_monotonic_and_preserves_exact_zero_background(
    calibrator: PosthocTargetCalibrator,
) -> None:
    prediction = torch.tensor(
        [[[0.0, 0.8], [0.2, 0.6]], [[0.4, 0.1], [0.7, 0.3]]],
        dtype=torch.float32,
    )
    target = Domain(5.0, "T2w")
    support = prediction != 0
    first = calibrator.apply(prediction, target, support_mask=support)
    second = calibrator.apply(prediction.clone(), target, support_mask=support)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first[0, 0, 0].item() == 0.0
    source_foreground = prediction[prediction != 0]
    mapped_foreground = first[prediction != 0]
    order = torch.argsort(source_foreground)
    ordered = mapped_foreground[order]
    assert bool((ordered[1:] >= ordered[:-1]).all())


def test_frozen_source_support_removes_tiny_decoder_background_identically(
    calibrator: PosthocTargetCalibrator,
) -> None:
    support = torch.zeros(4, 4, 4, dtype=torch.bool)
    support[1:3, 1:3, 1:3] = True
    identity = torch.full((4, 4, 4), 1e-7)
    sb = torch.full((4, 4, 4), 9e-7)
    identity[support] = torch.linspace(0.1, 0.8, int(support.sum()))
    sb[support] = torch.linspace(0.2, 0.9, int(support.sum()))
    domain = Domain(3.0, "T1w")

    calibrated_identity = calibrator.apply(identity, domain, support_mask=support)
    calibrated_sb = calibrator.apply(sb, domain, support_mask=support)

    assert bool((calibrated_identity[~support] == 0).all())
    assert bool((calibrated_sb[~support] == 0).all())
    assert bool((calibrated_identity[support] != 0).all())
    assert bool((calibrated_sb[support] != 0).all())


def test_robust_affine_is_deterministic_and_background_safe(
    calibrator: PosthocTargetCalibrator,
) -> None:
    prediction = _training_volume(1.5, 2, 0)
    target = Domain(3.0, "T2-FLAIR")
    support = prediction != 0
    first = calibrator.apply(
        prediction, target, support_mask=support, mode="robust_affine"
    )
    second = calibrator.apply(
        prediction, target, support_mask=support, mode="robust_affine"
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool((first[prediction == 0] == 0).all())


def test_gate01_calibrator_fails_closed_on_shape_finite_and_unknown_domain(
    calibrator: PosthocTargetCalibrator,
) -> None:
    with pytest.raises(ValueError, match="full volume"):
        calibrator.apply(
            torch.ones(3, 3),
            Domain(3.0, "T1w"),
            support_mask=torch.ones(3, 3, dtype=torch.bool),
        )

    nonfinite = torch.ones(3, 3, 3)
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        calibrator.apply(
            nonfinite,
            Domain(3.0, "T1w"),
            support_mask=torch.ones_like(nonfinite, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="support-mask shape mismatch"):
        calibrator.apply(
            torch.ones(3, 3, 3),
            Domain(3.0, "T1w"),
            support_mask=torch.ones(2, 3, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        calibrator.apply(
            torch.ones(3, 3, 3),
            Domain(3.0, "T1w"),
            support_mask=torch.ones(3, 3, 3),
        )

    # Domain itself fails closed before lookup for an unsupported field strength.
    with pytest.raises(ValueError, match="Unsupported field strength"):
        Domain(9.4, "T1w")


def test_training_only_and_retrospective_only_fit_guards() -> None:
    base = [
        TrainingTemplateVolume(
            volume=torch.ones(3, 3, 3),
            domain=Domain(field, contrast),
            record_identity=f"{contrast.value}-{field:g}",
        )
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    ]
    validation = list(base)
    validation[0] = TrainingTemplateVolume(
        volume=validation[0].volume,
        domain=validation[0].domain,
        record_identity=validation[0].record_identity,
        split="validation",
    )
    with pytest.raises(ValueError, match="training records only"):
        fit_posthoc_target_calibrator(
            validation,
            split_fingerprint=RESPLIT_FINGERPRINT,
            training_cohort_identity="synthetic",
            code_commit="abc",
        )

    prospective = list(base)
    prospective[0] = TrainingTemplateVolume(
        volume=prospective[0].volume,
        domain=prospective[0].domain,
        record_identity=prospective[0].record_identity,
        cohort="P",
    )
    with pytest.raises(ValueError, match="retrospective cohort only"):
        fit_posthoc_target_calibrator(
            prospective,
            split_fingerprint=RESPLIT_FINGERPRINT,
            training_cohort_identity="synthetic",
            code_commit="abc",
        )


def test_calibrator_fit_is_streaming_with_bounded_full_volume_liveness() -> None:
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    maximum_alive = 0

    def volumes():
        nonlocal maximum_alive
        for repeat in range(3):
            for contrast_index, contrast in enumerate(CONTRASTS):
                for field in FIELD_STRENGTHS_T:
                    tensor = _training_volume(field, contrast_index, repeat)
                    references.append(weakref.ref(tensor))
                    gc.collect()
                    maximum_alive = max(
                        maximum_alive,
                        sum(reference() is not None for reference in references),
                    )
                    yield TrainingTemplateVolume(
                        volume=tensor,
                        domain=Domain(field, contrast),
                        record_identity=f"stream-{repeat}-{contrast.value}-{field:g}",
                    )

    fitted = fit_posthoc_target_calibrator(
        volumes(),
        split_fingerprint=RESPLIT_FINGERPRINT,
        training_cohort_identity="synthetic-stream",
        code_commit="stream-test",
        num_quantiles=9,
    )
    gc.collect()
    assert fitted.provenance["domain_volume_counts"]
    assert maximum_alive <= 2
    assert sum(reference() is not None for reference in references) == 0
    source = inspect.getsource(fit_posthoc_target_calibrator)
    assert "list(volumes)" not in source


def test_stale_split_template_and_artifact_provenance_are_rejected(
    calibrator: PosthocTargetCalibrator,
) -> None:
    payload = calibrator.to_dict()
    with pytest.raises(ValueError, match="split fingerprint mismatch"):
        PosthocTargetCalibrator.from_dict(
            payload, expected_split_fingerprint="stale-split"
        )

    altered = copy.deepcopy(payload)
    label = next(iter(altered["templates"]))
    altered["templates"][label]["quantiles"][2] += 0.01
    with pytest.raises(ValueError, match="hash mismatch"):
        PosthocTargetCalibrator.from_dict(altered)

    altered_provenance = copy.deepcopy(payload)
    altered_provenance["provenance"]["code_commit"] = "stale-code"
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        PosthocTargetCalibrator.from_dict(altered_provenance)

    artifacts = frozen_artifact_provenance()
    artifacts["sb_v2_checkpoint_sha256"] = "stale-checkpoint"
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        validate_frozen_artifact_provenance(artifacts)


def test_calibrator_json_round_trip_preserves_exact_mapping(
    calibrator: PosthocTargetCalibrator, tmp_path
) -> None:
    path = calibrator.save(tmp_path / "calibrator.json")
    loaded = PosthocTargetCalibrator.load(
        path,
        expected_split_fingerprint=RESPLIT_FINGERPRINT,
        expected_template_sha256=calibrator.template_sha256,
    )
    prediction = _training_volume(3.0, 0, 0)
    torch.testing.assert_close(
        calibrator.apply(
            prediction, Domain(7.0, "T1w"), support_mask=prediction != 0
        ),
        loaded.apply(
            prediction, Domain(7.0, "T1w"), support_mask=prediction != 0
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_legacy_gate0_histogram_baseline_remains_backward_compatible(tmp_path) -> None:
    source = Domain(0.1, "T1w")
    target = Domain(3.0, "T1w")
    baselines = fit_image_intensity_baselines(
        [
            (_training_volume(0.1, 0, 0), source),
            (_training_volume(3.0, 0, 0), target),
        ],
        num_quantiles=17,
    )
    legacy = baselines["histogram"]
    path = legacy.save(tmp_path / "legacy.json")
    reloaded = ImageIntensityBaseline.load(path)
    prediction = _training_volume(0.1, 0, 0)
    torch.testing.assert_close(
        legacy.apply(prediction, target), reloaded.apply(prediction, target)
    )
    # Gate-0 v1 mapped the entire image, including zeros. This is intentionally preserved;
    # Gate 0.1 uses PosthocTargetCalibrator for the corrected background contract.
    assert legacy.to_dict()["contract_version"] == "image-intensity-baseline-v1"
