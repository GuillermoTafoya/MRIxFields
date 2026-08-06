from __future__ import annotations

import json

import pytest
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.evaluation.intensity_baselines import (
    INTENSITY_BASELINE_CONTRACT_VERSION,
    ImageIntensityBaseline,
    fit_image_intensity_baselines,
    reference_volume_records,
)

SRC = Domain(0.1, "T1w")
TGT = Domain(3.0, "T1w")


def _volume(seed: int, scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 1, 8, 12, 12, generator=generator)
    return (base * scale + shift).clamp(0.0, 1.0)


def _fit(**kwargs):
    volumes = [
        (_volume(1, scale=0.4), SRC),
        (_volume(2, scale=0.4), SRC),
        (_volume(3, scale=1.0), TGT),
        (_volume(4, scale=1.0), TGT),
    ]
    return fit_image_intensity_baselines(volumes, **kwargs)


def test_both_legacy_modes_are_fitted_from_one_pass() -> None:
    baselines = _fit()
    assert set(baselines) == {"robust_affine", "histogram"}
    for baseline in baselines.values():
        assert set(baseline.references) == {SRC.label, TGT.label}
        assert baseline.references[TGT.label].volumes == 2
        assert baseline.provenance["reference_volumes"] == 4


def test_legacy_histogram_matching_moves_distribution_to_target() -> None:
    baseline = _fit()["histogram"]
    source = _volume(11, scale=0.4)
    mapped = baseline.apply(source, TGT)
    reference = baseline.references[TGT.label].quantiles
    probabilities = baseline.probabilities
    source_error = (
        torch.quantile(source.reshape(-1), probabilities) - reference
    ).abs().mean()
    mapped_error = (
        torch.quantile(mapped.reshape(-1), probabilities) - reference
    ).abs().mean()
    assert mapped_error < 0.25 * source_error


def test_legacy_robust_affine_matches_target_percentile_window() -> None:
    baseline = _fit()["robust_affine"]
    mapped = baseline.apply(_volume(12, scale=0.4), TGT)
    probabilities = baseline.probabilities
    low_index = int(torch.argmin((probabilities - baseline.low_probability).abs()))
    high_index = int(torch.argmin((probabilities - baseline.high_probability).abs()))
    reference = baseline.references[TGT.label].quantiles
    mapped_quantiles = torch.quantile(mapped.reshape(-1), probabilities)
    assert mapped_quantiles[low_index] == pytest.approx(
        float(reference[low_index]), abs=0.05
    )
    assert mapped_quantiles[high_index] == pytest.approx(
        float(reference[high_index]), abs=0.05
    )


def test_legacy_intensity_map_preserves_shape_and_monotonicity() -> None:
    baseline = _fit()["histogram"]
    source = _volume(13, scale=0.4)
    mapped = baseline.apply(source, TGT)
    assert mapped.shape == source.shape
    order = torch.argsort(source.reshape(-1))
    ordered = mapped.reshape(-1)[order]
    assert bool((ordered[1:] - ordered[:-1] >= -1e-5).all())


def test_legacy_unfitted_domain_raises() -> None:
    with pytest.raises(KeyError, match="No intensity reference"):
        _fit()["histogram"].apply(_volume(14), Domain(7.0, "T2w"))


def test_legacy_json_round_trip_preserves_mapping(tmp_path) -> None:
    baseline = _fit()["histogram"]
    reloaded = ImageIntensityBaseline.load(baseline.save(tmp_path / "intensity.json"))
    source = _volume(15, scale=0.4)
    torch.testing.assert_close(baseline.apply(source, TGT), reloaded.apply(source, TGT))


def test_legacy_contract_version_mismatch_is_refused(tmp_path) -> None:
    payload = _fit()["histogram"].to_dict()
    assert payload["contract_version"] == INTENSITY_BASELINE_CONTRACT_VERSION
    payload["contract_version"] = "image-intensity-baseline-v0"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        ImageIntensityBaseline.load(path)


def test_legacy_empty_reference_set_is_refused() -> None:
    with pytest.raises(ValueError, match="no reference volumes"):
        fit_image_intensity_baselines([])


def test_legacy_reference_records_come_only_from_training_splits() -> None:
    records = [
        VolumeRecord(case_id="R_a", image_path="a.nii", domain=SRC, subject_id="1"),
        VolumeRecord(case_id="R_b", image_path="b.nii", domain=SRC, subject_id="2"),
        VolumeRecord(case_id="R_c", image_path="c.nii", domain=SRC, subject_id="3"),
        VolumeRecord(
            case_id="P_held",
            image_path="d.nii",
            domain=SRC,
            subject_id="held-out-synthetic",
        ),
    ]
    split_of_case = {
        "R_a": "train",
        "R_b": "train",
        "R_c": "train",
        "P_held": "validation",
    }
    chosen = reference_volume_records(records, split_of_case=split_of_case, per_domain=2)
    assert {record.case_id for record in chosen} == {"R_a", "R_b"}
    assert "P_held" not in {record.case_id for record in chosen}
