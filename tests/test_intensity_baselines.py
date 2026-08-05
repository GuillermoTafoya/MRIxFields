from __future__ import annotations

import json

import pytest
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.evaluation.intensity_baselines import (
    INTENSITY_BASELINE_CONTRACT_VERSION,
    ImageIntensityBaseline,
    fit_image_intensity_baselines,
    reference_volume_records,
)

SRC = Domain(0.1, Contrast.T1W)
TGT = Domain(3.0, Contrast.T1W)


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


def test_both_modes_are_fitted_from_one_pass() -> None:
    baselines = _fit()
    assert set(baselines) == {"robust_affine", "histogram"}
    for baseline in baselines.values():
        assert set(baseline.references) == {SRC.label, TGT.label}
        assert baseline.references[TGT.label].volumes == 2
        assert baseline.provenance["reference_volumes"] == 4


def test_histogram_matching_moves_the_intensity_distribution_onto_the_target() -> None:
    baseline = _fit()["histogram"]
    source = _volume(11, scale=0.4)
    mapped = baseline.apply(source, TGT)

    reference = baseline.references[TGT.label].quantiles
    probabilities = baseline.probabilities
    mapped_quantiles = torch.quantile(mapped.reshape(-1), probabilities)
    # The mapped volume's quantiles should track the target reference far more closely than
    # the untouched source's did.
    source_error = (torch.quantile(source.reshape(-1), probabilities) - reference).abs().mean()
    mapped_error = (mapped_quantiles - reference).abs().mean()
    assert mapped_error < 0.25 * source_error


def test_robust_affine_matches_the_target_percentile_window() -> None:
    baseline = _fit()["robust_affine"]
    source = _volume(12, scale=0.4)
    mapped = baseline.apply(source, TGT)

    probabilities = baseline.probabilities
    low_index = int(torch.argmin((probabilities - baseline.low_probability).abs()))
    high_index = int(torch.argmin((probabilities - baseline.high_probability).abs()))
    reference = baseline.references[TGT.label].quantiles
    mapped_quantiles = torch.quantile(mapped.reshape(-1), probabilities)

    assert mapped_quantiles[low_index] == pytest.approx(float(reference[low_index]), abs=0.05)
    assert mapped_quantiles[high_index] == pytest.approx(float(reference[high_index]), abs=0.05)


def test_intensity_map_preserves_shape_and_monotonicity() -> None:
    baseline = _fit()["histogram"]
    source = _volume(13, scale=0.4)
    mapped = baseline.apply(source, TGT)

    assert mapped.shape == source.shape
    # A photometric map must not reorder voxels: that is what makes it a pure intensity
    # baseline rather than something that could invent structure.
    flat_source = source.reshape(-1)
    flat_mapped = mapped.reshape(-1)
    order = torch.argsort(flat_source)
    ordered = flat_mapped[order]
    assert bool((ordered[1:] - ordered[:-1] >= -1e-5).all())


def test_unfitted_domain_raises() -> None:
    baseline = _fit()["histogram"]
    with pytest.raises(KeyError, match="No intensity reference"):
        baseline.apply(_volume(14), Domain(7.0, Contrast.T2W))


def test_json_round_trip_preserves_the_mapping(tmp_path) -> None:
    baseline = _fit()["histogram"]
    path = baseline.save(tmp_path / "intensity.json")
    reloaded = ImageIntensityBaseline.load(path)

    source = _volume(15, scale=0.4)
    torch.testing.assert_close(baseline.apply(source, TGT), reloaded.apply(source, TGT))


def test_contract_version_mismatch_is_refused(tmp_path) -> None:
    payload = _fit()["histogram"].to_dict()
    assert payload["contract_version"] == INTENSITY_BASELINE_CONTRACT_VERSION
    payload["contract_version"] = "image-intensity-baseline-v0"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        ImageIntensityBaseline.load(path)


def test_empty_reference_set_is_refused() -> None:
    with pytest.raises(ValueError, match="no reference volumes"):
        fit_image_intensity_baselines([])


def test_reference_records_come_only_from_training_splits() -> None:
    records = [
        VolumeRecord(case_id="R_a", image_path="a.nii", domain=SRC, subject_id="1"),
        VolumeRecord(case_id="R_b", image_path="b.nii", domain=SRC, subject_id="2"),
        VolumeRecord(case_id="R_c", image_path="c.nii", domain=SRC, subject_id="3"),
        VolumeRecord(case_id="P_held", image_path="d.nii", domain=SRC, subject_id="0006"),
    ]
    split_of_case = {"R_a": "train", "R_b": "train", "R_c": "train", "P_held": "validation"}

    chosen = reference_volume_records(records, split_of_case=split_of_case, per_domain=2)
    case_ids = {r.case_id for r in chosen}
    assert case_ids == {"R_a", "R_b"}
    # The evaluated traveller must never define the reference it is scored against.
    assert "P_held" not in case_ids
