import torch
from torch.utils.data import WeightedRandomSampler

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.sampling import domain_oversampling_weights


def _record(field_strength_t: float, case_id: str) -> VolumeRecord:
    return VolumeRecord(
        case_id=case_id,
        image_path=f"{case_id}.nii.gz",
        domain=Domain(field_strength_t, "T2-FLAIR"),
    )


def test_domain_oversampling_weights_applies_boost_and_default() -> None:
    records = [_record(0.1, "a"), _record(1.5, "b"), _record(0.1, "c"), _record(7.0, "d")]

    weights = domain_oversampling_weights(records, boost_by_field={0.1: 3.0}, default_weight=1.0)

    assert weights == [3.0, 1.0, 3.0, 1.0]


def test_domain_oversampling_weights_feeds_weighted_random_sampler() -> None:
    records = [_record(0.1, "a"), _record(1.5, "b")]
    weights = domain_oversampling_weights(records, boost_by_field={0.1: 3.0})

    sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True)

    indices = list(iter(sampler))
    assert len(indices) == len(records)
    assert all(isinstance(index, int) or torch.is_tensor(index) for index in indices)
