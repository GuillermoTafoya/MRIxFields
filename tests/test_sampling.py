import torch
from torch.utils.data import WeightedRandomSampler

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.sampling import domain_oversampling_weights, field_balanced_weights


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


def _skewed_field_records() -> list[VolumeRecord]:
    # Mirror the real imbalance shape: 5T and 0.1T under-represented vs 1.5T.
    counts = {0.1: 12, 1.5: 30, 3.0: 20, 5.0: 8, 7.0: 18}
    records: list[VolumeRecord] = []
    for field, n in counts.items():
        for i in range(n):
            records.append(_record(field, f"f{field}_{i}"))
    return records


def test_field_balanced_weights_equalize_per_field_mass() -> None:
    records = _skewed_field_records()
    weights = field_balanced_weights(records)

    # Each field's total weight mass must be equal (inverse-frequency), so a weighted draw
    # samples every field equally in expectation regardless of its volume count.
    mass: dict[float, float] = {}
    for record, w in zip(records, weights):
        mass[record.domain.field_strength_t] = mass.get(record.domain.field_strength_t, 0.0) + w
    values = list(mass.values())
    assert max(values) - min(values) < 1e-9


def test_field_balanced_weights_mean_is_default_weight() -> None:
    records = _skewed_field_records()
    weights = field_balanced_weights(records, default_weight=1.0)
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-9


def test_field_balanced_weights_upsamples_rare_field() -> None:
    records = _skewed_field_records()
    weights = field_balanced_weights(records)
    by_field = {r.domain.field_strength_t: w for r, w in zip(records, weights)}
    # 5T (8 volumes) is rarer than 1.5T (30) => each 5T record must carry more weight.
    assert by_field[5.0] > by_field[1.5]
