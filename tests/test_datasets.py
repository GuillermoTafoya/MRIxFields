import torch
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.datasets import (
    StreamingPatchDataset,
    SyntheticVolumeDataset,
    collate_raw_batches,
    random_any_to_any_selector,
)
from fieldbridge.data.domains import Domain
from fieldbridge.data.sampling import field_balanced_weights


def test_synthetic_dataset_batch_shape() -> None:
    dataset = SyntheticVolumeDataset(num_samples=3, volume_shape=(1, 6, 7, 8), seed=1)
    sample = dataset[0]
    assert sample.image.shape == (1, 6, 7, 8)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_raw_batches)
    batch = next(iter(loader))
    assert batch.image.shape == (2, 1, 6, 7, 8)
    assert len(batch.source_domain) == 2
    assert len(batch.target_domain) == 2


def test_synthetic_dataset_random_any_to_any_covers_distinct_and_identity_pairs() -> None:
    dataset = SyntheticVolumeDataset(num_samples=200, seed=7, pair_sampling="random_any_to_any")

    pairs = [(dataset[i].source_domain, dataset[i].target_domain) for i in range(len(dataset))]

    assert any(source == target for source, target in pairs)
    assert any(source != target for source, target in pairs)


def test_synthetic_dataset_random_any_to_any_is_reproducible() -> None:
    first = SyntheticVolumeDataset(num_samples=20, seed=7, pair_sampling="random_any_to_any")
    second = SyntheticVolumeDataset(num_samples=20, seed=7, pair_sampling="random_any_to_any")

    for index in range(len(first)):
        assert first[index].source_domain == second[index].source_domain
        assert first[index].target_domain == second[index].target_domain


def _volume_records(count: int) -> list[VolumeRecord]:
    return [
        VolumeRecord(case_id=f"case-{i:04d}", image_path=f"vol-{i}.nii.gz", domain=Domain(3.0, "T1w"))
        for i in range(count)
    ]


def test_streaming_patch_dataset_shape_count_and_reads_each_volume_once_per_pass() -> None:
    records = _volume_records(3)
    load_calls: dict[str, int] = {}

    def counting_loader(path, record) -> torch.Tensor:  # type: ignore[no-untyped-def]
        load_calls[record.case_id] = load_calls.get(record.case_id, 0) + 1
        return torch.randn(1, 20, 20, 20)

    dataset = StreamingPatchDataset(
        records,
        image_loader=counting_loader,
        patch_size=(8, 8, 8),
        patches_per_volume=4,
        seed=0,
    )

    patches = list(dataset)
    assert len(patches) == 3 * 4  # patches_per_volume drawn from each volume
    assert all(p.image.shape == (1, 8, 8, 8) for p in patches)
    assert all(torch.isfinite(p.image).all() for p in patches)
    # One disk read per volume per pass, not one per patch.
    assert load_calls == {"case-0000": 1, "case-0001": 1, "case-0002": 1}


def test_streaming_patch_dataset_applies_volume_transform_once_before_cropping() -> None:
    transform_calls = {"count": 0}

    def volume_transform(image: torch.Tensor) -> torch.Tensor:
        transform_calls["count"] += 1
        return image * 2.0

    dataset = StreamingPatchDataset(
        _volume_records(1),
        image_loader=lambda path, record: torch.ones(1, 10, 10, 10),
        patch_size=(4, 4, 4),
        patches_per_volume=5,
        volume_transform=volume_transform,
    )

    patches = list(dataset)
    assert transform_calls["count"] == 1  # once per volume, not per patch
    assert all(torch.allclose(p.image, torch.full((1, 4, 4, 4), 2.0)) for p in patches)


def test_streaming_patch_dataset_reshuffles_between_passes() -> None:
    dataset = StreamingPatchDataset(
        _volume_records(8),
        image_loader=lambda path, record: torch.zeros(1, 4, 4, 4),
        patch_size=(2, 2, 2),
        patches_per_volume=1,
        seed=0,
    )

    first_pass = [p.metadata["case_id"] for p in dataset]
    second_pass = [p.metadata["case_id"] for p in dataset]

    assert sorted(first_pass) == sorted(second_pass)  # same volumes each pass
    assert first_pass != second_pass  # but a different order (reseeded per pass)


def test_streaming_patch_dataset_feeds_dataloader_batches() -> None:
    dataset = StreamingPatchDataset(
        _volume_records(4),
        image_loader=lambda path, record: torch.randn(1, 12, 12, 12),
        patch_size=(6, 6, 6),
        patches_per_volume=3,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_raw_batches)
    batch = next(iter(loader))
    assert batch.image.shape == (2, 1, 6, 6, 6)
    assert len(batch.source_domain) == 2


def test_random_any_to_any_selector_is_reproducible_and_respects_identity_flag() -> None:
    record = VolumeRecord(case_id="case-0001", image_path="unused.nii.gz", domain=Domain(3.0, "T1w"))

    first_selector = random_any_to_any_selector(seed=13)
    second_selector = random_any_to_any_selector(seed=13)
    assert first_selector(record) == second_selector(record)

    no_identity_selector = random_any_to_any_selector(seed=13, allow_identity=False)
    assert no_identity_selector(record) != record.domain



# --- field-balanced streaming order (item 3) -----------------------------------------

_FIELD_COUNTS = {0.1: 12, 1.5: 30, 3.0: 20, 5.0: 8, 7.0: 18}


def _field_records() -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    for field, n in _FIELD_COUNTS.items():
        for i in range(n):
            records.append(
                VolumeRecord(case_id=f"f{field}_{i}", image_path=f"f{field}_{i}.nii.gz", domain=Domain(field, "T1w"))
            )
    return records


def _zeros_loader(path, record):  # type: ignore[no-untyped-def]
    return torch.zeros(1, 4, 4, 4)


def _field_fractions(dataset: StreamingPatchDataset, passes: int) -> dict[float, float]:
    from collections import Counter

    counter: Counter = Counter()
    for _ in range(passes):
        counter.update(patch.source_domain.field_strength_t for patch in dataset)
    total = sum(counter.values())
    return {field: counter[field] / total for field in _FIELD_COUNTS}


def test_streaming_field_balance_moves_sampled_distribution_toward_uniform() -> None:
    records = _field_records()
    weights = field_balanced_weights(records)
    balanced = StreamingPatchDataset(
        records, image_loader=_zeros_loader, patch_size=None, patches_per_volume=1, seed=0, sampling_weights=weights
    )
    natural = StreamingPatchDataset(
        records, image_loader=_zeros_loader, patch_size=None, patches_per_volume=1, seed=0
    )

    bal = _field_fractions(balanced, passes=8)
    nat = _field_fractions(natural, passes=8)
    uniform = 1.0 / len(_FIELD_COUNTS)

    # The balanced draw is much flatter across fields than the natural distribution...
    assert (max(bal.values()) - min(bal.values())) < (max(nat.values()) - min(nat.values()))
    # ...and every field lands near uniform (finite-sample tolerance over 8 passes).
    for field in _FIELD_COUNTS:
        assert abs(bal[field] - uniform) < 0.06, (field, bal[field])
    # The under-represented 5T is genuinely up-sampled vs its natural share.
    assert bal[5.0] > nat[5.0]


def test_streaming_field_balance_is_reproducible() -> None:
    records = _field_records()
    weights = field_balanced_weights(records)
    a = StreamingPatchDataset(
        records, image_loader=_zeros_loader, patch_size=None, patches_per_volume=1, seed=7, sampling_weights=weights
    )
    b = StreamingPatchDataset(
        records, image_loader=_zeros_loader, patch_size=None, patches_per_volume=1, seed=7, sampling_weights=weights
    )
    seq_a = [p.source_domain.field_strength_t for p in a]
    seq_b = [p.source_domain.field_strength_t for p in b]
    assert seq_a == seq_b


def test_streaming_without_weights_reads_each_volume_once() -> None:
    # Default (no sampling_weights) is the unchanged uniform randperm: every volume exactly once.
    records = _field_records()
    load_calls: dict[str, int] = {}

    def counting_loader(path, record):  # type: ignore[no-untyped-def]
        load_calls[record.case_id] = load_calls.get(record.case_id, 0) + 1
        return torch.zeros(1, 4, 4, 4)

    dataset = StreamingPatchDataset(
        records, image_loader=counting_loader, patch_size=None, patches_per_volume=1, seed=0
    )
    patches = list(dataset)
    assert len(patches) == len(records)
    assert all(count == 1 for count in load_calls.values())
    assert len(load_calls) == len(records)
