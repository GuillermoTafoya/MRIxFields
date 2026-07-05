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

