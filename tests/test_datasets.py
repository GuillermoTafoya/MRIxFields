from torch.utils.data import DataLoader

from clbfield.data.contracts import VolumeRecord
from clbfield.data.datasets import (
    SyntheticVolumeDataset,
    collate_raw_batches,
    random_any_to_any_selector,
)
from clbfield.data.domains import Domain


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


def test_random_any_to_any_selector_is_reproducible_and_respects_identity_flag() -> None:
    record = VolumeRecord(case_id="case-0001", image_path="unused.nii.gz", domain=Domain(3.0, "T1w"))

    first_selector = random_any_to_any_selector(seed=13)
    second_selector = random_any_to_any_selector(seed=13)
    assert first_selector(record) == second_selector(record)

    no_identity_selector = random_any_to_any_selector(seed=13, allow_identity=False)
    assert no_identity_selector(record) != record.domain

