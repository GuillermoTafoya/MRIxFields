from torch.utils.data import DataLoader

from clbfield.data.datasets import SyntheticVolumeDataset, collate_raw_batches


def test_synthetic_dataset_batch_shape() -> None:
    dataset = SyntheticVolumeDataset(num_samples=3, volume_shape=(1, 6, 7, 8), seed=1)
    sample = dataset[0]
    assert sample.image.shape == (1, 6, 7, 8)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_raw_batches)
    batch = next(iter(loader))
    assert batch.image.shape == (2, 1, 6, 7, 8)
    assert len(batch.source_domain) == 2
    assert len(batch.target_domain) == 2

