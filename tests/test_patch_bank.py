import json

import pytest
import torch
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.datasets import collate_raw_batches
from fieldbridge.data.domains import Domain
from fieldbridge.data.patch_bank import (
    PatchBankDataset,
    build_patch_bank,
    patch_bank_size,
)


@pytest.mark.parametrize("open_bank", [PatchBankDataset, patch_bank_size])
def test_missing_bank_directory_names_the_build_command(tmp_path, open_bank) -> None:
    with pytest.raises(FileNotFoundError, match="build-patch-bank"):
        open_bank(tmp_path / "patch_bank_ppv32")


@pytest.mark.parametrize("open_bank", [PatchBankDataset, patch_bank_size])
def test_half_built_bank_reports_missing_files(tmp_path, open_bank) -> None:
    # Directory exists (an interrupted build) but has no meta/index: must say so rather
    # than raising a bare FileNotFoundError on bank_meta.json.
    bank = tmp_path / "patch_bank_ppv32"
    bank.mkdir()
    with pytest.raises(FileNotFoundError, match="bank_meta.json"):
        open_bank(bank)


def _records(count: int) -> list[VolumeRecord]:
    fields = [0.1, 1.5, 3.0, 5.0, 7.0]
    return [
        VolumeRecord(
            case_id=f"case-{i:03d}",
            image_path=f"vol-{i}.nii.gz",
            domain=Domain(fields[i % len(fields)], "T1w"),
        )
        for i in range(count)
    ]


def _identity_loader(shape=(1, 8, 8, 8)):
    def _load(path, record):  # type: ignore[no-untyped-def]
        return torch.randn(shape)

    return _load


def test_build_patch_bank_writes_shards_index_and_meta(tmp_path) -> None:
    result = build_patch_bank(
        _records(4),
        image_loader=_identity_loader(),
        out_dir=tmp_path,
        patch_size=(4, 4, 4),
        patches_per_volume=3,
        volume_transform=None,
        seed=7,
    )

    assert result.num_volumes_written == 4
    assert result.num_volumes_failed == 0
    assert result.total_patches == 12

    assert (tmp_path / "bank_meta.json").exists()
    shards = list((tmp_path / "shards").glob("*.npy"))
    assert len(shards) == 4
    index_lines = [json.loads(x) for x in (tmp_path / "bank_index.jsonl").read_text().splitlines() if x.strip()]
    assert len(index_lines) == 4
    assert index_lines[0]["num_patches"] == 3
    assert patch_bank_size(tmp_path) == (4, 3)


def test_build_patch_bank_is_resumable(tmp_path) -> None:
    load_counts = {"n": 0}

    def counting_loader(path, record):  # type: ignore[no-untyped-def]
        load_counts["n"] += 1
        return torch.randn(1, 8, 8, 8)

    records = _records(3)
    build_patch_bank(
        records, image_loader=counting_loader, out_dir=tmp_path,
        patch_size=(4, 4, 4), patches_per_volume=2, volume_transform=None,
    )
    assert load_counts["n"] == 3

    # Second run over the same dir must skip everything and not re-read from disk.
    second = build_patch_bank(
        records, image_loader=counting_loader, out_dir=tmp_path,
        patch_size=(4, 4, 4), patches_per_volume=2, volume_transform=None,
    )
    assert load_counts["n"] == 3  # no additional reads
    assert second.num_volumes_written == 0
    assert second.num_volumes_skipped == 3


def test_build_patch_bank_tolerates_read_failures(tmp_path) -> None:
    def flaky_loader(path, record):  # type: ignore[no-untyped-def]
        if record.case_id == "case-001":
            raise OSError(107, "Transport endpoint is not connected")
        return torch.randn(1, 8, 8, 8)

    result = build_patch_bank(
        _records(3), image_loader=flaky_loader, out_dir=tmp_path,
        patch_size=(4, 4, 4), patches_per_volume=2, volume_transform=None, max_read_retries=1,
    )

    assert result.num_volumes_written == 2
    assert result.num_volumes_failed == 1
    failures = [json.loads(x) for x in (tmp_path / "bank_failures.jsonl").read_text().splitlines() if x.strip()]
    assert failures[0]["case_id"] == "case-001"
    # The failed volume is absent from the bank (not zero-filled).
    assert patch_bank_size(tmp_path) == (2, 2)


def test_patch_bank_dataset_serves_patches_and_domains(tmp_path) -> None:
    records = _records(3)
    build_patch_bank(
        records, image_loader=_identity_loader((1, 10, 10, 10)), out_dir=tmp_path,
        patch_size=(4, 4, 4), patches_per_volume=4, volume_transform=None,
    )

    dataset = PatchBankDataset(tmp_path)
    assert len(dataset) == 3 * 4

    sample = dataset[0]
    assert sample.image.shape == (1, 4, 4, 4)
    assert sample.image.dtype == torch.float32
    assert torch.isfinite(sample.image).all()
    # Patch i belongs to volume i // ppv -> its domain.
    assert dataset[0].source_domain == records[0].domain
    assert dataset[4].source_domain == records[1].domain
    assert dataset[8].source_domain == records[2].domain

    loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_raw_batches)
    batch = next(iter(loader))
    assert batch.image.shape == (5, 1, 4, 4, 4)
    assert len(batch.source_domain) == 5


def test_patch_bank_build_is_deterministic_for_a_seed(tmp_path) -> None:
    # Same seed + same input volume => identical patches (needed for resume consistency).
    def fixed_loader(path, record):  # type: ignore[no-untyped-def]
        g = torch.Generator().manual_seed(123)
        return torch.randn(1, 12, 12, 12, generator=g)

    bank_a = tmp_path / "a"
    bank_b = tmp_path / "b"
    for out in (bank_a, bank_b):
        build_patch_bank(
            _records(1), image_loader=fixed_loader, out_dir=out,
            patch_size=(4, 4, 4), patches_per_volume=3, volume_transform=None, seed=42,
        )

    a = PatchBankDataset(bank_a)
    b = PatchBankDataset(bank_b)
    assert torch.equal(a[0].image, b[0].image)
    assert torch.equal(a[2].image, b[2].image)
