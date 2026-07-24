"""Sanity guards for enabling num_workers>0 on the streaming patch loader.

Two fixes are locked in here, both of which only bite after workers actually spawn (i.e.
after GPU time is already committed on the training box):

* StreamingPatchDataset / ManifestVolumeDataset must stay picklable. DataLoader worker
  `spawn` (the default on Windows) pickles the dataset to each worker; a lambda default
  `target_domain_selector` made that fail, crashing num_workers>0 at worker start.
* _build_streaming_patch_loader_from_records must force `persistent_workers` on and set
  `prefetch_factor` when num_workers>0 (so per-epoch reshuffling survives spawn and reads
  overlap compute), and leave both off at num_workers=0.
"""

from __future__ import annotations

import hashlib
import pickle

import torch
from torch.utils.data import DataLoader

from fieldbridge.cli import _build_streaming_patch_loader_from_records
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.datasets import (
    ManifestVolumeDataset,
    StreamingPatchDataset,
    _identity_target_domain,
    collate_raw_batches,
)
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain


def _dummy_loader(path, record) -> torch.Tensor:  # module-level => picklable
    return torch.zeros(1, 8, 8, 8)


def _coordinate_loader(path, record) -> torch.Tensor:  # module-level => picklable
    del path, record
    return torch.arange(16**3, dtype=torch.float32).reshape(1, 16, 16, 16)


def _record() -> VolumeRecord:
    return VolumeRecord(
        case_id="c0",
        image_path="/does/not/exist.nii.gz",
        domain=Domain(3.0, "T1w"),
        subject_id="s0",
    )


def _config(**training: object) -> dict:
    return {
        "seed": 13,
        "data": {
            "patch_size": [8, 8, 8],
            "patches_per_volume": 2,
            "foreground_threshold": 0.0,
            "stratified_crop": {"foreground": 0.7, "border": 0.2, "air": 0.1},
        },
        "training": dict(training),
    }


def test_streaming_dataset_default_selector_is_module_level_and_picklable() -> None:
    dataset = StreamingPatchDataset(
        [_record()], image_loader=_dummy_loader, patch_size=(8, 8, 8), patches_per_volume=2
    )
    assert dataset.target_domain_selector is _identity_target_domain
    # Must round-trip through pickle exactly as DataLoader does for spawn workers.
    pickle.loads(pickle.dumps(dataset))


def test_manifest_dataset_default_selector_is_picklable() -> None:
    dataset = ManifestVolumeDataset([_record()], image_loader=_dummy_loader)
    assert dataset.target_domain_selector is _identity_target_domain
    pickle.loads(pickle.dumps(dataset))


def test_loader_forces_persistent_and_prefetch_when_workers_positive() -> None:
    loader = _build_streaming_patch_loader_from_records(
        [_record()], batch_size=2, config=_config(prefetch_factor=3), num_workers=2
    )
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3


def test_loader_leaves_persistent_off_when_single_process() -> None:
    loader = _build_streaming_patch_loader_from_records(
        [_record()], batch_size=2, config=_config(), num_workers=0
    )
    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def _worker_invariance_records() -> list[VolumeRecord]:
    return [
        VolumeRecord(
            case_id=f"case-{index}",
            image_path=f"case-{index}.nii.gz",
            domain=Domain(field, contrast),
            subject_id=f"subject-{index}",
            metadata={"prefix": "P"},
        )
        for index, (field, contrast) in enumerate(
            (field, contrast)
            for field in FIELD_STRENGTHS_T
            for contrast in CONTRASTS
        )
    ]


def _crop_signatures(num_workers: int) -> list[tuple[str, str]]:
    dataset = StreamingPatchDataset(
        _worker_invariance_records(),
        image_loader=_coordinate_loader,
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        seed=29,
        joint_domain_balance=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=num_workers,
        persistent_workers=False,
        collate_fn=collate_raw_batches,
    )
    signatures: list[tuple[str, str]] = []
    for batch in loader:
        assert isinstance(batch.metadata, list)
        for index, metadata in enumerate(batch.metadata):
            digest = hashlib.sha256(
                batch.image[index].numpy().tobytes()
            ).hexdigest()
            signatures.append((str(metadata["case_id"]), digest))
    return sorted(signatures)


def test_crop_sequence_is_worker_assignment_invariant_and_item_decorrelated() -> None:
    single_process = _crop_signatures(0)
    repeated = _crop_signatures(0)
    two_workers = _crop_signatures(2)

    assert single_process == repeated == two_workers
    # Scheduled items do not all inherit one correlated initial RNG stream.
    assert len({digest for _, digest in single_process}) > 1
