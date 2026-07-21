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

import pickle

import torch

from fieldbridge.cli import _build_streaming_patch_loader_from_records
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.datasets import (
    ManifestVolumeDataset,
    StreamingPatchDataset,
    _identity_target_domain,
)
from fieldbridge.data.domains import Domain


def _dummy_loader(path, record) -> torch.Tensor:  # module-level => picklable
    return torch.zeros(1, 8, 8, 8)


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
