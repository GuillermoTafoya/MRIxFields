"""Storage source abstractions for manifests and volume records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.manifests import Manifest, load_manifest
from fieldbridge.official.mrixfields2026 import parse_mrixfields_filename


class DataSource(Protocol):
    """Protocol for storage backends that expose records through manifests."""

    def records(self) -> Sequence[VolumeRecord]:
        """Return volume records without loading image arrays."""

    def manifest(self) -> Manifest:
        """Return the source manifest."""


@dataclass(slots=True)
class LocalExtractedNiftiSource:
    """Local extracted NIfTI source backed by an explicit manifest."""

    root: Path | str
    manifest_path: Path | str

    def records(self) -> Sequence[VolumeRecord]:
        return self.manifest().records

    def manifest(self) -> Manifest:
        manifest = load_manifest(self.manifest_path)
        return _resolve_relative_paths(manifest, Path(self.root))


@dataclass(slots=True)
class LocalLatentSource:
    """Local latent tensor source backed by an explicit manifest."""

    root: Path | str
    manifest_path: Path | str

    def records(self) -> Sequence[VolumeRecord]:
        return self.manifest().records

    def manifest(self) -> Manifest:
        manifest = load_manifest(self.manifest_path)
        return _resolve_relative_paths(manifest, Path(self.root))


@dataclass(slots=True)
class DriveZipSource:
    """Stub for future Drive zip ingestion."""

    drive_uri: str

    def records(self) -> Sequence[VolumeRecord]:
        raise NotImplementedError("DriveZipSource is a stub; extract data locally first.")

    def manifest(self) -> Manifest:
        raise NotImplementedError("DriveZipSource does not implement cloud access yet.")


@dataclass(slots=True)
class GCSBucketSource:
    """Stub for future Google Cloud Storage ingestion."""

    bucket_uri: str

    def records(self) -> Sequence[VolumeRecord]:
        raise NotImplementedError("GCSBucketSource is a stub; use a local manifest for now.")

    def manifest(self) -> Manifest:
        raise NotImplementedError("GCSBucketSource does not implement cloud access yet.")


def _resolve_relative_paths(manifest: Manifest, root: Path) -> Manifest:
    records = []
    for record in manifest.records:
        image_path = record.image_path
        if not image_path.is_absolute():
            image_path = root / image_path
        records.append(
            VolumeRecord(
                case_id=record.case_id,
                image_path=image_path,
                domain=record.domain,
                subject_id=record.subject_id,
                split=record.split,
                metadata=record.metadata,
            )
        )
    return Manifest.from_records(records, name=manifest.name, metadata=manifest.metadata)


def nifti_image_loader(path: Path, record: VolumeRecord) -> torch.Tensor:
    """Load an official MRIxFields NIfTI volume as a `(1, D, H, W)` float32 tensor.

    Requires the optional `nibabel` dependency (`pip install -e ".[nifti]"`); it is not
    part of the core install so CPU/synthetic-only workflows never need it.
    """

    del record
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError("nifti_image_loader requires nibabel: pip install -e \".[nifti]\".") from exc

    volume = torch.from_numpy(nib.load(str(path)).get_fdata(dtype="float32"))
    if not torch.isfinite(volume).all():
        raise ValueError(f"Non-finite values in NIfTI volume: {path}")
    return volume.unsqueeze(0)


_DEFAULT_PRIORITY_DIRS: tuple[str, ...] = (
    "Validating_prospective",
    "Training_prospective",
    "Training_retrospective",
)


def records_from_directory(
    root: Path | str,
    *,
    max_records: int | None = 8,
    priority_dirs: Sequence[str] = _DEFAULT_PRIORITY_DIRS,
) -> list[VolumeRecord]:
    """Build `VolumeRecord`s from a directory of official-format NIfTI files.

    Meant for ad-hoc dry runs against a small slice of real data (e.g. a Drive-mounted
    Colab runtime), not for checked-in manifests — per `AGENTS.md`, real data/paths never
    land in this repo. Files under `priority_dirs` (matched by any path component) are
    preferred over the rest, mirroring how validation/paired data is prioritized for
    quick sanity checks; ties break by sorted path for determinism.
    """

    root = Path(root)
    all_paths = sorted(p for p in root.rglob("*.nii.gz") if not p.name.endswith("_seg.nii.gz"))

    def _priority(path: Path) -> int:
        parts = set(path.relative_to(root).parts)
        for rank, keyword in enumerate(priority_dirs):
            if keyword in parts:
                return rank
        return len(priority_dirs)

    ordered = sorted(all_paths, key=lambda p: (_priority(p), str(p)))
    if max_records is not None:
        ordered = ordered[:max_records]

    records = []
    for path in ordered:
        parsed = parse_mrixfields_filename(path.name)
        case_id = path.stem.removesuffix(".nii")
        split = next((part for part in path.relative_to(root).parts if part in priority_dirs), None)
        records.append(
            VolumeRecord(
                case_id=case_id,
                image_path=path,
                domain={"field_strength_t": float(parsed.field[:-1]), "contrast": parsed.modality},
                subject_id=parsed.subject_id,
                split=split,
                metadata={"prefix": parsed.prefix},
            )
        )
    return records

