"""Storage source abstractions for manifests and volume records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clbfield.data.contracts import VolumeRecord
from clbfield.data.manifests import Manifest, load_manifest


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

