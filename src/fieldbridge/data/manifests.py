"""Manifest loading and auditing utilities."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fieldbridge.data.contracts import VolumeRecord


@dataclass(frozen=True, slots=True)
class Manifest:
    """Collection of volume records independent of storage backend."""

    records: tuple[VolumeRecord, ...]
    name: str = "manifest"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        records: Iterable[VolumeRecord],
        *,
        name: str = "manifest",
        metadata: Mapping[str, Any] | None = None,
    ) -> "Manifest":
        return cls(records=tuple(records), name=name, metadata=dict(metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "records": [record.to_dict() for record in self.records],
        }


def record_from_mapping(data: Mapping[str, Any]) -> VolumeRecord:
    return VolumeRecord(
        case_id=str(data["case_id"]),
        image_path=Path(str(data["image_path"])),
        domain=dict(data["domain"]),
        subject_id=None if data.get("subject_id") is None else str(data["subject_id"]),
        split=None if data.get("split") is None else str(data["split"]),
        metadata=dict(data.get("metadata", {})),
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load a JSON or YAML manifest into records."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        if manifest_path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)

    if isinstance(payload, list):
        records_payload = payload
        name = manifest_path.stem
        metadata: Mapping[str, Any] = {}
    elif isinstance(payload, dict):
        records_payload = payload.get("records", [])
        name = str(payload.get("name", manifest_path.stem))
        metadata = dict(payload.get("metadata", {}))
    else:
        raise ValueError(f"Unsupported manifest payload in {manifest_path}.")

    if not isinstance(records_payload, list):
        raise ValueError("Manifest records must be a list.")

    return Manifest.from_records(
        (record_from_mapping(record) for record in records_payload),
        name=name,
        metadata=metadata,
    )


def audit_manifest(manifest: Manifest, *, strict_paths: bool = False) -> dict[str, Any]:
    """Return manifest health counts without reading image contents."""

    case_ids = [record.case_id for record in manifest.records]
    duplicate_case_ids = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    missing_paths = [str(record.image_path) for record in manifest.records if not record.image_path.exists()]
    domain_counts = Counter(record.domain.label for record in manifest.records)
    split_counts = Counter(record.split or "unspecified" for record in manifest.records)
    ok = not duplicate_case_ids and (not strict_paths or not missing_paths)

    return {
        "name": manifest.name,
        "record_count": len(manifest.records),
        "ok": ok,
        "duplicate_case_ids": duplicate_case_ids,
        "missing_path_count": len(missing_paths),
        "missing_paths": missing_paths,
        "domain_counts": dict(sorted(domain_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }

