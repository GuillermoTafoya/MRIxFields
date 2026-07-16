"""Fail-closed adapter from the official MRIxFields JSONL schema to volume records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.manifests import Manifest, audit_manifest
from fieldbridge.official.data_manifest import (
    MRIxFieldsDataRecord,
    MRIxFieldsManifestAuditReport,
    audit_mrixfields_manifest,
    read_manifest_jsonl,
)


@dataclass(frozen=True, slots=True)
class AdaptedMRIxFieldsManifest:
    """A standard manifest plus both audits that authorized the conversion."""

    manifest: Manifest
    official_audit: MRIxFieldsManifestAuditReport
    volume_audit: dict[str, Any]


def adapt_mrixfields_manifest(
    records: Sequence[MRIxFieldsDataRecord],
    *,
    name: str = "mrixfields2026-official",
    strict_paths: bool = False,
) -> AdaptedMRIxFieldsManifest:
    """Convert audited official records without inventing or mutating identities.

    ``sample_id`` is the unique volume identity and therefore becomes ``case_id``.
    ``subject_id`` and ``split_name`` remain distinct split/leakage metadata.
    """

    if not records:
        raise ValueError("Official MRIxFields manifest is empty.")
    official_audit = audit_mrixfields_manifest(records)
    if not official_audit.ok:
        raise ValueError(
            "Official MRIxFields manifest audit failed; diagnostic execution stopped "
            f"before payload loading ({len(official_audit.errors)} error(s))."
        )

    volume_records = tuple(_adapt_record(record) for record in records)
    manifest = Manifest.from_records(
        volume_records,
        name=name,
        metadata={
            "source_schema": "mrixfields2026_jsonl",
            "identity_contract": "case_id_is_official_sample_id",
        },
    )
    volume_audit = audit_manifest(manifest, strict_paths=strict_paths)
    if not bool(volume_audit["ok"]):
        raise ValueError(
            "Adapted MRIxFields volume-manifest audit failed; diagnostic execution "
            "stopped before payload loading."
        )
    return AdaptedMRIxFieldsManifest(
        manifest=manifest,
        official_audit=official_audit,
        volume_audit=volume_audit,
    )


def load_adapted_mrixfields_manifest(
    path: str | Path,
    *,
    strict_paths: bool = False,
) -> AdaptedMRIxFieldsManifest:
    """Read an official JSONL manifest and apply the fail-closed adapter."""

    manifest_path = Path(path)
    return adapt_mrixfields_manifest(
        read_manifest_jsonl(manifest_path),
        name=manifest_path.stem,
        strict_paths=strict_paths,
    )


def _adapt_record(record: MRIxFieldsDataRecord) -> VolumeRecord:
    return VolumeRecord(
        case_id=record.sample_id,
        image_path=record.raw_uri,
        domain=Domain(record.field_value, record.internal_modality),
        subject_id=record.subject_id,
        split=record.split_name,
        metadata={
            "sample_id": record.sample_id,
            "split_name": record.split_name,
            "cohort": record.cohort,
            "is_paired": record.is_paired,
            "domain_id": record.domain_id,
            "official_modality": record.modality,
            "official_field": record.field,
            "relative_path": record.relative_path,
        },
    )
