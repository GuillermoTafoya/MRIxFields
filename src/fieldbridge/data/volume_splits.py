"""Deterministic volume-level split construction before slice expansion."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Literal

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast
from fieldbridge.data.manifests import record_from_mapping

SplitName = Literal["train", "validation", "test"]
_SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


class VolumeSplitError(ValueError):
    """Raised when split construction or auditing fails."""


@dataclass(frozen=True, slots=True)
class PseudoPairManifestValidation:
    ok: bool
    record_count: int
    sequence: str
    target_fields: tuple[float, ...]
    counts_by_field: dict[str, int]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "record_count": self.record_count,
            "sequence": self.sequence,
            "target_fields": list(self.target_fields),
            "counts_by_field": dict(self.counts_by_field),
            "errors": list(self.errors),
        }

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        raise VolumeSplitError("Invalid pseudo-pair manifest: " + "; ".join(self.errors))


@dataclass(frozen=True, slots=True)
class VolumeSplits:
    train: tuple[VolumeRecord, ...]
    validation: tuple[VolumeRecord, ...]
    test: tuple[VolumeRecord, ...]
    sequence: str
    target_fields: tuple[float, ...]
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def records_for(self, split: SplitName) -> tuple[VolumeRecord, ...]:
        return getattr(self, split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "target_fields": list(self.target_fields),
            "seed": self.seed,
            "metadata": dict(self.metadata),
            "splits": {
                "train": [record.to_dict() for record in self.train],
                "validation": [record.to_dict() for record in self.validation],
                "test": [record.to_dict() for record in self.test],
            },
        }


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    ok: bool
    leaked_case_ids: dict[str, list[str]]
    leaked_paths: dict[str, list[str]]
    leaked_subject_ids: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "leaked_case_ids": self.leaked_case_ids,
            "leaked_paths": self.leaked_paths,
            "leaked_subject_ids": self.leaked_subject_ids,
        }

    def raise_for_leakage(self) -> None:
        if self.ok:
            return
        raise VolumeSplitError(
            "Volume split leakage detected: "
            f"case_ids={sorted(self.leaked_case_ids)}, "
            f"paths={sorted(self.leaked_paths)}, "
            f"subject_ids={sorted(self.leaked_subject_ids)}."
        )


def build_volume_splits(
    records: Sequence[VolumeRecord],
    *,
    sequence: str,
    target_fields: Sequence[float],
    train_volumes_per_field: int,
    val_volumes_per_field: int,
    test_volumes_per_field: int,
    seed: int,
) -> VolumeSplits:
    """Build deterministic field-balanced splits from high-field records."""

    if not records:
        raise VolumeSplitError("Cannot build volume splits from an empty record list.")
    sequence_label = Contrast.parse(sequence).value
    fields = tuple(float(field) for field in target_fields)
    validate_pseudo_pair_manifest_records(
        records,
        sequence=sequence_label,
        target_fields=fields,
    ).raise_for_errors()
    if not fields:
        raise VolumeSplitError("target_fields must contain at least one field strength.")
    counts = {
        "train": _positive_count(train_volumes_per_field, "train_volumes_per_field"),
        "validation": _positive_count(val_volumes_per_field, "val_volumes_per_field"),
        "test": _positive_count(test_volumes_per_field, "test_volumes_per_field"),
    }
    required_per_field = sum(counts.values())

    by_field: dict[float, list[VolumeRecord]] = {field: [] for field in fields}
    for record in records:
        if Contrast.parse(record.domain.contrast).value != sequence_label:
            continue
        field = float(record.domain.field_strength_t)
        if field in by_field:
            by_field[field].append(record)

    split_records: dict[str, list[VolumeRecord]] = {name: [] for name in _SPLIT_NAMES}
    for field in fields:
        candidates = sorted(by_field[field], key=_record_sort_key)
        if len(candidates) < required_per_field:
            raise VolumeSplitError(
                f"Insufficient {sequence_label} records for {field:g}T: need "
                f"{required_per_field}, found {len(candidates)}."
            )
        rng = random.Random(_field_seed(seed, field))
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        cursor = 0
        for split_name in _SPLIT_NAMES:
            take = counts[split_name]
            split_records[split_name].extend(shuffled[cursor : cursor + take])
            cursor += take

    splits = VolumeSplits(
        train=tuple(sorted(split_records["train"], key=_record_sort_key)),
        validation=tuple(sorted(split_records["validation"], key=_record_sort_key)),
        test=tuple(sorted(split_records["test"], key=_record_sort_key)),
        sequence=sequence_label,
        target_fields=fields,
        seed=int(seed),
        metadata={
            "train_volumes_per_field": counts["train"],
            "val_volumes_per_field": counts["validation"],
            "test_volumes_per_field": counts["test"],
        },
    )
    audit_volume_splits(splits).raise_for_leakage()
    return splits


def validate_pseudo_pair_manifest_records(
    records: Sequence[VolumeRecord],
    *,
    sequence: str,
    target_fields: Sequence[float],
) -> PseudoPairManifestValidation:
    """Validate the VolumeRecord manifest schema used by pseudo-pair training.

    The pseudo-pair commands consume the existing FieldBridge manifest schema:
    top-level ``records`` with ``case_id``, ``image_path``, ``domain`` containing
    ``field_strength_t`` and ``contrast``, and ``subject_id``. The subject id is
    required here because leakage is audited at subject level before slice expansion.
    """

    sequence_label = Contrast.parse(sequence).value
    fields = tuple(float(field) for field in target_fields)
    errors: list[str] = []
    if not records:
        errors.append("manifest contains no records")
    if not fields:
        errors.append("target_fields is empty")

    case_counts = Counter(record.case_id for record in records)
    path_counts = Counter(str(record.image_path) for record in records)
    counts_by_field: Counter[float] = Counter()
    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        if not str(record.case_id).strip():
            errors.append(f"{prefix} has empty case_id")
        if not str(record.image_path).strip():
            errors.append(f"{prefix} has empty image_path")
        if record.subject_id is None or not str(record.subject_id).strip():
            errors.append(f"{prefix} ({record.case_id}) is missing subject_id")
        if Contrast.parse(record.domain.contrast).value == sequence_label:
            field = float(record.domain.field_strength_t)
            if field in fields:
                counts_by_field[field] += 1

    duplicates = sorted(case_id for case_id, count in case_counts.items() if count > 1)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate case_id values: {duplicates}")
    if duplicate_paths:
        errors.append(f"duplicate image_path values: {duplicate_paths}")

    for field in fields:
        if counts_by_field.get(field, 0) == 0:
            errors.append(f"no {sequence_label} records found for target field {field:g}T")

    return PseudoPairManifestValidation(
        ok=not errors,
        record_count=len(records),
        sequence=sequence_label,
        target_fields=fields,
        counts_by_field={f"{field:g}T": counts_by_field.get(field, 0) for field in fields},
        errors=tuple(errors),
    )


def audit_volume_splits(splits: VolumeSplits) -> LeakageAudit:
    """Audit case/path/subject identities across train/validation/test."""

    case_seen: dict[str, set[str]] = defaultdict(set)
    path_seen: dict[str, set[str]] = defaultdict(set)
    subject_seen: dict[str, set[str]] = defaultdict(set)
    for split_name in _SPLIT_NAMES:
        for record in splits.records_for(split_name):
            case_seen[record.case_id].add(split_name)
            path_seen[str(record.image_path)].add(split_name)
            if record.subject_id:
                subject_seen[record.subject_id].add(split_name)

    leaked_case_ids = _leaked(case_seen)
    leaked_paths = _leaked(path_seen)
    leaked_subject_ids = _leaked(subject_seen)
    return LeakageAudit(
        ok=not leaked_case_ids and not leaked_paths and not leaked_subject_ids,
        leaked_case_ids=leaked_case_ids,
        leaked_paths=leaked_paths,
        leaked_subject_ids=leaked_subject_ids,
    )


def summarize_volume_splits(splits: VolumeSplits, *, slices_per_volume: int | None = None) -> dict[str, Any]:
    """Return volume and derived-slice counts by split and target field."""

    summary: dict[str, Any] = {
        "sequence": splits.sequence,
        "target_fields": list(splits.target_fields),
        "splits": {},
    }
    for split_name in _SPLIT_NAMES:
        records = splits.records_for(split_name)
        counts = Counter(float(record.domain.field_strength_t) for record in records)
        field_summary = {
            f"{field:g}T": {
                "volumes": counts.get(field, 0),
                "slices": None if slices_per_volume is None else counts.get(field, 0) * slices_per_volume,
            }
            for field in splits.target_fields
        }
        total_volumes = len(records)
        split_payload = {
            "volumes": total_volumes,
            "slices": None if slices_per_volume is None else total_volumes * slices_per_volume,
            "by_field": field_summary,
        }
        summary["splits"][split_name] = split_payload
    return summary


def save_volume_splits(splits: VolumeSplits, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def volume_splits_fingerprint(splits: VolumeSplits) -> str:
    payload = json.dumps(splits.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_volume_splits(path: str | Path) -> VolumeSplits:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise VolumeSplitError(f"Volume split JSON must be a mapping: {path}")
    split_payload = payload.get("splits")
    if not isinstance(split_payload, Mapping):
        raise VolumeSplitError("Volume split JSON is missing a 'splits' mapping.")

    def _records(name: str) -> tuple[VolumeRecord, ...]:
        raw = split_payload.get(name)
        if not isinstance(raw, list):
            raise VolumeSplitError(f"Volume split JSON is missing list splits.{name}.")
        return tuple(record_from_mapping(record) for record in raw)

    splits = VolumeSplits(
        train=_records("train"),
        validation=_records("validation"),
        test=_records("test"),
        sequence=str(payload["sequence"]),
        target_fields=tuple(float(field) for field in payload.get("target_fields", [])),
        seed=int(payload.get("seed", 0)),
        metadata=dict(payload.get("metadata", {})),
    )
    audit_volume_splits(splits).raise_for_leakage()
    return splits


def _positive_count(value: int, name: str) -> int:
    count = int(value)
    if count < 0:
        raise VolumeSplitError(f"{name} must be non-negative.")
    return count


def _record_sort_key(record: VolumeRecord) -> tuple[str, str]:
    return (f"{record.domain.field_strength_t:g}T", record.case_id)


def _field_seed(seed: int, field: float) -> int:
    return int(seed) * 1009 + int(round(float(field) * 1000))


def _leaked(seen: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    return {
        key: sorted(set(split_names))
        for key, split_names in seen.items()
        if len(set(split_names)) > 1
    }
