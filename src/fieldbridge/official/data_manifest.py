"""Metadata-first MRIxFields2026 raw data manifest utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fieldbridge.official.mrixfields2026 import (
    FIELD_VALUES,
    FIELDS,
    FULL_SHAPE,
    OFFICIAL_MODALITIES,
    internal_modality_from_official,
    normalize_field_label,
    normalize_modality,
    parse_mrixfields_filename,
)
from fieldbridge.official.validation import (
    validate_dtype,
    validate_intensity_range,
    validate_shape,
)

OFFICIAL_SPLITS: tuple[str, ...] = (
    "Training_retrospective",
    "Training_prospective",
    "Validating_prospective",
    "Testing_prospective",
)


@dataclass(frozen=True, slots=True)
class MRIxFieldsDomainSpec:
    domain_id: int
    modality: str
    internal_modality: str
    field: str
    field_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PayloadMetadata:
    shape: tuple[int, int, int]
    dtype: str
    intensity_min: float
    intensity_max: float
    affine_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MRIxFieldsDataRecord:
    sample_id: str
    split_name: str
    cohort: Literal["retrospective", "prospective"]
    is_paired: bool
    prefix: Literal["R", "P"]
    modality: str
    internal_modality: str
    field: str
    field_value: float
    subject_id: str
    domain_id: int
    relative_path: str
    raw_uri: str
    filename: str
    shape: tuple[int, int, int] | None = None
    dtype: str | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    affine_hash: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.shape is not None:
            data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MRIxFieldsDataRecord":
        shape_value = data.get("shape")
        shape = None if shape_value is None else tuple(int(dim) for dim in shape_value)
        return cls(
            sample_id=str(data["sample_id"]),
            split_name=str(data["split_name"]),
            cohort=_cohort_for_split(str(data["split_name"])),
            is_paired=bool(data["is_paired"]),
            prefix=_validate_prefix(str(data["prefix"])),
            modality=normalize_modality(str(data["modality"])),
            internal_modality=str(data["internal_modality"]),
            field=normalize_field_label(str(data["field"])),
            field_value=float(data["field_value"]),
            subject_id=str(data["subject_id"]),
            domain_id=int(data["domain_id"]),
            relative_path=_normalize_relative_path(str(data["relative_path"])),
            raw_uri=str(data["raw_uri"]),
            filename=str(data["filename"]),
            shape=shape,  # type: ignore[arg-type]
            dtype=None if data.get("dtype") is None else str(data["dtype"]),
            intensity_min=None if data.get("intensity_min") is None else float(data["intensity_min"]),
            intensity_max=None if data.get("intensity_max") is None else float(data["intensity_max"]),
            affine_hash=None if data.get("affine_hash") is None else str(data["affine_hash"]),
            sha256=None if data.get("sha256") is None else str(data["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class MRIxFieldsManifestAuditReport:
    ok: bool
    total_records: int
    counts_by_split: dict[str, int]
    counts_by_modality: dict[str, int]
    counts_by_field: dict[str, int]
    counts_by_domain_id: dict[int, int]
    counts_by_split_modality_field: dict[str, int]
    duplicate_raw_uris: list[str]
    duplicate_sample_ids: list[str]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def all_domain_specs() -> tuple[MRIxFieldsDomainSpec, ...]:
    specs: list[MRIxFieldsDomainSpec] = []
    domain_id = 0
    for modality in OFFICIAL_MODALITIES:
        for field in FIELDS:
            specs.append(
                MRIxFieldsDomainSpec(
                    domain_id=domain_id,
                    modality=modality,
                    internal_modality=internal_modality_from_official(modality),
                    field=field,
                    field_value=FIELD_VALUES[field],
                )
            )
            domain_id += 1
    return tuple(specs)


def domain_id_for(modality: str, field: str) -> int:
    official_modality = normalize_modality(modality)
    official_field = normalize_field_label(field)
    return OFFICIAL_MODALITIES.index(official_modality) * len(FIELDS) + FIELDS.index(official_field)


def domain_label_for(domain_id: int) -> tuple[str, str]:
    value = int(domain_id)
    specs = all_domain_specs()
    if value < 0 or value >= len(specs):
        raise ValueError(f"Domain ID must be in [0, {len(specs) - 1}], got {domain_id}.")
    spec = specs[value]
    return spec.modality, spec.field


def parse_mrixfields_data_path(
    path: str | Path,
    data_root: str | Path | None = None,
    *,
    allow_unknown_split: bool = False,
    include_payload_metadata: bool = False,
) -> MRIxFieldsDataRecord:
    relative_path = _relative_path(path, data_root)
    parts = PurePosixPath(relative_path).parts
    if len(parts) != 4:
        raise ValueError(
            "MRIxFields data paths must use '<split>/<modality>/<field>/<filename>.nii.gz', "
            f"got {relative_path!r}."
        )

    split_name, modality_folder, field_folder, filename = parts
    if split_name not in OFFICIAL_SPLITS and not allow_unknown_split:
        raise ValueError(f"Unknown MRIxFields split {split_name!r}. Expected one of {OFFICIAL_SPLITS}.")

    folder_modality = normalize_modality(modality_folder)
    folder_field = normalize_field_label(field_folder)
    parsed = parse_mrixfields_filename(filename)
    if parsed.is_segmentation:
        raise ValueError(f"Raw MRIxFields data manifests do not accept segmentation files: {relative_path}.")
    if parsed.modality != folder_modality:
        raise ValueError(
            f"Folder modality {folder_modality!r} does not match filename modality "
            f"{parsed.modality!r}: {relative_path}."
        )
    if parsed.field != folder_field:
        raise ValueError(
            f"Folder field {folder_field!r} does not match filename field "
            f"{parsed.field!r}: {relative_path}."
        )
    _validate_split_prefix(split_name, parsed.prefix, relative_path)

    payload = None
    raw_path = _raw_path(path, data_root)
    if include_payload_metadata:
        payload = inspect_nifti_payload_metadata(raw_path)

    return MRIxFieldsDataRecord(
        sample_id=_sample_id(split_name, parsed.prefix, parsed.modality, parsed.field, parsed.subject_id),
        split_name=split_name,
        cohort=_cohort_for_split(split_name),
        is_paired=split_name != "Training_retrospective",
        prefix=parsed.prefix,
        modality=parsed.modality,
        internal_modality=internal_modality_from_official(parsed.modality),
        field=parsed.field,
        field_value=FIELD_VALUES[parsed.field],
        subject_id=parsed.subject_id,
        domain_id=domain_id_for(parsed.modality, parsed.field),
        relative_path=relative_path,
        raw_uri=str(raw_path),
        filename=filename,
        shape=None if payload is None else payload.shape,
        dtype=None if payload is None else payload.dtype,
        intensity_min=None if payload is None else payload.intensity_min,
        intensity_max=None if payload is None else payload.intensity_max,
        affine_hash=None if payload is None else payload.affine_hash,
    )


def build_mrixfields_manifest_from_paths(
    paths: Iterable[str | Path],
    data_root: str | Path | None = None,
    *,
    include_payload_metadata: bool = False,
) -> list[MRIxFieldsDataRecord]:
    records = [
        parse_mrixfields_data_path(path, data_root, include_payload_metadata=include_payload_metadata)
        for path in paths
    ]
    return sorted(records, key=_record_sort_key)


def write_manifest_jsonl(records: Sequence[MRIxFieldsDataRecord], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def read_manifest_jsonl(path: str | Path) -> list[MRIxFieldsDataRecord]:
    records: list[MRIxFieldsDataRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
            records.append(MRIxFieldsDataRecord.from_mapping(payload))
    return records


def scan_mrixfields_data_root(
    data_root: str | Path,
    *,
    splits: Sequence[str] | None = None,
    include_payload_metadata: bool = False,
) -> list[MRIxFieldsDataRecord]:
    root = Path(data_root)
    if not root.exists():
        raise ValueError(f"MRIxFields data root does not exist: {root}.")
    if not root.is_dir():
        raise ValueError(f"MRIxFields data root must be a directory: {root}.")

    selected_splits = set(splits or OFFICIAL_SPLITS)
    for split in selected_splits:
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"Unknown MRIxFields split {split!r}. Expected one of {OFFICIAL_SPLITS}.")

    paths = [
        path
        for path in root.rglob("*.nii.gz")
        if (
            path.is_file()
            and not _has_hidden_part(path.relative_to(root))
            and path.relative_to(root).parts[0] in selected_splits
        )
    ]
    return build_mrixfields_manifest_from_paths(paths, root, include_payload_metadata=include_payload_metadata)


def inspect_nifti_payload_metadata(path: Path) -> PayloadMetadata:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "NIfTI payload inspection requires nibabel. Install optional NIfTI dependencies "
            "or run without --inspect-payload."
        ) from exc

    image = nib.load(str(path))
    data = image.get_fdata(dtype="float32")
    affine_hash = hashlib.sha256(image.affine.astype("float64").tobytes()).hexdigest()
    return PayloadMetadata(
        shape=tuple(int(dim) for dim in image.shape[:3]),  # type: ignore[arg-type]
        dtype=str(image.get_data_dtype()),
        intensity_min=float(data.min()),
        intensity_max=float(data.max()),
        affine_hash=affine_hash,
    )


def audit_mrixfields_manifest(records: Sequence[MRIxFieldsDataRecord]) -> MRIxFieldsManifestAuditReport:
    errors: list[str] = []
    warnings: list[str] = []
    raw_uri_counts = Counter(record.raw_uri for record in records)
    sample_id_counts = Counter(record.sample_id for record in records)
    duplicate_raw_uris = sorted(uri for uri, count in raw_uri_counts.items() if count > 1)
    duplicate_sample_ids = sorted(sample_id for sample_id, count in sample_id_counts.items() if count > 1)

    for uri in duplicate_raw_uris:
        errors.append(f"Duplicate raw_uri: {uri}.")
    for sample_id in duplicate_sample_ids:
        errors.append(f"Duplicate sample_id: {sample_id}.")

    prospective_subject_fields: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in records:
        errors.extend(_audit_record(record))
        if record.is_paired:
            prospective_subject_fields[(record.split_name, record.subject_id, record.modality)].add(record.field)

    for (split_name, subject_id, modality), fields in sorted(prospective_subject_fields.items()):
        if len(fields) == 1:
            warnings.append(
                f"Prospective subject {subject_id} in {split_name}/{modality} has only one observed field."
            )

    return MRIxFieldsManifestAuditReport(
        ok=not errors,
        total_records=len(records),
        counts_by_split=dict(sorted(Counter(record.split_name for record in records).items())),
        counts_by_modality=dict(sorted(Counter(record.modality for record in records).items())),
        counts_by_field=dict(sorted(Counter(record.field for record in records).items())),
        counts_by_domain_id=dict(sorted(Counter(record.domain_id for record in records).items())),
        counts_by_split_modality_field=dict(
            sorted(Counter(f"{record.split_name}|{record.modality}|{record.field}" for record in records).items())
        ),
        duplicate_raw_uris=duplicate_raw_uris,
        duplicate_sample_ids=duplicate_sample_ids,
        errors=errors,
        warnings=warnings,
    )


def _audit_record(record: MRIxFieldsDataRecord) -> list[str]:
    errors: list[str] = []
    if record.split_name not in OFFICIAL_SPLITS:
        errors.append(f"{record.sample_id}: unknown split {record.split_name!r}.")
    try:
        modality = normalize_modality(record.modality)
    except ValueError as exc:
        errors.append(f"{record.sample_id}: {exc}")
        modality = ""
    try:
        field = normalize_field_label(record.field)
    except ValueError as exc:
        errors.append(f"{record.sample_id}: {exc}")
        field = ""
    if modality and field:
        expected_domain_id = domain_id_for(modality, field)
        if record.domain_id != expected_domain_id:
            errors.append(
                f"{record.sample_id}: domain_id {record.domain_id} does not match "
                f"{modality}/{field} ({expected_domain_id})."
            )
    try:
        domain_label_for(record.domain_id)
    except ValueError as exc:
        errors.append(f"{record.sample_id}: {exc}")
    if record.shape is not None:
        errors.extend(f"{record.sample_id}: {error}" for error in validate_shape(record.shape, FULL_SHAPE))
    if record.dtype is not None:
        errors.extend(f"{record.sample_id}: {error}" for error in validate_dtype(record.dtype))
    if record.intensity_min is not None and record.intensity_max is not None:
        errors.extend(
            f"{record.sample_id}: {error}"
            for error in validate_intensity_range(record.intensity_min, record.intensity_max)
        )
    return errors


def _record_sort_key(record: MRIxFieldsDataRecord) -> tuple[int, int, int, str, str]:
    return (
        OFFICIAL_SPLITS.index(record.split_name)
        if record.split_name in OFFICIAL_SPLITS
        else len(OFFICIAL_SPLITS),
        OFFICIAL_MODALITIES.index(record.modality),
        FIELDS.index(record.field),
        record.subject_id,
        record.filename,
    )


def _relative_path(path: str | Path, data_root: str | Path | None) -> str:
    path_obj = Path(path)
    if data_root is not None:
        root = Path(data_root)
        candidate = path_obj if path_obj.is_absolute() else root / path_obj
        try:
            resolved_relative = candidate.resolve().relative_to(root.resolve()).as_posix()
            return _normalize_relative_path(resolved_relative)
        except ValueError as exc:
            raise ValueError(f"Path {path_obj} is not under MRIxFields data root {root}.") from exc

    parts = path_obj.parts
    split_indexes = [index for index, part in enumerate(parts) if part in OFFICIAL_SPLITS]
    if split_indexes:
        return PurePosixPath(*parts[split_indexes[0] :]).as_posix()
    return _normalize_relative_path(path_obj.as_posix())


def _raw_path(path: str | Path, data_root: str | Path | None) -> Path:
    path_obj = Path(path)
    if data_root is not None and not path_obj.is_absolute():
        return Path(data_root) / path_obj
    return path_obj


def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _validate_split_prefix(split_name: str, prefix: str, relative_path: str) -> None:
    if split_name == "Training_retrospective" and prefix != "R":
        raise ValueError(f"Training_retrospective expects R-prefixed files: {relative_path}.")
    if split_name != "Training_retrospective" and prefix != "P":
        raise ValueError(f"{split_name} expects P-prefixed files: {relative_path}.")


def _cohort_for_split(split_name: str) -> Literal["retrospective", "prospective"]:
    return "retrospective" if split_name == "Training_retrospective" else "prospective"


def _sample_id(split_name: str, prefix: str, modality: str, field: str, subject_id: str) -> str:
    return f"{split_name}:{prefix}:{modality}:{field}:{subject_id}"


def _validate_prefix(prefix: str) -> Literal["R", "P"]:
    if prefix not in {"R", "P"}:
        raise ValueError(f"Prefix must be 'R' or 'P', got {prefix!r}.")
    return prefix  # type: ignore[return-value]


def _has_hidden_part(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)
