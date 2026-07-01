"""MRIxFields2026 submission tree, zip, and manifest preflight utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from fieldbridge.official.mrixfields2026 import (
    FIELDS,
    OFFICIAL_MODALITIES,
    build_prediction_filename,
    expected_prediction_file_count,
    expected_segmentation_file_count,
    expected_subject_ids_for_pair,
    get_task_pairs,
    normalize_field_label,
    normalize_modality,
    pair_name,
    parse_mrixfields_filename,
    parse_pair_name,
    requires_segmentation,
)


@dataclass(frozen=True, slots=True)
class ExpectedSubmissionEntry:
    task: str
    modality: str
    pair: str
    source_field: str
    target_field: str
    subject_id: str
    kind: str
    relative_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubmissionValidationReport:
    task: str
    root: str
    ok: bool
    expected_pred_count: int
    expected_seg_count: int
    found_pred_count: int
    found_seg_count: int
    missing: list[str]
    extra: list[str]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_submission_entries(
    task: str,
    include_segmentation: bool | None = None,
) -> list[ExpectedSubmissionEntry]:
    """Return official relative paths expected for a task submission."""

    canonical_task = _normalize_task(task)
    include_seg = requires_segmentation(canonical_task) if include_segmentation is None else include_segmentation
    include_seg = include_seg and requires_segmentation(canonical_task)
    entries: list[ExpectedSubmissionEntry] = []

    for source_field, target_field in get_task_pairs(canonical_task):
        current_pair = pair_name(source_field, target_field)
        for modality in OFFICIAL_MODALITIES:
            for subject_id in expected_subject_ids_for_pair(canonical_task, source_field, target_field):
                entries.append(
                    _entry(
                        canonical_task,
                        modality,
                        current_pair,
                        source_field,
                        target_field,
                        subject_id,
                        "pred",
                    )
                )
                if include_seg:
                    entries.append(
                        _entry(
                            canonical_task,
                            modality,
                            current_pair,
                            source_field,
                            target_field,
                            subject_id,
                            "seg",
                        )
                    )
    return entries


def validate_submission_dir(
    root: Path,
    task: str,
    *,
    strict_segmentation: bool = True,
    allow_extra_files: bool = False,
) -> SubmissionValidationReport:
    """Validate an on-disk submission tree without reading file payloads."""

    canonical_task = _normalize_task(task)
    root_path = Path(root)
    paths, errors, extra, seg_dirs = _collect_directory_entries(
        root_path,
        canonical_task,
        allow_extra_files=allow_extra_files,
    )
    return _validate_relative_paths(
        paths,
        canonical_task,
        root=str(root_path),
        strict_segmentation=strict_segmentation,
        allow_extra_files=allow_extra_files,
        initial_errors=errors,
        initial_extra=extra,
        seg_directories=seg_dirs,
    )


def validate_submission_zip(
    zip_path: Path,
    task: str,
    *,
    strict_segmentation: bool = True,
) -> SubmissionValidationReport:
    """Validate a submission zip by inspecting archive member names only."""

    canonical_task = _normalize_task(task)
    archive_path = Path(zip_path)
    paths: list[str] = []
    errors: list[str] = []
    extra: list[str] = []
    seg_dirs: list[str] = []

    if not archive_path.exists():
        errors.append(f"Zip file does not exist: {archive_path}.")
    else:
        try:
            with ZipFile(archive_path, "r") as archive:
                names = [_normalize_relative_name(info.filename) for info in archive.infolist()]
        except OSError as exc:
            names = []
            errors.append(f"Could not open zip file {archive_path}: {exc}.")

        root_parts = {name.split("/", 1)[0] for name in names if name}
        if canonical_task not in root_parts:
            errors.append(f"Zip archive must contain {canonical_task}/ at archive root.")
        modality_roots = sorted(root_parts.intersection(OFFICIAL_MODALITIES))
        if modality_roots:
            errors.append(
                "Zip archive has modality folders at root; expected archive names to start "
                f"with {canonical_task}/. Found: {modality_roots}."
            )
        task_roots = sorted(root for root in root_parts if root.startswith("task") and root != canonical_task)
        if task_roots:
            errors.append(f"Zip archive contains wrong or extra task roots: {task_roots}.")
            extra.extend(task_roots)

        for name in names:
            if not name:
                continue
            if name.endswith("/"):
                parts = PurePosixPath(name.rstrip("/")).parts
                if canonical_task in parts and "seg" in parts:
                    seg_dirs.append(name.rstrip("/"))
                continue
            if name.startswith(f"{canonical_task}/"):
                paths.append(name)
            else:
                extra.append(name)

    return _validate_relative_paths(
        paths,
        canonical_task,
        root=str(archive_path),
        strict_segmentation=strict_segmentation,
        allow_extra_files=False,
        initial_errors=errors,
        initial_extra=extra,
        seg_directories=seg_dirs,
    )


def build_submission_zip(
    submission_root: Path,
    task: str,
    out_zip: Path,
    *,
    validate_first: bool = True,
    strict_segmentation: bool = True,
) -> Path:
    """Create a submission zip whose archive members start with taskN/."""

    canonical_task = _normalize_task(task)
    root_path = Path(submission_root)
    report = validate_submission_dir(
        root_path,
        canonical_task,
        strict_segmentation=strict_segmentation,
    )
    if validate_first and not report.ok:
        raise ValueError(f"Submission tree failed validation: {report.to_dict()}")

    task_root = _task_root_for(root_path, canonical_task)
    archive_path = Path(out_zip)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted(path for path in task_root.rglob("*") if path.is_file()):
            if file_path.resolve() == archive_path.resolve():
                continue
            archive_name = PurePosixPath(canonical_task) / file_path.relative_to(task_root).as_posix()
            archive.write(file_path, archive_name.as_posix())

    return archive_path


def audit_prediction_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    task: str,
) -> SubmissionValidationReport:
    """Audit manifest-style prediction rows without requiring files to exist."""

    canonical_task = _normalize_task(task)
    paths: list[str] = []
    errors: list[str] = []

    for index, row in enumerate(rows):
        path, row_errors = _row_to_relative_path(row, canonical_task, index)
        if path is not None:
            paths.append(path)
        errors.extend(row_errors)

    return _validate_relative_paths(
        paths,
        canonical_task,
        root="<manifest rows>",
        strict_segmentation=True,
        allow_extra_files=False,
        initial_errors=errors,
    )


def _entry(
    task: str,
    modality: str,
    current_pair: str,
    source_field: str,
    target_field: str,
    subject_id: str,
    kind: str,
) -> ExpectedSubmissionEntry:
    segmentation = kind == "seg"
    filename = build_prediction_filename(modality, target_field, subject_id, segmentation=segmentation)
    relative_path = f"{task}/{modality}/{current_pair}/{kind}/{filename}"
    return ExpectedSubmissionEntry(
        task=task,
        modality=modality,
        pair=current_pair,
        source_field=source_field,
        target_field=target_field,
        subject_id=subject_id,
        kind=kind,
        relative_path=relative_path,
    )


def _validate_relative_paths(
    paths: Iterable[str],
    task: str,
    *,
    root: str,
    strict_segmentation: bool,
    allow_extra_files: bool,
    initial_errors: Sequence[str] | None = None,
    initial_extra: Sequence[str] | None = None,
    seg_directories: Sequence[str] | None = None,
) -> SubmissionValidationReport:
    canonical_task = _normalize_task(task)
    present = {_normalize_relative_name(path) for path in paths if _normalize_relative_name(path)}
    expected_pred = {
        entry.relative_path for entry in expected_submission_entries(canonical_task, include_segmentation=False)
    }
    expected_seg = {
        entry.relative_path for entry in expected_submission_entries(canonical_task, include_segmentation=True)
        if entry.kind == "seg"
    }
    expected_all = set(expected_pred)
    errors = list(initial_errors or [])
    warnings: list[str] = []
    extra = sorted(set(initial_extra or []))

    seg_present = _segmentation_paths(present).union(_segmentation_dirs(seg_directories or ()))
    if requires_segmentation(canonical_task):
        if strict_segmentation:
            expected_all.update(expected_seg)
        elif not seg_present:
            warnings.append(
                f"{canonical_task} has no segmentation files; Dice/Volume outputs will be missing or null."
            )
        else:
            expected_all.update(expected_seg)
            warnings.append(
                f"{canonical_task} has partial/optional segmentation enabled; all expected seg files are required."
            )
    elif seg_present:
        errors.append(f"{canonical_task} expects prediction files only; segmentation paths are not allowed.")

    valid_expected_pred: set[str] = set()
    valid_expected_seg: set[str] = set()
    for relative_path in sorted(present):
        if allow_extra_files and not _looks_like_submission_payload_path(relative_path):
            continue
        inspection = _inspect_submission_path(relative_path, canonical_task)
        errors.extend(inspection.errors)
        if relative_path not in expected_all and not allow_extra_files:
            extra.append(relative_path)
        if inspection.is_valid and relative_path in expected_pred:
            valid_expected_pred.add(relative_path)
        if inspection.is_valid and relative_path in expected_seg:
            valid_expected_seg.add(relative_path)

    missing = sorted(expected_all - present)
    extra = sorted(set(extra))
    ok = not errors and not missing and not extra

    return SubmissionValidationReport(
        task=canonical_task,
        root=root,
        ok=ok,
        expected_pred_count=len(expected_pred),
        expected_seg_count=expected_segmentation_file_count(canonical_task),
        found_pred_count=len(valid_expected_pred),
        found_seg_count=len(valid_expected_seg),
        missing=missing,
        extra=extra,
        errors=errors,
        warnings=warnings,
    )


@dataclass(frozen=True, slots=True)
class _PathInspection:
    is_valid: bool
    kind: str | None
    errors: list[str]


def _inspect_submission_path(relative_path: str, task: str) -> _PathInspection:
    errors: list[str] = []
    parts = PurePosixPath(relative_path).parts
    if len(parts) != 5:
        return _PathInspection(False, None, [f"Malformed submission path: {relative_path}."])

    task_part, modality_part, pair_part, kind, filename = parts
    if task_part != task:
        errors.append(f"Path {relative_path} is under {task_part!r}, expected {task!r}.")
    if kind not in {"pred", "seg"}:
        errors.append(f"Path {relative_path} uses invalid kind {kind!r}; expected 'pred' or 'seg'.")

    try:
        modality = normalize_modality(modality_part)
    except ValueError as exc:
        modality = ""
        errors.append(f"Path {relative_path} has invalid modality folder: {exc}")

    try:
        source_field, target_field = parse_pair_name(pair_part)
    except ValueError as exc:
        source_field, target_field = "", ""
        errors.append(f"Path {relative_path} has invalid pair folder: {exc}")

    try:
        parsed = parse_mrixfields_filename(filename)
    except ValueError as exc:
        parsed = None
        errors.append(f"Path {relative_path} has malformed filename: {exc}")

    if source_field and target_field and (source_field, target_field) not in get_task_pairs(task):
        errors.append(f"Pair {pair_part!r} is not allowed for {task}.")

    if parsed is not None:
        if parsed.prefix != "P":
            errors.append(f"Submission filename must use prefix 'P', got {parsed.prefix!r}: {relative_path}.")
        if modality and parsed.modality != modality:
            errors.append(
                f"Modality folder {modality_part!r} does not match filename modality "
                f"{parsed.modality!r}: {relative_path}."
            )
        if target_field and parsed.field != target_field:
            errors.append(
                f"Filename target field {parsed.field!r} does not match pair target "
                f"{target_field!r}: {relative_path}."
            )
        if kind == "pred" and parsed.is_segmentation:
            errors.append(f"Segmentation filename is placed under pred/: {relative_path}.")
        if kind == "seg" and not parsed.is_segmentation:
            errors.append(f"Prediction filename is placed under seg/: {relative_path}.")
        if kind == "seg" and not requires_segmentation(task):
            errors.append(f"{task} does not allow segmentation files: {relative_path}.")
        if source_field:
            try:
                expected_ids = expected_subject_ids_for_pair(task, source_field, target_field)
            except ValueError:
                expected_ids = ()
            if expected_ids and parsed.subject_id not in expected_ids:
                errors.append(
                    f"Subject ID {parsed.subject_id!r} is not expected for source field "
                    f"{source_field!r}: {relative_path}."
                )

    return _PathInspection(is_valid=not errors, kind=kind if kind in {"pred", "seg"} else None, errors=errors)


def _looks_like_submission_payload_path(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    return len(parts) == 5 and parts[3] in {"pred", "seg"}


def _collect_directory_entries(
    root: Path,
    task: str,
    *,
    allow_extra_files: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    paths: list[str] = []
    errors: list[str] = []
    extra: list[str] = []
    seg_dirs: list[str] = []

    if not root.exists():
        errors.append(f"Submission root does not exist: {root}.")
        return paths, errors, extra, seg_dirs
    if not root.is_dir():
        errors.append(f"Submission root must be a directory: {root}.")
        return paths, errors, extra, seg_dirs

    task_root = _task_root_for(root, task)
    if not task_root.exists():
        errors.append(f"Submission root must contain {task}/ or be the {task}/ directory itself.")
        root_names = {child.name for child in root.iterdir()}
        modality_roots = sorted(root_names.intersection(OFFICIAL_MODALITIES))
        if modality_roots:
            errors.append(
                f"Found modality folders at root {modality_roots}; expected them under {task}/."
            )
        return paths, errors, extra, seg_dirs

    if root.name != task:
        for child in root.iterdir():
            if child.name == task:
                continue
            if child.is_dir() and child.name in OFFICIAL_MODALITIES:
                errors.append(f"Found modality folder at root instead of under {task}/: {child.name}.")
            if child.is_dir() and child.name.startswith("task"):
                errors.append(f"Found wrong or extra task root: {child.name}.")
            if not allow_extra_files:
                extra.append(child.name)

    for path in task_root.rglob("*"):
        rel = (PurePosixPath(task) / path.relative_to(task_root).as_posix()).as_posix()
        if path.is_dir():
            if path.name == "seg":
                seg_dirs.append(rel)
            continue
        paths.append(rel)

    return paths, errors, extra, seg_dirs


def _task_root_for(root: Path, task: str) -> Path:
    return root if root.name == task else root / task


def _segmentation_paths(paths: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).parts
        if "seg" in parts or path.endswith("_seg.nii.gz"):
            result.add(path)
    return result


def _segmentation_dirs(paths: Iterable[str]) -> set[str]:
    return {path for path in paths if "seg" in PurePosixPath(path).parts}


def _row_to_relative_path(
    row: Mapping[str, Any],
    task: str,
    index: int,
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    raw_path = row.get("relative_path", row.get("path"))
    if raw_path is None:
        try:
            modality = normalize_modality(str(row["modality"]))
            source_field, target_field = _row_pair(row)
            subject_id = str(row["subject_id"])
            kind = str(row.get("kind", "pred")).strip().lower()
            if kind not in {"pred", "seg"}:
                raise ValueError(f"Invalid kind {kind!r}; expected 'pred' or 'seg'.")
            filename = build_prediction_filename(
                modality,
                target_field,
                subject_id,
                segmentation=kind == "seg",
            )
            path = f"{task}/{modality}/{pair_name(source_field, target_field)}/{kind}/{filename}"
        except (KeyError, ValueError) as exc:
            return None, [f"Row {index} cannot be converted to an official path: {exc}"]
    else:
        path = _normalize_relative_name(str(raw_path))
        if not path.startswith(f"{task}/"):
            path = f"{task}/{path}" if not path.startswith("task") else path

    errors.extend(_validate_row_metadata_against_path(row, path, index))
    return path, errors


def _row_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    if row.get("pair") is not None:
        return parse_pair_name(str(row["pair"]))
    return normalize_field_label(str(row["source_field"])), normalize_field_label(str(row["target_field"]))


def _validate_row_metadata_against_path(
    row: Mapping[str, Any],
    path: str,
    index: int,
) -> list[str]:
    errors: list[str] = []
    parts = PurePosixPath(path).parts
    if len(parts) != 5:
        return errors
    _task, modality_part, pair_part, kind, filename = parts
    try:
        source_field, target_field = parse_pair_name(pair_part)
        parsed = parse_mrixfields_filename(filename)
    except ValueError:
        return errors

    if row.get("modality") is not None:
        try:
            row_modality = normalize_modality(str(row["modality"]))
        except ValueError as exc:
            errors.append(f"Row {index} has invalid modality metadata: {exc}")
        else:
            if row_modality != modality_part:
                errors.append(
                    f"Row {index} modality does not match path: {row['modality']!r} "
                    f"vs {modality_part!r}."
                )
    if row.get("pair") is not None:
        try:
            row_pair = parse_pair_name(str(row["pair"]))
        except ValueError as exc:
            errors.append(f"Row {index} has invalid pair metadata: {exc}")
        else:
            if row_pair != (source_field, target_field):
                errors.append(f"Row {index} pair does not match path: {row['pair']!r} vs {pair_part!r}.")
    if row.get("source_field") is not None:
        try:
            row_source = normalize_field_label(str(row["source_field"]))
        except ValueError as exc:
            errors.append(f"Row {index} has invalid source_field metadata: {exc}")
        else:
            if row_source != source_field:
                errors.append(
                    f"Row {index} source_field does not match path: "
                    f"{row['source_field']!r} vs {source_field!r}."
                )
    if row.get("target_field") is not None:
        try:
            row_target = normalize_field_label(str(row["target_field"]))
        except ValueError as exc:
            errors.append(f"Row {index} has invalid target_field metadata: {exc}")
        else:
            if row_target != target_field:
                errors.append(
                    f"Row {index} target_field does not match path: "
                    f"{row['target_field']!r} vs {target_field!r}."
                )
    if row.get("subject_id") is not None and str(row["subject_id"]) != parsed.subject_id:
        errors.append(
            f"Row {index} subject_id does not match filename: {row['subject_id']!r} "
            f"vs {parsed.subject_id!r}."
        )
    if row.get("kind") is not None and str(row["kind"]).strip().lower() != kind:
        errors.append(f"Row {index} kind does not match path: {row['kind']!r} vs {kind!r}.")
    return errors


def _normalize_relative_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def _normalize_task(task: str) -> str:
    key = str(task).strip().lower().replace(" ", "")
    if key in {"1", "task1"}:
        return "task1"
    if key in {"2", "task2"}:
        return "task2"
    if key in {"3", "task3"}:
        return "task3"
    raise ValueError("Task must be one of 'task1', 'task2', or 'task3'.")
