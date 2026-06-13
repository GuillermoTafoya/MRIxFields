"""Official MRIxFields2026 constants and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FIELDS: tuple[str, ...] = ("0.1T", "1.5T", "3T", "5T", "7T")
FIELD_VALUES: dict[str, float] = {"0.1T": 0.1, "1.5T": 1.5, "3T": 3.0, "5T": 5.0, "7T": 7.0}

OFFICIAL_MODALITIES: tuple[str, ...] = ("T1W", "T2W", "T2FLAIR")
_OFFICIAL_TO_INTERNAL_MODALITY: dict[str, str] = {
    "T1W": "T1w",
    "T2W": "T2w",
    "T2FLAIR": "T2-FLAIR",
}

FULL_SHAPE: tuple[int, int, int] = (364, 436, 364)
SUBMISSION_Z_CLIP: tuple[int, int] = (150, 180)
SUBMISSION_SHAPE: tuple[int, int, int] = (364, 436, 30)
INTENSITY_RANGE: tuple[float, float] = (0.0, 1.0)

# Baseline preprocessing axial range used for training examples; this is not the
# official submission z-slab.
TRAIN_SLICE_RANGE: tuple[int, int] = (72, 292)

TASK1_PAIRS: tuple[tuple[str, str], ...] = (
    ("0.1T", "7T"),
    ("1.5T", "7T"),
    ("3T", "7T"),
    ("5T", "7T"),
)
TASK2_PAIRS: tuple[tuple[str, str], ...] = (
    ("0.1T", "1.5T"),
    ("0.1T", "3T"),
    ("0.1T", "5T"),
    ("0.1T", "7T"),
)
TASK3_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (source, target) for source in FIELDS for target in FIELDS if source != target
)

VALIDATION_RELEASED_IDS: dict[str, tuple[str, ...]] = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}

_TASK_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "task1": TASK1_PAIRS,
    "task2": TASK2_PAIRS,
    "task3": TASK3_PAIRS,
}

_FIELD_ALIASES: dict[str, str] = {
    "0.1T": "0.1T",
    "1.5T": "1.5T",
    "3T": "3T",
    "3.0T": "3T",
    "5T": "5T",
    "5.0T": "5T",
    "7T": "7T",
    "7.0T": "7T",
}


@dataclass(frozen=True, slots=True)
class MRIxFieldsFilename:
    """Parsed official MRIxFields filename metadata."""

    prefix: Literal["R", "P"]
    modality: str
    field: str
    subject_id: str
    is_segmentation: bool


def normalize_modality(modality: str) -> str:
    """Normalize supported modality aliases to official MRIxFields labels."""

    key = str(modality).strip().replace("-", "").replace("_", "").upper()
    aliases = {
        "T1W": "T1W",
        "T2W": "T2W",
        "T2FLAIR": "T2FLAIR",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported modality {modality!r}. Expected one of {OFFICIAL_MODALITIES} "
            "or known aliases."
        ) from exc


def internal_modality_from_official(modality: str) -> str:
    """Map an official modality or alias to the repo's existing internal label."""

    official = normalize_modality(modality)
    return _OFFICIAL_TO_INTERNAL_MODALITY[official]


def normalize_field_label(field: str) -> str:
    """Normalize a field label to the official MRIxFields spelling."""

    key = str(field).strip().upper()
    try:
        return _FIELD_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported field {field!r}. Expected one of {FIELDS}.") from exc


def get_task_pairs(task: str) -> tuple[tuple[str, str], ...]:
    return _TASK_PAIRS[_normalize_task(task)]


def requires_segmentation(task: str) -> bool:
    return _normalize_task(task) in {"task1", "task2"}


def expected_modalities(task: str) -> tuple[str, ...]:
    _normalize_task(task)
    return OFFICIAL_MODALITIES


def is_allowed_pair(task: str, source_field: str, target_field: str) -> bool:
    pair = (normalize_field_label(source_field), normalize_field_label(target_field))
    return pair in get_task_pairs(task)


def pair_name(source_field: str, target_field: str) -> str:
    return f"{normalize_field_label(source_field)}_to_{normalize_field_label(target_field)}"


def parse_pair_name(pair: str) -> tuple[str, str]:
    parts = str(pair).split("_to_")
    if len(parts) != 2:
        raise ValueError("Pair name must use the format '<source>_to_<target>'.")
    source, target = normalize_field_label(parts[0]), normalize_field_label(parts[1])
    if source == target:
        raise ValueError("Pair source and target fields must differ.")
    return source, target


def expected_subject_ids_for_pair(
    task: str,
    source_field: str,
    target_field: str,
) -> tuple[str, ...]:
    source = normalize_field_label(source_field)
    target = normalize_field_label(target_field)
    if not is_allowed_pair(task, source, target):
        raise ValueError(f"Pair {pair_name(source, target)!r} is not allowed for {_normalize_task(task)}.")
    return VALIDATION_RELEASED_IDS[source]


def expected_subtask_count(task: str) -> int:
    return len(get_task_pairs(task)) * len(expected_modalities(task))


def expected_prediction_file_count(task: str) -> int:
    return sum(
        len(expected_subject_ids_for_pair(task, source, target)) * len(expected_modalities(task))
        for source, target in get_task_pairs(task)
    )


def expected_segmentation_file_count(task: str) -> int:
    if not requires_segmentation(task):
        return 0
    return expected_prediction_file_count(task)


def parse_mrixfields_filename(name: str) -> MRIxFieldsFilename:
    """Parse an official MRIxFields filename without reading a NIfTI payload."""

    filename = str(name)
    suffix = ".nii.gz"
    if not filename.endswith(suffix):
        raise ValueError(f"MRIxFields filename must end with {suffix!r}: {filename!r}.")

    stem = filename[: -len(suffix)]
    is_segmentation = stem.endswith("_seg")
    if is_segmentation:
        stem = stem[: -len("_seg")]

    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(
            "MRIxFields filename must use '<R|P>_<MOD>_<FIELD>_<ID>.nii.gz' "
            "or '<R|P>_<MOD>_<FIELD>_<ID>_seg.nii.gz'."
        )

    prefix, modality, field, subject_id = parts
    if prefix not in {"R", "P"}:
        raise ValueError(f"Invalid MRIxFields filename prefix {prefix!r}. Expected 'R' or 'P'.")

    return MRIxFieldsFilename(
        prefix=prefix,  # type: ignore[arg-type]
        modality=normalize_modality(modality),
        field=normalize_field_label(field),
        subject_id=_validate_subject_id(subject_id),
        is_segmentation=is_segmentation,
    )


def build_prediction_filename(
    modality: str,
    target_field: str,
    subject_id: str,
    segmentation: bool = False,
) -> str:
    """Build an official prediction filename using the target field."""

    official_modality = normalize_modality(modality)
    field = normalize_field_label(target_field)
    validated_id = _validate_subject_id(subject_id)
    suffix = "_seg" if segmentation else ""
    return f"P_{official_modality}_{field}_{validated_id}{suffix}.nii.gz"


def spec_as_dict() -> dict[str, Any]:
    """Return a JSON-serializable MRIxFields2026 spec summary."""

    return {
        "fields": list(FIELDS),
        "field_values": dict(FIELD_VALUES),
        "modalities": list(OFFICIAL_MODALITIES),
        "full_shape": list(FULL_SHAPE),
        "submission_z_clip": list(SUBMISSION_Z_CLIP),
        "submission_shape": list(SUBMISSION_SHAPE),
        "intensity_range": list(INTENSITY_RANGE),
        "train_slice_range": {
            "range": list(TRAIN_SLICE_RANGE),
            "label": "baseline preprocessing axial range, not submission range",
        },
        "task1": _task_summary("task1"),
        "task2": _task_summary("task2"),
        "task3": _task_summary("task3"),
        "validation_released_ids": {field: list(ids) for field, ids in VALIDATION_RELEASED_IDS.items()},
    }


def _task_summary(task: str) -> dict[str, Any]:
    pairs = get_task_pairs(task)
    return {
        "pairs": [list(pair) for pair in pairs],
        "pair_names": [pair_name(source, target) for source, target in pairs],
        "pair_count": len(pairs),
        "subtask_count": expected_subtask_count(task),
        "prediction_file_count": expected_prediction_file_count(task),
        "segmentation_file_count": expected_segmentation_file_count(task),
        "requires_segmentation": requires_segmentation(task),
        "modalities": list(expected_modalities(task)),
    }


def _normalize_task(task: str) -> str:
    key = str(task).strip().lower().replace(" ", "")
    if key in {"1", "task1"}:
        return "task1"
    if key in {"2", "task2"}:
        return "task2"
    if key in {"3", "task3"}:
        return "task3"
    raise ValueError("Task must be one of 'task1', 'task2', or 'task3'.")


def _validate_subject_id(subject_id: str) -> str:
    value = str(subject_id).strip()
    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"Subject ID must be a 4-digit string, got {subject_id!r}.")
    return value
