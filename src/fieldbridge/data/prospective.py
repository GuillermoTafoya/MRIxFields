"""Prospective paired-file grouping helpers.

These utilities operate on paths and filenames only. They do not read image files.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re


_PROSPECTIVE_PATTERN = re.compile(
    r"^P_(?P<sequence>.+)_(?P<field>\d+(?:\.\d+)?)T_(?P<case_id>\d+)\.nii(?:\.gz)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ProspectivePathRecord:
    path: Path
    sequence: str
    case_id: str
    field_strength_t: float


@dataclass(frozen=True, slots=True)
class LeaveOneSubjectOutFold:
    held_out_case_id: str
    train_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]


def parse_prospective_path(path: str | Path) -> ProspectivePathRecord:
    """Parse names like ``P_T2FLAIR_7T_0006.nii.gz``."""

    parsed_path = Path(path)
    match = _PROSPECTIVE_PATTERN.match(parsed_path.name)
    if match is None:
        raise ValueError(f"Path {parsed_path.name!r} does not match the prospective pattern.")
    return ProspectivePathRecord(
        path=parsed_path,
        sequence=match.group("sequence").upper(),
        case_id=match.group("case_id"),
        field_strength_t=float(match.group("field")),
    )


def group_prospective_paths(
    paths: Iterable[str | Path],
) -> dict[tuple[str, str], tuple[ProspectivePathRecord, ...]]:
    """Group prospective paths by ``(case_id, sequence)``."""

    groups: dict[tuple[str, str], list[ProspectivePathRecord]] = {}
    for path in paths:
        record = parse_prospective_path(path)
        groups.setdefault((record.case_id, record.sequence), []).append(record)
    return {
        key: tuple(sorted(records, key=lambda record: record.field_strength_t))
        for key, records in sorted(groups.items())
    }


def find_multifield_groups(
    paths: Iterable[str | Path],
    *,
    min_fields: int = 2,
) -> dict[tuple[str, str], tuple[ProspectivePathRecord, ...]]:
    """Return groups with at least ``min_fields`` unique field strengths."""

    if min_fields < 2:
        raise ValueError(f"min_fields must be at least 2, got {min_fields}.")
    return {
        key: records
        for key, records in group_prospective_paths(paths).items()
        if len({record.field_strength_t for record in records}) >= min_fields
    }


def leave_one_subject_out_folds(case_ids: Sequence[str]) -> tuple[LeaveOneSubjectOutFold, ...]:
    """Build leave-one-case/subject-out folds from case IDs."""

    unique_case_ids = tuple(sorted(dict.fromkeys(str(case_id) for case_id in case_ids)))
    return tuple(
        LeaveOneSubjectOutFold(
            held_out_case_id=held_out,
            train_case_ids=tuple(case_id for case_id in unique_case_ids if case_id != held_out),
            test_case_ids=(held_out,),
        )
        for held_out in unique_case_ids
    )
