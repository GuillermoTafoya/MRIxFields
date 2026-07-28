"""Move whole subjects between the train/validation/test arrays of a VAE split JSON.

Used to relocate the prospective travellers (the only cross-field paired subjects) into the
validation split so the Stage-2 transport eval gate has a held-out anchor, and to keep them out of
train (2-3 traveller anatomies in train only overfit — v3.0's failure mode). This is a pure record
relocation: it does NOT touch the bank latents (no re-encode), only which split-array a case sits
in. It refuses to overwrite its input and writes a new, versioned file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SPLIT_KEYS = ("train", "validation", "test")


def promote_subjects_to_split(
    split_data: Mapping[str, Any],
    subjects: Sequence[str],
    to_split: str,
) -> dict[str, Any]:
    """Return a copy of ``split_data`` with every record of ``subjects`` moved into ``to_split``."""

    if to_split not in SPLIT_KEYS:
        raise ValueError(f"to_split must be one of {SPLIT_KEYS}; got {to_split!r}.")
    subject_set = {str(s) for s in subjects}
    if not subject_set:
        raise ValueError("No subjects given to promote.")

    splits = split_data.get("splits")
    if not isinstance(splits, Mapping) or not all(key in splits for key in SPLIT_KEYS):
        raise ValueError("split_data.splits must contain train/validation/test arrays.")

    moved: dict[str, int] = {key: 0 for key in SPLIT_KEYS}
    kept: dict[str, list[dict[str, Any]]] = {key: [] for key in SPLIT_KEYS}
    promoted: list[dict[str, Any]] = []
    seen_subjects: set[str] = set()
    for key in SPLIT_KEYS:
        for record in splits[key]:
            subject_id = str(record.get("subject_id"))
            if subject_id in subject_set:
                seen_subjects.add(subject_id)
                if key != to_split:
                    moved[key] += 1
                promoted.append(record)
            else:
                kept[key].append(record)

    missing = sorted(subject_set - seen_subjects)
    if missing:
        raise ValueError(f"Subjects not found in the split: {missing}.")

    result: dict[str, Any] = {k: v for k, v in split_data.items() if k != "splits"}
    result["splits"] = {key: list(kept[key]) for key in SPLIT_KEYS}
    result["splits"][to_split].extend(promoted)
    result["resplit"] = {
        "promoted_subjects": sorted(subject_set),
        "to_split": to_split,
        "moved_records_from": {k: v for k, v in moved.items() if v},
    }
    return result


def resplit_file(
    in_path: str | Path,
    out_path: str | Path,
    subjects: Sequence[str],
    to_split: str,
) -> dict[str, Any]:
    """Read a split JSON, promote ``subjects`` to ``to_split``, and write a new file."""

    source = Path(in_path)
    destination = Path(out_path)
    if destination.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite the input split; choose a different --out path.")
    data = json.loads(source.read_text(encoding="utf-8"))
    updated = promote_subjects_to_split(data, subjects, to_split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    counts = {key: len(updated["splits"][key]) for key in SPLIT_KEYS}
    return {"out": str(destination), "counts": counts, "resplit": updated["resplit"]}


__all__ = ["promote_subjects_to_split", "resplit_file", "SPLIT_KEYS"]
