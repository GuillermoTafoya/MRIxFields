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
_FINGERPRINT_KEYS = ("fingerprint", "recovery_fingerprint_v3")


def cohort_of(record: Mapping[str, Any]) -> str:
    """``P`` or ``R`` — the cohort prefix of a record's case_id."""

    return str(record.get("case_id", "")).split("_", 1)[0]


def _parse_subject_spec(spec: str) -> tuple[str | None, str]:
    """``"P:0006"`` / ``"P_0006"`` -> ("P", "0006"); a bare ``"0006"`` -> (None, "0006")."""

    text = str(spec)
    for separator in (":", "_"):
        if separator in text:
            cohort, _, subject = text.partition(separator)
            if cohort in ("P", "R"):
                return cohort, subject
    return None, text


def promote_subjects_to_split(
    split_data: Mapping[str, Any],
    subjects: Sequence[str],
    to_split: str,
) -> dict[str, Any]:
    """Return a copy of ``split_data`` with every record of ``subjects`` moved into ``to_split``.

    Subjects are identified by **(cohort, subject_id)**, not by the bare number. The official
    data description gives the two cohorts overlapping numeric ranges — retrospective IDs run
    0001-1056 (field-scoped) and prospective 0001-0040 — so ``0006`` names two different
    people: traveller ``P_..._0006`` and a 0.1T retrospective volunteer ``R_..._0006``. Both
    live in this split's train array. Matching on the number alone would silently drag a
    stranger's volumes along with the traveller.

    Pass ``"P:0006"`` (or ``"P_0006"``) to disambiguate. A bare id is accepted only when it is
    unambiguous in this split; otherwise it raises rather than guessing.
    """

    if to_split not in SPLIT_KEYS:
        raise ValueError(f"to_split must be one of {SPLIT_KEYS}; got {to_split!r}.")
    if not subjects:
        raise ValueError("No subjects given to promote.")

    splits = split_data.get("splits")
    if not isinstance(splits, Mapping) or not all(key in splits for key in SPLIT_KEYS):
        raise ValueError("split_data.splits must contain train/validation/test arrays.")

    requested = [_parse_subject_spec(s) for s in subjects]

    cohorts_by_id: dict[str, set[str]] = {}
    for key in SPLIT_KEYS:
        for record in splits[key]:
            cohorts_by_id.setdefault(str(record.get("subject_id")), set()).add(cohort_of(record))
    ambiguous = {
        subject: sorted(cohorts_by_id.get(subject, set()))
        for cohort, subject in requested
        if cohort is None and len(cohorts_by_id.get(subject, set())) > 1
    }
    if ambiguous:
        raise ValueError(
            f"Ambiguous subject id(s) {ambiguous}: the same number exists in more than one "
            "cohort and they are DIFFERENT people. Qualify them, e.g. 'P:0006'."
        )

    wanted = {
        (cohort or next(iter(cohorts_by_id.get(subject, {""}))), subject)
        for cohort, subject in requested
    }

    moved: dict[str, int] = {key: 0 for key in SPLIT_KEYS}
    kept: dict[str, list[dict[str, Any]]] = {key: [] for key in SPLIT_KEYS}
    promoted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for key in SPLIT_KEYS:
        for record in splits[key]:
            identity = (cohort_of(record), str(record.get("subject_id")))
            if identity in wanted:
                seen.add(identity)
                if key != to_split:
                    moved[key] += 1
                promoted.append(record)
            else:
                kept[key].append(record)

    missing = sorted(f"{cohort}:{subject}" for cohort, subject in wanted - seen)
    if missing:
        raise ValueError(f"Subjects not found in the split: {missing}.")
    subject_set = {f"{cohort}:{subject}" for cohort, subject in wanted}

    result: dict[str, Any] = {
        k: v for k, v in split_data.items() if k not in ("splits", *_FINGERPRINT_KEYS)
    }
    result["splits"] = {key: list(kept[key]) for key in SPLIT_KEYS}
    result["splits"][to_split].extend(promoted)
    result["resplit"] = {
        "promoted_subjects": sorted(subject_set),
        "to_split": to_split,
        "moved_records_from": {k: v for k, v in moved.items() if v},
    }
    # The split fingerprints are membership-sensitive, and this function just changed
    # membership. Carrying the input's fingerprints forward produced a file that
    # `load_vae_splits` refuses as "stale or altered" — which is every Stage-2 consumer of a
    # resplit split. They are dropped here and recomputed by `resplit_file`, so an in-memory
    # result never carries a fingerprint that lies about its own contents.
    return result


def recomputed_fingerprints(split_data: Mapping[str, Any]) -> dict[str, str]:
    """Both membership fingerprints for a relocated split, from the canonical implementations.

    Imported lazily so this module stays usable for plain record relocation without pulling in
    the split builder, and so the fingerprint definitions live in exactly one place.
    """

    from fieldbridge.data.manifests import record_from_mapping
    from fieldbridge.data.vae_splits import (
        VaeSplits,
        vae_splits_fingerprint,
        vae_splits_recovery_fingerprint_v3,
    )

    splits = split_data["splits"]
    fractions = [float(value) for value in split_data["fractions"]]
    rebuilt = VaeSplits(
        train=tuple(record_from_mapping(r) for r in splits["train"]),
        validation=tuple(record_from_mapping(r) for r in splits["validation"]),
        test=tuple(record_from_mapping(r) for r in splits["test"]),
        seed=int(split_data["seed"]),
        fractions=(fractions[0], fractions[1], fractions[2]),
        metadata=dict(split_data.get("metadata", {})),
    )
    return {
        "fingerprint": vae_splits_fingerprint(rebuilt),
        "recovery_fingerprint_v3": vae_splits_recovery_fingerprint_v3(rebuilt),
    }


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
    updated.update(recomputed_fingerprints(updated))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    counts = {key: len(updated["splits"][key]) for key in SPLIT_KEYS}
    return {"out": str(destination), "counts": counts, "resplit": updated["resplit"]}


__all__ = [
    "promote_subjects_to_split",
    "recomputed_fingerprints",
    "resplit_file",
    "cohort_of",
    "SPLIT_KEYS",
]
