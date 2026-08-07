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

from fieldbridge.data.vae_splits import load_vae_splits

SPLIT_KEYS = ("train", "validation", "test")

# Both are membership-sensitive hashes written by `save_vae_splits` and enforced by
# `load_vae_splits`. Relocating records invalidates both.
_FINGERPRINT_KEYS = ("fingerprint", "recovery_fingerprint_v3")


def _is_prospective_record(record: Mapping[str, Any]) -> bool:
    """Match the prospective-record identity used by Stage-2 evaluation."""

    metadata = record.get("metadata")
    return str(record.get("case_id", "")).startswith("P_") or (
        isinstance(metadata, Mapping) and metadata.get("prefix") == "P"
    )


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
            if not isinstance(record, Mapping):
                raise ValueError(f"split_data.splits.{key} must contain record objects.")
            subject_id = str(record.get("subject_id"))
            if subject_id in subject_set and _is_prospective_record(record):
                seen_subjects.add(subject_id)
                if key != to_split:
                    moved[key] += 1
                promoted.append(record)
            else:
                kept[key].append(record)

    missing = sorted(subject_set - seen_subjects)
    if missing:
        raise ValueError(f"Subjects not found in the split: {missing}.")

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
    """Validate, resplit, revalidate, and atomically publish a VAE split JSON."""

    source = Path(in_path)
    destination = Path(out_path)
    if destination.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite the input split; choose a different --out path.")
    # Validate the persisted identities before reading the raw mapping. Otherwise a stale or
    # altered split could be transformed and receive freshly computed, apparently valid hashes.
    load_vae_splits(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    updated = promote_subjects_to_split(data, subjects, to_split)
    updated.update(recomputed_fingerprints(updated))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
        # Treat the temporary artifact as an untrusted persisted split. Publication is permitted
        # only after the same canonical loader used by downstream consumers accepts it.
        load_vae_splits(temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    counts = {key: len(updated["splits"][key]) for key in SPLIT_KEYS}
    return {"out": str(destination), "counts": counts, "resplit": updated["resplit"]}


__all__ = [
    "promote_subjects_to_split",
    "recomputed_fingerprints",
    "resplit_file",
    "SPLIT_KEYS",
]
