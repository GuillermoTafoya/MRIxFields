"""Subject-level train/validation/test splits for the Etapa 1 KL-VAE.

Distinct from `data/volume_splits.py`, which builds *pseudo-pair* splits for one contrast
of paired traveller data (`sequence` + `target_fields`, field-balanced). The VAE is blind
to (field, contrast) and trains on the whole pool of the official Training data (15
domains: 5 fields x 3 contrasts, retrospective + prospective), so it needs a general
splitter with three properties:

* **Subject-level, not volume-level.** A prospective traveller (`P_` prefix) is the same
  physical subject imaged at all 5 fields x 3 contrasts; every one of that subject's
  volumes must land in the same split or the held-out sets leak. Retrospective (`R_`)
  subjects are unique per field, so each is effectively its own group.
* **Domain-stratified.** Single-domain (retrospective) subjects are split *within* each
  (field, contrast) bucket so validation/test cover every domain — 0.1T in particular,
  the hardest to reconstruct and the one whose held-out score matters most.
* **Deterministic + auditable.** Seeded assignment, persisted JSON with a fingerprint, and
  a leakage audit (reusing `volume_splits.LeakageAudit`) so a split is reproducible and a
  subject can never straddle two splits unnoticed.

This is our *internal* split carved from the released Training data — not the challenge's
own Validating/Testing sets (whose targets are withheld). Point eval at those directly
once they are available.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.manifests import record_from_mapping
from fieldbridge.data.volume_splits import LeakageAudit, SplitName, VolumeSplitError, _SPLIT_NAMES

import torch


@dataclass(frozen=True, slots=True)
class VaeSplits:
    train: tuple[VolumeRecord, ...]
    validation: tuple[VolumeRecord, ...]
    test: tuple[VolumeRecord, ...]
    seed: int
    fractions: tuple[float, float, float]  # (train, validation, test)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def records_for(self, split: SplitName) -> tuple[VolumeRecord, ...]:
        return getattr(self, split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fractions": list(self.fractions),
            "metadata": dict(self.metadata),
            "splits": {name: [r.to_dict() for r in self.records_for(name)] for name in _SPLIT_NAMES},
        }


def _subject_key(record: VolumeRecord) -> str:
    """Group key that keeps one physical subject's volumes together.

    `{prefix}:{subject_id}` — the prefix (R/P) separates a retrospective `R_0006` from a
    prospective `P_0006`, and groups all of a P traveller's field/contrast volumes. Falls
    back to `case_id` (unique per record) when the official metadata is absent, so a
    non-official manifest still splits without leakage (each record its own subject).
    """

    prefix = record.metadata.get("prefix") if isinstance(record.metadata, Mapping) else None
    if prefix and record.subject_id:
        return f"{prefix}:{record.subject_id}"
    if record.subject_id:
        return str(record.subject_id)
    return record.case_id


def _domain_label(record: VolumeRecord) -> str:
    return getattr(record.domain, "label", str(record.domain))


def _split_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Integer (train, val, test) counts for `n` items, val/test floored, train gets the rest.

    Flooring val/test (rather than rounding) means a bucket only donates to a held-out set
    once it is large enough — a domain with 1-2 subjects keeps them in train rather than
    starving train to fill val/test. Train always gets the remainder so nothing is dropped.
    """

    n_val = int(fractions[1] * n)
    n_test = int(fractions[2] * n)
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def build_vae_splits(
    records: Sequence[VolumeRecord],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 13,
) -> VaeSplits:
    """Subject-level, domain-stratified train/val/test split over the whole record pool.

    Single-domain subjects are stratified within their (field, contrast) bucket; multi-
    domain subjects (prospective travellers) are split as their own pooled bucket since
    they contribute to every domain regardless. Assignment is at the subject level, so all
    of a subject's volumes share a split.
    """

    if not records:
        raise VolumeSplitError("Cannot build VAE splits from an empty record list.")
    fractions = (float(train_frac), float(val_frac), float(test_frac))
    if any(f < 0 for f in fractions):
        raise VolumeSplitError(f"fractions must be non-negative, got {fractions}.")
    total = sum(fractions)
    if abs(total - 1.0) > 1e-6:
        raise VolumeSplitError(f"fractions must sum to 1.0, got {total} {fractions}.")

    # Group records by subject; record which domains each subject spans.
    by_subject: dict[str, list[VolumeRecord]] = defaultdict(list)
    subject_domains: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = _subject_key(record)
        by_subject[key].append(record)
        subject_domains[key].add(_domain_label(record))

    # Bucket subjects: single-domain subjects by their domain (for per-domain
    # stratification); multi-domain subjects (travellers) into one shared bucket.
    buckets: dict[str, list[str]] = defaultdict(list)
    for key, domains in subject_domains.items():
        bucket = next(iter(domains)) if len(domains) == 1 else "__multi_domain__"
        buckets[bucket].append(key)

    assigned: dict[SplitName, list[VolumeRecord]] = {name: [] for name in _SPLIT_NAMES}
    for bucket_name in sorted(buckets):
        subjects = sorted(buckets[bucket_name])  # sorted first => shuffle is the only randomness
        generator = torch.Generator().manual_seed(seed + _bucket_seed(bucket_name))
        order = torch.randperm(len(subjects), generator=generator).tolist()
        shuffled = [subjects[i] for i in order]
        n_train, n_val, _ = _split_counts(len(shuffled), fractions)
        for position, subject_key in enumerate(shuffled):
            if position < n_train:
                split: SplitName = "train"
            elif position < n_train + n_val:
                split = "validation"
            else:
                split = "test"
            assigned[split].extend(by_subject[subject_key])

    splits = VaeSplits(
        train=tuple(_sorted_records(assigned["train"])),
        validation=tuple(_sorted_records(assigned["validation"])),
        test=tuple(_sorted_records(assigned["test"])),
        seed=int(seed),
        fractions=fractions,
        metadata={
            "num_subjects": len(by_subject),
            "num_records": len(records),
            "num_multi_domain_subjects": len(buckets.get("__multi_domain__", [])),
        },
    )
    audit_vae_splits(splits).raise_for_leakage()
    return splits


def audit_vae_splits(splits: VaeSplits) -> LeakageAudit:
    """Fail if any subject/case/path identity appears in more than one split."""

    case_seen: dict[str, set[str]] = defaultdict(set)
    path_seen: dict[str, set[str]] = defaultdict(set)
    subject_seen: dict[str, set[str]] = defaultdict(set)
    for name in _SPLIT_NAMES:
        for record in splits.records_for(name):
            case_seen[record.case_id].add(name)
            path_seen[str(record.image_path)].add(name)
            subject_seen[_subject_key(record)].add(name)
    leaked_case = {k: sorted(v) for k, v in case_seen.items() if len(v) > 1}
    leaked_path = {k: sorted(v) for k, v in path_seen.items() if len(v) > 1}
    leaked_subject = {k: sorted(v) for k, v in subject_seen.items() if len(v) > 1}
    return LeakageAudit(
        ok=not (leaked_case or leaked_path or leaked_subject),
        leaked_case_ids=leaked_case,
        leaked_paths=leaked_path,
        leaked_subject_ids=leaked_subject,
    )


def summarize_vae_splits(splits: VaeSplits) -> dict[str, Any]:
    """Per-split record + subject counts and per-domain record counts (a stratification check)."""

    summary: dict[str, Any] = {"seed": splits.seed, "fractions": list(splits.fractions), "splits": {}}
    for name in _SPLIT_NAMES:
        records = splits.records_for(name)
        per_domain: dict[str, int] = defaultdict(int)
        subjects: set[str] = set()
        for record in records:
            per_domain[_domain_label(record)] += 1
            subjects.add(_subject_key(record))
        summary["splits"][name] = {
            "num_records": len(records),
            "num_subjects": len(subjects),
            "per_domain": dict(sorted(per_domain.items())),
        }
    return summary


def vae_splits_fingerprint(splits: VaeSplits) -> str:
    """Frozen case-membership hash used by the completed audit-v1 contract."""

    payload = {
        name: sorted(record.case_id for record in splits.records_for(name))
        for name in _SPLIT_NAMES
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vae_splits_recovery_fingerprint_v3(splits: VaeSplits) -> str:
    """Strict v3 recovery hash over complete record and split identity.

    This is intentionally separate from ``vae_splits_fingerprint``: the latter is used
    by the frozen be60d75 audit-v1 selection contract and must retain its case-only
    arithmetic. The v3 recovery hash detects changed paths, domains, subjects, seed,
    fractions, or split metadata under unchanged case IDs.
    """

    payload = {
        "seed": int(splits.seed),
        "fractions": [float(value) for value in splits.fractions],
        "metadata": dict(splits.metadata),
        "splits": {
            name: [
                record.to_dict()
                for record in sorted(
                    splits.records_for(name),
                    key=lambda record: (record.case_id, str(record.image_path)),
                )
            ]
            for name in _SPLIT_NAMES
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_vae_splits(splits: VaeSplits, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = splits.to_dict()
    payload["fingerprint"] = vae_splits_fingerprint(splits)
    payload["recovery_fingerprint_v3"] = vae_splits_recovery_fingerprint_v3(splits)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_vae_splits(path: str | Path) -> VaeSplits:
    split_path = Path(path)
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VolumeSplitError(f"Could not read VAE split file {split_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise VolumeSplitError(f"VAE split file {split_path} must contain a JSON object.")
    splits_payload = payload.get("splits")
    if not isinstance(splits_payload, Mapping):
        raise VolumeSplitError(f"VAE split file {split_path} is missing a 'splits' object.")
    missing_names = [name for name in _SPLIT_NAMES if name not in splits_payload]
    if missing_names:
        raise VolumeSplitError(
            f"VAE split file {split_path} is missing split(s): {missing_names}."
        )

    def _records(name: str) -> tuple[VolumeRecord, ...]:
        values = splits_payload.get(name)
        if not isinstance(values, list):
            raise VolumeSplitError(
                f"VAE split file {split_path} entry {name!r} must be a list."
            )
        try:
            return tuple(record_from_mapping(record) for record in values)
        except (KeyError, TypeError, ValueError) as exc:
            raise VolumeSplitError(
                f"VAE split file {split_path} contains a malformed {name!r} record: {exc}"
            ) from exc

    fractions = payload.get("fractions")
    if not isinstance(fractions, list) or len(fractions) != 3:
        raise VolumeSplitError(
            f"VAE split file {split_path} must contain three split fractions."
        )
    try:
        fraction_values = tuple(float(value) for value in fractions)
        seed = int(payload["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VolumeSplitError(
            f"VAE split file {split_path} has invalid seed or fractions."
        ) from exc
    if (
        any(not math.isfinite(value) or value < 0.0 for value in fraction_values)
        or abs(sum(fraction_values) - 1.0) > 1e-6
    ):
        raise VolumeSplitError(
            f"VAE split file {split_path} fractions must be finite, non-negative, "
            f"and sum to 1.0; got {fraction_values}."
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise VolumeSplitError(
            f"VAE split file {split_path} metadata must be a JSON object."
        )
    splits = VaeSplits(
        train=_records("train"),
        validation=_records("validation"),
        test=_records("test"),
        seed=seed,
        fractions=(
            fraction_values[0],
            fraction_values[1],
            fraction_values[2],
        ),
        metadata=dict(metadata),
    )
    expected_membership_fingerprint = payload.get("fingerprint")
    if (
        not isinstance(expected_membership_fingerprint, str)
        or not expected_membership_fingerprint
    ):
        raise VolumeSplitError(
            f"VAE split file {split_path} has no persisted fingerprint; rebuild it."
        )
    actual_membership_fingerprint = vae_splits_fingerprint(splits)
    if actual_membership_fingerprint != expected_membership_fingerprint:
        raise VolumeSplitError(
            "VAE split fingerprint mismatch; the file is stale or was altered: "
            f"{expected_membership_fingerprint} != {actual_membership_fingerprint}."
        )
    expected_recovery_fingerprint = payload.get("recovery_fingerprint_v3")
    if (
        not isinstance(expected_recovery_fingerprint, str)
        or not expected_recovery_fingerprint
    ):
        raise VolumeSplitError(
            f"VAE split file {split_path} has no v3 recovery fingerprint; rebuild it."
        )
    actual_recovery_fingerprint = vae_splits_recovery_fingerprint_v3(splits)
    if actual_recovery_fingerprint != expected_recovery_fingerprint:
        raise VolumeSplitError(
            "VAE split v3 recovery fingerprint mismatch; the file is stale or was "
            f"altered: {expected_recovery_fingerprint} != "
            f"{actual_recovery_fingerprint}."
        )
    for name in _SPLIT_NAMES:
        records = splits.records_for(name)
        case_ids = [record.case_id for record in records]
        paths = [str(record.image_path) for record in records]
        if len(case_ids) != len(set(case_ids)) or len(paths) != len(set(paths)):
            raise VolumeSplitError(
                f"VAE split file {split_path} contains duplicate case/path identity "
                f"inside the {name!r} split."
            )
    audit_vae_splits(splits).raise_for_leakage()
    return splits


def _sorted_records(records: Iterable[VolumeRecord]) -> list[VolumeRecord]:
    return sorted(records, key=lambda r: (str(r.image_path), r.case_id))


def _bucket_seed(bucket_name: str) -> int:
    """Small deterministic per-bucket offset so each domain bucket shuffles independently."""

    return int(hashlib.sha256(bucket_name.encode("utf-8")).hexdigest(), 16) % (2**31)
