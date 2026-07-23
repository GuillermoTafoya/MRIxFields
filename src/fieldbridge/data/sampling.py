"""Deterministic hierarchical sampling and legacy per-record weighting helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fieldbridge.data.contracts import VolumeRecord


def subject_key(record: VolumeRecord) -> str:
    """Stable subject key, falling back to the case when subject metadata is absent."""

    prefix = record.metadata.get("prefix") if isinstance(record.metadata, Mapping) else None
    if prefix and record.subject_id:
        return f"{prefix}:{record.subject_id}"
    return str(record.subject_id or record.case_id)


def joint_domain_subject_balanced_indices(
    records: Sequence[VolumeRecord],
    *,
    num_samples: int | None = None,
    seed: int = 0,
    pass_index: int = 0,
) -> list[int]:
    """Return a deterministic 15-way domain-balanced, subject-fair record schedule.

    Draws are allocated as evenly as integer arithmetic permits across complete
    ``field x contrast`` domains. Within each domain they are allocated evenly across
    subjects, then round-robin across that subject's volumes. Remainders rotate with the
    pass index, preventing the same domain, subject, or volume from receiving the extra
    draw forever. Small domains necessarily repeat records, but the repetition is
    explicit and fair at both subject and volume level.
    """

    if not records:
        return []
    draws = len(records) if num_samples is None else int(num_samples)
    if draws < 0:
        raise ValueError("num_samples must be non-negative.")
    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_domain[record.domain.label].append(index)
    domains = sorted(by_domain)
    if len(domains) != 15:
        raise ValueError(
            "joint-domain balancing requires all 15 field x contrast domains; "
            f"found {len(domains)}: {domains}."
        )

    generator = torch.Generator().manual_seed(int(seed) + int(pass_index))
    domain_order = torch.randperm(len(domains), generator=generator).tolist()
    ordered_domains = [domains[i] for i in domain_order]
    domain_counts = _rotating_counts(draws, len(domains), pass_index)
    scheduled: list[int] = []
    for domain_position, domain in enumerate(ordered_domains):
        count = domain_counts[domain_position]
        by_subject: dict[str, list[int]] = defaultdict(list)
        for index in by_domain[domain]:
            by_subject[subject_key(records[index])].append(index)
        subjects = sorted(by_subject)
        subject_order = torch.randperm(len(subjects), generator=generator).tolist()
        ordered_subjects = [subjects[i] for i in subject_order]
        subject_counts = _rotating_counts(
            count, len(ordered_subjects), pass_index + domain_position
        )
        for subject_position, subject in enumerate(ordered_subjects):
            volume_indices = sorted(by_subject[subject])
            volume_order = torch.randperm(len(volume_indices), generator=generator).tolist()
            ordered_volumes = [volume_indices[i] for i in volume_order]
            volume_count = subject_counts[subject_position]
            start = (pass_index + subject_position) % len(ordered_volumes)
            scheduled.extend(
                ordered_volumes[(start + offset) % len(ordered_volumes)]
                for offset in range(volume_count)
            )
    # Interleave domains rather than emitting one domain in a long contiguous block.
    buckets: dict[str, list[int]] = defaultdict(list)
    for index in scheduled:
        buckets[records[index].domain.label].append(index)
    result: list[int] = []
    while any(buckets.values()):
        for domain in ordered_domains:
            if buckets[domain]:
                result.append(buckets[domain].pop(0))
    return result


def exposure_report(
    records: Sequence[VolumeRecord], indices: Sequence[int]
) -> dict[str, Any]:
    """Observed record draws plus domain and within-domain subject exposure."""

    domains: Counter[str] = Counter()
    subjects: dict[str, Counter[str]] = defaultdict(Counter)
    volumes: Counter[str] = Counter()
    for index in indices:
        record = records[int(index)]
        label = record.domain.label
        domains[label] += 1
        subjects[label][subject_key(record)] += 1
        volumes[record.case_id] += 1
    return {
        "total_draws": len(indices),
        "by_domain": dict(sorted(domains.items())),
        "by_domain_subject": {
            domain: dict(sorted(counts.items())) for domain, counts in sorted(subjects.items())
        },
        "by_volume": dict(sorted(volumes.items())),
    }


def _rotating_counts(total: int, groups: int, rotation: int) -> list[int]:
    if groups <= 0:
        return []
    base, remainder = divmod(int(total), int(groups))
    counts = [base] * groups
    for offset in range(remainder):
        counts[(int(rotation) + offset) % groups] += 1
    return counts


def field_balanced_weights(records: Sequence[VolumeRecord], *, default_weight: float = 1.0) -> list[float]:
    """Inverse-frequency per-record weights that equalize field strengths in expectation.

    Each record gets weight proportional to `1 / count(its field)` within `records`, scaled so
    the weights average to `default_weight`. Sampling with these weights makes every field
    strength appear ~equally often per pass regardless of how many volumes each field has — the
    under-represented fields (here 5T at 220 and 0.1T at 240 vs 1.5T at 441) get up-sampled with
    NO hand-tuned multipliers (the ratios come from the data). Feeds
    `StreamingPatchDataset(sampling_weights=...)`; the multiplier map variant is
    `domain_oversampling_weights`.

    Because each field's total weight mass is equalized here, striding the weight vector by
    worker (records[worker.id::num_workers]) keeps the per-worker field balance too: a field's
    mass in a shard is ~1/num_workers of its global mass, equal across fields.
    """

    counts: Counter = Counter(record.domain.field_strength_t for record in records)
    num_fields = len(counts)
    if num_fields == 0:
        return []
    total = len(records)
    return [
        default_weight * (total / num_fields) / counts[record.domain.field_strength_t]
        for record in records
    ]


def domain_oversampling_weights(
    records: Sequence[VolumeRecord],
    *,
    boost_by_field: Mapping[float, float],
    default_weight: float = 1.0,
) -> list[float]:
    """Per-record weights for `torch.utils.data.WeightedRandomSampler`.

    `boost_by_field` maps `field_strength_t -> multiplier`, e.g. `{0.1: 3.0}` samples
    0.1T records three times as often as `default_weight`. Fields not present in
    `boost_by_field` use `default_weight` unchanged. No default map ships here — the
    oversampling ratio is a real experiment hyperparameter, not something to guess.
    Keys should match the canonical `FIELD_STRENGTHS_T` values (`Domain.__post_init__`
    canonicalizes `field_strength_t`, so exact float equality against those values is
    safe).
    """

    return [boost_by_field.get(record.domain.field_strength_t, default_weight) for record in records]
