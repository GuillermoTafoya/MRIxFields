"""Per-record sampling weights for domain oversampling (e.g. boosting 0.1T)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from fieldbridge.data.contracts import VolumeRecord


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
