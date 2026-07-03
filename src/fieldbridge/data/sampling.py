"""Per-record sampling weights for domain oversampling (e.g. boosting 0.1T)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fieldbridge.data.contracts import VolumeRecord


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
