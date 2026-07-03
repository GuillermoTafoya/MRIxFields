import pytest
import torch

from fieldbridge.data.transforms import (
    compose,
    identity_transform,
    normalize_percentile_clip_to_unit_range,
    normalize_zero_mean_unit_variance,
)


def test_percentile_clip_output_within_unit_range() -> None:
    image = torch.cat([torch.randn(1000), torch.tensor([1000.0])])

    normalized = normalize_percentile_clip_to_unit_range(image)

    assert torch.isfinite(normalized).all()
    assert normalized.min() >= -1.0 - 1e-5
    assert normalized.max() <= 1.0 + 1e-5


def test_percentile_clip_suppresses_outlier_influence() -> None:
    # Without percentile clipping, this single extreme outlier would dominate a min-max
    # scale and compress the rest of the distribution near -1.
    image = torch.cat([torch.linspace(0.0, 1.0, 1000), torch.tensor([1000.0])])

    normalized = normalize_percentile_clip_to_unit_range(image)

    # Most of the distribution (away from the injected outlier) should still spread
    # meaningfully across the range, not be crushed into a narrow band near -1.
    assert normalized[:1000].std() > 0.3


def test_percentile_clip_raises_on_degenerate_image() -> None:
    image = torch.full((16, 16), 5.0)

    with pytest.raises(ValueError):
        normalize_percentile_clip_to_unit_range(image)


def test_percentile_clip_composes_with_identity_transform() -> None:
    image = torch.randn(8, 8)

    composed = compose([identity_transform, normalize_percentile_clip_to_unit_range])(image)
    direct = normalize_percentile_clip_to_unit_range(image)

    assert torch.equal(composed, direct)


def test_normalize_zero_mean_unit_variance_matches_manual_computation() -> None:
    image = torch.randn(4, 4)

    normalized = normalize_zero_mean_unit_variance(image)

    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(normalized.std(), torch.tensor(1.0), atol=1e-4)
