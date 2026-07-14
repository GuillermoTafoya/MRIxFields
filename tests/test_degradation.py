import torch

from fieldbridge.data.degradation import (
    additive_gaussian_noise,
    compose_degradation,
    degradation_strength,
    downsample_then_upsample,
    gaussian_blur,
)
from fieldbridge.data.domains import Domain


def test_degradation_functions_preserve_shape_dtype_and_input() -> None:
    x = torch.rand(1, 1, 12, 14)
    original = x.clone()

    outputs = [
        downsample_then_upsample(x, scale_factor=0.5),
        gaussian_blur(x, sigma=0.4),
        additive_gaussian_noise(x, std=0.05, generator=torch.Generator().manual_seed(1)),
        compose_degradation(x, strength=0.4, generator=torch.Generator().manual_seed(1)),
    ]

    assert torch.equal(x, original)
    assert all(output.shape == x.shape for output in outputs)
    assert all(output.dtype == x.dtype for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_degradation_is_deterministic_with_fixed_generator() -> None:
    x = torch.rand(1, 1, 16, 16)

    first = compose_degradation(x, strength=0.6, generator=torch.Generator().manual_seed(9))
    second = compose_degradation(x, strength=0.6, generator=torch.Generator().manual_seed(9))

    assert torch.allclose(first, second)


def test_degradation_strength_uses_field_ratio() -> None:
    severe = degradation_strength(Domain(7.0, "T2-FLAIR"), Domain(0.1, "T2-FLAIR"))
    mild = degradation_strength(Domain(3.0, "T2-FLAIR"), Domain(1.5, "T2-FLAIR"))
    none = degradation_strength(Domain(1.5, "T2-FLAIR"), Domain(3.0, "T2-FLAIR"))

    assert 0.0 < mild < severe <= 1.0
    assert none == 0.0
