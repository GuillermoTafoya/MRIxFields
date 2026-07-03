import torch

from fieldbridge.data.degradation import (
    additive_gaussian_noise,
    compose_degradation,
    degradation_strength,
    downsample_then_upsample,
    gaussian_blur,
    intensity_compression,
    low_pass_filter,
    multiplicative_smooth_bias_field,
)
from fieldbridge.data.domains import Domain


def test_downsample_then_upsample_preserves_2d_shape() -> None:
    x = torch.randn(2, 1, 16, 20)

    degraded = downsample_then_upsample(x, scale_factor=0.5)

    assert degraded.shape == x.shape
    assert torch.isfinite(degraded).all()


def test_degradation_functions_preserve_dtype_device_and_do_not_mutate_input() -> None:
    x = torch.randn(1, 2, 12, 14, dtype=torch.float64)
    original = x.clone()
    generator_seed = 13
    functions = [
        lambda value: downsample_then_upsample(value, scale_factor=0.6),
        lambda value: gaussian_blur(value, sigma=0.5),
        lambda value: additive_gaussian_noise(
            value,
            std=0.05,
            generator=torch.Generator().manual_seed(generator_seed),
        ),
        lambda value: multiplicative_smooth_bias_field(
            value,
            strength=0.2,
            generator=torch.Generator().manual_seed(generator_seed),
        ),
        lambda value: intensity_compression(value, amount=0.2),
        lambda value: low_pass_filter(value, cutoff_fraction=0.7),
        lambda value: compose_degradation(
            value,
            strength=0.4,
            generator=torch.Generator().manual_seed(generator_seed),
        ),
    ]

    for function in functions:
        output = function(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.device == x.device
        assert torch.equal(x, original)


def test_compose_degradation_preserves_3d_shape() -> None:
    x = torch.randn(2, 1, 6, 16, 20)
    generator = torch.Generator().manual_seed(3)

    degraded = compose_degradation(x, strength=0.4, generator=generator)

    assert degraded.shape == x.shape
    assert torch.isfinite(degraded).all()


def test_bias_field_broadcasts_across_2d_and_3d_channels() -> None:
    x_2d = torch.ones(2, 3, 16, 18)
    x_3d = torch.ones(2, 3, 6, 16, 18)

    out_2d = multiplicative_smooth_bias_field(
        x_2d,
        strength=0.3,
        generator=torch.Generator().manual_seed(4),
    )
    out_3d = multiplicative_smooth_bias_field(
        x_3d,
        strength=0.3,
        generator=torch.Generator().manual_seed(4),
    )

    assert out_2d.shape == x_2d.shape
    assert out_3d.shape == x_3d.shape
    assert torch.isfinite(out_2d).all()
    assert torch.isfinite(out_3d).all()


def test_additive_noise_uses_provided_generator() -> None:
    x = torch.zeros(1, 1, 8, 8)

    out_a = additive_gaussian_noise(x, std=0.2, generator=torch.Generator().manual_seed(21))
    out_b = additive_gaussian_noise(x, std=0.2, generator=torch.Generator().manual_seed(21))
    out_c = additive_gaussian_noise(x, std=0.2, generator=torch.Generator().manual_seed(22))

    assert torch.allclose(out_a, out_b)
    assert not torch.allclose(out_a, out_c)


def test_compose_degradation_is_deterministic_with_fixed_generator_seed() -> None:
    x = torch.randn(2, 1, 16, 20)
    gen_a = torch.Generator().manual_seed(17)
    gen_b = torch.Generator().manual_seed(17)

    degraded_a = compose_degradation(x, strength=0.6, generator=gen_a)
    degraded_b = compose_degradation(x, strength=0.6, generator=gen_b)

    assert torch.allclose(degraded_a, degraded_b)


def test_stronger_degradation_changes_tensor_more_than_weaker_degradation() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 1, 24, 24)
    weak = compose_degradation(x, strength=0.15, generator=torch.Generator().manual_seed(9))
    strong = compose_degradation(x, strength=0.85, generator=torch.Generator().manual_seed(9))

    weak_delta = torch.mean(torch.abs(weak - x))
    strong_delta = torch.mean(torch.abs(strong - x))

    assert strong_delta > weak_delta


def test_degradation_strength_uses_log_field_ratio() -> None:
    severe = degradation_strength(Domain(7.0, "T1w"), Domain(0.1, "T1w"))
    mild = degradation_strength(Domain(3.0, "T1w"), Domain(1.5, "T1w"))
    no_degrade = degradation_strength(Domain(1.5, "T1w"), Domain(3.0, "T1w"))

    assert 0.0 < mild < severe <= 1.0
    assert no_degrade == 0.0
