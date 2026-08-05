"""The official-nRMSE training term must agree with the published evaluator, and must not
diverge on the air patches the stratified crop produces."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fieldbridge.evaluation.mrixfields2026_official import official_task3_nrmse
from fieldbridge.training.losses import nrmse_loss, official_nrmse_loss
from fieldbridge.training.stage1_vae import Stage1VAEConfig, _nrmse_term


def _pair(seed: int, shape: tuple[int, ...], scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.rand(shape, generator=generator) * scale
    prediction = target + 0.02 * torch.randn(shape, generator=generator)
    return prediction, target


def test_matches_published_evaluator_on_a_single_sample() -> None:
    prediction, target = _pair(0, (1, 1, 8, 8, 8), scale=0.3)
    expected = official_task3_nrmse(
        prediction.squeeze().numpy().astype(np.float64),
        target.squeeze().numpy().astype(np.float64),
    )

    value = official_nrmse_loss(prediction, target, rms_floor=1e-6)

    assert float(value) == pytest.approx(expected, rel=1e-5)


def test_is_the_per_sample_mean_not_the_pooled_ratio() -> None:
    prediction, target = _pair(1, (4, 1, 8, 8, 8), scale=0.3)

    value = official_nrmse_loss(prediction, target, rms_floor=1e-6)

    per_sample = [
        official_task3_nrmse(
            prediction[i].squeeze().numpy().astype(np.float64),
            target[i].squeeze().numpy().astype(np.float64),
        )
        for i in range(4)
    ]
    assert float(value) == pytest.approx(float(np.mean(per_sample)), rel=1e-5)


def test_dark_target_is_penalized_far_more_than_a_bright_one_for_equal_absolute_error() -> None:
    """The whole point of the switch: range-normalized nRMSE cannot tell these apart."""

    generator = torch.Generator().manual_seed(2)
    shape = (1, 1, 8, 8, 8)
    error = 0.02 * torch.randn(shape, generator=generator)
    bright = torch.rand(shape, generator=generator) * 0.9 + 0.05
    dark = bright * 0.05

    official_bright = float(official_nrmse_loss(bright + error, bright, rms_floor=1e-6))
    official_dark = float(official_nrmse_loss(dark + error, dark, rms_floor=1e-6))
    range_bright = float(nrmse_loss(bright + error, bright, data_range=1.0))
    range_dark = float(nrmse_loss(dark + error, dark, data_range=1.0))

    assert official_dark > 10 * official_bright
    assert range_dark == pytest.approx(range_bright, rel=1e-6)


def test_air_patch_does_not_diverge_and_the_floor_is_what_bounds_it() -> None:
    air = torch.zeros(2, 1, 8, 8, 8)
    prediction = air + 0.01

    value = official_nrmse_loss(prediction, air, rms_floor=0.02)

    assert torch.isfinite(value)
    # ||p - t|| / (rms_floor * sqrt(N)) == RMSE / rms_floor exactly, once the floor binds.
    assert float(value) == pytest.approx(0.01 / 0.02, rel=1e-4)


def test_gradient_is_finite_at_a_perfect_reconstruction() -> None:
    target = torch.rand(2, 1, 8, 8, 8) * 0.3
    prediction = target.clone().requires_grad_(True)

    official_nrmse_loss(prediction, target, rms_floor=0.02).backward()

    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_rms_floor_is_patch_size_independent() -> None:
    """A floor expressed in RMS units must bind at the same intensity for any patch size."""

    values = [
        float(official_nrmse_loss(torch.full(shape, 0.01), torch.zeros(shape), rms_floor=0.02))
        for shape in ((1, 1, 8, 8, 8), (1, 1, 16, 16, 16), (1, 1, 32, 32, 32))
    ]

    assert values[0] == pytest.approx(values[1], rel=1e-4)
    assert values[1] == pytest.approx(values[2], rel=1e-4)


def test_rejects_a_nonpositive_floor() -> None:
    with pytest.raises(ValueError, match="rms_floor must be positive"):
        official_nrmse_loss(torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4), rms_floor=0.0)


@pytest.mark.parametrize("mode", ["range", "official"])
def test_config_selects_the_term_and_the_default_preserves_run_c(mode: str) -> None:
    prediction, target = _pair(3, (2, 1, 8, 8, 8), scale=0.2)
    cfg = Stage1VAEConfig(nrmse_mode=mode, nrmse_rms_floor=0.02)  # type: ignore[arg-type]

    value = _nrmse_term(prediction, target, cfg)

    expected = (
        official_nrmse_loss(prediction, target, rms_floor=0.02)
        if mode == "official"
        else nrmse_loss(prediction, target, data_range=cfg.data_range)
    )
    assert float(value) == pytest.approx(float(expected), rel=1e-6)
    assert Stage1VAEConfig().nrmse_mode == "range"


def test_config_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="nrmse_mode must be"):
        Stage1VAEConfig(nrmse_mode="l2")  # type: ignore[arg-type]
