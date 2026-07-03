import pytest
import torch

from fieldbridge.evaluation.metrics import (
    gradient_mae,
    lpips_metric,
    masked_mae,
    masked_mse,
    masked_psnr,
    normalized_cross_correlation,
    nrmse,
    outside_mask_mean_abs,
    ssim,
)


def test_nrmse_zero_for_identical_tensors() -> None:
    x = torch.rand(2, 1, 8, 8)

    assert torch.isclose(nrmse(x, x), torch.tensor(0.0), atol=1e-6)


def test_nrmse_positive_for_different_tensors() -> None:
    prediction = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)

    assert nrmse(prediction, target) > 0.0


def test_masked_mae_zero_for_identical_tensors() -> None:
    x = torch.rand(2, 1, 8, 8)
    mask = torch.ones_like(x)

    assert torch.isclose(masked_mae(x, x, mask), torch.tensor(0.0), atol=1e-6)


def test_outside_mask_mean_abs_detects_nonzero_background_prediction() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    prediction[:, :, 0, 0] = 3.0
    mask = torch.ones_like(prediction)
    mask[:, :, 0, 0] = 0.0

    value = outside_mask_mean_abs(prediction, mask)

    assert torch.isclose(value, torch.tensor(3.0))


def test_masked_metrics_return_scalar_tensors() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    mask = torch.ones_like(prediction)

    assert masked_mse(prediction, target, mask).ndim == 0
    assert masked_psnr(prediction, target, mask).ndim == 0
    assert normalized_cross_correlation(target, target, mask).ndim == 0
    assert gradient_mae(prediction, target, mask).ndim == 0


def test_masked_metrics_are_stable_for_single_voxel_masks() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    mask = torch.zeros_like(prediction)
    mask[:, :, 2, 2] = 1.0

    assert torch.isfinite(masked_mae(prediction, target, mask))
    assert torch.isfinite(masked_mse(prediction, target, mask))
    assert torch.isfinite(masked_psnr(prediction, target, mask))
    assert torch.isfinite(normalized_cross_correlation(target, target, mask))


def test_masked_metrics_reject_empty_masks() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    mask = torch.zeros_like(prediction)

    with pytest.raises(ValueError, match="mask must select"):
        masked_mae(prediction, target, mask)
    with pytest.raises(ValueError, match="mask must select"):
        normalized_cross_correlation(prediction, target, mask)


def test_ssim_one_for_identical_tensors() -> None:
    x = torch.rand(2, 1, 16, 16)

    value = ssim(x, x)

    assert torch.isclose(value, torch.tensor(1.0), atol=1e-4)


def test_ssim_rejects_non_2d_batches() -> None:
    x = torch.rand(2, 1, 8, 8, 8)

    with pytest.raises(ValueError):
        ssim(x, x)


def test_lpips_metric_requires_optional_dependency() -> None:
    prediction = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)

    with pytest.raises(ImportError):
        lpips_metric(prediction, target)
