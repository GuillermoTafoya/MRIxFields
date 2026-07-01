import pytest
import torch

from clbfield.evaluation.metrics import lpips_metric, nrmse, ssim


def test_nrmse_zero_for_identical_tensors() -> None:
    x = torch.rand(2, 1, 8, 8)

    assert torch.isclose(nrmse(x, x), torch.tensor(0.0), atol=1e-6)


def test_nrmse_positive_for_different_tensors() -> None:
    prediction = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)

    assert nrmse(prediction, target) > 0.0


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
