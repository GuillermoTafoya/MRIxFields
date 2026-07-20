import pytest
import torch

from fieldbridge.evaluation.metrics import lpips_metric, nrmse, ssim, ssim3d


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


def test_ssim3d_one_for_identical_volumes() -> None:
    x = torch.rand(2, 1, 16, 16, 16)

    value = ssim3d(x, x)

    assert torch.isclose(value, torch.tensor(1.0), atol=1e-4)


def test_ssim3d_rejects_non_5d_batches() -> None:
    x = torch.rand(2, 1, 16, 16)

    with pytest.raises(ValueError):
        ssim3d(x, x)


def test_lpips_metric_requires_optional_dependency(monkeypatch) -> None:
    # Simulate the dependency being absent instead of relying on the env: any run that
    # installs the 'perceptual' extra (Colab does) has lpips present, and this test would
    # then fall through to a real VGG forward instead of the ImportError path it is about.
    import sys

    monkeypatch.setitem(sys.modules, "lpips", None)  # makes `import lpips` raise ImportError
    prediction = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)

    with pytest.raises(ImportError):
        lpips_metric(prediction, target)
