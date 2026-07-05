import pytest
import torch

from fieldbridge.evaluation.metrics import nrmse
from fieldbridge.training.losses import (
    adversarial_hinge_loss_discriminator,
    adversarial_hinge_loss_generator,
    background_penalty,
    combined_reconstruction_loss,
    cycle_consistency_loss,
    gradient_loss,
    identity_loss,
    kl_divergence,
    lpips_loss,
    lpips_loss_3d,
    masked_l1_loss,
    nrmse_loss,
    ssim_loss,
    synthseg_inloss_stub,
    transport_cost_loss,
)


def _assert_finite_and_backprop(loss: torch.Tensor, *leaves: torch.Tensor) -> None:
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    for leaf in leaves:
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()


def test_kl_divergence_zero_for_standard_normal() -> None:
    mean = torch.zeros(2, 4, requires_grad=True)
    logvar = torch.zeros(2, 4, requires_grad=True)

    loss = kl_divergence(mean, logvar)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    _assert_finite_and_backprop(loss, mean, logvar)


def test_transport_cost_loss() -> None:
    z_source = torch.randn(2, 4, 8, 8, requires_grad=True)
    z_translated = z_source.detach().clone().requires_grad_(True)

    loss = transport_cost_loss(z_source, z_translated)

    _assert_finite_and_backprop(loss, z_source, z_translated)


def test_cycle_consistency_loss() -> None:
    x = torch.randn(2, 1, 8, 8, requires_grad=True)
    x_cycled = torch.randn(2, 1, 8, 8, requires_grad=True)

    loss = cycle_consistency_loss(x, x_cycled)

    _assert_finite_and_backprop(loss, x, x_cycled)


def test_identity_loss() -> None:
    x = torch.randn(2, 1, 8, 8, requires_grad=True)
    x_identity_output = torch.randn(2, 1, 8, 8, requires_grad=True)

    loss = identity_loss(x, x_identity_output)

    _assert_finite_and_backprop(loss, x, x_identity_output)


def test_masked_l1_loss_ignores_outside_mask_voxels() -> None:
    prediction = torch.tensor([[[[1.0, 100.0], [3.0, 100.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    loss = masked_l1_loss(prediction, target, mask)

    assert torch.isclose(loss, torch.tensor(1.5))
    _assert_finite_and_backprop(loss, prediction)


def test_background_penalty_penalizes_outside_mask_prediction() -> None:
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    loss = background_penalty(prediction, mask)

    assert torch.isclose(loss, torch.tensor(3.0))
    _assert_finite_and_backprop(loss, prediction)


def test_gradient_loss_returns_nonnegative_scalar() -> None:
    prediction = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.randn(1, 1, 4, 4)

    loss = gradient_loss(prediction, target)

    assert loss.ndim == 0
    assert loss.item() >= 0.0
    _assert_finite_and_backprop(loss, prediction)


def test_combined_reconstruction_loss_backpropagates() -> None:
    prediction = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.randn(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)

    loss = combined_reconstruction_loss(prediction, target, mask)

    _assert_finite_and_backprop(loss, prediction)


def test_adversarial_hinge_losses() -> None:
    real_logits = torch.randn(4, requires_grad=True)
    fake_logits = torch.randn(4, requires_grad=True)

    generator_loss = adversarial_hinge_loss_generator(fake_logits)
    assert torch.isfinite(generator_loss)

    discriminator_loss = adversarial_hinge_loss_discriminator(real_logits, fake_logits)
    _assert_finite_and_backprop(discriminator_loss, real_logits, fake_logits)


def test_lpips_loss_requires_optional_dependency() -> None:
    prediction = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)

    with pytest.raises(ImportError):
        lpips_loss(prediction, target)


def test_lpips_loss_3d_rejects_non_5d() -> None:
    prediction = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)

    with pytest.raises(ValueError):
        lpips_loss_3d(prediction, target)


def test_lpips_loss_3d_finite_scalar() -> None:
    pytest.importorskip("lpips")
    prediction = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
    target = torch.randn(2, 1, 16, 16, 16)

    loss = lpips_loss_3d(prediction, target, num_slices=4)

    _assert_finite_and_backprop(loss, prediction)


def test_ssim_loss_dispatches_to_3d_for_volumes() -> None:
    x = torch.rand(2, 1, 8, 8, 8, requires_grad=True)

    loss = ssim_loss(x, x.detach())

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)
    _assert_finite_and_backprop(loss, x)


def test_synthseg_inloss_stub_fails_explicitly() -> None:
    with pytest.raises(NotImplementedError):
        synthseg_inloss_stub()


def test_ssim_loss_zero_for_identical_tensors() -> None:
    x = torch.rand(2, 1, 16, 16, requires_grad=True)

    loss = ssim_loss(x, x.detach())

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)
    _assert_finite_and_backprop(loss, x)


def test_ssim_loss_positive_for_different_tensors() -> None:
    prediction = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)

    loss = ssim_loss(prediction, target)

    assert loss > 0.0
    _assert_finite_and_backprop(loss, prediction)


def test_nrmse_loss_matches_evaluation_metric() -> None:
    prediction = torch.rand(2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 8, 8)

    loss = nrmse_loss(prediction, target)
    expected = nrmse(prediction.detach(), target)

    assert torch.isclose(loss, expected)
    _assert_finite_and_backprop(loss, prediction)
