import pytest
import torch

from fieldbridge.evaluation.metrics import nrmse
from fieldbridge.training.losses import (
    adversarial_hinge_loss_discriminator,
    adversarial_hinge_loss_generator,
    background_penalty,
    build_lpips_net,
    combined_reconstruction_loss,
    combined_reconstruction_loss_components,
    cycle_consistency_loss,
    gradient_loss,
    identity_loss,
    kl_divergence,
    kl_divergence_free_bits,
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


def test_background_penalty_matches_minus_one_one_target_outside_mask() -> None:
    target = torch.tensor([[[[0.0, -1.0], [0.0, -1.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    matching_prediction = target.detach().clone().requires_grad_(True)
    gray_prediction = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], requires_grad=True)

    matching_loss = background_penalty(matching_prediction, mask, target=target)
    gray_loss = background_penalty(gray_prediction, mask, target=target)

    assert torch.isclose(matching_loss, torch.tensor(0.0))
    assert gray_loss > 0.0
    _assert_finite_and_backprop(gray_loss, gray_prediction)


def test_background_penalty_zero_one_behavior_matches_target_outside_mask() -> None:
    target = torch.tensor([[[[0.5, 0.0], [0.5, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    matching_prediction = target.detach().clone().requires_grad_(True)
    bright_prediction = torch.tensor([[[[0.5, 0.25], [0.5, 0.25]]]], requires_grad=True)

    assert torch.isclose(background_penalty(matching_prediction, mask, target=target), torch.tensor(0.0))
    assert background_penalty(bright_prediction, mask, target=target) > 0.0


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


def test_combined_reconstruction_loss_components_backpropagate() -> None:
    prediction = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.randn(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)

    components = combined_reconstruction_loss_components(prediction, target, mask)

    assert set(components) == {"masked_l1", "gradient", "background", "total"}
    _assert_finite_and_backprop(components["total"], prediction)


def test_adversarial_hinge_losses() -> None:
    real_logits = torch.randn(4, requires_grad=True)
    fake_logits = torch.randn(4, requires_grad=True)

    generator_loss = adversarial_hinge_loss_generator(fake_logits)
    assert torch.isfinite(generator_loss)

    discriminator_loss = adversarial_hinge_loss_discriminator(real_logits, fake_logits)
    _assert_finite_and_backprop(discriminator_loss, real_logits, fake_logits)


def test_lpips_loss_requires_optional_dependency(monkeypatch) -> None:
    # Simulate the dependency being absent instead of relying on the env: Colab//any run
    # that installs the 'perceptual' extra has lpips present, and this test would then fall
    # through to a real VGG forward instead of exercising the ImportError path it is about.
    import sys

    monkeypatch.setitem(sys.modules, "lpips", None)  # makes `import lpips` raise ImportError
    prediction = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)

    with pytest.raises(ImportError):
        lpips_loss(prediction, target)


def test_build_lpips_net_keeps_stdout_clean(monkeypatch, capsys) -> None:
    # lpips.LPIPS prints "Setting up [LPIPS]..." to stdout; under --json that corrupts the
    # machine-readable output. build_lpips_net must redirect that chatter to stderr.
    import sys
    import types

    fake = types.ModuleType("lpips")

    class _FakeLPIPS(torch.nn.Module):
        def __init__(self, net: str = "vgg") -> None:
            super().__init__()
            print("Setting up [LPIPS] perceptual loss: trunk [vgg]")

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a - b).abs().mean()

    fake.LPIPS = _FakeLPIPS
    monkeypatch.setitem(sys.modules, "lpips", fake)

    from fieldbridge.training.losses import build_lpips_net

    net = build_lpips_net(torch.device("cpu"))
    captured = capsys.readouterr()

    assert captured.out == ""  # nothing leaked to stdout
    assert "Setting up [LPIPS]" in captured.err  # redirected to stderr
    assert isinstance(net, _FakeLPIPS)


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


def test_build_lpips_net_freezes_params_but_keeps_input_gradient_path() -> None:
    # LPIPS is a fixed metric, but the VAE's perceptual term still has to backprop through
    # it. Freezing must cut grads to the *net's* params without cutting the path to the
    # input — a silent no-op perceptual loss would train fine and look fine.
    pytest.importorskip("lpips")
    net = build_lpips_net(torch.device("cpu"))

    assert not any(parameter.requires_grad for parameter in net.parameters())
    assert not net.training

    prediction = torch.rand(1, 1, 4, 16, 16, requires_grad=True)
    target = torch.rand(1, 1, 4, 16, 16)

    loss = lpips_loss_3d(prediction, target, net=net, num_slices=2)

    assert loss.requires_grad
    _assert_finite_and_backprop(loss, prediction)
    assert prediction.grad.abs().sum() > 0


# --- free-bits KL (item 4) -----------------------------------------------------------


def test_free_bits_zero_equals_kl_divergence_5d() -> None:
    # free_bits=0 must reproduce the plain kl_divergence term exactly (float order aside),
    # since the per-element KL is >= 0 so the per-channel clamp is a no-op.
    torch.manual_seed(0)
    mean = torch.randn(3, 4, 5, 6, 7, requires_grad=True)
    logvar = torch.randn(3, 4, 5, 6, 7, requires_grad=True)

    fb = kl_divergence_free_bits(mean, logvar, 0.0)
    plain = kl_divergence(mean, logvar)

    assert torch.allclose(fb, plain, atol=1e-5, rtol=1e-5)
    _assert_finite_and_backprop(fb, mean, logvar)


def test_free_bits_zero_equals_kl_divergence_4d() -> None:
    torch.manual_seed(1)
    mean = torch.randn(2, 4, 9, 9, requires_grad=True)
    logvar = torch.randn(2, 4, 9, 9, requires_grad=True)

    assert torch.allclose(kl_divergence_free_bits(mean, logvar, 0.0), kl_divergence(mean, logvar), atol=1e-5, rtol=1e-5)


def test_free_bits_floor_raises_the_term_and_stays_finite() -> None:
    # A large free_bits floors every channel's per-element-mean KL at free_bits, so the term
    # equals free_bits * (spatial elements/channel) * num_channels and exceeds the unclamped KL.
    torch.manual_seed(2)
    mean = 0.05 * torch.randn(2, 4, 8, 8, 8, requires_grad=True)  # near-prior => tiny raw KL
    logvar = torch.zeros(2, 4, 8, 8, 8, requires_grad=True)

    free_bits = 0.5
    fb = kl_divergence_free_bits(mean, logvar, free_bits)
    spatial = 8 * 8 * 8
    channels = 4
    assert torch.isfinite(fb)
    # every channel below the floor => term == free_bits * spatial * channels
    assert torch.isclose(fb, torch.tensor(free_bits * spatial * channels), rtol=1e-4)
    assert fb > kl_divergence(mean, logvar)


def test_free_bits_matches_latent_stats_per_dim_kl_scale() -> None:
    # The clamped quantity is per-channel per-element-mean KL == LatentStatsAccumulator.per_dim_kl,
    # so free_bits is read in the same units as the collapse diagnostics. Check the per-channel
    # mean the function clamps equals the accumulator's per_dim_kl.
    from fieldbridge.training.latent_stats import LatentStatsAccumulator

    torch.manual_seed(3)
    mean = torch.randn(4, 4, 6, 6, 6)
    logvar = torch.randn(4, 4, 6, 6, 6)
    acc = LatentStatsAccumulator(latent_channels=4)
    acc.update(mean, logvar)
    per_dim_kl = acc.compute()["per_dim_kl"]

    kl_elem = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())
    per_channel_mean = kl_elem.mean(dim=(0, 2, 3, 4))
    for a, b in zip(per_channel_mean.tolist(), per_dim_kl):
        assert abs(a - b) < 1e-6
