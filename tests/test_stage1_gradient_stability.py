from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from fieldbridge.config import load_yaml_config
from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.domains import Domain
from fieldbridge.training.losses import ssim_loss
from fieldbridge.training.ssim import stable_training_ssim3d
from fieldbridge.training.stage1_gradient_smoke import (
    run_stage1_gradient_smoke,
)
from fieldbridge.training.stage1_vae import (
    Stage1VAEConfig,
    _clip_gradients_fail_closed,
    _compute_vae_loss_components,
)


def _ssim_fixture(case: str) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(1, 1, 12, 12, 12)
    generator = torch.Generator().manual_seed(20260724)
    if case == "all_zero_background":
        prediction = 0.03 * torch.randn(target.shape, generator=generator)
    elif case == "foreground_cube":
        target[..., 3:9, 3:9, 3:9] = 0.72
        prediction = (
            target + 0.04 * torch.randn(target.shape, generator=generator)
        ).clamp(-0.1, 1.1)
    elif case == "identical":
        target[..., 3:9, 3:9, 3:9] = 0.72
        prediction = target.clone()
    elif case == "constant_local_windows":
        target.fill_(0.4)
        prediction = torch.full_like(target, 0.55)
    else:  # pragma: no cover - test definition guard
        raise AssertionError(case)
    return prediction.requires_grad_(True), target


@pytest.mark.parametrize(
    "case",
    (
        "all_zero_background",
        "foreground_cube",
        "identical",
        "constant_local_windows",
    ),
)
@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_training_ssim_exact_zero_windows_have_finite_backward(
    case: str, precision: str
) -> None:
    prediction, target = _ssim_fixture(case)
    context = (
        nullcontext()
        if precision == "fp32"
        else torch.autocast("cpu", dtype=torch.bfloat16)
    )

    with context:
        similarity = stable_training_ssim3d(
            prediction,
            target,
            data_range=1.0,
            window_size=7,
        )
        loss = ssim_loss(
            prediction,
            target,
            data_range=1.0,
            window_size=7,
        )
    loss.backward()

    assert torch.isfinite(similarity)
    assert -1.0 <= float(similarity.detach()) <= 1.0
    assert torch.isfinite(loss)
    assert 0.0 <= float(loss.detach()) <= 2.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    if case == "identical":
        assert float(similarity.detach()) == pytest.approx(1.0, abs=1e-6)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)


class _ExactIdentityEncoder(nn.Module):
    latent_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode_dist(self, image, domain):
        del domain
        mean = image * self.scale
        return mean, torch.zeros_like(mean)


class _IdentityDecoder(nn.Module):
    def decode(self, latent, domain):
        del domain
        return latent


def test_singular_zero_weight_nrmse_diagnostic_is_detached_from_total() -> None:
    target = torch.zeros(1, 1, 8, 8, 8)
    target[..., 2:6, 2:6, 2:6] = 0.7
    domain = Domain(3.0, "T1w")
    encoder = _ExactIdentityEncoder()
    cfg = Stage1VAEConfig(
        latent_mode="deterministic",
        loss_weights={
            "masked_l1": 1.0,
            "background": 0.0,
            "ssim": 0.0,
            "nrmse": 0.0,
            "lpips": 0.0,
            "kl": 0.0,
        },
    )

    components = _compute_vae_loss_components(
        encoder,
        _IdentityDecoder(),
        RawBatch(target, domain, domain),
        cfg,
        lpips_net=None,
    )
    assert float(components["nrmse"]) == 0.0
    assert components["nrmse"].requires_grad is False

    components["total"].backward()

    assert encoder.scale.grad is not None
    assert torch.isfinite(encoder.scale.grad)
    assert float(encoder.scale.grad) == 0.0


@pytest.mark.parametrize("precision", ("fp32", "bf16"))
def test_full_arm_a_gradient_smoke_updates_with_all_parameter_gradients(
    precision: str,
) -> None:
    config = load_yaml_config(
        "configs/experiment/stage1_ae_v3_joint_domain.yaml"
    )

    result = run_stage1_gradient_smoke(
        config,
        device="cpu",
        precision=precision,
        patch_size=16,
    )

    assert result.device == "cpu"
    assert result.precision == precision
    assert result.loss_weights == Stage1VAEConfig.from_mapping(config).loss_weights
    assert result.finite_gradient_parameter_tensors == (
        result.trainable_parameter_tensors
    )
    assert result.updated_parameter_tensors > 0
    assert torch.isfinite(torch.tensor(result.gradient_norm))
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in result.components.values()
    )


def test_gradient_failure_names_nonfinite_parameters() -> None:
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([float("nan"), float("inf")])

    with pytest.raises(
        FloatingPointError,
        match=r"encoder\.stem\.weight\(nan=1,inf=1\)",
    ):
        _clip_gradients_fail_closed(
            [("encoder.stem.weight", parameter)],
            1.0,
        )
