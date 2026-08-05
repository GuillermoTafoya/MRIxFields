"""Synthetic exact-zero-background gradient smoke for the Stage-1 v3 Arm-A loss."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.stage1_vae import (
    Stage1VAEConfig,
    _autocast_context,
    _clip_gradients_fail_closed,
    _compute_vae_loss_components,
    _resolve_device,
)
from fieldbridge.utils.seeding import seed_everything


@dataclass(frozen=True, slots=True)
class Stage1GradientSmokeResult:
    """Machine-readable result from one synthetic Arm-A optimizer step."""

    device: str
    precision: str
    patch_size: int
    loss_weights: dict[str, float]
    components: dict[str, float]
    gradient_norm: float
    trainable_parameter_tensors: int
    finite_gradient_parameter_tensors: int
    updated_parameter_tensors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "patch_size": self.patch_size,
            "loss_weights": dict(self.loss_weights),
            "components": dict(self.components),
            "gradient_norm": self.gradient_norm,
            "trainable_parameter_tensors": self.trainable_parameter_tensors,
            "finite_gradient_parameter_tensors": self.finite_gradient_parameter_tensors,
            "updated_parameter_tensors": self.updated_parameter_tensors,
        }


def run_stage1_gradient_smoke(
    config: Mapping[str, Any],
    *,
    device: str | None = None,
    precision: str | None = None,
    patch_size: int = 16,
) -> Stage1GradientSmokeResult:
    """Run backward, fail-closed clipping, and one update on a synthetic 3D brain cube.

    The checked-in Arm-A configuration supplies the scientific loss composition. The
    smoke deliberately uses a reduced-width instance of the same 3D encoder/decoder
    classes so it is fast enough to run before allocating time to private-data training.
    """

    cfg = Stage1VAEConfig.from_mapping(config)
    if cfg.latent_mode != "deterministic":
        raise ValueError("Stage-1 gradient smoke requires the deterministic Arm-A config.")
    resolved_device = _resolve_device(device or cfg.device)
    resolved_precision = precision or cfg.precision
    if resolved_precision not in ("fp32", "bf16"):
        raise ValueError("precision must be 'fp32' or 'bf16'.")
    if patch_size < 8 or patch_size % 4 != 0:
        raise ValueError("patch_size must be at least 8 and divisible by 4.")

    model_mapping = config.get("model", {})
    if not isinstance(model_mapping, Mapping):
        raise ValueError("Config section 'model' must be a mapping.")
    if int(model_mapping.get("spatial_dims", 3)) != 3:
        raise ValueError("Stage-1 gradient smoke requires the 3D Arm-A model.")
    if int(model_mapping.get("domain_conditioning_dim", 0)) != 0:
        raise ValueError("Stage-1 Arm-A gradient smoke must be unconditioned.")

    seed_everything(cfg.seed)
    latent_channels = int(model_mapping.get("latent_channels", 4))
    encoder = KLVAEEncoder(
        in_channels=int(model_mapping.get("in_channels", 1)),
        base_channels=4,
        latent_channels=latent_channels,
        spatial_dims=3,
        activation=str(model_mapping.get("activation", "silu")),
        num_res_blocks=1,
    ).to(resolved_device)
    decoder = KLVAEDecoder(
        out_channels=int(model_mapping.get("in_channels", 1)),
        base_channels=4,
        latent_channels=latent_channels,
        spatial_dims=3,
        activation=str(model_mapping.get("activation", "silu")),
        num_res_blocks=1,
        output_activation=str(model_mapping.get("output_activation", "none")),
        domain_conditioning_dim=0,
    ).to(resolved_device)
    named_parameters = [
        (f"encoder.{name}", parameter)
        for name, parameter in encoder.named_parameters()
    ] + [
        (f"decoder.{name}", parameter)
        for name, parameter in decoder.named_parameters()
    ]
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in named_parameters],
        lr=cfg.lr,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }

    target = torch.zeros(
        1,
        int(model_mapping.get("in_channels", 1)),
        patch_size,
        patch_size,
        patch_size,
        device=resolved_device,
    )
    lower = patch_size // 4
    upper = patch_size - lower
    target[..., lower:upper, lower:upper, lower:upper] = 0.72
    domain = Domain(3.0, "T1w")
    batch = RawBatch(
        image=target,
        source_domain=domain,
        target_domain=domain,
        metadata={"fixture": "synthetic_exact_zero_background"},
    )

    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(resolved_device, resolved_precision):
        components = _compute_vae_loss_components(
            encoder,
            decoder,
            batch,
            cfg,
            lpips_net=None,
            global_step=1,
        )
    components["total"].backward()
    missing = [
        name for name, parameter in named_parameters if parameter.grad is None
    ]
    if missing:
        raise FloatingPointError(
            "Stage-1 gradient smoke found trainable parameters without gradients: "
            + ", ".join(missing[:12])
        )
    finite_count = sum(
        bool(torch.isfinite(parameter.grad).all())
        for _, parameter in named_parameters
        if parameter.grad is not None
    )
    gradient_norm = _clip_gradients_fail_closed(
        named_parameters,
        cfg.grad_clip_norm,
    )
    optimizer.step()
    nonfinite_parameters = [
        name
        for name, parameter in named_parameters
        if not bool(torch.isfinite(parameter).all())
    ]
    if nonfinite_parameters:
        raise FloatingPointError(
            "Stage-1 gradient smoke produced non-finite updated parameters: "
            + ", ".join(nonfinite_parameters[:12])
        )
    updated_count = sum(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in named_parameters
    )
    if updated_count == 0:
        raise FloatingPointError(
            "Stage-1 gradient smoke completed without updating any parameter tensor."
        )

    return Stage1GradientSmokeResult(
        device=str(resolved_device),
        precision=str(resolved_precision),
        patch_size=patch_size,
        loss_weights={
            name: float(value) for name, value in cfg.loss_weights.items()
        },
        components={
            name: float(value.detach().cpu())
            for name, value in components.items()
        },
        gradient_norm=float(gradient_norm.detach().cpu()),
        trainable_parameter_tensors=len(named_parameters),
        finite_gradient_parameter_tensors=finite_count,
        updated_parameter_tensors=updated_count,
    )
