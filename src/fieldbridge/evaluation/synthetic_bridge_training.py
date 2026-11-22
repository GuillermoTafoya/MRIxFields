"""CPU-only conditional Brownian bridge training for the synthetic factorial experiment."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import Domain
from fieldbridge.evaluation.stage2_unified import _module_state_sha256
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.stage2_unified import UnifiedStage2Config, _bridge_sample

if TYPE_CHECKING:
    from fieldbridge.evaluation.synthetic_bridge_phantoms import SubjectBatch


SOURCE_DOMAIN = Domain(3.0, "T1w")
TARGET_DOMAINS = (Domain(1.5, "T1w"), Domain(7.0, "T1w"))
FIELD_CODES = {TARGET_DOMAINS[0]: -1.0, TARGET_DOMAINS[1]: 1.0}


def sample_bridge_target(
    source: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    *,
    objective: str,
    config: UnifiedStage2Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare supervisands on exactly the same production Brownian bridge draw."""
    if config.bridge != "schrodinger" or not bool(((0 < t) & (t < 1)).all()):
        raise ValueError("Brownian targets require a Schrodinger bridge and interior times.")
    state, drift = _bridge_sample(source, target, t, config, noise=noise)
    if objective == "forward_drift":
        return state, drift
    if objective != "current_velocity":
        raise ValueError(f"Unknown training objective: {objective}")
    time = t.reshape(-1, *([1] * (source.ndim - 1)))
    # Differentiate the interpolant for fixed source, target and standard normal noise.
    # Conditional expectation given state and time is the probability-flow velocity.
    scale_derivative = config.sigma * (1 - 2 * time) / (2 * torch.sqrt(time * (1 - time)))
    return state, (target - source) + scale_derivative * noise


def batch_indices(
    n_subjects: int, per_field: int, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distinct source IDs and a uniform derangement of each field's same IDs."""
    if not 2 <= per_field <= n_subjects:
        raise ValueError("Each requested field needs at least two distinct training subjects.")
    positions = torch.arange(per_field)
    sources, shuffled = [], []
    for _ in TARGET_DOMAINS:
        source = torch.randperm(n_subjects, generator=generator)[:per_field]
        permutation = torch.randperm(per_field, generator=generator)
        while torch.any(permutation == positions):
            permutation = torch.randperm(per_field, generator=generator)
        sources.append(source)
        shuffled.append(source[permutation])
    return torch.stack(sources), torch.stack(shuffled)


class SyntheticVelocity(BaseTranslator):
    """A local 3D velocity field conditioned on time and the two requested fields."""

    def __init__(self) -> None:
        super().__init__()
        channels = 16
        self.spatial = nn.Conv3d(1, channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(2, channels)
        self.hidden = nn.Conv3d(channels, channels, kernel_size=1)
        self.head = nn.Conv3d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, z: torch.Tensor, source_domain: Sequence[Domain], target_domain: Sequence[Domain],
        t: torch.Tensor,
    ) -> torch.Tensor:
        if len(source_domain) != len(z) or any(label != SOURCE_DOMAIN for label in source_domain):
            raise ValueError("Expected one 3T T1w source label per image.")
        field = z.new_tensor([FIELD_CODES[label] for label in target_domain])
        conditioning = self.condition(torch.stack((t, field), dim=1))
        hidden = F.silu(self.spatial(z) + conditioning[:, :, None, None, None])
        return self.head(F.silu(self.hidden(hidden)))


def train_arm(
    train: SubjectBatch, *, pairing: str, objective: str, seed: int, updates: int,
    per_field: int, sigma: float, time_eps: float, lr: float,
) -> tuple[SyntheticVelocity, dict]:
    """Reset all stochastic inputs for a matched comparison. Use no held-out data."""
    if pairing not in ("paired", "shuffled") or objective not in ("forward_drift", "current_velocity"):
        raise ValueError("Unknown factorial arm.")
    if updates < 1 or sigma <= 0 or lr <= 0 or not 0 < time_eps < 0.5:
        raise ValueError("Invalid Brownian training parameters.")
    source = train.source
    targets = [train.targets[label.field_strength_t] for label in TARGET_DOMAINS]
    if any(x.shape != source.shape or x.device.type != "cpu" or x.dtype != torch.float32
           for x in [source, *targets]) or source.ndim != 5 or source.shape[1] != 1:
        raise ValueError("Expected aligned CPU float32 volumes with shape [N,1,D,H,W].")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = SyntheticVelocity().cpu()
    initial_hash = _module_state_sha256(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cfg = UnifiedStage2Config(bridge="schrodinger", sigma=sigma, time_eps=time_eps)
    batch_size = per_field * len(TARGET_DOMAINS)
    source_labels = [SOURCE_DOMAIN] * batch_size
    target_labels = [label for label in TARGET_DOMAINS for _ in range(per_field)]
    draws, endpoints = hashlib.sha256(), hashlib.sha256()
    total_loss = 0.0
    for _ in range(updates):
        ids, deranged = batch_indices(len(source), per_field, generator)
        target_ids = ids if pairing == "paired" else deranged
        x = source[ids.flatten()]
        y = torch.cat([images[indices] for images, indices in zip(targets, target_ids)])
        t = torch.rand(batch_size, generator=generator).clamp(time_eps, 1 - time_eps)
        noise = torch.randn(x.shape, generator=generator)
        for draw in (ids, deranged, t, noise):
            draws.update(draw.numpy().tobytes())
        endpoints.update(target_ids.numpy().tobytes())
        state, velocity = sample_bridge_target(x, y, t, noise, objective=objective, config=cfg)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(state, source_labels, target_labels, t)
        loss = F.mse_loss(prediction, velocity)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return model.eval().requires_grad_(False), {
        "model": "local Conv3d 3x3x3, 16 channels; time and field projection; SiLU, 1x1x1 SiLU, zero-init head",
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam", "initial_state_sha256": initial_hash,
        "draws_sha256": draws.hexdigest(), "pairing_indices_sha256": endpoints.hexdigest(),
        "final_state_sha256": _module_state_sha256(model), "mean_training_loss": total_loss / updates,
    }
