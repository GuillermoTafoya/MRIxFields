"""Forward-diffusion (DDPM) schedule and closed-form noising.

T~100 (not the DDM paper's T=1000) — our latent is a small 2D grid, not their 3D
volumes, and this scale hasn't been validated at T=100 specifically; the paper's own
beta endpoints (1e-4 -> 2e-2) are reused as a starting point over the shorter schedule,
not re-derived, so this is a hyperparameter to retune once real training starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


def make_schedule(
    num_timesteps: int = 100, *, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> DiffusionSchedule:
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.

    Returns (x_t, noise) — the noise actually used, so the caller can compute the
    noise-prediction loss against this exact sample rather than a freshly-generated
    (and therefore decorrelated) one.
    """

    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bars = schedule.alpha_bars.to(device=x0.device, dtype=x0.dtype)
    alpha_bar_t = alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise
