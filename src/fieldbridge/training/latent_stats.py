"""Posterior-collapse diagnostics for the Etapa 1 KL-VAE latent.

The training loss is one scalar; it cannot tell "the KL term is small because the model
found a tight, informative posterior" apart from "the KL term is small because the
posterior collapsed to the prior and the latent carries nothing". These statistics do,
per latent *channel*:

* **per-dim std of the posterior mean** — how much a channel actually varies across
  samples/voxels. A collapsed channel sits at ~0 everywhere, so its std -> 0. Stage 2 does
  optimal transport in this latent and assumes it is standardized (std ~ 1 per dim), so
  this is also the readout for "is the latent close to unit variance".
* **per-dim KL** — mean KL contribution of each channel to N(0, I). A dead channel
  contributes ~0.
* **active units** — count of channels whose per-dim KL exceeds a small threshold. The
  standard collapse headline number: `active_units < num_dims` means some channels are
  dead.

Accumulated online (running sums per channel) so it works over a whole validation pass
without materializing every latent. Reused by both the training loop (per-epoch) and the
offline checkpoint evaluation, so the numbers are defined once and are comparable.
"""

from __future__ import annotations

from typing import Any

import torch

# Default per-dim KL below which a latent channel is called "dead". Small and absolute:
# a live channel on this VAE contributes O(0.1-1) nats/dim, a collapsed one ~1e-3. Not a
# swept hyperparameter — a legibility threshold for the "active units" count.
DEFAULT_ACTIVE_KL_THRESHOLD = 0.01


class LatentStatsAccumulator:
    """Running per-channel latent statistics over one or more (mean, logvar) batches.

    All reductions keep the channel axis (dim=1) and collapse batch + spatial, so the
    result is one number per latent channel regardless of 2D vs 3D. Sums are kept in
    float64 because a validation pass accumulates over millions of voxels and a float32
    sum-of-squares loses precision at that scale.
    """

    def __init__(self, latent_channels: int) -> None:
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}.")
        self.latent_channels = int(latent_channels)
        self._count = 0  # elements accumulated *per channel*
        self._sum = torch.zeros(self.latent_channels, dtype=torch.float64)
        self._sumsq = torch.zeros(self.latent_channels, dtype=torch.float64)
        self._kl_sum = torch.zeros(self.latent_channels, dtype=torch.float64)

    @torch.no_grad()
    def update(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        if mean.shape != logvar.shape:
            raise ValueError(
                f"mean and logvar must share shape; got {tuple(mean.shape)} and {tuple(logvar.shape)}."
            )
        if mean.ndim < 2 or int(mean.shape[1]) != self.latent_channels:
            raise ValueError(
                f"expected a tensor with {self.latent_channels} channels at dim 1, got shape {tuple(mean.shape)}."
            )
        m = mean.detach().to(torch.float64)
        lv = logvar.detach().to(torch.float64)
        reduce_dims = tuple(d for d in range(m.ndim) if d != 1)
        # Per-element KL(N(mean, exp(logvar)) || N(0, I)); the same closed form the training
        # kl_divergence term sums over — kept per-element here so we can average per channel.
        kl_elem = -0.5 * (1.0 + lv - m.pow(2) - lv.exp())
        self._count += m.numel() // self.latent_channels
        self._sum += m.sum(dim=reduce_dims).cpu()
        self._sumsq += m.pow(2).sum(dim=reduce_dims).cpu()
        self._kl_sum += kl_elem.sum(dim=reduce_dims).cpu()

    def compute(self, *, active_threshold: float = DEFAULT_ACTIVE_KL_THRESHOLD) -> dict[str, Any]:
        count = max(self._count, 1)
        mean_c = self._sum / count
        var_c = (self._sumsq / count - mean_c.pow(2)).clamp_min(0.0)
        std_c = var_c.sqrt()
        kl_c = self._kl_sum / count
        active_mask = kl_c > active_threshold
        total_count = count * self.latent_channels
        global_mean = float(self._sum.sum() / total_count)
        global_var = float((self._sumsq.sum() / total_count - global_mean**2))
        global_std = float(max(global_var, 0.0) ** 0.5)
        return {
            "num_dims": self.latent_channels,
            "active_units": int(active_mask.sum()),
            "active_threshold": float(active_threshold),
            "dead_units": int((~active_mask).sum()),
            "per_dim_std": [float(v) for v in std_c],
            "per_dim_kl": [float(v) for v in kl_c],
            "global_std": global_std,
            "mean_per_dim_std": float(std_c.mean()),
            "min_per_dim_std": float(std_c.min()),
            "max_per_dim_std": float(std_c.max()),
            "mean_per_dim_kl": float(kl_c.mean()),
            "elements_per_dim": int(self._count),
        }


def summarize_latent_stats(stats: dict[str, Any]) -> str:
    """One compact human-readable line for the training log."""

    return (
        f"active_units={stats['active_units']}/{stats['num_dims']} "
        f"global_std={stats['global_std']:.3f} "
        f"per_dim_std[min={stats['min_per_dim_std']:.3f} mean={stats['mean_per_dim_std']:.3f} "
        f"max={stats['max_per_dim_std']:.3f}] mean_dim_kl={stats['mean_per_dim_kl']:.4f}"
    )
