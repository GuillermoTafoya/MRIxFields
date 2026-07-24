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

from dataclasses import dataclass
from typing import Any

import torch

# Default per-dim KL below which a latent channel is called "dead". Small and absolute:
# a live channel on this VAE contributes O(0.1-1) nats/dim, a collapsed one ~1e-3. Not a
# swept hyperparameter — a legibility threshold for the "active units" count.
DEFAULT_ACTIVE_KL_THRESHOLD = 0.01
DEFAULT_INPUT_DEPENDENCE_THRESHOLD = 0.01


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

    def compute(
        self,
        *,
        active_threshold: float = DEFAULT_ACTIVE_KL_THRESHOLD,
        active_std_threshold: float = 0.0,
        activity_rule: str = "kl",
    ) -> dict[str, Any]:
        count = max(self._count, 1)
        mean_c = self._sum / count
        var_c = (self._sumsq / count - mean_c.pow(2)).clamp_min(0.0)
        std_c = var_c.sqrt()
        kl_c = self._kl_sum / count
        kl_active = kl_c > active_threshold
        std_active = std_c > active_std_threshold
        if activity_rule == "kl":
            active_mask = kl_active
        elif activity_rule == "std":
            active_mask = std_active
        elif activity_rule == "std_and_kl":
            active_mask = std_active & kl_active
        else:
            raise ValueError(
                "activity_rule must be 'kl', 'std', or 'std_and_kl', "
                f"got {activity_rule!r}."
            )
        total_count = count * self.latent_channels
        global_mean = float(self._sum.sum() / total_count)
        global_var = float((self._sumsq.sum() / total_count - global_mean**2))
        global_std = float(max(global_var, 0.0) ** 0.5)
        return {
            "num_dims": self.latent_channels,
            "active_units": int(active_mask.sum()),
            "active_threshold": float(active_threshold),
            "active_std_threshold": float(active_std_threshold),
            "activity_rule": activity_rule,
            "dead_units": int((~active_mask).sum()),
            "per_dim_std": [float(v) for v in std_c],
            "per_dim_kl": [float(v) for v in kl_c],
            "global_std": global_std,
            "global_mean": global_mean,
            "mean_per_dim_std": float(std_c.mean()),
            "min_per_dim_std": float(std_c.min()),
            "max_per_dim_std": float(std_c.max()),
            "mean_per_dim_kl": float(kl_c.mean()),
            "elements_per_dim": int(self._count),
        }


@dataclass
class _VolumeLatentEvidence:
    """Patch accumulator for one validation volume."""

    latent_channels: int
    count_per_channel: int = 0
    patches: int = 0
    channel_sum: torch.Tensor | None = None
    channel_sumsq: torch.Tensor | None = None
    channel_kl_sum: torch.Tensor | None = None

    def update(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        if mean.shape != logvar.shape or mean.ndim < 3 or int(mean.shape[0]) != 1:
            raise ValueError(
                "Domain-balanced latent evidence expects equal-shape single-sample "
                f"(1,C,...) tensors; got {tuple(mean.shape)} and {tuple(logvar.shape)}."
            )
        if int(mean.shape[1]) != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {int(mean.shape[1])}."
            )
        m = mean.detach().to(device="cpu", dtype=torch.float64)
        lv = logvar.detach().to(device="cpu", dtype=torch.float64)
        reduce_dims = tuple(d for d in range(m.ndim) if d != 1)
        channel_sum = m.sum(dim=reduce_dims)
        channel_sumsq = m.square().sum(dim=reduce_dims)
        kl_sum = (
            -0.5 * (1.0 + lv - m.square() - lv.exp())
        ).sum(dim=reduce_dims)
        self.channel_sum = (
            channel_sum
            if self.channel_sum is None
            else self.channel_sum + channel_sum
        )
        self.channel_sumsq = (
            channel_sumsq
            if self.channel_sumsq is None
            else self.channel_sumsq + channel_sumsq
        )
        self.channel_kl_sum = (
            kl_sum
            if self.channel_kl_sum is None
            else self.channel_kl_sum + kl_sum
        )
        self.count_per_channel += m.numel() // self.latent_channels
        self.patches += 1

    def compute(self) -> dict[str, Any]:
        if (
            self.count_per_channel <= 0
            or self.patches <= 0
            or self.channel_sum is None
            or self.channel_sumsq is None
            or self.channel_kl_sum is None
        ):
            raise ValueError("Cannot compute empty volume latent evidence.")
        count = float(self.count_per_channel)
        channel_mean = self.channel_sum / count
        return {
            "per_channel_mean": channel_mean,
            "per_channel_second_moment": self.channel_sumsq / count,
            "per_channel_raw_kl": self.channel_kl_sum / count,
            "patches": int(self.patches),
        }


class DomainBalancedLatentStatsAccumulator:
    """Latent-health evidence with volume and equal-domain macro weighting.

    Patch evidence is first reduced within each volume.  Volume evidence is averaged
    equally within each field-by-contrast domain, then all domains are macro-averaged.

    ``input_dependence`` is the standard deviation, in posterior-mean latent units, of
    each volume's channel-wise mean posterior from its domain mean. It is deterministic
    and zero for a fixed spatial template emitted for every input, even when that
    template has large spatial standard deviation or raw KL.
    """

    def __init__(self, latent_channels: int) -> None:
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")
        self.latent_channels = int(latent_channels)
        self._volumes: dict[
            str, dict[str, _VolumeLatentEvidence]
        ] = {}

    @torch.no_grad()
    def update(
        self,
        *,
        domain: str,
        volume_id: str,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> None:
        domain_volumes = self._volumes.setdefault(str(domain), {})
        accumulator = domain_volumes.setdefault(
            str(volume_id), _VolumeLatentEvidence(self.latent_channels)
        )
        accumulator.update(mean, logvar)

    def compute(
        self,
        *,
        active_kl_threshold: float,
        active_std_threshold: float,
        input_dependence_threshold: float = DEFAULT_INPUT_DEPENDENCE_THRESHOLD,
        require_raw_kl: bool,
    ) -> dict[str, Any]:
        if not self._volumes:
            return {}
        per_domain: dict[str, dict[str, Any]] = {}
        for domain, volumes in sorted(self._volumes.items()):
            computed = [volume.compute() for volume in volumes.values()]
            channel_means = torch.stack(
                [item["per_channel_mean"] for item in computed], dim=0
            )
            channel_second_moments = torch.stack(
                [item["per_channel_second_moment"] for item in computed], dim=0
            )
            domain_channel_mean = channel_means.mean(dim=0)
            std = (
                channel_second_moments.mean(dim=0)
                - domain_channel_mean.square()
            ).clamp_min(0.0).sqrt()
            raw_kl = torch.stack(
                [item["per_channel_raw_kl"] for item in computed], dim=0
            ).mean(dim=0)
            if int(channel_means.shape[0]) < 2:
                input_dependence = torch.zeros(
                    self.latent_channels, dtype=torch.float64
                )
            else:
                input_dependence = (
                    channel_means - domain_channel_mean.unsqueeze(0)
                ).square().mean(dim=0).sqrt()
            domain_active = (
                (std > float(active_std_threshold))
                & (input_dependence > float(input_dependence_threshold))
            )
            if require_raw_kl:
                domain_active &= raw_kl > float(active_kl_threshold)
            per_domain[domain] = {
                "num_volumes": len(computed),
                "num_patches": sum(int(item["patches"]) for item in computed),
                "per_dim_std": [float(value) for value in std],
                "per_dim_raw_kl": [float(value) for value in raw_kl],
                "per_dim_input_dependence": [
                    float(value) for value in input_dependence
                ],
                "active_mask": [bool(value) for value in domain_active],
            }

        macro_std = torch.stack(
            [
                torch.tensor(value["per_dim_std"], dtype=torch.float64)
                for value in per_domain.values()
            ]
        ).mean(dim=0)
        macro_kl = torch.stack(
            [
                torch.tensor(value["per_dim_raw_kl"], dtype=torch.float64)
                for value in per_domain.values()
            ]
        ).mean(dim=0)
        macro_input = torch.stack(
            [
                torch.tensor(
                    value["per_dim_input_dependence"], dtype=torch.float64
                )
                for value in per_domain.values()
            ]
        ).mean(dim=0)
        active_mask = (
            (macro_std > float(active_std_threshold))
            & (macro_input > float(input_dependence_threshold))
        )
        if require_raw_kl:
            active_mask &= macro_kl > float(active_kl_threshold)
        rule = (
            "input_dependence_and_std_and_raw_kl"
            if require_raw_kl
            else "input_dependence_and_std"
        )
        return {
            "num_dims": self.latent_channels,
            "active_units": int(active_mask.sum()),
            "dead_units": int((~active_mask).sum()),
            "active_mask": [bool(value) for value in active_mask],
            "activity_rule": rule,
            "active_threshold": float(active_kl_threshold),
            "active_std_threshold": float(active_std_threshold),
            "input_dependence_threshold": float(input_dependence_threshold),
            "input_dependence_units": "posterior_mean_latent_rms",
            "aggregation": "patches_to_volumes_to_domains_equal_domain_macro",
            "per_dim_std": [float(value) for value in macro_std],
            "per_dim_kl": [float(value) for value in macro_kl],
            "per_dim_raw_kl": [float(value) for value in macro_kl],
            "per_dim_input_dependence": [
                float(value) for value in macro_input
            ],
            "mean_per_dim_std": float(macro_std.mean()),
            "global_std": float(macro_std.mean()),
            "global_mean": 0.0,
            "min_per_dim_std": float(macro_std.min()),
            "max_per_dim_std": float(macro_std.max()),
            "mean_per_dim_kl": float(macro_kl.mean()),
            "mean_input_dependence": float(macro_input.mean()),
            "num_domains": len(per_domain),
            "num_volumes": sum(
                int(value["num_volumes"]) for value in per_domain.values()
            ),
            "per_domain": per_domain,
        }


def summarize_latent_stats(stats: dict[str, Any]) -> str:
    """One compact human-readable line for the training log."""

    return (
        f"active_units={stats['active_units']}/{stats['num_dims']} "
        f"global_std={stats['global_std']:.3f} "
        f"per_dim_std[min={stats['min_per_dim_std']:.3f} mean={stats['mean_per_dim_std']:.3f} "
        f"max={stats['max_per_dim_std']:.3f}] mean_dim_kl={stats['mean_per_dim_kl']:.4f}"
    )
