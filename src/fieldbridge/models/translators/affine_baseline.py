"""Closed-form per-channel affine latent baseline: ``z_t ~= a_c * z_s + b_c``.

This is NOT a trained model. It is the cheapest thing that can reproduce what the v2
transport was diagnosed as having learned — a per-field intensity rescaling — so that any
learned model has to beat it before "it moved the intensities correctly" counts as a result.

Why moment matching and not least squares. Least squares needs correspondences, and the
retrospective pool is unpaired: there is no ``(z_s, z_t)`` of the same anatomy to regress.
What the pool does give is each domain's per-channel marginal. Matching those marginals,

    a_c = sigma_t[c] / sigma_s[c]        b_c = mu_t[c] - a_c * mu_s[c]

is the exact 1-D optimal-transport map between the two Gaussians with those moments, and it
is the unique affine map that carries the source marginal onto the target marginal. So it is
the closed-form unpaired analogue of the paired least-squares fit, not an approximation of
it. :func:`fit_paired_affine` provides the genuine paired least-squares coefficients from the
travellers, which upper-bounds what *any* per-channel affine can do and is used as a
diagnostic reference, never as the baseline itself (it has seen paired supervision).

Two moment tables are fitted in the same pass and both are kept:

- ``all``        — every latent voxel, background included. This is the naive baseline and
                   the one a network conditioned on ``log(f_t/f_s)`` gets for free.
- ``foreground`` — only latent voxels whose across-channel norm exceeds a per-volume
                   percentile. Background is near-constant across field strengths, so it
                   drags the ``all`` moments toward "do nothing"; the foreground table is the
                   stronger baseline and the fairer bar.

Latents are handled in RAW bank space (not standardized). Standardization is itself a
per-channel affine, so a raw-space affine is the same map either way, and raw is the space
that actually gets decoded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from fieldbridge.data.domains import Domain

MomentSpace = Literal["all", "foreground"]
MOMENT_SPACES: tuple[MomentSpace, ...] = ("all", "foreground")
AFFINE_BASELINE_CONTRACT_VERSION = "affine-latent-baseline-v1"


@dataclass(frozen=True, slots=True)
class DomainMoments:
    """Per-channel first and second moments of one domain's latents."""

    mean: torch.Tensor  # (C,)
    std: torch.Tensor  # (C,)
    voxels: int
    volumes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "voxels": int(self.voxels),
            "volumes": int(self.volumes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainMoments":
        return cls(
            mean=torch.tensor([float(v) for v in data["mean"]], dtype=torch.float32),
            std=torch.tensor([float(v) for v in data["std"]], dtype=torch.float32),
            voxels=int(data["voxels"]),
            volumes=int(data["volumes"]),
        )


class _MomentAccumulator:
    """Streaming per-channel mean/std in float64 (1.4e9 voxels per channel overflows f32)."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._sum = torch.zeros(channels, dtype=torch.float64)
        self._sumsq = torch.zeros(channels, dtype=torch.float64)
        self._count = 0
        self._volumes = 0

    def update(self, latent: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """``latent`` is (C, x, y, z); ``mask`` is a broadcastable (x, y, z) boolean."""

        flat = latent.reshape(self.channels, -1).to(torch.float64)
        if mask is not None:
            selected = mask.reshape(-1)
            flat = flat[:, selected]
        if flat.shape[1] == 0:
            return
        self._sum += flat.sum(dim=1)
        self._sumsq += flat.square().sum(dim=1)
        self._count += flat.shape[1]
        self._volumes += 1

    def compute(self, *, eps: float) -> DomainMoments:
        if self._count == 0:
            raise ValueError("No voxels accumulated for these domain moments.")
        mean = self._sum / self._count
        var = (self._sumsq / self._count) - mean.square()
        std = var.clamp_min(0.0).sqrt().clamp_min(eps)
        return DomainMoments(
            mean=mean.to(torch.float32),
            std=std.to(torch.float32),
            voxels=int(self._count),
            volumes=int(self._volumes),
        )


def latent_foreground_mask(latent: torch.Tensor, *, percentile: float) -> torch.Tensor:
    """Boolean (x, y, z) mask of the top ``100 - percentile``% latent voxels by channel norm.

    The VAE latent of image background is near-constant but not zero, so a ``> 0`` threshold
    does not separate it. A per-volume quantile of the across-channel norm does, without
    needing the image or a brain mask.
    """

    if latent.ndim != 4:
        raise ValueError(f"latent_foreground_mask expects (C, x, y, z), got {tuple(latent.shape)}.")
    if not 0.0 <= percentile < 100.0:
        raise ValueError(f"percentile must be in [0, 100), got {percentile}.")
    norm = latent.to(torch.float32).square().sum(dim=0).sqrt()
    threshold = torch.quantile(norm.reshape(-1).float(), percentile / 100.0)
    return norm > threshold


@dataclass(frozen=True, slots=True)
class AffineLatentBaseline:
    """Per-domain moment tables + the affine map they induce between any two domains."""

    moments: Mapping[str, DomainMoments]
    space: MomentSpace
    foreground_percentile: float
    eps: float = 1e-6
    provenance: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.moments:
            raise ValueError("AffineLatentBaseline needs at least one domain's moments.")
        if self.provenance is None:
            object.__setattr__(self, "provenance", {})

    def _moments_for(self, domain: Domain) -> DomainMoments:
        label = domain.label
        found = self.moments.get(label)
        if found is None:
            raise KeyError(
                f"No fitted moments for domain {label!r}. Fitted domains: "
                f"{sorted(self.moments)}. The pool has no volumes in that domain, so no "
                "closed-form affine exists for it."
            )
        return found

    def coefficients(self, source: Domain, target: Domain) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel ``(a, b)`` carrying the source marginal onto the target marginal."""

        src, tgt = self._moments_for(source), self._moments_for(target)
        a = tgt.std / src.std.clamp_min(self.eps)
        b = tgt.mean - a * src.mean
        return a, b

    def transport(self, z: torch.Tensor, source: Domain, target: Domain) -> torch.Tensor:
        """Apply the affine to a RAW-space latent, (C, x, y, z) or (B, C, x, y, z)."""

        if z.ndim not in (4, 5):
            raise ValueError(f"transport expects (C,x,y,z) or (B,C,x,y,z), got {tuple(z.shape)}.")
        a, b = self.coefficients(source, target)
        shape = [1] * z.ndim
        shape[1 if z.ndim == 5 else 0] = a.numel()
        a = a.to(device=z.device, dtype=z.dtype).reshape(shape)
        b = b.to(device=z.device, dtype=z.dtype).reshape(shape)
        return a * z + b

    def __call__(self, z: torch.Tensor, source: Domain, target: Domain) -> torch.Tensor:
        return self.transport(z, source, target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": AFFINE_BASELINE_CONTRACT_VERSION,
            "space": self.space,
            "foreground_percentile": float(self.foreground_percentile),
            "eps": float(self.eps),
            "provenance": dict(self.provenance),
            "moments": {label: m.to_dict() for label, m in sorted(self.moments.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AffineLatentBaseline":
        version = str(data.get("contract_version", ""))
        if version != AFFINE_BASELINE_CONTRACT_VERSION:
            raise ValueError(
                f"Affine baseline contract mismatch: file has {version!r}, code expects "
                f"{AFFINE_BASELINE_CONTRACT_VERSION!r}."
            )
        return cls(
            moments={k: DomainMoments.from_dict(v) for k, v in data["moments"].items()},
            space=str(data["space"]),  # type: ignore[arg-type]
            foreground_percentile=float(data["foreground_percentile"]),
            eps=float(data.get("eps", 1e-6)),
            provenance=dict(data.get("provenance", {})),
        )

    def save(self, path: str | Path) -> Path:
        return _write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "AffineLatentBaseline":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_affine_baselines(
    latents: Iterable[tuple[torch.Tensor, Domain]],
    *,
    channels: int,
    foreground_percentile: float = 50.0,
    eps: float = 1e-6,
    provenance: Mapping[str, Any] | None = None,
    log_every: int = 0,
) -> dict[MomentSpace, AffineLatentBaseline]:
    """One streaming pass over the pool producing BOTH the ``all`` and ``foreground`` tables.

    ``latents`` yields ``(latent (C,x,y,z), domain)``. Both tables come from the same pass
    because the expensive part is reading ~14 GB of latents once, not the arithmetic.
    """

    accumulators: dict[MomentSpace, dict[str, _MomentAccumulator]] = {
        space: {} for space in MOMENT_SPACES
    }
    seen = 0
    for latent, domain in latents:
        if latent.ndim == 5:
            latent = latent[0]
        if latent.shape[0] != channels:
            raise ValueError(
                f"Latent has {latent.shape[0]} channels but {channels} were declared."
            )
        latent = latent.to(torch.float32)
        label = domain.label
        mask = latent_foreground_mask(latent, percentile=foreground_percentile)
        for space, voxel_mask in (("all", None), ("foreground", mask)):
            bucket = accumulators[space]  # type: ignore[index]
            if label not in bucket:
                bucket[label] = _MomentAccumulator(channels)
            bucket[label].update(latent, voxel_mask)
        seen += 1
        if log_every and (seen == 1 or seen % log_every == 0):
            print(f"affine_baseline fit {seen} volumes (latest domain={label})", flush=True)

    base_provenance = dict(provenance or {})
    base_provenance["pool_volumes"] = seen
    return {
        space: AffineLatentBaseline(
            moments={label: acc.compute(eps=eps) for label, acc in accumulators[space].items()},
            space=space,
            foreground_percentile=foreground_percentile,
            eps=eps,
            provenance=base_provenance,
        )
        for space in MOMENT_SPACES
    }


def fit_paired_affine(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    channels: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Genuine per-channel least squares ``a_c, b_c`` from PAIRED ``(z_s, z_t)`` volumes.

    Diagnostic only. This has seen paired supervision, so it is an upper bound on what any
    per-channel affine can achieve — never a legitimate unpaired baseline.
    """

    if not pairs:
        raise ValueError("fit_paired_affine needs at least one (z_s, z_t) pair.")
    n = torch.zeros(channels, dtype=torch.float64)
    sx = torch.zeros(channels, dtype=torch.float64)
    sy = torch.zeros(channels, dtype=torch.float64)
    sxx = torch.zeros(channels, dtype=torch.float64)
    sxy = torch.zeros(channels, dtype=torch.float64)
    for z_s, z_t in pairs:
        xs = z_s.reshape(channels, -1).to(torch.float64)
        ys = z_t.reshape(channels, -1).to(torch.float64)
        if xs.shape != ys.shape:
            raise ValueError(f"Paired latents disagree in shape: {xs.shape} vs {ys.shape}.")
        n += xs.shape[1]
        sx += xs.sum(dim=1)
        sy += ys.sum(dim=1)
        sxx += xs.square().sum(dim=1)
        sxy += (xs * ys).sum(dim=1)
    mean_x, mean_y = sx / n, sy / n
    cov = sxy / n - mean_x * mean_y
    var = (sxx / n - mean_x.square()).clamp_min(eps)
    a = cov / var
    b = mean_y - a * mean_x
    return a.to(torch.float32), b.to(torch.float32)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = [
    "AFFINE_BASELINE_CONTRACT_VERSION",
    "MOMENT_SPACES",
    "AffineLatentBaseline",
    "DomainMoments",
    "fit_affine_baselines",
    "fit_paired_affine",
    "latent_foreground_mask",
]
