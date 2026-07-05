"""Deterministic reconstruction eval + diagnostic plots for the Etapa 1 KL-VAE.

Deliberately does NOT reuse the training forward pass: eval reconstructs from the latent
*mean* (no reparameterization sampling — the sampled `mean + eps*sigma` path is what made
the notebook reconstructions look like noise), under no_grad, and tiles the full volume
with a sliding window so a real 3D volume never gets decoded whole (the OOM/RAM blowup).
Inputs are normalized with the exact same [-1, 1] transform as training.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import RawBatch
from fieldbridge.evaluation.metrics import mae, mse, nrmse, ssim3d
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.losses import build_lpips_net, lpips_loss_3d

# Inputs live in [-1, 1] (decoder Tanh / normalize_percentile_clip_to_unit_range contract),
# so the intensity range for range-normalized metrics is 2.0, not 1.0.
_DATA_RANGE = 2.0


@dataclass(frozen=True, slots=True)
class SampleMetrics:
    case_id: str
    domain: str
    nrmse: float
    ssim3d: float
    lpips: float
    mse: float
    mae: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "nrmse": self.nrmse,
            "ssim3d": self.ssim3d,
            "lpips": self.lpips,
            "mse": self.mse,
            "mae": self.mae,
        }


def _tiled_starts(dim: int, patch: int, stride: int) -> list[int]:
    """Tile starts covering [0, dim) with `patch`-sized windows at `stride`; last clamps to the edge."""

    if patch >= dim:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts


def _hann_window_3d(patch: tuple[int, int, int], device: "Any", dtype: "Any") -> torch.Tensor:
    """Separable 3D Hann weight window (tapers to ~0 at tile faces) for overlap blending.

    Clamped away from exactly 0 so a voxel covered by a single tile (its face lands on the
    volume boundary) still normalizes cleanly — with single coverage the weight cancels in
    the num/den ratio, so its magnitude is irrelevant there anyway.
    """

    axes: list[torch.Tensor] = []
    for size in patch:
        if size <= 1:
            axes.append(torch.ones(max(size, 1), device=device, dtype=dtype))
            continue
        n = torch.arange(size, device=device, dtype=dtype)
        axes.append(0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (size - 1)))
    window = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return window.clamp_min(1e-3)


@torch.no_grad()
def sliding_window_reconstruct(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    image: torch.Tensor,
    *,
    patch_size: Sequence[int],
    domain: Any,
    overlap: float = 0.5,
) -> torch.Tensor:
    """Reconstruct a full (B, C, D, H, W) volume from latent means, tile by tile.

    Each tile is encoded/decoded independently (no cross-tile context), so with stride ==
    patch (overlap 0) the tile faces show up as hard seams — a regular panel grid every
    `patch` voxels. `overlap` (fraction in [0, 1)) shrinks the stride so tiles overlap, and
    a Hann weight window tapers each tile's contribution to ~0 at its faces, blending the
    seams away. Uses `encode_dist(...)[0]` (the mean) — no sampling — so it's deterministic.
    """

    if image.ndim != 5:
        raise ValueError(f"sliding_window_reconstruct expects 5D (B,C,D,H,W), got {image.ndim}D.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    pd, ph, pw = (int(p) for p in patch_size)
    _, _, depth, height, width = image.shape
    strides = tuple(max(1, round(p * (1.0 - overlap))) for p in (pd, ph, pw))
    window = _hann_window_3d((pd, ph, pw), image.device, image.dtype)

    weighted_sum = torch.zeros_like(image)
    weight_sum = torch.zeros_like(image)
    for z in _tiled_starts(depth, pd, strides[0]):
        for y in _tiled_starts(height, ph, strides[1]):
            for x in _tiled_starts(width, pw, strides[2]):
                tile = image[..., z : z + pd, y : y + ph, x : x + pw]
                mean, _ = encoder.encode_dist(tile, domain)
                rec = decoder.decode(mean, domain)
                weighted_sum[..., z : z + pd, y : y + ph, x : x + pw] += rec * window
                weight_sum[..., z : z + pd, y : y + ph, x : x + pw] += window
    return weighted_sum / weight_sum.clamp_min(1e-8)


def run_stage1_eval(
    *,
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    loader: DataLoader[RawBatch],
    patch_size: Sequence[int],
    out_dir: Path,
    num_samples: int = 4,
    device: torch.device | None = None,
    lpips_num_slices: int = 8,
    per_domain: bool = False,
    overlap: float = 0.5,
    loss_curve: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic reconstruction over up to `num_samples` volumes.

    With `per_domain=True`, keep at most one volume per distinct field strength (the
    cross-field "domain") — one 0.1T, one 1.5T, ... — instead of the first N in manifest
    order, so the diagnostics cover the field range. `num_samples` then caps the number of
    distinct domains collected.

    Writes `metrics.json` and `diagnostics.png` (and `loss_curve.png` if a loss history is
    given) to `out_dir`. Returns the metrics payload.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the LPIPS net once (not per sample) and reuse it; None if disabled or the
    # optional dependency is missing.
    lpips_net = None
    if lpips_num_slices > 0:
        try:
            lpips_net = build_lpips_net(device)
        except ImportError:
            lpips_net = None

    samples: list[dict[str, torch.Tensor]] = []
    metrics: list[SampleMetrics] = []
    seen_domains: set[float] = set()
    for batch in loader:
        if len(samples) >= num_samples:
            break
        if per_domain:
            key = _domain_field_key(batch.source_domain)
            if key in seen_domains:
                continue
            seen_domains.add(key)
        image = batch.image.to(device)
        domain = batch.source_domain
        recon = sliding_window_reconstruct(
            encoder, decoder, image, patch_size=patch_size, domain=domain, overlap=overlap
        )

        case_id, domain_label = _batch_labels(batch)
        metrics.append(
            SampleMetrics(
                case_id=case_id,
                domain=domain_label,
                nrmse=float(nrmse(recon, image, data_range=_DATA_RANGE)),
                ssim3d=float(ssim3d(recon, image, data_range=_DATA_RANGE)),
                lpips=_maybe_lpips(recon, image, lpips_num_slices, lpips_net),
                mse=float(mse(recon, image)),
                mae=float(mae(recon, image)),
            )
        )
        samples.append({"original": image.detach().cpu(), "recon": recon.detach().cpu()})

    payload: dict[str, Any] = {
        "num_samples": len(metrics),
        "data_range": _DATA_RANGE,
        "per_sample": [m.to_dict() for m in metrics],
        "mean": _aggregate(metrics),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    _plot_diagnostics(samples, metrics, out_dir / "diagnostics.png")
    if loss_curve is not None:
        _plot_loss_curve(loss_curve, out_dir / "loss_curve.png")
    return payload


def _domain_field_key(domain: Any) -> float:
    """Field strength of a (possibly batched) domain, used to dedup by cross-field domain."""

    if isinstance(domain, Sequence) and domain:
        domain = domain[0]
    field = getattr(domain, "field_strength_t", None)
    return float(field) if field is not None else float("nan")


def _maybe_lpips(
    recon: torch.Tensor, image: torch.Tensor, num_slices: int, net: "Any"
) -> float:
    """Slice-based LPIPS, or NaN if disabled (num_slices<=0) or the LPIPS net is unavailable."""

    if num_slices <= 0 or net is None:
        return float("nan")
    return float(lpips_loss_3d(recon, image, num_slices=num_slices, net=net))


def _batch_labels(batch: RawBatch) -> tuple[str, str]:
    metadata = batch.metadata
    case_id = "unknown"
    if isinstance(metadata, Sequence) and metadata and isinstance(metadata[0], dict):
        case_id = str(metadata[0].get("case_id", "unknown"))
    domain = batch.source_domain
    if isinstance(domain, Sequence) and domain:
        domain = domain[0]
    domain_label = getattr(domain, "label", str(domain))
    return case_id, domain_label


def _aggregate(metrics: Sequence[SampleMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = ("nrmse", "ssim3d", "lpips", "mse", "mae")
    return {key: float(sum(getattr(m, key) for m in metrics) / len(metrics)) for key in keys}


def _mid_slice(volume: torch.Tensor) -> "Any":
    # volume: (B, C, D, H, W) -> central axial slice of the first item/channel.
    import numpy as np

    array = volume[0, 0].numpy()
    mid = array.shape[0] // 2
    return np.rot90(array[mid])


def _plot_diagnostics(
    samples: Sequence[dict[str, torch.Tensor]],
    metrics: Sequence[SampleMetrics],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = max(len(samples), 1)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows), squeeze=False)
    for row, (sample, metric) in enumerate(zip(samples, metrics)):
        original = _mid_slice(sample["original"])
        recon = _mid_slice(sample["recon"])
        error = np.abs(original - recon)

        axes[row][0].imshow(original, cmap="gray", vmin=-1, vmax=1)
        axes[row][0].set_title(f"Original — {metric.domain}\n{metric.case_id}", fontsize=11)
        axes[row][0].axis("off")

        axes[row][1].imshow(recon, cmap="gray", vmin=-1, vmax=1)
        axes[row][1].set_title(
            f"Recon (deterministic mean)\nnRMSE {metric.nrmse:.4f} | SSIM3D {metric.ssim3d:.4f} | "
            f"LPIPS {metric.lpips:.4f}",
            fontsize=11,
        )
        axes[row][1].axis("off")

        im = axes[row][2].imshow(error, cmap="hot")
        axes[row][2].set_title("Absolute Error", fontsize=11)
        axes[row][2].axis("off")
        fig.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.04)

        axes[row][3].hist(
            sample["original"].flatten().numpy()[::10], bins=60, alpha=0.5, label="Original", color="tab:blue"
        )
        axes[row][3].hist(
            sample["recon"].flatten().numpy()[::10], bins=60, alpha=0.5, label="Recon", color="tab:orange"
        )
        axes[row][3].set_yscale("log")
        axes[row][3].set_title("Intensity Distribution [-1, 1] (log)", fontsize=11)
        axes[row][3].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_loss_curve(losses: Sequence[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(losses), marker="o", markersize=3, color="forestgreen")
    ax.set_title("Stage 1 VAE — training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
