"""Deterministic reconstruction eval + diagnostic plots for the Etapa 1 KL-VAE.

Deliberately does NOT reuse the training forward pass: eval reconstructs from the latent
*mean* (no reparameterization sampling — the sampled `mean + eps*sigma` path is what made
the notebook reconstructions look like noise), under no_grad, and tiles the full volume
with a sliding window so a real 3D volume never gets decoded whole (the OOM/RAM blowup).
Inputs are the official [0, 1] volumes, passed through unchanged exactly as in training
(the official format forbids rescaling intensity).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import RawBatch
from fieldbridge.evaluation.metrics import mae, mse, normalized_cross_correlation, nrmse, psnr, ssim3d
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.latent_stats import LatentStatsAccumulator
from fieldbridge.training.losses import build_lpips_net, lpips_loss_3d

# Official MRIxFields2026 volumes ship in [0, 1] and must not be rescaled (see
# data/transforms.py:assert_official_unit_range), so the intensity range for
# range-normalized metrics is 1.0. This is the range the challenge scores against —
# changing it makes the reported nRMSE/SSIM incomparable to the leaderboard.
_DATA_RANGE = 1.0
_REAL_EVALUATION_INSTALL = 'pip install -e ".[nifti,evaluation]"'

# Tolerance on the [0, 1] contract before the range guard fires. Loose enough for fp/bf16
# round-off, tight enough to catch a de-normalized tensor.
_RANGE_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class SampleMetrics:
    case_id: str
    domain: str
    # Field strength (T) and contrast pulled out of the domain so per-field and
    # per-contrast aggregation (and the CSV) don't have to re-parse the "0.1T/T1w" label.
    field_strength_t: float
    contrast: str
    nrmse: float
    ssim3d: float
    lpips: float
    psnr: float
    mse: float
    mae: float
    # Pearson correlation between recon and target. Reported because SSIM cannot separate
    # "anatomy reconstructed but intensity range wrong" from "decoder emits a constant":
    # on [0, 1] MRI the target mean is low (background = 0), so a recon with the wrong DC
    # level distorts SSIM's luminance term whether or not structure is present. Correlation is invariant to any affine intensity change, so it
    # separates them outright: ~1 => anatomy present (a pure calibration fault), ~0 => no
    # structure at all (a model/training fault). Diagnostic only, not a challenge metric.
    correlation: float
    recon_stats: dict[str, float]
    target_stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "field_strength_t": self.field_strength_t,
            "contrast": self.contrast,
            "nrmse": self.nrmse,
            "ssim3d": self.ssim3d,
            "lpips": self.lpips,
            "psnr": self.psnr,
            "mse": self.mse,
            "mae": self.mae,
            "l1": self.mae,  # L1 == MAE; surfaced under both names for the requested table
            "correlation": self.correlation,
            "recon_stats": dict(self.recon_stats),
            "target_stats": dict(self.target_stats),
        }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


def _assert_same_space(recon: torch.Tensor, target: torch.Tensor) -> None:
    """Fail loudly if recon and target don't share shape and the [0, 1] contract.

    A de-normalization mismatch (one side left in raw intensities, or a decoder whose
    output stopped being bounded) otherwise shows up only as silently wrong metrics.
    """

    if recon.shape != target.shape:
        raise ValueError(
            f"recon and target must share shape; got {tuple(recon.shape)} and {tuple(target.shape)}."
        )
    low_limit, high_limit = -_RANGE_TOLERANCE, 1.0 + _RANGE_TOLERANCE
    for name, tensor in (("recon", recon), ("target", target)):
        low, high = float(tensor.min()), float(tensor.max())
        if low < low_limit or high > high_limit:
            raise ValueError(
                f"{name} violates the [0, 1] contract: range [{low:.4f}, {high:.4f}]. "
                "Official volumes ship in [0, 1] and must not be rescaled "
                "(see data/transforms.py:assert_official_unit_range)."
            )


def _log_diagnostic(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _range_report(metric: SampleMetrics) -> str:
    """One-line recon-vs-target range + correlation summary, for the Etapa-1 diagnostic gate.

    `verdict` names what the numbers imply, so a broken run is legible from the log alone
    without re-deriving the SSIM-luminance argument in `SampleMetrics.correlation`.
    """

    recon, target = metric.recon_stats, metric.target_stats
    # Expressed as a fraction of the data range, not an absolute offset: the same 25%-of-
    # range gap that flagged a compressed recon on the old [-1, 1] contract must keep
    # flagging it on the official [0, 1] one.
    floor_gap = 0.25 * _DATA_RANGE
    if abs(metric.correlation) < 0.2:
        verdict = "NO STRUCTURE (recon ~ constant; model/training fault, not calibration)"
    elif recon["min"] > target["min"] + floor_gap:
        verdict = "STRUCTURE OK but RANGE COMPRESSED (recon never reaches background; calibration fault)"
    else:
        verdict = "structure and range both plausible"
    return (
        f"stage1_eval {metric.domain} {metric.case_id}: "
        f"recon[min={recon['min']:+.3f} max={recon['max']:+.3f} mean={recon['mean']:+.3f} std={recon['std']:.3f}] "
        f"target[min={target['min']:+.3f} max={target['max']:+.3f} mean={target['mean']:+.3f} std={target['std']:.3f}] "
        f"corr={metric.correlation:+.4f} ssim3d={metric.ssim3d:+.4f} -> {verdict}"
    )


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
    clamp_output: bool = True,
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
    # Clamp only here, at the boundary where a bounded tensor is actually required. The
    # decoder head is linear by default (KLVAEDecoder's docstring explains why a
    # saturating head cannot reach the exact-0 background), so a raw reconstruction may
    # overshoot [0, 1]; the challenge scores clipped images, and _assert_same_space
    # enforces the contract downstream.
    reconstruction = weighted_sum / weight_sum.clamp_min(1e-8)
    return reconstruction.clamp(0.0, 1.0) if clamp_output else reconstruction


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
    per_domain: bool | None = None,
    per_field_contrast: bool = False,
    per_domain_samples: int = 1,
    oversample_field: float | None = None,
    oversample_factor: int = 1,
    eval_seed: int = 13,
    compute_latent_stats: bool = True,
    latent_active_kl_threshold: float = 0.01,
    write_csv: bool = True,
    print_tables: bool = True,
    overlap: float = 0.5,
    loss_curve: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic reconstruction over a fixed, domain-balanced set of volumes.

    Selection (when `per_field_contrast=True`, the alias `per_domain=True`, or any
    oversampling is requested): keep up to `per_domain_samples` volumes per distinct
    field/contrast pair, in the loader's (deterministic) order, so coverage spans every
    domain present. `oversample_field` (e.g. 0.1 for 0.1T) raises that cap to
    `per_domain_samples * oversample_factor` for the hardest field. With the defaults
    (`per_domain_samples=1`, `oversample_field=None`) this reduces to one volume per pair —
    the historical behavior. `num_samples` is an overall cap. `eval_seed` is recorded for
    provenance (selection itself is loader-order-deterministic, not random).

    Metrics per sample: L1(=MAE), PSNR, SSIM3D, nRMSE, LPIPS, correlation. Reconstruction is
    full-volume (sliding window from latent means). Writes `metrics.json`, `metrics.csv`,
    `diagnostics.png` (input/recon/|error| panel with a fixed center slice), plus per-field
    and per-contrast aggregate tables to stdout, and posterior-collapse latent stats.
    Returns the metrics payload.
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

    latent_channels = getattr(encoder, "latent_channels", None)
    latent_acc = (
        LatentStatsAccumulator(int(latent_channels))
        if compute_latent_stats and latent_channels
        else None
    )

    select_field_contrast = per_field_contrast or bool(per_domain) or oversample_field is not None
    samples: list[dict[str, torch.Tensor]] = []
    metrics: list[SampleMetrics] = []
    per_key_counts: dict[tuple[float, str], int] = {}
    for batch in loader:
        if len(samples) >= num_samples:
            break
        if select_field_contrast:
            key = _domain_field_contrast_key(batch.source_domain)
            cap = per_domain_samples
            if oversample_field is not None and math.isclose(key[0], float(oversample_field), abs_tol=1e-6):
                cap = per_domain_samples * max(1, oversample_factor)
            if per_key_counts.get(key, 0) >= cap:
                continue
            per_key_counts[key] = per_key_counts.get(key, 0) + 1
        image = batch.image.to(device)
        domain = batch.source_domain
        recon = sliding_window_reconstruct(
            encoder, decoder, image, patch_size=patch_size, domain=domain, overlap=overlap
        )

        _assert_same_space(recon, image)

        if latent_acc is not None:
            center = _center_crop(image, patch_size)
            with torch.no_grad():
                latent_mean, latent_logvar = encoder.encode_dist(center, domain)
            latent_acc.update(latent_mean, latent_logvar)

        case_id, domain_label = _batch_labels(batch)
        field_t, contrast = _domain_field_contrast_labels(batch.source_domain)
        sample_metrics = SampleMetrics(
            case_id=case_id,
            domain=domain_label,
            field_strength_t=field_t,
            contrast=contrast,
            nrmse=float(nrmse(recon, image, data_range=_DATA_RANGE)),
            ssim3d=float(ssim3d(recon, image, data_range=_DATA_RANGE)),
            lpips=_maybe_lpips(recon, image, lpips_num_slices, lpips_net),
            psnr=float(psnr(recon, image, data_range=_DATA_RANGE)),
            mse=float(mse(recon, image)),
            mae=float(mae(recon, image)),
            correlation=float(normalized_cross_correlation(recon, image)),
            recon_stats=_tensor_stats(recon),
            target_stats=_tensor_stats(image),
        )
        metrics.append(sample_metrics)
        _log_diagnostic(_range_report(sample_metrics))
        samples.append({"original": image.detach().cpu(), "recon": recon.detach().cpu()})

    by_field = _aggregate_by(metrics, lambda m: f"{m.field_strength_t:g}T")
    by_contrast = _aggregate_by(metrics, lambda m: m.contrast)
    payload: dict[str, Any] = {
        "num_samples": len(metrics),
        "data_range": _DATA_RANGE,
        "eval_seed": eval_seed,
        "sampling_coverage_unit": "field_contrast" if select_field_contrast else "manifest_order",
        "oversample_field": oversample_field,
        "oversample_factor": oversample_factor,
        "per_sample": [m.to_dict() for m in metrics],
        "mean": _aggregate(metrics),
        "by_field_strength": by_field,
        "by_contrast": by_contrast,
        "by_field_contrast": _aggregate_by(metrics, lambda m: m.domain),
    }
    if latent_acc is not None and metrics:
        payload["latent"] = latent_acc.compute(active_threshold=latent_active_kl_threshold)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if write_csv:
        _write_metrics_csv(metrics, out_dir / "metrics.csv")
    if print_tables:
        _print_metric_tables(by_field, by_contrast, payload.get("latent"))

    _plot_diagnostics(samples, metrics, out_dir / "diagnostics.png")
    if loss_curve is not None:
        _plot_loss_curve(loss_curve, out_dir / "loss_curve.png")
    return payload


def _center_crop(image: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    """Center crop of `patch_size` from a 5D volume (for representative latent encoding).

    Clamps each requested extent to the actual spatial size, so a volume smaller than the
    patch along an axis is simply used whole on that axis."""

    if image.ndim != 5:
        raise ValueError(f"_center_crop expects a 5D (B,C,D,H,W) tensor, got {image.ndim}D.")
    starts_sizes: list[tuple[int, int]] = []
    for spatial, want in zip(image.shape[2:], patch_size):
        size = min(int(want), int(spatial))
        start = (int(spatial) - size) // 2
        starts_sizes.append((start, size))
    (sd, nd), (sh, nh), (sw, nw) = starts_sizes
    return image[..., sd : sd + nd, sh : sh + nh, sw : sw + nw]


def _domain_field_contrast_labels(domain: Any) -> tuple[float, str]:
    """(field strength as float, contrast as its clean value string) from a domain."""

    if isinstance(domain, Sequence) and not isinstance(domain, (str, bytes)) and domain:
        domain = domain[0]
    field = getattr(domain, "field_strength_t", None)
    contrast = getattr(domain, "contrast", None)
    contrast_str = getattr(contrast, "value", str(contrast)) if contrast is not None else "unknown"
    return (float(field) if field is not None else float("nan"), contrast_str)


def _aggregate_by(
    metrics: Sequence[SampleMetrics], key_fn: "Any"
) -> dict[str, dict[str, float]]:
    """Group metrics by `key_fn(sample)` and average each group (nan-aware, plus a count)."""

    groups: dict[str, list[SampleMetrics]] = {}
    for metric in metrics:
        groups.setdefault(key_fn(metric), []).append(metric)
    aggregated: dict[str, dict[str, float]] = {}
    for label in sorted(groups):
        group = groups[label]
        summary = {key: _nanmean([getattr(m, key) for m in group]) for key in _AGG_KEYS}
        summary["l1"] = summary["mae"]
        summary["count"] = float(len(group))
        aggregated[label] = summary
    return aggregated


def _nanmean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v == v]  # drop NaN
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _write_metrics_csv(metrics: Sequence[SampleMetrics], path: Path) -> None:
    fields = ["case_id", "domain", "field_strength_t", "contrast", "l1", "psnr", "ssim3d", "nrmse", "lpips", "correlation"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "case_id": metric.case_id,
                    "domain": metric.domain,
                    "field_strength_t": f"{metric.field_strength_t:g}",
                    "contrast": metric.contrast,
                    "l1": f"{metric.mae:.6f}",
                    "psnr": f"{metric.psnr:.4f}",
                    "ssim3d": f"{metric.ssim3d:.6f}",
                    "nrmse": f"{metric.nrmse:.6f}",
                    "lpips": f"{metric.lpips:.6f}",
                    "correlation": f"{metric.correlation:.6f}",
                }
            )


def _print_metric_tables(
    by_field: Mapping[str, Mapping[str, float]],
    by_contrast: Mapping[str, Mapping[str, float]],
    latent: Mapping[str, Any] | None,
) -> None:
    """Human-readable per-field / per-contrast tables (L1, PSNR, SSIM, LPIPS) to stdout."""

    def render(title: str, table: Mapping[str, Mapping[str, float]]) -> None:
        print(f"\n== {title} ==")
        print(f"{'group':<12} {'n':>3} {'L1':>10} {'PSNR':>8} {'SSIM':>8} {'nRMSE':>8} {'LPIPS':>8}")
        for label, row in table.items():
            print(
                f"{label:<12} {int(row['count']):>3} {row['l1']:>10.5f} {row['psnr']:>8.3f} "
                f"{row['ssim3d']:>8.4f} {row['nrmse']:>8.4f} {row['lpips']:>8.4f}"
            )

    render("metrics by field strength", by_field)
    render("metrics by contrast", by_contrast)
    if latent:
        print(
            f"\n== latent ==\nactive_units={latent['active_units']}/{latent['num_dims']} "
            f"global_std={latent['global_std']:.3f} "
            f"per_dim_std(min/mean/max)={latent['min_per_dim_std']:.3f}/"
            f"{latent['mean_per_dim_std']:.3f}/{latent['max_per_dim_std']:.3f}"
        )


def render_reconstruction_panel(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    labels: Sequence[str],
    path: Path,
) -> None:
    """Render input / reconstruction / |error| rows for a batch to `path` (one row/sample).

    Handles 4D (B,C,H,W) and 5D (B,C,D,H,W) tensors — for volumes a fixed center slice
    along the first spatial axis is shown. Used both by the offline eval and the in-training
    recon hook, so the visual is defined once. Requires matplotlib (the 'evaluation' extra).
    """

    import numpy as np

    plt = _matplotlib_pyplot()
    batch = int(originals.shape[0])
    fig, axes = plt.subplots(batch, 3, figsize=(12, 4 * batch), squeeze=False)
    for row in range(batch):
        original = _panel_slice(originals[row : row + 1])
        recon = _panel_slice(reconstructions[row : row + 1])
        error = np.abs(original - recon)
        label = labels[row] if row < len(labels) else f"sample {row}"

        axes[row][0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][0].set_title(f"Input — {label}", fontsize=11)
        axes[row][0].axis("off")
        axes[row][1].imshow(recon, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][1].set_title(
            f"Recon\nL1 {float(np.abs(original - recon).mean()):.4f}", fontsize=11
        )
        axes[row][1].axis("off")
        image = axes[row][2].imshow(error, cmap="hot")
        axes[row][2].set_title("|error|", fontsize=11)
        axes[row][2].axis("off")
        fig.colorbar(image, ax=axes[row][2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _panel_slice(volume: torch.Tensor) -> "Any":
    """Display-rotated center slice for a single (1,C,...) sample, 2D or 3D."""

    import numpy as np

    if volume.ndim == 5:
        return _central_first_spatial_axis_slice(volume)
    array = volume[0, 0].numpy()
    return np.rot90(array)


def _domain_field_contrast_key(domain: Any) -> tuple[float, str]:
    """Field and contrast of a possibly batched acquisition label."""

    if isinstance(domain, Sequence) and domain:
        domain = domain[0]
    field = getattr(domain, "field_strength_t", None)
    contrast = getattr(domain, "contrast", None)
    return (
        float(field) if field is not None else float("nan"),
        str(contrast) if contrast is not None else "unknown",
    )


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


# Numeric SampleMetrics fields averaged in the overall + per-group aggregates.
_AGG_KEYS = ("nrmse", "ssim3d", "lpips", "psnr", "mse", "mae", "correlation")


def _aggregate(metrics: Sequence[SampleMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    return {key: float(sum(getattr(m, key) for m in metrics) / len(metrics)) for key in _AGG_KEYS}


def _central_first_spatial_axis_slice(volume: torch.Tensor) -> "Any":
    """Return a display-rotated center slice along the first raw spatial axis.

    No anatomical plane is implied without orientation or affine metadata.
    """

    import numpy as np

    array = volume[0, 0].numpy()
    mid = array.shape[0] // 2
    return np.rot90(array[mid])


def _matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Stage-1 diagnostic rendering requires Matplotlib from the 'evaluation' extra. "
            f"For real Stage-1 evaluation, run: {_REAL_EVALUATION_INSTALL}"
        ) from exc
    return plt


def _plot_diagnostics(
    samples: Sequence[dict[str, torch.Tensor]],
    metrics: Sequence[SampleMetrics],
    path: Path,
) -> None:
    import numpy as np

    plt = _matplotlib_pyplot()

    rows = max(len(samples), 1)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows), squeeze=False)
    for row, (sample, metric) in enumerate(zip(samples, metrics)):
        original = _central_first_spatial_axis_slice(sample["original"])
        recon = _central_first_spatial_axis_slice(sample["recon"])
        error = np.abs(original - recon)

        # Official [0, 1] contract: display on [0, 1], not [-1, 1] (which halved the
        # apparent brightness of every panel and washed the anatomy out).
        axes[row][0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][0].set_title(f"Original — {metric.domain}\n{metric.case_id}", fontsize=11)
        axes[row][0].axis("off")

        axes[row][1].imshow(recon, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][1].set_title(
            f"Recon (deterministic mean)\nL1 {metric.mae:.4f} | PSNR {metric.psnr:.2f} | "
            f"SSIM3D {metric.ssim3d:.4f} | LPIPS {metric.lpips:.4f}\ncorr {metric.correlation:+.4f} | "
            f"range [{metric.recon_stats['min']:+.2f}, {metric.recon_stats['max']:+.2f}]",
            fontsize=11,
        )
        axes[row][1].axis("off")

        im = axes[row][2].imshow(error, cmap="hot")
        axes[row][2].set_title("Absolute Error", fontsize=11)
        axes[row][2].axis("off")
        fig.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.04)

        axes[row][3].hist(
            sample["original"].flatten().numpy()[::10],
            bins=60,
            range=(0.0, 1.0),
            alpha=0.5,
            label="Original",
            color="tab:blue",
        )
        axes[row][3].hist(
            sample["recon"].flatten().numpy()[::10],
            bins=60,
            range=(0.0, 1.0),
            alpha=0.5,
            label="Recon",
            color="tab:orange",
        )
        axes[row][3].set_yscale("log")
        axes[row][3].set_title("Intensity Distribution [0, 1] (log)", fontsize=11)
        axes[row][3].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_loss_curve(losses: Sequence[float], path: Path) -> None:
    plt = _matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(losses), marker="o", markersize=3, color="forestgreen")
    ax.set_title("Stage 1 VAE — training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
