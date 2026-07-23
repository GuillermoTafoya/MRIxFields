"""Loss functions."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Mapping

import torch
from torch.nn import functional as F


def reconstruction_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def latent_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error over the selected mask region."""

    _validate_same_shape(prediction, target)
    if mask is None:
        return F.l1_loss(prediction, target)
    prepared_mask = _prepare_mask(mask, prediction)
    denominator = _positive_mask_sum(prepared_mask, "masked_l1_loss")
    return (torch.abs(prediction - target) * prepared_mask).sum() / denominator


def foreground_weighted_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.0,
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    """L1 with foreground voxels up-weighted relative to background, target-derived.

    Foreground is `target > threshold` (on the official [0, 1] contract, background is
    exactly 0, so threshold 0 selects brain). Each voxel's error is weighted
    `1 + (foreground_weight - 1) * is_foreground`, then normalized by the total weight —
    so `foreground_weight == 1.0` reduces *exactly* to the plain mean L1 (the property
    that keeps this a no-op when the feature flag is off). Motivation: uniform L1 lets a
    black-background-dominated patch produce a trivially low loss; up-weighting foreground
    stops those batches from masking real reconstruction error. Diagnostic/opt-in — see
    Stage1VAEConfig.foreground_loss_weighting.
    """

    _validate_same_shape(prediction, target)
    absolute_error = torch.abs(prediction - target)
    foreground = (target > threshold).to(prediction.dtype)
    weight = 1.0 + (float(foreground_weight) - 1.0) * foreground
    return (absolute_error * weight).sum() / weight.sum().clamp_min(1.0)


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over the selected mask region."""

    _validate_same_shape(prediction, target)
    if mask is None:
        return F.mse_loss(prediction, target)
    prepared_mask = _prepare_mask(mask, prediction)
    denominator = _positive_mask_sum(prepared_mask, "masked_mse_loss")
    return ((prediction - target).pow(2) * prepared_mask).sum() / denominator


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 loss between finite spatial gradients of prediction and target."""

    _validate_same_shape(prediction, target)
    if prediction.ndim < 3:
        raise ValueError(
            "gradient_loss expects tensors with batch, channel, and at least one spatial dimension."
        )
    prepared_mask = _prepare_mask(mask, prediction) if mask is not None else None
    losses: list[torch.Tensor] = []
    for dim in range(2, prediction.ndim):
        if int(prediction.shape[dim]) < 2:
            continue
        pred_diff = prediction.diff(dim=dim)
        target_diff = target.diff(dim=dim)
        diff_mask = _gradient_mask(prepared_mask, dim) if prepared_mask is not None else None
        losses.append(masked_l1_loss(pred_diff, target_diff, diff_mask))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def background_penalty(
    prediction: torch.Tensor,
    mask: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize prediction error outside the foreground mask.

    When ``target`` is omitted this preserves the historical behavior of penalizing
    nonzero predictions. Pseudo-pair training passes the target explicitly so the
    outside-support term is correct in both ``[0, 1]`` and ``[-1, 1]`` model ranges.
    """

    if target is not None:
        _validate_same_shape(prediction, target)
    residual = prediction if target is None else prediction - target
    if mask is None:
        return torch.abs(residual).mean()
    prepared_mask = _prepare_mask(mask, prediction)
    outside_mask = (1.0 - prepared_mask).clamp_min(0.0)
    denominator = outside_mask.sum()
    if not bool(torch.any(outside_mask > 0).detach().cpu().item()):
        return prediction.sum() * 0.0
    return (torch.abs(residual) * outside_mask).sum() / denominator


def combined_reconstruction_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return unweighted pseudo-pair reconstruction components plus weighted total."""

    _validate_same_shape(prediction, target)
    active_weights = {"masked_l1": 1.0, "gradient": 0.1, "background": 0.05}
    if weights is not None:
        active_weights.update(dict(weights))

    masked = masked_l1_loss(prediction, target, mask)
    gradient = gradient_loss(prediction, target, mask)
    background = background_penalty(prediction, mask, target=target)
    total = prediction.sum() * 0.0
    total = total + active_weights.get("masked_l1", 0.0) * masked
    total = total + active_weights.get("gradient", 0.0) * gradient
    total = total + active_weights.get("background", 0.0) * background
    return {
        "masked_l1": masked,
        "gradient": gradient,
        "background": background,
        "total": total,
    }


def combined_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Weighted reconstruction loss for synthetic translator interface tests.

    SSIM is intentionally omitted here; it can be added later if a lightweight,
    dependency-free implementation is needed for training.
    """

    return combined_reconstruction_loss_components(prediction, target, mask, weights)["total"]


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mean, exp(logvar)) || N(0, I)), summed over features and averaged over the batch."""

    per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=tuple(range(1, mean.ndim)))
    return per_sample.mean()


def kl_divergence_free_bits(mean: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL to N(0, I) with a per-channel free-bits floor, on the same scale as `kl_divergence`.

    Motivation (item 4): the Etapa-1 latent collapses to 1 active channel of 4 — the KL term,
    tiny as its weight is, still pushes low-information channels to the prior and euthanizes
    them. Free-bits reserves a per-channel KL budget the optimizer is not penalized for: below
    `free_bits` the channel's contribution is clamped, so its gradient through the KL term
    vanishes and the term stops pulling it toward the prior. It PERMITS (does not force) the
    reconstruction loss to keep all 4 channels alive.

    Scale reconciliation — the subtle part. `free_bits` is expressed on the *per-element mean*
    scale, i.e. directly comparable to `LatentStatsAccumulator.per_dim_kl` (the number the
    collapse diagnostics report: ~0.24 on the live channel, <0.008 on the dead ones). But the
    training KL term lives on the *summed* scale (`kl_divergence` ~1e3). So we clamp on the
    per-element-mean per channel, then multiply back by the per-channel spatial element count
    to land on `kl_divergence`'s scale. Two consequences, both intended:

    * `free_bits == 0.0` reduces to `kl_divergence` exactly (kl_elem is >= 0 elementwise, so
      the clamp is a no-op), up to float summation order — the equivalence test uses allclose.
    * `free_bits` is read in the same units as the per-dim-KL you see in the logs, so 0.5 means
      "reserve ~0.5 nats/element per channel" — roughly the KL of a unit-variance informative
      posterior, i.e. the standardized-latent target Etapa-2 assumes.

    Reduction matches `LatentStatsAccumulator` (reduce over every dim but channel=1).
    """

    if mean.ndim < 2:
        raise ValueError(f"kl_divergence_free_bits expects (batch, channels, ...); got {tuple(mean.shape)}.")
    kl_elem = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())  # >= 0 elementwise
    reduce_dims = tuple(d for d in range(mean.ndim) if d != 1)
    per_channel_mean = kl_elem.mean(dim=reduce_dims)  # [C]; == LatentStatsAccumulator per_dim_kl
    # Spatial elements per channel per sample (S): rescales the per-element-mean back onto the
    # summed kl_divergence scale so free_bits=0 is byte-for-byte the same term (float order aside).
    spatial_numel = mean.numel() / (mean.shape[0] * mean.shape[1])
    return spatial_numel * per_channel_mean.clamp_min(float(free_bits)).sum()


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """1 - ssim(...) — ssim is "higher is better", losses in this module are "minimize".

    Dispatches by rank: 4D (B,C,H,W) -> 2D `ssim`, 5D (B,C,D,H,W) -> `ssim3d`. Lets the
    same loss term drive both the 2D-slice and full-3D-volume training paths.

    Deferred import: evaluation/metrics.py imports training.losses.lpips_loss (also
    deferred, inside lpips_metric) — importing evaluation.metrics at module level here
    would work today (no actual cycle, since that import is function-local on the other
    side too) but keeping both sides deferred avoids ever depending on which one loads
    first.
    """

    from fieldbridge.evaluation.metrics import ssim, ssim3d

    metric = ssim3d if prediction.ndim == 5 else ssim
    similarity = metric(prediction, target, **kwargs)
    if not bool(torch.isfinite(similarity)):
        raise ValueError("SSIM similarity must be finite.")
    if not bool(((similarity >= -1.0) & (similarity <= 1.0)).detach().cpu().item()):
        raise ValueError(f"SSIM similarity escaped [-1, 1]: {float(similarity)}.")
    loss = 1.0 - similarity
    if not bool(torch.isfinite(loss)) or bool((loss < 0).detach().cpu().item()):
        raise ValueError(f"SSIM loss must be finite and nonnegative, got {float(loss)}.")
    return loss


def nrmse_loss(prediction: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Alias for evaluation.metrics.nrmse — already "lower is better", no sign flip needed.

    Kept here (rather than importing evaluation.metrics.nrmse directly at call sites) so
    every term in a loss composition comes from training.losses, not half from here and
    half reached into evaluation.metrics directly.
    """

    from fieldbridge.evaluation.metrics import nrmse

    return nrmse(prediction, target, **kwargs)


def transport_cost_loss(z_source: torch.Tensor, z_translated: torch.Tensor) -> torch.Tensor:
    """Penalize latent displacement between source and translated latents."""

    return F.mse_loss(z_translated, z_source)


def cycle_consistency_loss(x: torch.Tensor, x_cycled: torch.Tensor) -> torch.Tensor:
    """A -> B -> A should reconstruct A."""

    return F.l1_loss(x_cycled, x)


def identity_loss(x: torch.Tensor, x_identity_output: torch.Tensor) -> torch.Tensor:
    """When source domain == target domain, the output should equal the input."""

    return F.l1_loss(x_identity_output, x)


def adversarial_hinge_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    return -torch.mean(fake_logits)


def adversarial_hinge_loss_discriminator(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))


def lpips_loss(prediction: torch.Tensor, target: torch.Tensor, *, net: torch.nn.Module | None = None) -> torch.Tensor:
    """Perceptual loss via LPIPS. Requires the optional `lpips` package.

    Inputs are the project's official [0, 1] volumes; `lpips.LPIPS` with its default
    `normalize=False` expects [-1, 1], so they are affine-mapped here. This is a
    metric-space conversion inside the perceptual net, NOT a rescaling of the data (which
    the official format forbids) — the tensors the losses and metrics see stay in [0, 1].
    Skipping it would silently halve the effective input contrast and report an LPIPS
    that is not the challenge's LPIPS.

    Constructing the default net loads pretrained VGG weights, which is expensive —
    build it once (e.g. in the training loop) and pass it in via `net` on every call
    rather than relying on the lazy default in a hot loop.
    """

    if net is None:
        net = _default_lpips_net(prediction.device)
    return net(
        _to_three_channel(_unit_range_to_signed(prediction)),
        _to_three_channel(_unit_range_to_signed(target)),
    ).mean()


def _unit_range_to_signed(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1], the input convention of `lpips.LPIPS(normalize=False)`."""

    return x * 2.0 - 1.0


def lpips_loss_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    net: torch.nn.Module | None = None,
    num_slices: int = 8,
    axis: int = 2,
) -> torch.Tensor:
    """Slice-based LPIPS for 3D volumes: average 2D LPIPS over sampled slices.

    LPIPS wraps a 2D VGG net (`_to_three_channel` assumes 4D). For (B,C,D,H,W) volumes we
    sample `num_slices` equispaced slices along `axis` (default the depth axis, index 2),
    fold them into the batch, and run the 2D LPIPS once. Equispaced (not random) so the
    term is deterministic per forward — no seed coupling with the reparameterization.
    """

    if prediction.ndim != 5:
        raise ValueError(f"lpips_loss_3d expects 5D (B,C,D,H,W) tensors, got {prediction.ndim}D.")
    _validate_same_shape(prediction, target)
    depth = int(prediction.shape[axis])
    count = min(num_slices, depth)
    idx = torch.linspace(0, depth - 1, count, device=prediction.device).round().long().unique()
    pred_slices = prediction.index_select(axis, idx)
    target_slices = target.index_select(axis, idx)
    # Fold the sampled-slice axis into the batch -> (B*S, C, H, W) for the 2D net.
    pred_2d = pred_slices.movedim(axis, 1).flatten(0, 1)
    target_2d = target_slices.movedim(axis, 1).flatten(0, 1)
    return lpips_loss(pred_2d, target_2d, net=net)


def build_lpips_net(device: torch.device) -> torch.nn.Module:
    """Construct the pretrained LPIPS(vgg) net, keeping its chatter off stdout.

    `lpips.LPIPS(...)` prints "Setting up [LPIPS]..." / "Loading model from..." to stdout.
    Under `--json` (stdout is the machine-readable channel, redirected to a file) that
    corrupts the JSON — so we redirect the constructor's stdout to stderr, where all other
    diagnostics already go.

    All parameters are frozen: LPIPS is a fixed perceptual *metric*, never a trained
    component. `lpips.LPIPS` already freezes the VGG trunk but leaves its `lin` calibration
    convs trainable, which (a) makes eval's `float(lpips_value)` warn about reading a
    grad-tracking tensor and (b) makes training backprop into params no optimizer owns.
    Gradients still reach the VAE through the *input* activations, which is the only path
    the perceptual term needs.
    """

    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "lpips_loss requires the optional 'lpips' package. "
            "Install with `pip install -e '.[perceptual]'`."
        ) from exc
    with contextlib.redirect_stdout(sys.stderr):
        net = lpips.LPIPS(net="vgg")
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net.to(device).eval()


def _default_lpips_net(device: torch.device) -> torch.nn.Module:
    return build_lpips_net(device)


def _to_three_channel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"lpips_loss expects 1 or 3 channels, got {x.shape[1]}.")


def synthseg_inloss_stub(*args: object, **kwargs: object) -> torch.Tensor:
    """Stub for a SynthSeg-label-based anatomy in-loss.

    Depends on SynthSeg labels from the official preprocessing pipeline, which are
    not yet confirmed as available in this scaffold. Fails explicitly until real
    labels are wired in, per AGENTS.md's stub convention.
    """

    del args, kwargs
    raise NotImplementedError("SynthSeg in-loss depends on official preproc labels not yet available.")


def _validate_same_shape(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )


def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    prepared = mask.to(device=reference.device, dtype=reference.dtype)
    if prepared.ndim == reference.ndim - 1 and int(prepared.shape[0]) == int(reference.shape[0]):
        prepared = prepared.unsqueeze(1)
    try:
        prepared = torch.broadcast_to(prepared, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}."
        ) from exc
    if not torch.isfinite(prepared).all():
        raise ValueError("mask must contain only finite values.")
    return prepared


def _positive_mask_sum(mask: torch.Tensor, loss_name: str) -> torch.Tensor:
    denominator = mask.sum()
    if not bool((denominator > 0).detach().cpu().item()):
        raise ValueError(f"{loss_name} mask must select at least one element.")
    return denominator


def _gradient_mask(mask: torch.Tensor, dim: int) -> torch.Tensor:
    before = mask.narrow(dim, 0, int(mask.shape[dim]) - 1)
    after = mask.narrow(dim, 1, int(mask.shape[dim]) - 1)
    return before * after
