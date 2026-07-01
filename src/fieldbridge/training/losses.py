"""Loss functions."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def reconstruction_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def latent_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mean, exp(logvar)) || N(0, I)), summed over features and averaged over the batch."""

    per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=tuple(range(1, mean.ndim)))
    return per_sample.mean()


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

    Constructing the default net loads pretrained VGG weights, which is expensive —
    build it once (e.g. in the training loop) and pass it in via `net` on every call
    rather than relying on the lazy default in a hot loop.
    """

    if net is None:
        net = _default_lpips_net(prediction.device)
    return net(_to_three_channel(prediction), _to_three_channel(target)).mean()


def _default_lpips_net(device: torch.device) -> torch.nn.Module:
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "lpips_loss requires the optional 'lpips' package. "
            "Install with `pip install -e '.[perceptual]'`."
        ) from exc
    return lpips.LPIPS(net="vgg").to(device)


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
