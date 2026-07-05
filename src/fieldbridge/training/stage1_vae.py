"""VAE-only training entry point for Etapa 1 (encoder/decoder, no diffuser yet).

Not `train_loop.py` — that assumes an encode -> translate -> decode pipeline with a
`BaseTranslator`; this stage has no translator and a different loss set
(SSIM + nRMSE + LPIPS + KL, no transport-cost/cycle/identity). Reusing `train_loop.py`
would force awkward no-op translator plumbing.

Once this stage is validated (checkpoint saved, reconstruction quality confirmed), the
conditional diffuser (see `training/stage2_diffuser.py`) trains on top of this VAE's
latent, VAE frozen by default — staged, not end-to-end.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import RawBatch
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.batch import move_raw_batch
from fieldbridge.training.checkpoints import checkpoint_filename, load_checkpoint, save_checkpoint
from fieldbridge.training.losses import (
    build_lpips_net,
    kl_divergence,
    lpips_loss,
    lpips_loss_3d,
    nrmse_loss,
    ssim_loss,
)
from fieldbridge.training.warm_start import load_state_dict_tolerant
from fieldbridge.utils.seeding import seed_everything

Precision = Literal["fp32", "bf16"]
Device = Literal["auto", "cpu", "cuda"]

# Unlike Etapa 2's "everything 0 except reconstruction" ladder convention, all four
# terms here are active by default: SSIM+nRMSE+LPIPS+KL is a fixed composition whose
# *relative* weights get swept experimentally, not a term-by-term ladder to turn on one
# at a time.
DEFAULT_VAE_LOSS_WEIGHTS: dict[str, float] = {
    "ssim": 1.0,
    "nrmse": 1.0,
    "lpips": 1.0,
    "kl": 1e-4,
}


@dataclass(frozen=True, slots=True)
class Stage1VAEConfig:
    steps: int = 2
    batch_size: int = 2
    seed: int = 13
    lr: float = 1e-4
    device: Device = "auto"
    precision: Precision = "fp32"
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_VAE_LOSS_WEIGHTS))
    ssim_window_size: int = 7
    lpips_num_slices: int = 8
    grad_clip_norm: float = 1.0
    warm_start_checkpoint: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 0
    checkpoint_at_end: bool = False
    checkpoint_max_bytes: int = 10_000_000
    log_every_steps: int = 0
    resume_from: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage1VAEConfig":
        # NOTE: `cls.<field>` is not a usable default here — with frozen+slots
        # dataclasses, accessing a field on the *class* returns a slot
        # `member_descriptor`, not the field's default value. Use a real instance.
        defaults = cls()
        training = dict(data.get("training", {})) if isinstance(data.get("training", {}), Mapping) else {}
        loss_weights = dict(DEFAULT_VAE_LOSS_WEIGHTS)
        loss_weights.update(training.get("loss_weights", data.get("loss_weights", {})))
        checkpoint_dir = training.get("checkpoint_dir", data.get("checkpoint_dir"))
        warm_start_checkpoint = training.get("warm_start_checkpoint", data.get("warm_start_checkpoint"))
        resume_from = training.get("resume_from", data.get("resume_from"))
        return cls(
            steps=int(training.get("steps", data.get("steps", defaults.steps))),
            batch_size=int(training.get("batch_size", data.get("batch_size", defaults.batch_size))),
            seed=int(data.get("seed", training.get("seed", defaults.seed))),
            lr=float(training.get("lr", data.get("lr", defaults.lr))),
            device=training.get("device", data.get("device", defaults.device)),
            precision=training.get("precision", data.get("precision", defaults.precision)),
            loss_weights=loss_weights,
            ssim_window_size=int(
                training.get("ssim_window_size", data.get("ssim_window_size", defaults.ssim_window_size))
            ),
            lpips_num_slices=int(
                training.get("lpips_num_slices", data.get("lpips_num_slices", defaults.lpips_num_slices))
            ),
            grad_clip_norm=float(
                training.get("grad_clip_norm", data.get("grad_clip_norm", defaults.grad_clip_norm))
            ),
            warm_start_checkpoint=Path(warm_start_checkpoint) if warm_start_checkpoint else None,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_every_steps=int(
                training.get(
                    "checkpoint_every_steps", data.get("checkpoint_every_steps", defaults.checkpoint_every_steps)
                )
            ),
            checkpoint_at_end=bool(
                training.get("checkpoint_at_end", data.get("checkpoint_at_end", defaults.checkpoint_at_end))
            ),
            checkpoint_max_bytes=int(
                training.get("checkpoint_max_bytes", data.get("checkpoint_max_bytes", defaults.checkpoint_max_bytes))
            ),
            log_every_steps=int(training.get("log_every_steps", data.get("log_every_steps", defaults.log_every_steps))),
            resume_from=Path(resume_from) if resume_from else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "lr": self.lr,
            "device": self.device,
            "precision": self.precision,
            "loss_weights": dict(self.loss_weights),
            "ssim_window_size": self.ssim_window_size,
            "lpips_num_slices": self.lpips_num_slices,
            "grad_clip_norm": self.grad_clip_norm,
            "checkpoint_at_end": self.checkpoint_at_end,
            "log_every_steps": self.log_every_steps,
        }


@dataclass(frozen=True, slots=True)
class Stage1VAETrainResult:
    steps: int
    losses: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def seconds_per_step(self) -> float:
        return self.elapsed_seconds / self.steps if self.steps else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "losses": self.losses,
            "final_loss": self.final_loss,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_step": self.seconds_per_step,
        }


def run_stage1_vae_train(
    config: Stage1VAEConfig | Mapping[str, Any] | None,
    *,
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    loader: DataLoader[RawBatch],
) -> Stage1VAETrainResult:
    # Typed against KLVAEEncoder/KLVAEDecoder specifically (not BaseEncoder/BaseDecoder)
    # since this loop needs encode_dist(), which isn't part of the base ABC contract.
    cfg = _coerce_config(config)
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if cfg.warm_start_checkpoint is not None:
        state = load_checkpoint(cfg.warm_start_checkpoint, map_location=device)
        if "encoder" in state:
            load_state_dict_tolerant(encoder, state["encoder"])
        if "decoder" in state:
            load_state_dict_tolerant(decoder, state["decoder"])

    trainable_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)

    start_step = 0
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))

    lpips_net: nn.Module | None = None
    if cfg.loss_weights.get("lpips", 0.0) > 0:
        lpips_net = build_lpips_net(device)

    iterator = iter(loader)
    autocast_ctx = _autocast_context(device, cfg.precision)
    batches_per_epoch = _loader_length(loader)
    if cfg.log_every_steps > 0:
        _log_training_progress(
            f"stage1_vae train start: steps={cfg.steps} batch_size={cfg.batch_size} "
            f"device={device.type} batches_per_epoch={batches_per_epoch or 'unknown'}"
        )

    losses: list[float] = []
    last_checkpoint_step: int | None = None
    train_start = time.perf_counter()
    for step in range(start_step, start_step + cfg.steps):
        step_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_raw_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            total_loss = _compute_vae_loss(encoder, decoder, batch, cfg, lpips_net=lpips_net)
        total_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
        optimizer.step()
        _sync_if_cuda(device)
        loss_value = float(total_loss.detach().cpu())
        losses.append(loss_value)

        if (
            cfg.checkpoint_dir is not None
            and cfg.checkpoint_every_steps
            and (step + 1) % cfg.checkpoint_every_steps == 0
        ):
            _save_step_checkpoint(cfg, encoder, decoder, optimizer, step + 1)
            last_checkpoint_step = step + 1

        current_step = step + 1
        if cfg.log_every_steps > 0 and (
            len(losses) == 1 or current_step % cfg.log_every_steps == 0 or len(losses) == cfg.steps
        ):
            elapsed = time.perf_counter() - train_start
            step_elapsed = time.perf_counter() - step_start
            _log_training_progress(
                f"stage1_vae step={current_step}/{start_step + cfg.steps} loss={loss_value:.6f} "
                f"step_sec={step_elapsed:.3f} avg_sec_per_step={elapsed / len(losses):.3f}"
            )

    final_step = start_step + len(losses)
    if (
        cfg.checkpoint_dir is not None
        and cfg.checkpoint_at_end
        and losses
        and last_checkpoint_step != final_step
    ):
        _save_step_checkpoint(cfg, encoder, decoder, optimizer, final_step)

    elapsed_seconds = time.perf_counter() - train_start
    return Stage1VAETrainResult(steps=len(losses), losses=losses, elapsed_seconds=elapsed_seconds)


def _compute_vae_loss(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    batch: RawBatch,
    cfg: Stage1VAEConfig,
    *,
    lpips_net: nn.Module | None,
) -> torch.Tensor:
    # Reparameterize once, here, rather than calling encode() and then encode_dist()
    # separately — that would run the encoder twice with two different random samples,
    # silently decorrelating the KL term from the reconstruction term.
    weights = cfg.loss_weights
    mean, logvar = encoder.encode_dist(batch.image, batch.source_domain)
    eps = torch.randn_like(mean)
    z = mean + eps * torch.exp(0.5 * logvar)
    reconstructed = decoder.decode(z, batch.source_domain)

    total = weights.get("nrmse", 0.0) * nrmse_loss(reconstructed, batch.image)
    if weights.get("ssim", 0.0) > 0:
        # ssim_loss dispatches on rank: 5D volumes -> ssim3d (avg_pool3d), 4D -> 2D ssim.
        total = total + weights["ssim"] * ssim_loss(reconstructed, batch.image, window_size=cfg.ssim_window_size)
    if weights.get("lpips", 0.0) > 0:
        # LPIPS wraps a 2D VGG net: for 5D volumes use the slice-averaged variant, for 4D
        # (2D slices) the plain one.
        if reconstructed.ndim == 5:
            lpips_value = lpips_loss_3d(
                reconstructed, batch.image, net=lpips_net, num_slices=cfg.lpips_num_slices
            )
        else:
            lpips_value = lpips_loss(reconstructed, batch.image, net=lpips_net)
        total = total + weights["lpips"] * lpips_value
    total = total + weights.get("kl", 0.0) * kl_divergence(mean, logvar)
    return total


def _save_step_checkpoint(
    cfg: Stage1VAEConfig,
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    assert cfg.checkpoint_dir is not None
    filename = checkpoint_filename("vae", "kl_vae", step)
    save_checkpoint(
        cfg.checkpoint_dir / filename,
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        max_bytes=cfg.checkpoint_max_bytes,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _coerce_config(config: Stage1VAEConfig | Mapping[str, Any] | None) -> Stage1VAEConfig:
    if config is None:
        return Stage1VAEConfig()
    if isinstance(config, Stage1VAEConfig):
        return config
    return Stage1VAEConfig.from_mapping(config)


def _loader_length(loader: DataLoader[RawBatch]) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _log_training_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is 'cuda', but CUDA is not available.")
    if device != "cpu" and device != "cuda":
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("precision must be 'fp32' or 'bf16'.")
