"""Conditional-diffuser training entry point for Etapa 1's second sub-stage.

Staged, not end-to-end: this trains `DenoisingUNet` on top of a VAE (`KLVAEEncoder`)
already trained by `stage1_vae.py`. The VAE is FROZEN by default (`train_vae_jointly`
defaults to False), reusing `assert_frozen` from `train_loop.py` — the same mechanism
this project already uses for the Etapa1(VAE) -> Etapa2(translator) boundary, applied
one stage earlier. Frozen-by-default avoids the diffuser chasing a latent distribution
that's still moving under its own optimizer.

Open question, deliberately NOT resolved here: if `train_vae_jointly=True` is ever
used, should the VAE's own SSIM/nRMSE/LPIPS/KL losses stay active alongside the
noise-prediction loss, or does joint training mean only the diffusion loss
backpropagates into the VAE? Left for when that option is actually turned on.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import RawBatch
from fieldbridge.models.autoencoders.kl_vae import KLVAEEncoder
from fieldbridge.models.diffusion.denoising_unet import DenoisingUNet
from fieldbridge.models.diffusion.schedule import DiffusionSchedule, make_schedule, q_sample
from fieldbridge.training.batch import move_raw_batch
from fieldbridge.training.checkpoints import checkpoint_filename, load_checkpoint, save_checkpoint
from fieldbridge.training.train_loop import assert_frozen
from fieldbridge.utils.seeding import seed_everything

Precision = Literal["fp32", "bf16"]
Device = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class Stage2DiffuserConfig:
    steps: int = 2
    batch_size: int = 2
    seed: int = 13
    lr: float = 1e-4
    device: Device = "auto"
    precision: Precision = "fp32"
    num_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    train_vae_jointly: bool = False
    vae_checkpoint: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 0
    checkpoint_at_end: bool = False
    checkpoint_max_bytes: int = 10_000_000
    log_every_steps: int = 0
    resume_from: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage2DiffuserConfig":
        defaults = cls()
        training = dict(data.get("training", {})) if isinstance(data.get("training", {}), Mapping) else {}
        checkpoint_dir = training.get("checkpoint_dir", data.get("checkpoint_dir"))
        vae_checkpoint = training.get("vae_checkpoint", data.get("vae_checkpoint"))
        resume_from = training.get("resume_from", data.get("resume_from"))
        return cls(
            steps=int(training.get("steps", data.get("steps", defaults.steps))),
            batch_size=int(training.get("batch_size", data.get("batch_size", defaults.batch_size))),
            seed=int(data.get("seed", training.get("seed", defaults.seed))),
            lr=float(training.get("lr", data.get("lr", defaults.lr))),
            device=training.get("device", data.get("device", defaults.device)),
            precision=training.get("precision", data.get("precision", defaults.precision)),
            num_timesteps=int(training.get("num_timesteps", data.get("num_timesteps", defaults.num_timesteps))),
            beta_start=float(training.get("beta_start", data.get("beta_start", defaults.beta_start))),
            beta_end=float(training.get("beta_end", data.get("beta_end", defaults.beta_end))),
            train_vae_jointly=bool(
                training.get("train_vae_jointly", data.get("train_vae_jointly", defaults.train_vae_jointly))
            ),
            vae_checkpoint=Path(vae_checkpoint) if vae_checkpoint else None,
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
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "train_vae_jointly": self.train_vae_jointly,
            "checkpoint_at_end": self.checkpoint_at_end,
            "log_every_steps": self.log_every_steps,
        }


@dataclass(frozen=True, slots=True)
class Stage2DiffuserTrainResult:
    steps: int
    losses: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "losses": self.losses,
            "final_loss": self.final_loss,
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_stage2_diffuser_train(
    config: Stage2DiffuserConfig | Mapping[str, Any] | None,
    *,
    unet: DenoisingUNet,
    encoder: KLVAEEncoder,
    loader: DataLoader[RawBatch],
) -> Stage2DiffuserTrainResult:
    cfg = _coerce_config(config)
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    unet = unet.to(device)
    encoder = encoder.to(device)

    if cfg.vae_checkpoint is not None:
        state = load_checkpoint(cfg.vae_checkpoint, map_location=device)
        encoder.load_state_dict(state["encoder"])

    trainable_params = list(unet.parameters())
    if cfg.train_vae_jointly:
        trainable_params += list(encoder.parameters())
    else:
        assert_frozen(_freeze(encoder))
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)

    start_step = 0
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        unet.load_state_dict(state["unet"])
        if cfg.train_vae_jointly and "encoder" in state:
            encoder.load_state_dict(state["encoder"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))

    schedule = make_schedule(cfg.num_timesteps, beta_start=cfg.beta_start, beta_end=cfg.beta_end)

    iterator = iter(loader)
    autocast_ctx = _autocast_context(device, cfg.precision)
    batches_per_epoch = _loader_length(loader)
    if cfg.log_every_steps > 0:
        _log_training_progress(
            f"stage2_diffuser train start: steps={cfg.steps} batch_size={cfg.batch_size} "
            f"device={device.type} train_vae_jointly={cfg.train_vae_jointly} "
            f"batches_per_epoch={batches_per_epoch or 'unknown'}"
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
            total_loss = _compute_diffuser_loss(unet, encoder, batch, schedule, cfg)
        total_loss.backward()
        optimizer.step()
        _sync_if_cuda(device)
        loss_value = float(total_loss.detach().cpu())
        losses.append(loss_value)

        if (
            cfg.checkpoint_dir is not None
            and cfg.checkpoint_every_steps
            and (step + 1) % cfg.checkpoint_every_steps == 0
        ):
            _save_step_checkpoint(cfg, unet, encoder, optimizer, step + 1)
            last_checkpoint_step = step + 1

        current_step = step + 1
        if cfg.log_every_steps > 0 and (
            len(losses) == 1 or current_step % cfg.log_every_steps == 0 or len(losses) == cfg.steps
        ):
            elapsed = time.perf_counter() - train_start
            step_elapsed = time.perf_counter() - step_start
            _log_training_progress(
                f"stage2_diffuser step={current_step}/{start_step + cfg.steps} loss={loss_value:.6f} "
                f"step_sec={step_elapsed:.3f} avg_sec_per_step={elapsed / len(losses):.3f}"
            )

    final_step = start_step + len(losses)
    if (
        cfg.checkpoint_dir is not None
        and cfg.checkpoint_at_end
        and losses
        and last_checkpoint_step != final_step
    ):
        _save_step_checkpoint(cfg, unet, encoder, optimizer, final_step)

    elapsed_seconds = time.perf_counter() - train_start
    return Stage2DiffuserTrainResult(steps=len(losses), losses=losses, elapsed_seconds=elapsed_seconds)


def _compute_diffuser_loss(
    unet: DenoisingUNet,
    encoder: KLVAEEncoder,
    batch: RawBatch,
    schedule: DiffusionSchedule,
    cfg: Stage2DiffuserConfig,
) -> torch.Tensor:
    with torch.set_grad_enabled(cfg.train_vae_jointly):
        z0 = encoder.encode(batch.image, batch.source_domain)
    t = torch.randint(0, schedule.num_timesteps, (z0.shape[0],), device=z0.device)
    z_t, noise = q_sample(z0, t, schedule)
    noise_pred = unet(z_t, t, batch.source_domain)
    return F.mse_loss(noise_pred, noise)


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _save_step_checkpoint(
    cfg: Stage2DiffuserConfig,
    unet: DenoisingUNet,
    encoder: KLVAEEncoder,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    assert cfg.checkpoint_dir is not None
    filename = checkpoint_filename("diffuser", "field_cond_unet", step)
    state: dict[str, Any] = {
        "unet": unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if cfg.train_vae_jointly:
        state["encoder"] = encoder.state_dict()
    save_checkpoint(
        cfg.checkpoint_dir / filename,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _coerce_config(config: Stage2DiffuserConfig | Mapping[str, Any] | None) -> Stage2DiffuserConfig:
    if config is None:
        return Stage2DiffuserConfig()
    if isinstance(config, Stage2DiffuserConfig):
        return config
    return Stage2DiffuserConfig.from_mapping(config)


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
