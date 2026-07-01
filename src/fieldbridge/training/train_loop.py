"""Real training loop for the Etapa 2 latent translator (configurable losses, precision, resume).

Unlike `smoke_train.py` (a fixed CPU integration smoke test), this loop is meant to be
reused across the ablation ladder (StarGAN-v2, OT-CFM, SB): it owns precision, gradient
checkpointing, resume-from-checkpoint, and the loss terms that are common to any latent
translator (reconstruction, transport-cost, cycle, identity). Model-specific losses
(adversarial, flow-matching, KL) are composed by each ladder stage on top of this loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import make_synthetic_loader
from fieldbridge.models.autoencoders.base import BaseDecoder, BaseEncoder
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.batch import move_raw_batch
from fieldbridge.training.checkpoints import checkpoint_filename, load_checkpoint, save_checkpoint
from fieldbridge.training.losses import (
    cycle_consistency_loss,
    identity_loss,
    reconstruction_mse,
    transport_cost_loss,
)
from fieldbridge.utils.seeding import seed_everything

Precision = Literal["fp32", "bf16"]
Stage = Literal["autoencoder", "translator"]

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "reconstruction": 1.0,
    "transport_cost": 0.0,
    "cycle": 0.0,
    "identity": 0.0,
}


@dataclass(frozen=True, slots=True)
class TrainLoopConfig:
    steps: int = 2
    batch_size: int = 2
    num_samples: int = 4
    volume_shape: tuple[int, int, int, int] = (1, 8, 8, 8)
    seed: int = 13
    lr: float = 1e-2
    stage: Stage = "translator"
    variant: str = "identity"
    precision: Precision = "fp32"
    gradient_checkpointing: bool = False
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 0
    resume_from: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainLoopConfig":
        # NOTE: `cls.<field>` is not a usable default here — with frozen+slots dataclasses,
        # accessing a field on the *class* (rather than an instance) returns a slot
        # `member_descriptor`, not the field's default value. Use a real instance instead.
        defaults = cls()
        training = dict(data.get("training", {})) if isinstance(data.get("training", {}), Mapping) else {}
        dataset = dict(data.get("data", {})) if isinstance(data.get("data", {}), Mapping) else {}
        model = dict(data.get("model", {})) if isinstance(data.get("model", {}), Mapping) else {}
        loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        loss_weights.update(training.get("loss_weights", data.get("loss_weights", {})))
        checkpoint_dir = training.get("checkpoint_dir", data.get("checkpoint_dir"))
        resume_from = training.get("resume_from", data.get("resume_from"))
        return cls(
            steps=int(training.get("steps", data.get("steps", defaults.steps))),
            batch_size=int(training.get("batch_size", data.get("batch_size", defaults.batch_size))),
            num_samples=int(dataset.get("num_samples", data.get("num_samples", defaults.num_samples))),
            volume_shape=tuple(dataset.get("volume_shape", data.get("volume_shape", defaults.volume_shape))),
            seed=int(data.get("seed", training.get("seed", defaults.seed))),
            lr=float(training.get("lr", data.get("lr", defaults.lr))),
            stage=training.get("stage", data.get("stage", defaults.stage)),
            variant=model.get("variant", data.get("variant", defaults.variant)),
            precision=training.get("precision", data.get("precision", defaults.precision)),
            gradient_checkpointing=bool(
                training.get(
                    "gradient_checkpointing", data.get("gradient_checkpointing", defaults.gradient_checkpointing)
                )
            ),
            loss_weights=loss_weights,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_every_steps=int(
                training.get(
                    "checkpoint_every_steps", data.get("checkpoint_every_steps", defaults.checkpoint_every_steps)
                )
            ),
            resume_from=Path(resume_from) if resume_from else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "volume_shape": list(self.volume_shape),
            "seed": self.seed,
            "lr": self.lr,
            "stage": self.stage,
            "variant": self.variant,
            "precision": self.precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "loss_weights": dict(self.loss_weights),
        }


@dataclass(frozen=True, slots=True)
class TrainLoopResult:
    steps: int
    losses: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "losses": self.losses, "final_loss": self.final_loss}


def assert_frozen(module: nn.Module) -> None:
    """Verify a module has no trainable parameters (e.g. the Etapa 1 VAE before Etapa 2 training)."""

    trainable = [name for name, param in module.named_parameters() if param.requires_grad]
    if trainable:
        raise RuntimeError(f"Expected {type(module).__name__} to be frozen, but found trainable params: {trainable}")


def run_train_loop(
    config: TrainLoopConfig | Mapping[str, Any] | None,
    *,
    encoder: BaseEncoder,
    decoder: BaseDecoder,
    translator: BaseTranslator,
    loader: DataLoader[RawBatch] | None = None,
) -> TrainLoopResult:
    cfg = _coerce_config(config)
    seed_everything(cfg.seed)
    device = torch.device("cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    translator = translator.to(device)

    if cfg.stage == "translator":
        assert_frozen(encoder)
        assert_frozen(decoder)
        trainable_params = list(translator.parameters())
    else:
        trainable_params = list(encoder.parameters()) + list(decoder.parameters()) + list(translator.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)

    start_step = 0
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        translator.load_state_dict(state["translator"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))

    # Resume restores model/optimizer state and the step counter, but the data loader
    # itself always restarts from index 0 — it does not persist iterator position.
    data_loader = loader if loader is not None else make_synthetic_loader(
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        volume_shape=cfg.volume_shape,
        seed=cfg.seed,
        pair_sampling="random_any_to_any",
    )
    iterator = iter(data_loader)
    autocast_ctx = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16) if cfg.precision == "bf16" else nullcontext()
    )

    losses: list[float] = []
    for step in range(start_step, start_step + cfg.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)
        batch = move_raw_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            total_loss = _compute_total_loss(encoder, decoder, translator, batch, cfg)
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))

        if cfg.checkpoint_dir is not None and cfg.checkpoint_every_steps and (step + 1) % cfg.checkpoint_every_steps == 0:
            _save_step_checkpoint(cfg, translator, optimizer, step + 1)

    return TrainLoopResult(steps=len(losses), losses=losses)


def _compute_total_loss(
    encoder: BaseEncoder,
    decoder: BaseDecoder,
    translator: BaseTranslator,
    batch: RawBatch,
    cfg: TrainLoopConfig,
) -> torch.Tensor:
    weights = cfg.loss_weights
    z_source = encoder.encode(batch.image, batch.source_domain)
    z_translated = _translate(translator, z_source, batch.source_domain, batch.target_domain, cfg)
    reconstructed = decoder.decode(z_translated, batch.target_domain)

    total = weights.get("reconstruction", 0.0) * reconstruction_mse(reconstructed, batch.image)
    total = total + weights.get("transport_cost", 0.0) * transport_cost_loss(z_source, z_translated)

    if weights.get("cycle", 0.0) > 0:
        z_cycled = _translate(translator, z_translated, batch.target_domain, batch.source_domain, cfg)
        cycled = decoder.decode(z_cycled, batch.source_domain)
        total = total + weights["cycle"] * cycle_consistency_loss(batch.image, cycled)

    if weights.get("identity", 0.0) > 0:
        z_identity = _translate(translator, z_source, batch.source_domain, batch.source_domain, cfg)
        identity_reconstruction = decoder.decode(z_identity, batch.source_domain)
        total = total + weights["identity"] * identity_loss(batch.image, identity_reconstruction)

    return total


def _translate(
    translator: BaseTranslator,
    z: torch.Tensor,
    source_domain: Any,
    target_domain: Any,
    cfg: TrainLoopConfig,
) -> torch.Tensor:
    if not cfg.gradient_checkpointing:
        return translator(z, source_domain, target_domain)
    return torch.utils.checkpoint.checkpoint(translator, z, source_domain, target_domain, use_reentrant=False)


def _save_step_checkpoint(
    cfg: TrainLoopConfig, translator: BaseTranslator, optimizer: torch.optim.Optimizer, step: int
) -> None:
    assert cfg.checkpoint_dir is not None
    filename = checkpoint_filename(cfg.stage, cfg.variant, step)
    save_checkpoint(
        cfg.checkpoint_dir / filename,
        {"translator": translator.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _coerce_config(config: TrainLoopConfig | Mapping[str, Any] | None) -> TrainLoopConfig:
    if config is None:
        return TrainLoopConfig()
    if isinstance(config, TrainLoopConfig):
        return config
    return TrainLoopConfig.from_mapping(config)
