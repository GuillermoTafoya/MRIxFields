"""CPU-only synthetic smoke training loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from fieldbridge.data.datasets import make_synthetic_loader
from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.translators.identity import IdentityTranslator
from fieldbridge.training.batch import move_raw_batch
from fieldbridge.training.losses import reconstruction_mse
from fieldbridge.utils.seeding import seed_everything


@dataclass(frozen=True, slots=True)
class SmokeTrainConfig:
    steps: int = 2
    batch_size: int = 2
    num_samples: int = 4
    volume_shape: tuple[int, int, int, int] = (1, 8, 8, 8)
    seed: int = 13
    lr: float = 1e-2
    initial_scale: float = 0.95

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SmokeTrainConfig":
        training = dict(data.get("training", {})) if isinstance(data.get("training", {}), Mapping) else {}
        dataset = dict(data.get("data", {})) if isinstance(data.get("data", {}), Mapping) else {}
        model = dict(data.get("model", {})) if isinstance(data.get("model", {}), Mapping) else {}
        return cls(
            steps=int(training.get("steps", data.get("steps", cls.steps))),
            batch_size=int(training.get("batch_size", data.get("batch_size", cls.batch_size))),
            num_samples=int(dataset.get("num_samples", data.get("num_samples", cls.num_samples))),
            volume_shape=tuple(dataset.get("volume_shape", data.get("volume_shape", cls.volume_shape))),
            seed=int(data.get("seed", training.get("seed", cls.seed))),
            lr=float(training.get("lr", data.get("lr", cls.lr))),
            initial_scale=float(model.get("initial_scale", data.get("initial_scale", cls.initial_scale))),
        )


@dataclass(frozen=True, slots=True)
class SmokeTrainResult:
    steps: int
    losses: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "losses": self.losses, "final_loss": self.final_loss}


def run_smoke_train(config: SmokeTrainConfig | Mapping[str, Any] | None = None) -> SmokeTrainResult:
    cfg = _coerce_config(config)
    seed_everything(cfg.seed)
    device = torch.device("cpu")
    loader = make_synthetic_loader(
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        volume_shape=cfg.volume_shape,
        seed=cfg.seed,
    )
    encoder = IdentityEncoder().to(device)
    decoder = IdentityDecoder().to(device)
    translator = IdentityTranslator(learnable_scale=True, initial_scale=cfg.initial_scale).to(device)
    optimizer = torch.optim.SGD(translator.parameters(), lr=cfg.lr)
    iterator = iter(loader)
    losses: list[float] = []

    for _ in range(cfg.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = move_raw_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        latent = encoder.encode(batch.image, batch.source_domain)
        translated = translator(latent, batch.source_domain, batch.target_domain)
        reconstructed = decoder.decode(translated, batch.target_domain)
        loss = reconstruction_mse(reconstructed, batch.image)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return SmokeTrainResult(steps=cfg.steps, losses=losses)


def _coerce_config(config: SmokeTrainConfig | Mapping[str, Any] | None) -> SmokeTrainConfig:
    if config is None:
        return SmokeTrainConfig()
    if isinstance(config, SmokeTrainConfig):
        return config
    return SmokeTrainConfig.from_mapping(config)

