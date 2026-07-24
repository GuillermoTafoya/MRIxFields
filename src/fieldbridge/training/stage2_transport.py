"""Etapa-2 latent transport training (OT-CFM / Schrodinger bridge) on the frozen-VAE bank.

Trains a single shared velocity field ``v_theta(z_t, t, c_source, c_target)`` on precomputed
latents (the VAE stays frozen and is not even loaded here). The ablation ladder is expressed
as config, not new networks:

- ``coupling``: how a minibatch of source latents is paired to target latents.
  ``independent`` (random) or ``ot`` (exact minibatch optimal-transport assignment, the
  squared-L2 Hungarian plan) — the unpaired coupling the plan calls for.
- ``bridge``: the conditional path + regression target.
  ``ot_cfm`` (deterministic linear interpolation, target velocity ``z1 - z0``) or
  ``schrodinger`` (Brownian bridge with volatility ``sigma``, target drift
  ``(z1 - z_t)/(1 - t)`` — simulation-free bridge matching).

Loss ladder (training-conventions): every non-flow term defaults to weight 0 and is a
one-number switch. ``flow`` is the flow/bridge-matching MSE; ``transport_cost`` penalizes
``||v||^2`` (minimal displacement); ``identity`` enforces ``v(z, t, c, c) = 0`` on real
latents. ``cycle`` needs the ODE sampler and is intentionally not yet implemented.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F

from fieldbridge.data.domains import Domain
from fieldbridge.data.latent_bank_dataset import LatentBankIndex, LatentStats
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.checkpoints import checkpoint_filename, load_checkpoint, save_checkpoint
from fieldbridge.utils.seeding import seed_everything

Precision = Literal["fp32", "bf16"]
Device = Literal["auto", "cpu", "cuda"]
Coupling = Literal["independent", "ot"]
Bridge = Literal["ot_cfm", "schrodinger"]

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "flow": 1.0,
    "transport_cost": 0.0,
    "identity": 0.0,
    "cycle": 0.0,
}


@dataclass(frozen=True, slots=True)
class Stage2TransportConfig:
    steps: int = 2
    batch_size: int = 4
    seed: int = 13
    lr: float = 2e-4
    device: Device = "auto"
    precision: Precision = "bf16"
    coupling: Coupling = "ot"
    bridge: Bridge = "ot_cfm"
    sigma: float = 0.1  # Schrodinger-bridge volatility (bridge="schrodinger" only)
    time_eps: float = 1e-3  # clamp on t and (1 - t)
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 0
    checkpoint_at_end: bool = True
    checkpoint_max_bytes: int = 500_000_000
    val_every_steps: int = 0
    val_batches: int = 4
    log_every_steps: int = 0
    resume_from: Path | None = None
    variant: str = "flow_matching_latent"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage2TransportConfig":
        defaults = cls()
        training = data.get("training", {})
        training = dict(training) if isinstance(training, Mapping) else {}

        def pick(key: str, current: Any) -> Any:
            return training.get(key, data.get(key, current))

        weights = dict(DEFAULT_LOSS_WEIGHTS)
        weights.update(training.get("loss_weights", data.get("loss_weights", {})))
        checkpoint_dir = pick("checkpoint_dir", None)
        resume_from = pick("resume_from", None)
        return cls(
            steps=int(pick("steps", defaults.steps)),
            batch_size=int(pick("batch_size", defaults.batch_size)),
            seed=int(pick("seed", defaults.seed)),
            lr=float(pick("lr", defaults.lr)),
            device=pick("device", defaults.device),
            precision=pick("precision", defaults.precision),
            coupling=pick("coupling", defaults.coupling),
            bridge=pick("bridge", defaults.bridge),
            sigma=float(pick("sigma", defaults.sigma)),
            time_eps=float(pick("time_eps", defaults.time_eps)),
            loss_weights=weights,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_every_steps=int(pick("checkpoint_every_steps", defaults.checkpoint_every_steps)),
            checkpoint_at_end=bool(pick("checkpoint_at_end", defaults.checkpoint_at_end)),
            checkpoint_max_bytes=int(pick("checkpoint_max_bytes", defaults.checkpoint_max_bytes)),
            val_every_steps=int(pick("val_every_steps", defaults.val_every_steps)),
            val_batches=int(pick("val_batches", defaults.val_batches)),
            log_every_steps=int(pick("log_every_steps", defaults.log_every_steps)),
            resume_from=Path(resume_from) if resume_from else None,
            variant=str(pick("variant", defaults.variant)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "lr": self.lr,
            "device": self.device,
            "precision": self.precision,
            "coupling": self.coupling,
            "bridge": self.bridge,
            "sigma": self.sigma,
            "time_eps": self.time_eps,
            "loss_weights": dict(self.loss_weights),
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class Stage2TransportResult:
    steps: int
    losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    best_val: float | None = None

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def seconds_per_step(self) -> float:
        return self.elapsed_seconds / len(self.losses) if self.losses else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "losses": self.losses,
            "val_losses": self.val_losses,
            "final_loss": self.final_loss,
            "seconds_per_step": self.seconds_per_step,
            "elapsed_seconds": self.elapsed_seconds,
            "best_val": self.best_val,
        }


def run_stage2_transport_train(
    config: Stage2TransportConfig | Mapping[str, Any] | None,
    *,
    translator: BaseTranslator,
    train_index: LatentBankIndex,
    stats: LatentStats,
    val_index: LatentBankIndex | None = None,
) -> Stage2TransportResult:
    cfg = _coerce_config(config)
    _validate_config(cfg)
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    translator = translator.to(device)
    optimizer = torch.optim.Adam(translator.parameters(), lr=cfg.lr)
    sampler = torch.Generator().manual_seed(cfg.seed)

    start_step = 0
    best_val: float | None = None
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        translator.load_state_dict(state["translator"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        best_val = state.get("best_val")

    autocast_ctx = _autocast_context(device, cfg.precision)
    if cfg.log_every_steps > 0:
        _log(
            f"stage2_transport start steps={cfg.steps} batch={cfg.batch_size} device={device.type} "
            f"coupling={cfg.coupling} bridge={cfg.bridge} train_records={len(train_index)} "
            f"val_records={len(val_index) if val_index else 0}"
        )

    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    last_ckpt_step: int | None = None
    train_start = time.perf_counter()
    for step in range(start_step, start_step + cfg.steps):
        step_start = time.perf_counter()
        translator.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            total, terms = _transport_loss(translator, train_index, stats, cfg, device, sampler)
        total.backward()
        optimizer.step()
        _sync_if_cuda(device)
        losses.append(float(total.detach().cpu()))

        current_step = step + 1
        if cfg.checkpoint_dir is not None and cfg.checkpoint_every_steps and current_step % cfg.checkpoint_every_steps == 0:
            _save_checkpoint(cfg, translator, optimizer, current_step, best_val, unique=True)
            last_ckpt_step = current_step

        if val_index is not None and cfg.val_every_steps and current_step % cfg.val_every_steps == 0:
            val = _validate(translator, val_index, stats, cfg, device)
            val_losses.append((current_step, val))
            if best_val is None or val < best_val:
                best_val = val
                if cfg.checkpoint_dir is not None:
                    _save_checkpoint(cfg, translator, optimizer, current_step, best_val, name="best")
            if cfg.log_every_steps > 0:
                _log(f"stage2_transport step={current_step} val_flow={val:.6f} best_val={best_val:.6f}")

        if cfg.log_every_steps > 0 and (len(losses) == 1 or current_step % cfg.log_every_steps == 0 or len(losses) == cfg.steps):
            elapsed = time.perf_counter() - train_start
            term_str = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
            _log(
                f"stage2_transport step={current_step}/{start_step + cfg.steps} loss={losses[-1]:.6f} "
                f"[{term_str}] step_sec={time.perf_counter() - step_start:.3f} "
                f"avg_sec_per_step={elapsed / len(losses):.3f}"
            )

    final_step = start_step + len(losses)
    if cfg.checkpoint_dir is not None and cfg.checkpoint_at_end and losses:
        _save_checkpoint(cfg, translator, optimizer, final_step, best_val, name="last")
        if last_ckpt_step != final_step and cfg.checkpoint_every_steps == 0:
            pass
    return Stage2TransportResult(
        steps=len(losses),
        losses=losses,
        val_losses=val_losses,
        elapsed_seconds=time.perf_counter() - train_start,
        best_val=best_val,
    )


def _transport_loss(
    translator: BaseTranslator,
    index: LatentBankIndex,
    stats: LatentStats,
    cfg: Stage2TransportConfig,
    device: torch.device,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    z0, dom_s = _sample_normalized_batch(index, stats, cfg.batch_size, device, sampler)
    z1, dom_t = _sample_normalized_batch(index, stats, cfg.batch_size, device, sampler)
    if cfg.coupling == "ot":
        perm = _ot_assignment(z0, z1)
        z1 = z1[perm]
        dom_t = [dom_t[i] for i in perm.tolist()]

    t = _sample_time(cfg.batch_size, device, sampler, cfg.time_eps)
    z_t, target = _bridge_sample(z0, z1, t, cfg, sampler)
    velocity = translator(z_t, dom_s, dom_t, t)

    weights = cfg.loss_weights
    flow = F.mse_loss(velocity, target)
    total = weights.get("flow", 1.0) * flow
    terms: dict[str, float] = {"flow": float(flow.detach().cpu())}

    if weights.get("transport_cost", 0.0) > 0.0:
        transport = velocity.square().mean()
        total = total + weights["transport_cost"] * transport
        terms["transport_cost"] = float(transport.detach().cpu())

    if weights.get("identity", 0.0) > 0.0:
        z_id, dom_id = _sample_normalized_batch(index, stats, cfg.batch_size, device, sampler)
        t_id = _sample_time(cfg.batch_size, device, sampler, cfg.time_eps)
        v_id = translator(z_id, dom_id, dom_id, t_id)  # source==target ⇒ velocity should be 0
        identity = v_id.square().mean()
        total = total + weights["identity"] * identity
        terms["identity"] = float(identity.detach().cpu())

    if weights.get("cycle", 0.0) > 0.0:
        raise NotImplementedError(
            "cycle loss for latent transport needs the ODE sampler (A->B->A); "
            "keep loss_weights.cycle at 0 until it is implemented."
        )
    return total, terms


def _bridge_sample(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: torch.Tensor,
    cfg: Stage2TransportConfig,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_b = t.reshape(-1, *([1] * (z0.ndim - 1)))
    if cfg.bridge == "ot_cfm":
        z_t = (1.0 - t_b) * z0 + t_b * z1
        target = z1 - z0
        return z_t, target
    if cfg.bridge == "schrodinger":
        mean = (1.0 - t_b) * z0 + t_b * z1
        std = cfg.sigma * torch.sqrt((t_b * (1.0 - t_b)).clamp_min(0.0))
        eps = torch.randn(z0.shape, generator=sampler, device="cpu").to(z0.device, z0.dtype)
        z_t = mean + std * eps
        target = (z1 - z_t) / (1.0 - t_b).clamp_min(cfg.time_eps)
        return z_t, target
    raise ValueError(f"Unknown bridge {cfg.bridge!r}.")


def _ot_assignment(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Exact minibatch OT coupling: squared-L2 Hungarian assignment (column perm for z1)."""

    from scipy.optimize import linear_sum_assignment

    flat0 = z0.reshape(z0.shape[0], -1).float()
    flat1 = z1.reshape(z1.shape[0], -1).float()
    cost = torch.cdist(flat0, flat1).square().detach().cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return torch.as_tensor(col, dtype=torch.long, device=z0.device)


def _sample_time(batch_size: int, device: torch.device, sampler: torch.Generator, eps: float) -> torch.Tensor:
    t = torch.rand(batch_size, generator=sampler, device="cpu")
    return t.clamp(eps, 1.0 - eps).to(device)


def _sample_normalized_batch(
    index: LatentBankIndex,
    stats: LatentStats,
    batch_size: int,
    device: torch.device,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, list[Domain]]:
    indices = torch.randint(0, len(index), (batch_size,), generator=sampler).tolist()
    latents, domains = index.load_batch(indices)
    latents = stats.normalize(latents).to(device)
    return latents, domains


@torch.no_grad()
def _validate(
    translator: BaseTranslator,
    index: LatentBankIndex,
    stats: LatentStats,
    cfg: Stage2TransportConfig,
    device: torch.device,
) -> float:
    translator.eval()
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    total = 0.0
    for _ in range(max(1, cfg.val_batches)):
        z0, dom_s = _sample_normalized_batch(index, stats, cfg.batch_size, device, generator)
        z1, dom_t = _sample_normalized_batch(index, stats, cfg.batch_size, device, generator)
        if cfg.coupling == "ot":
            perm = _ot_assignment(z0, z1)
            z1 = z1[perm]
            dom_t = [dom_t[i] for i in perm.tolist()]
        t = _sample_time(cfg.batch_size, device, generator, cfg.time_eps)
        z_t, target = _bridge_sample(z0, z1, t, cfg, generator)
        velocity = translator(z_t, dom_s, dom_t, t)
        total += float(F.mse_loss(velocity, target).cpu())
    return total / max(1, cfg.val_batches)


def _save_checkpoint(
    cfg: Stage2TransportConfig,
    translator: BaseTranslator,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float | None,
    *,
    name: str | None = None,
    unique: bool = False,
) -> None:
    assert cfg.checkpoint_dir is not None
    state: dict[str, Any] = {
        "translator": translator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
    }
    if unique:
        filename = checkpoint_filename("transport", cfg.variant, step)
        overwrite = False
    else:
        filename = f"transport_{cfg.variant}_{name}.pt"
        overwrite = True
    save_checkpoint(
        cfg.checkpoint_dir / filename,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=overwrite,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _validate_config(cfg: Stage2TransportConfig) -> None:
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if cfg.coupling not in ("independent", "ot"):
        raise ValueError(f"Unknown coupling {cfg.coupling!r}.")
    if cfg.bridge not in ("ot_cfm", "schrodinger"):
        raise ValueError(f"Unknown bridge {cfg.bridge!r}.")
    if cfg.bridge == "schrodinger" and cfg.sigma <= 0:
        raise ValueError("schrodinger bridge requires sigma > 0.")


def _coerce_config(config: Stage2TransportConfig | Mapping[str, Any] | None) -> Stage2TransportConfig:
    if config is None:
        return Stage2TransportConfig()
    if isinstance(config, Stage2TransportConfig):
        return config
    return Stage2TransportConfig.from_mapping(config)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is 'cuda', but CUDA is not available.")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("precision must be 'fp32' or 'bf16'.")


__all__ = [
    "Stage2TransportConfig",
    "Stage2TransportResult",
    "run_stage2_transport_train",
    "DEFAULT_LOSS_WEIGHTS",
]
