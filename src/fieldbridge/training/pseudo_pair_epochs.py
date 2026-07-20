"""Epoch-based pseudo-pair training for the conditional U-Net baseline."""

from __future__ import annotations

import json
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

from fieldbridge.data.pseudo_pairs import PseudoPairSliceBatch
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.checkpoints import load_checkpoint, save_checkpoint
from fieldbridge.training.losses import combined_reconstruction_loss_components
from fieldbridge.utils.seeding import seed_everything

Device = Literal["auto", "cpu", "cuda"]

DEFAULT_PSEUDO_PAIR_LOSS_WEIGHTS: dict[str, float] = {
    "masked_l1": 1.0,
    "gradient": 0.2,
    "background": 0.5,
}
PSEUDO_PAIR_PIPELINE_VERSION = 2
_LOSS_COMPONENT_KEYS = ("total", "masked_l1", "gradient", "background")


@dataclass(frozen=True, slots=True)
class PseudoPairEpochConfig:
    epochs: int = 1
    batch_size: int = 8
    seed: int = 13
    lr: float = 1e-4
    weight_decay: float = 1e-2
    device: Device = "auto"
    amp: bool = False
    grad_clip_norm: float = 1.0
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PSEUDO_PAIR_LOSS_WEIGHTS))
    scheduler: dict[str, Any] = field(default_factory=lambda: {"name": "none"})
    checkpoint_dir: Path | None = None
    checkpoint_max_bytes: int = 200_000_000
    resume_from: Path | None = None
    history_filename: str = "history.jsonl"
    log_every_steps: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PseudoPairEpochConfig":
        defaults = cls()
        training = dict(data.get("training", {})) if isinstance(data.get("training", {}), Mapping) else {}
        loss_weights = dict(DEFAULT_PSEUDO_PAIR_LOSS_WEIGHTS)
        loss_weights.update(training.get("loss_weights", data.get("loss_weights", {})))
        scheduler = training.get("scheduler", data.get("scheduler", defaults.scheduler))
        checkpoint_dir = training.get("checkpoint_dir", data.get("checkpoint_dir"))
        resume_from = training.get("resume_from", data.get("resume_from"))
        return cls(
            epochs=int(training.get("epochs", data.get("epochs", defaults.epochs))),
            batch_size=int(training.get("batch_size", data.get("batch_size", defaults.batch_size))),
            seed=int(data.get("seed", training.get("seed", defaults.seed))),
            lr=float(training.get("lr", data.get("lr", defaults.lr))),
            weight_decay=float(training.get("weight_decay", data.get("weight_decay", defaults.weight_decay))),
            device=training.get("device", data.get("device", defaults.device)),
            amp=bool(training.get("amp", data.get("amp", defaults.amp))),
            grad_clip_norm=float(
                training.get("grad_clip_norm", data.get("grad_clip_norm", defaults.grad_clip_norm))
            ),
            loss_weights=loss_weights,
            scheduler=dict(scheduler) if isinstance(scheduler, Mapping) else {"name": str(scheduler)},
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_max_bytes=int(
                training.get("checkpoint_max_bytes", data.get("checkpoint_max_bytes", defaults.checkpoint_max_bytes))
            ),
            resume_from=Path(resume_from) if resume_from else None,
            history_filename=str(
                training.get("history_filename", data.get("history_filename", defaults.history_filename))
            ),
            log_every_steps=int(training.get("log_every_steps", data.get("log_every_steps", defaults.log_every_steps))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "amp": self.amp,
            "grad_clip_norm": self.grad_clip_norm,
            "loss_weights": dict(self.loss_weights),
            "scheduler": dict(self.scheduler),
            "checkpoint_dir": None if self.checkpoint_dir is None else str(self.checkpoint_dir),
            "checkpoint_max_bytes": self.checkpoint_max_bytes,
            "history_filename": self.history_filename,
            "log_every_steps": self.log_every_steps,
        }


@dataclass(frozen=True, slots=True)
class PseudoPairEpochResult:
    epochs_completed: int
    global_step: int
    history: list[dict[str, Any]]
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    elapsed_seconds: float

    @property
    def final_validation_loss(self) -> float:
        if not self.history:
            return float("nan")
        return float(self.history[-1]["validation"]["loss"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_completed": self.epochs_completed,
            "global_step": self.global_step,
            "history": self.history,
            "best_checkpoint": None if self.best_checkpoint is None else str(self.best_checkpoint),
            "last_checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint),
            "final_validation_loss": self.final_validation_loss,
            "elapsed_seconds": self.elapsed_seconds,
        }


def train_pseudo_pair_epochs(
    config: PseudoPairEpochConfig | Mapping[str, Any] | None,
    *,
    model: BaseTranslator,
    train_loader: DataLoader[PseudoPairSliceBatch],
    val_loader: DataLoader[PseudoPairSliceBatch],
    run_metadata: Mapping[str, Any] | None = None,
) -> PseudoPairEpochResult:
    cfg = _coerce_config(config)
    _validate_loader(train_loader, "train")
    _validate_loader(val_loader, "validation")
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg.scheduler, cfg.epochs)
    scaler = _make_grad_scaler(enabled=cfg.amp and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        _validate_resume_state(state)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_validation_loss = float(state.get("best_validation_loss", best_validation_loss))

    steps_per_epoch = len(train_loader)
    train_samples = _loader_samples(train_loader)
    val_samples = _loader_samples(val_loader)
    _log(
        "pseudo_pair train start: "
        f"train_samples={train_samples} validation_samples={val_samples} "
        f"batch_size={cfg.batch_size} steps_per_epoch={steps_per_epoch} "
        f"epochs={cfg.epochs} device={device.type} amp={cfg.amp and device.type == 'cuda'}"
    )

    best_checkpoint = _checkpoint_path(cfg, "best")
    last_checkpoint = _checkpoint_path(cfg, "last")
    history: list[dict[str, Any]] = []
    history_path = _history_path(cfg)
    start_time = time.perf_counter()
    for epoch_index in range(start_epoch, cfg.epochs):
        train_summary, global_step = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            epoch=epoch_index + 1,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
        )
        validation_summary = _run_validation_epoch(model, val_loader, cfg, device)
        validation_loss = float(validation_summary["loss"])
        _step_scheduler(scheduler, validation_loss)
        record = {
            "epoch": epoch_index + 1,
            "global_step": global_step,
            "train": train_summary,
            "validation": validation_summary,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        _append_history(history_path, record)
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
        if cfg.checkpoint_dir is not None:
            _save_epoch_checkpoint(
                best_checkpoint if is_best else None,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index + 1,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                run_metadata=run_metadata,
            )
            _save_epoch_checkpoint(
                last_checkpoint,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_index + 1,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                run_metadata=run_metadata,
            )
        _log(
            f"pseudo_pair epoch={epoch_index + 1}/{cfg.epochs} "
            f"train_{_format_loss_components(train_summary)} "
            f"validation_{_format_loss_components(validation_summary)}"
        )

    elapsed_seconds = time.perf_counter() - start_time
    return PseudoPairEpochResult(
        epochs_completed=max(0, cfg.epochs - start_epoch),
        global_step=global_step,
        history=history,
        best_checkpoint=best_checkpoint if cfg.checkpoint_dir is not None else None,
        last_checkpoint=last_checkpoint if cfg.checkpoint_dir is not None else None,
        elapsed_seconds=elapsed_seconds,
    )


def _run_train_epoch(
    model: BaseTranslator,
    loader: DataLoader[PseudoPairSliceBatch],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: PseudoPairEpochConfig,
    device: torch.device,
    *,
    epoch: int,
    global_step: int,
    steps_per_epoch: int,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {key: 0.0 for key in _LOSS_COMPONENT_KEYS}
    total_samples = 0
    autocast_ctx = _autocast_context(device, cfg.amp)
    for step_in_epoch, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            prediction = model(batch.x_low, batch.source_domain, batch.target_domain)
            components = combined_reconstruction_loss_components(
                prediction,
                batch.x_high,
                batch.mask,
                cfg.loss_weights,
            )
            loss = components["total"]
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        _sync_if_cuda(device)
        batch_size = int(batch.x_low.shape[0])
        component_values = _component_values(components)
        for key in _LOSS_COMPONENT_KEYS:
            totals[key] += component_values[key] * batch_size
        total_samples += batch_size
        global_step += 1
        if cfg.log_every_steps > 0 and (
            step_in_epoch == 1 or step_in_epoch % cfg.log_every_steps == 0 or step_in_epoch == steps_per_epoch
        ):
            _log(
                f"pseudo_pair epoch={epoch} step={step_in_epoch}/{steps_per_epoch} "
                f"global_step={global_step} {_format_loss_components(component_values)}"
            )
    return _summarize_loss_components(totals, total_samples), global_step


def _run_validation_epoch(
    model: BaseTranslator,
    loader: DataLoader[PseudoPairSliceBatch],
    cfg: PseudoPairEpochConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in _LOSS_COMPONENT_KEYS}
    total_samples = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            prediction = model(batch.x_low, batch.source_domain, batch.target_domain)
            components = combined_reconstruction_loss_components(
                prediction,
                batch.x_high,
                batch.mask,
                cfg.loss_weights,
            )
            batch_size = int(batch.x_low.shape[0])
            component_values = _component_values(components)
            for key in _LOSS_COMPONENT_KEYS:
                totals[key] += component_values[key] * batch_size
            total_samples += batch_size
    return _summarize_loss_components(totals, total_samples)


def _save_epoch_checkpoint(
    path: Path | None,
    *,
    cfg: PseudoPairEpochConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    run_metadata: Mapping[str, Any] | None = None,
) -> None:
    if path is None:
        return
    state: dict[str, Any] = {
        "trainer": "pseudo_pair_epochs",
        "pseudo_pair_pipeline_version": PSEUDO_PAIR_PIPELINE_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scheduler_name": str(cfg.scheduler.get("name", "none")),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "pseudo_pair_config": cfg.to_dict(),
        "model_class": type(model).__name__,
        "run_metadata": dict(run_metadata or {}),
    }
    save_checkpoint(
        path,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=True,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _make_grad_scaler(*, enabled: bool):
    """Construct an AMP GradScaler across torch versions.

    torch>=2.4 exposes the unified `torch.amp.GradScaler(device, ...)`; older builds only
    have `torch.cuda.amp.GradScaler(...)`. Support both so the suite is green regardless of
    the environment's torch (e.g. 2.2.x locally vs 2.11 on the GPU box).
    """

    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = str(config.get("name", "none")).lower().replace("-", "_")
    if name in ("none", "null", ""):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", max(1, epochs))),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 1)),
            gamma=float(config.get("gamma", 0.1)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 2)),
        )
    raise ValueError(f"Unsupported pseudo-pair scheduler {name!r}.")


def _step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    validation_loss: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def _move_batch(batch: PseudoPairSliceBatch, device: torch.device) -> PseudoPairSliceBatch:
    return PseudoPairSliceBatch(
        x_low=batch.x_low.to(device),
        x_high=batch.x_high.to(device),
        mask=batch.mask.to(device),
        source_domain=batch.source_domain,
        target_domain=batch.target_domain,
        record_id=batch.record_id,
        subject_id=batch.subject_id,
        volume_path=batch.volume_path,
        slice_index=batch.slice_index.to(device),
        degradation_seed=batch.degradation_seed,
        degradation_strength=batch.degradation_strength.to(device),
        geometry=batch.geometry,
    )


def _validate_loader(loader: DataLoader[PseudoPairSliceBatch], name: str) -> None:
    try:
        length = len(loader)
    except TypeError as exc:
        raise ValueError(f"{name} loader must have a finite length for epoch training.") from exc
    if length <= 0:
        raise ValueError(f"{name} loader is empty.")
    samples = _loader_samples(loader)
    if samples <= 0:
        raise ValueError(f"{name} dataset is empty.")


def _loader_samples(loader: DataLoader[PseudoPairSliceBatch]) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return 0
    try:
        return int(len(dataset))
    except TypeError:
        return 0


def _history_path(cfg: PseudoPairEpochConfig) -> Path | None:
    if cfg.checkpoint_dir is None:
        return None
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg.checkpoint_dir / cfg.history_filename


def _append_history(path: Path | None, record: Mapping[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _checkpoint_path(cfg: PseudoPairEpochConfig, name: str) -> Path | None:
    if cfg.checkpoint_dir is None:
        return None
    return cfg.checkpoint_dir / f"{name}.pt"


def _validate_resume_state(state: Mapping[str, Any]) -> None:
    if state.get("trainer") != "pseudo_pair_epochs":
        raise ValueError("Resume checkpoint is not a pseudo-pair epoch checkpoint.")
    if int(state.get("pseudo_pair_pipeline_version", 1)) < PSEUDO_PAIR_PIPELINE_VERSION:
        raise ValueError(
            "Resume checkpoint was produced before the pseudo-pair loss/axis correction; "
            "start a fresh run instead of resuming it."
        )
    for key in ("model", "optimizer", "epoch", "global_step"):
        if key not in state:
            raise ValueError(f"Resume checkpoint is missing required key {key!r}.")


def _component_values(components: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(components[key].detach().cpu()) for key in _LOSS_COMPONENT_KEYS}


def _summarize_loss_components(totals: Mapping[str, float], total_samples: int) -> dict[str, float]:
    denominator = max(1, total_samples)
    summary = {key: float(totals[key]) / denominator for key in _LOSS_COMPONENT_KEYS}
    summary["loss"] = summary["total"]
    summary["samples"] = float(total_samples)
    return summary


def _format_loss_components(values: Mapping[str, float]) -> str:
    return (
        f"loss={values['total']:.6f} "
        f"masked_l1={values['masked_l1']:.6f} "
        f"gradient={values['gradient']:.6f} "
        f"background={values['background']:.6f}"
    )


def _coerce_config(config: PseudoPairEpochConfig | Mapping[str, Any] | None) -> PseudoPairEpochConfig:
    if config is None:
        return PseudoPairEpochConfig()
    if isinstance(config, PseudoPairEpochConfig):
        return config
    return PseudoPairEpochConfig.from_mapping(config)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is 'cuda', but CUDA is not available.")
    if device not in ("cpu", "cuda"):
        raise ValueError("training.device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)


def _autocast_context(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
