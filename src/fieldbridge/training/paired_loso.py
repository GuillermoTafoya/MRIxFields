"""Fixed-endpoint real-paired LOSO training for Track A."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from fieldbridge.data.pseudo_pairs import PseudoPairSliceBatch
from fieldbridge.models.factory import build_translator
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.checkpoints import load_checkpoint, save_checkpoint
from fieldbridge.training.losses import combined_reconstruction_loss_components
from fieldbridge.training.pseudo_pair_epochs import PseudoPairEpochConfig
from fieldbridge.utils.seeding import seed_everything

PAIRED_LOSO_PIPELINE_VERSION = 1
InitializationArm = Literal["identity_initialization", "synthetic_initialization"]


@dataclass(frozen=True, slots=True)
class PairedEndpointResult:
    fold_slot: str
    arm: InitializationArm
    epoch: int
    global_step: int
    steps_per_epoch: int
    endpoint_checkpoint: Path
    history: tuple[dict[str, Any], ...]


def initialize_residual_arm(
    model_config: Mapping[str, Any],
    *,
    arm: InitializationArm,
    synthetic_checkpoint: Mapping[str, Any] | None = None,
) -> BaseTranslator:
    parameters = dict(model_config)
    name = str(parameters.pop("name"))
    if name != "conditional_residual_unet_field_translator":
        raise ValueError("Paired LOSO v1 requires the residual U-Net translator.")
    model = build_translator(name, **parameters)
    if arm == "identity_initialization":
        if synthetic_checkpoint is not None:
            raise ValueError("Identity initialization must not receive a synthetic checkpoint.")
        return model
    if arm != "synthetic_initialization":
        raise ValueError(f"Unknown paired LOSO initialization arm {arm!r}.")
    if synthetic_checkpoint is None or "model" not in synthetic_checkpoint:
        raise ValueError("Synthetic initialization requires a validated checkpoint state.")
    model.load_state_dict(synthetic_checkpoint["model"], strict=True)
    return model


def train_fixed_endpoint(
    config: PseudoPairEpochConfig | Mapping[str, Any],
    *,
    model: BaseTranslator,
    train_loader: DataLoader[PseudoPairSliceBatch],
    checkpoint_dir: Path,
    fold_slot: str,
    arm: InitializationArm,
    experiment_fingerprint: str,
    expected_steps_per_epoch: int,
    expected_global_step: int,
    resume_from: Path | None = None,
) -> PairedEndpointResult:
    """Train without validation, early stopping, or checkpoint selection."""

    cfg = (
        config
        if isinstance(config, PseudoPairEpochConfig)
        else PseudoPairEpochConfig.from_mapping(config)
    )
    if len(train_loader) != int(expected_steps_per_epoch):
        raise ValueError(
            f"Fixed steps_per_epoch changed: {len(train_loader)} != {expected_steps_per_epoch}."
        )
    if cfg.epochs * int(expected_steps_per_epoch) != int(expected_global_step):
        raise ValueError("Configured epochs and steps do not match the fixed endpoint.")
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if str(cfg.scheduler.get("name", "none")).lower() not in ("none", "null", ""):
        raise ValueError("Paired LOSO v1 preregisters scheduler.name=none.")
    scaler = _grad_scaler(enabled=cfg.amp and device.type == "cuda")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / "resume.pt"
    endpoint_path = checkpoint_dir / "endpoint.pt"
    history_path = checkpoint_dir / "history.jsonl"

    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        state = load_checkpoint(resume_from, map_location=device)
        _validate_resume(
            state,
            cfg=cfg,
            fold_slot=fold_slot,
            arm=arm,
            experiment_fingerprint=experiment_fingerprint,
            expected_steps_per_epoch=expected_steps_per_epoch,
            expected_global_step=expected_global_step,
        )
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        loader_generator = getattr(train_loader, "generator", None)
        if loader_generator is None:
            raise ValueError("Resumable paired training requires a DataLoader generator.")
        loader_generator.set_state(state["data_loader_generator_state"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])

    for epoch_index in range(start_epoch, cfg.epochs):
        summary, global_step = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            global_step=global_step,
        )
        record = {
            "epoch": epoch_index + 1,
            "global_step": global_step,
            "train": summary,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        _save_paired_checkpoint(
            resume_path,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            fold_slot=fold_slot,
            arm=arm,
            epoch=epoch_index + 1,
            global_step=global_step,
            experiment_fingerprint=experiment_fingerprint,
            expected_steps_per_epoch=expected_steps_per_epoch,
            expected_global_step=expected_global_step,
            endpoint=False,
            data_loader_generator_state=_loader_generator_state(train_loader),
            scaler_state=scaler.state_dict(),
        )

    if global_step != expected_global_step:
        raise RuntimeError(f"Fixed endpoint changed: {global_step} != {expected_global_step}.")
    _save_paired_checkpoint(
        endpoint_path,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        fold_slot=fold_slot,
        arm=arm,
        epoch=cfg.epochs,
        global_step=global_step,
        experiment_fingerprint=experiment_fingerprint,
        expected_steps_per_epoch=expected_steps_per_epoch,
        expected_global_step=expected_global_step,
        endpoint=True,
        data_loader_generator_state=_loader_generator_state(train_loader),
        scaler_state=scaler.state_dict(),
    )
    return PairedEndpointResult(
        fold_slot=fold_slot,
        arm=arm,
        epoch=cfg.epochs,
        global_step=global_step,
        steps_per_epoch=expected_steps_per_epoch,
        endpoint_checkpoint=endpoint_path,
        history=tuple(history),
    )


def validate_endpoint_checkpoint(
    state: Mapping[str, Any],
    *,
    cfg: PseudoPairEpochConfig,
    fold_slot: str,
    arm: InitializationArm,
    experiment_fingerprint: str,
    expected_steps_per_epoch: int,
    expected_global_step: int,
) -> None:
    _validate_resume(
        state,
        cfg=cfg,
        fold_slot=fold_slot,
        arm=arm,
        experiment_fingerprint=experiment_fingerprint,
        expected_steps_per_epoch=expected_steps_per_epoch,
        expected_global_step=expected_global_step,
    )
    if state.get("endpoint") is not True:
        raise ValueError("Paired checkpoint is not the fixed endpoint.")
    if int(state["epoch"]) != cfg.epochs or int(state["global_step"]) != expected_global_step:
        raise ValueError("Paired endpoint epoch or global step changed.")


def _train_epoch(
    model: BaseTranslator,
    loader: DataLoader[PseudoPairSliceBatch],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: PseudoPairEpochConfig,
    device: torch.device,
    *,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {"total": 0.0, "masked_l1": 0.0, "gradient": 0.0, "background": 0.0}
    samples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.amp):
            prediction = model(batch.x_low, batch.source_domain, batch.target_domain)
            components = combined_reconstruction_loss_components(
                prediction,
                batch.x_high,
                batch.mask,
                cfg.loss_weights,
            )
        scaler.scale(components["total"]).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(batch.x_low.shape[0])
        for key in totals:
            totals[key] += float(components[key].detach().cpu()) * batch_size
        samples += batch_size
        global_step += 1
    return {
        **{key: value / max(1, samples) for key, value in totals.items()},
        "samples": float(samples),
    }, global_step


def _save_paired_checkpoint(
    path: Path,
    *,
    cfg: PseudoPairEpochConfig,
    model: BaseTranslator,
    optimizer: torch.optim.Optimizer,
    fold_slot: str,
    arm: InitializationArm,
    epoch: int,
    global_step: int,
    experiment_fingerprint: str,
    expected_steps_per_epoch: int,
    expected_global_step: int,
    endpoint: bool,
    data_loader_generator_state: torch.Tensor,
    scaler_state: Mapping[str, Any],
) -> None:
    state = {
        "trainer": "paired_loso_fixed_endpoint",
        "paired_loso_pipeline_version": PAIRED_LOSO_PIPELINE_VERSION,
        "model_class": type(model).__name__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_name": "AdamW",
        "scheduler": None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "endpoint": bool(endpoint),
        "fold_slot": fold_slot,
        "initialization_arm": arm,
        "experiment_fingerprint": experiment_fingerprint,
        "steps_per_epoch": int(expected_steps_per_epoch),
        "expected_global_step": int(expected_global_step),
        "training_config": cfg.to_dict(),
        "data_loader_generator_state": data_loader_generator_state,
        "scaler": dict(scaler_state),
    }
    save_checkpoint(
        path,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=True,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _validate_resume(
    state: Mapping[str, Any],
    *,
    cfg: PseudoPairEpochConfig,
    fold_slot: str,
    arm: InitializationArm,
    experiment_fingerprint: str,
    expected_steps_per_epoch: int,
    expected_global_step: int,
) -> None:
    expected = {
        "trainer": "paired_loso_fixed_endpoint",
        "paired_loso_pipeline_version": PAIRED_LOSO_PIPELINE_VERSION,
        "model_class": "ConditionalResidualUNetFieldTranslator",
        "fold_slot": fold_slot,
        "initialization_arm": arm,
        "experiment_fingerprint": experiment_fingerprint,
        "steps_per_epoch": expected_steps_per_epoch,
        "expected_global_step": expected_global_step,
        "optimizer_name": "AdamW",
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Resume checkpoint differs at {key!r}.")
    if state.get("training_config") != cfg.to_dict():
        raise ValueError("Resume checkpoint training config changed.")
    metadata = state.get("_meta")
    if not isinstance(metadata, Mapping) or metadata.get("config") != cfg.to_dict():
        raise ValueError("Resume checkpoint metadata config changed.")
    for key in (
        "model",
        "optimizer",
        "epoch",
        "global_step",
        "data_loader_generator_state",
        "scaler",
    ):
        if key not in state:
            raise ValueError(f"Resume checkpoint is missing {key!r}.")


def _loader_generator_state(loader: DataLoader[PseudoPairSliceBatch]) -> torch.Tensor:
    generator = getattr(loader, "generator", None)
    if generator is None:
        raise ValueError("Resumable paired training requires a DataLoader generator.")
    return generator.get_state()


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


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name not in ("cpu", "cuda"):
        raise ValueError("Training device must be auto, cpu, or cuda.")
    return torch.device(name)


def _grad_scaler(*, enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast()
