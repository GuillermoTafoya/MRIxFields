"""Fixed-endpoint real-paired LOSO training for Track A."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
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
ArmRecoveryAction = Literal["fresh", "resume", "endpoint"]
ProgressState = Literal[
    "pending", "running", "endpoint_complete", "evaluating", "complete"
]
_ANONYMOUS_SLOT = re.compile(r"^(fold|case)_\d{2}$")


@dataclass(frozen=True, slots=True)
class PairedEndpointResult:
    fold_slot: str
    arm: InitializationArm
    epoch: int
    global_step: int
    steps_per_epoch: int
    endpoint_checkpoint: Path
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ArmRecoveryPlan:
    action: ArmRecoveryAction
    checkpoint_state: Mapping[str, Any] | None
    resume_path: Path | None
    epoch: int
    global_step: int


class SanitizedRunProgress:
    """Atomically persist anonymous fold/arm orchestration state."""

    def __init__(
        self,
        path: Path,
        *,
        fold_case_slots: Mapping[str, str],
        resume: bool,
    ) -> None:
        self.path = Path(path)
        self._fold_case_slots = dict(fold_case_slots)
        _validate_anonymous_slots(self._fold_case_slots)
        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    "An existing sanitized run-progress artifact requires --resume."
                )
            self._payload = _load_progress_payload(
                self.path,
                fold_case_slots=self._fold_case_slots,
            )
        else:
            self._payload = {
                "folds": [
                    {
                        "fold_slot": fold_slot,
                        "case_slot": case_slot,
                        "arms": {
                            arm: {
                                "state": "pending",
                                "epoch": 0,
                                "global_step": 0,
                                "elapsed_seconds": 0.0,
                            }
                            for arm in NEURAL_INITIALIZATION_ARMS
                        },
                    }
                    for fold_slot, case_slot in self._fold_case_slots.items()
                ]
            }
            self._write()

    def update(
        self,
        *,
        fold_slot: str,
        case_slot: str,
        arm: InitializationArm,
        state: ProgressState,
        epoch: int,
        global_step: int,
        elapsed_seconds: float,
    ) -> None:
        if self._fold_case_slots.get(fold_slot) != case_slot:
            raise ValueError("Progress update used an unknown anonymous fold/case slot.")
        if arm not in NEURAL_INITIALIZATION_ARMS:
            raise ValueError("Progress update used an unknown initialization arm.")
        if state not in {"pending", "running", "endpoint_complete", "evaluating", "complete"}:
            raise ValueError("Progress update used an unknown state.")
        if epoch < 0 or global_step < 0 or elapsed_seconds < 0:
            raise ValueError("Progress counters must be non-negative.")
        for fold in self._payload["folds"]:
            if fold["fold_slot"] == fold_slot:
                fold["arms"][arm] = {
                    "state": state,
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "elapsed_seconds": float(elapsed_seconds),
                }
                self._write()
                return
        raise ValueError("Progress update fold was not initialized.")

    def elapsed_seconds(self, fold_slot: str, arm: InitializationArm) -> float:
        return float(self.status(fold_slot, arm)["elapsed_seconds"])

    def status(self, fold_slot: str, arm: InitializationArm) -> Mapping[str, Any]:
        for fold in self._payload["folds"]:
            if fold["fold_slot"] == fold_slot:
                return dict(fold["arms"][arm])
        raise ValueError("Progress fold was not initialized.")

    def validate_recovery(
        self,
        fold_slot: str,
        arm: InitializationArm,
        action: ArmRecoveryAction,
    ) -> None:
        status = self.status(fold_slot, arm)
        if action == "fresh" and (
            status["state"] not in {"pending", "running"}
            or int(status["epoch"]) != 0
            or int(status["global_step"]) != 0
        ):
            raise ValueError(
                "Progress records a partially started arm without a valid checkpoint."
            )

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(self._payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


NEURAL_INITIALIZATION_ARMS: tuple[InitializationArm, ...] = (
    "identity_initialization",
    "synthetic_initialization",
)


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


def resolve_arm_recovery(
    checkpoint_dir: Path,
    *,
    resume: bool,
    cfg: PseudoPairEpochConfig,
    fold_slot: str,
    arm: InitializationArm,
    experiment_fingerprint: str,
    expected_steps_per_epoch: int,
    expected_global_step: int,
    map_location: str | torch.device = "cpu",
) -> ArmRecoveryPlan:
    """Choose one fail-closed action from the artifacts for a single fold/arm."""

    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise ValueError("Paired arm checkpoint location is not a directory.")
    resume_path = checkpoint_dir / "resume.pt"
    endpoint_path = checkpoint_dir / "endpoint.pt"
    history_path = checkpoint_dir / "history.jsonl"
    known_paths = (resume_path, endpoint_path, history_path)
    unexpected = (
        sorted(path.name for path in checkpoint_dir.iterdir() if path not in known_paths)
        if checkpoint_dir.is_dir()
        else []
    )
    if unexpected:
        raise ValueError("Paired arm directory contains inconsistent partial artifacts.")
    existing = [path for path in known_paths if path.exists()]
    if not resume:
        if existing:
            raise FileExistsError(
                f"Existing fold/arm artifacts require --resume for {fold_slot}/{arm}."
            )
        return ArmRecoveryPlan("fresh", None, None, 0, 0)

    endpoint_state: Mapping[str, Any] | None = None
    resume_state: Mapping[str, Any] | None = None
    if endpoint_path.exists():
        endpoint_state = load_checkpoint(endpoint_path, map_location=map_location)
        validate_endpoint_checkpoint(
            endpoint_state,
            cfg=cfg,
            fold_slot=fold_slot,
            arm=arm,
            experiment_fingerprint=experiment_fingerprint,
            expected_steps_per_epoch=expected_steps_per_epoch,
            expected_global_step=expected_global_step,
        )
    if resume_path.exists():
        resume_state = load_checkpoint(resume_path, map_location=map_location)
        if endpoint_state is None:
            validate_resume_checkpoint(
                resume_state,
                cfg=cfg,
                fold_slot=fold_slot,
                arm=arm,
                experiment_fingerprint=experiment_fingerprint,
                expected_steps_per_epoch=expected_steps_per_epoch,
                expected_global_step=expected_global_step,
            )
        else:
            _validate_resume(
                resume_state,
                cfg=cfg,
                fold_slot=fold_slot,
                arm=arm,
                experiment_fingerprint=experiment_fingerprint,
                expected_steps_per_epoch=expected_steps_per_epoch,
                expected_global_step=expected_global_step,
            )
            if resume_state.get("endpoint") is not False:
                raise ValueError("Completed-arm resume checkpoint changed endpoint state.")
            resume_epoch = int(resume_state["epoch"])
            resume_global_step = int(resume_state["global_step"])
            if resume_epoch > cfg.epochs:
                raise ValueError("Completed-arm resume checkpoint epoch exceeds the endpoint.")
            if resume_global_step != resume_epoch * int(expected_steps_per_epoch):
                raise ValueError("Completed-arm resume checkpoint step alignment changed.")
    if history_path.exists():
        checkpoint_state = endpoint_state if endpoint_state is not None else resume_state
        if checkpoint_state is None:
            raise ValueError("History exists without a valid paired checkpoint.")
        _validate_history(
            history_path,
            expected_epoch=int(checkpoint_state["epoch"]),
            expected_global_step=int(checkpoint_state["global_step"]),
            expected_steps_per_epoch=expected_steps_per_epoch,
        )
    elif endpoint_state is not None or resume_state is not None:
        raise ValueError("Paired checkpoint exists without its completed-epoch history.")

    if endpoint_state is not None:
        return ArmRecoveryPlan(
            "endpoint",
            endpoint_state,
            None,
            int(endpoint_state["epoch"]),
            int(endpoint_state["global_step"]),
        )
    if resume_state is not None:
        return ArmRecoveryPlan(
            "resume",
            resume_state,
            resume_path,
            int(resume_state["epoch"]),
            int(resume_state["global_step"]),
        )
    if existing:
        raise ValueError("Inconsistent paired arm artifacts cannot be resumed.")
    return ArmRecoveryPlan("fresh", None, None, 0, 0)


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
    case_slot: str | None = None,
    epoch_progress: Callable[[int, int, float], None] | None = None,
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

    if endpoint_path.exists():
        raise FileExistsError("A paired fixed endpoint already exists and cannot be overwritten.")
    if resume_from is None and (resume_path.exists() or history_path.exists()):
        raise FileExistsError("Existing paired training artifacts require validated resume.")
    if resume_from is not None and Path(resume_from) != resume_path:
        raise ValueError("Paired resume must use this arm's exact resume.pt path.")

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
        if state.get("endpoint") is not False:
            raise ValueError("Resume checkpoint must be a non-endpoint checkpoint.")
        _validate_history(
            history_path,
            expected_epoch=int(state["epoch"]),
            expected_global_step=int(state["global_step"]),
            expected_steps_per_epoch=expected_steps_per_epoch,
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

    started_global_step = global_step
    started_at = time.perf_counter()
    for epoch_index in range(start_epoch, cfg.epochs):
        summary, global_step = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            fold_slot=fold_slot,
            case_slot=case_slot,
            arm=arm,
            epoch=epoch_index + 1,
            steps_per_epoch=expected_steps_per_epoch,
            expected_global_step=expected_global_step,
            started_at=started_at,
            started_global_step=started_global_step,
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
        if epoch_progress is not None:
            epoch_progress(epoch_index + 1, global_step, time.perf_counter() - started_at)

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


def validate_resume_checkpoint(
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
    epoch = int(state["epoch"])
    global_step = int(state["global_step"])
    if state.get("endpoint") is not False:
        raise ValueError("Resume checkpoint is incorrectly marked as an endpoint.")
    if epoch < 1 or epoch > cfg.epochs:
        raise ValueError("Resume checkpoint epoch is outside the resumable range.")
    if global_step != epoch * int(expected_steps_per_epoch):
        raise ValueError("Resume checkpoint epoch/global-step alignment changed.")


def _train_epoch(
    model: BaseTranslator,
    loader: DataLoader[PseudoPairSliceBatch],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: PseudoPairEpochConfig,
    device: torch.device,
    *,
    fold_slot: str,
    case_slot: str | None,
    arm: InitializationArm,
    epoch: int,
    steps_per_epoch: int,
    expected_global_step: int,
    started_at: float,
    started_global_step: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {"total": 0.0, "masked_l1": 0.0, "gradient": 0.0, "background": 0.0}
    samples = 0
    for step_in_epoch, raw_batch in enumerate(loader, start=1):
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
        if cfg.log_every_steps > 0 and (
            step_in_epoch == 1
            or step_in_epoch % cfg.log_every_steps == 0
            or step_in_epoch == steps_per_epoch
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - started_at, 1e-12)
            completed_steps = global_step - started_global_step
            values = {
                key: float(components[key].detach().cpu())
                for key in ("total", "masked_l1", "gradient", "background")
            }
            case_fragment = f" case={case_slot}" if case_slot is not None else ""
            print(
                f"paired_loso fold={fold_slot}{case_fragment} arm={arm} "
                f"epoch={epoch}/{cfg.epochs} step={step_in_epoch}/{steps_per_epoch} "
                f"global_step={global_step}/{expected_global_step} "
                f"total={values['total']:.6f} masked_l1={values['masked_l1']:.6f} "
                f"gradient={values['gradient']:.6f} background={values['background']:.6f} "
                f"elapsed_seconds={elapsed:.1f} steps_per_second={completed_steps / elapsed:.3f}",
                flush=True,
            )
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
        overwrite=not endpoint,
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
    if state.get("scheduler") is not None:
        raise ValueError("Resume checkpoint scheduler state changed.")
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


def _validate_history(
    path: Path,
    *,
    expected_epoch: int,
    expected_global_step: int,
    expected_steps_per_epoch: int,
) -> None:
    if not path.is_file():
        raise ValueError("Paired checkpoint history is missing.")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Paired history line {line_number} is invalid JSON.") from exc
        if not isinstance(record, Mapping):
            raise ValueError("Paired history records must be mappings.")
        records.append(record)
    if len(records) != expected_epoch:
        raise ValueError("Paired history epoch count differs from the checkpoint.")
    for index, record in enumerate(records, start=1):
        if int(record.get("epoch", -1)) != index:
            raise ValueError("Paired history epochs are not contiguous.")
        if int(record.get("global_step", -1)) != index * expected_steps_per_epoch:
            raise ValueError("Paired history global steps changed.")
    if expected_epoch and int(records[-1]["global_step"]) != expected_global_step:
        raise ValueError("Paired history endpoint differs from the checkpoint.")


def _validate_anonymous_slots(fold_case_slots: Mapping[str, str]) -> None:
    if not fold_case_slots:
        raise ValueError("Progress requires at least one anonymous fold/case slot.")
    for fold_slot, case_slot in fold_case_slots.items():
        if not _ANONYMOUS_SLOT.fullmatch(fold_slot) or not fold_slot.startswith("fold_"):
            raise ValueError("Progress fold slots must be anonymous fold_NN labels.")
        if not _ANONYMOUS_SLOT.fullmatch(case_slot) or not case_slot.startswith("case_"):
            raise ValueError("Progress case slots must be anonymous case_NN labels.")


def _load_progress_payload(
    path: Path,
    *,
    fold_case_slots: Mapping[str, str],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Sanitized run-progress artifact is invalid.") from exc
    if not isinstance(payload, dict) or set(payload) != {"folds"}:
        raise ValueError("Sanitized run-progress schema changed.")
    folds = payload["folds"]
    if not isinstance(folds, list) or len(folds) != len(fold_case_slots):
        raise ValueError("Sanitized run-progress fold set changed.")
    observed_slots: dict[str, str] = {}
    valid_states = {"pending", "running", "endpoint_complete", "evaluating", "complete"}
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != {"fold_slot", "case_slot", "arms"}:
            raise ValueError("Sanitized run-progress fold schema changed.")
        fold_slot = str(fold["fold_slot"])
        case_slot = str(fold["case_slot"])
        observed_slots[fold_slot] = case_slot
        arms = fold["arms"]
        if not isinstance(arms, dict) or set(arms) != set(NEURAL_INITIALIZATION_ARMS):
            raise ValueError("Sanitized run-progress arm set changed.")
        for status in arms.values():
            if not isinstance(status, dict) or set(status) != {
                "state", "epoch", "global_step", "elapsed_seconds"
            }:
                raise ValueError("Sanitized run-progress arm schema changed.")
            if status["state"] not in valid_states:
                raise ValueError("Sanitized run-progress contains an invalid state.")
            if (
                int(status["epoch"]) < 0
                or int(status["global_step"]) < 0
                or float(status["elapsed_seconds"]) < 0
            ):
                raise ValueError("Sanitized run-progress contains invalid counters.")
    if observed_slots != dict(fold_case_slots):
        raise ValueError("Sanitized run-progress anonymous slots changed.")
    return payload


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
