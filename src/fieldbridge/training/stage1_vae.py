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

import json
import math
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
from fieldbridge.training.latent_stats import LatentStatsAccumulator, summarize_latent_stats
from fieldbridge.training.losses import (
    build_lpips_net,
    foreground_weighted_l1_loss,
    kl_divergence,
    lpips_loss,
    lpips_loss_3d,
    masked_l1_loss,
    nrmse_loss,
    ssim_loss,
)
from fieldbridge.training.warm_start import load_state_dict_tolerant
from fieldbridge.utils.seeding import seed_everything

# Order in which per-term losses appear in the per-step console line (only terms with a
# positive weight are actually computed/printed). `total` is logged separately.
_LOSS_TERM_ORDER = ("l1", "ssim", "nrmse", "lpips", "kl")

Precision = Literal["fp32", "bf16"]
Device = Literal["auto", "cpu", "cuda"]

# Unlike Etapa 2's "everything 0 except reconstruction" ladder convention, all terms here
# are active by default: L1+SSIM+nRMSE+LPIPS+KL is a fixed composition whose *relative*
# weights get swept experimentally, not a term-by-term ladder to turn on one at a time.
#
# L1 is the flat absolute-intensity anchor the recipe was missing: nRMSE has a single
# global sqrt (near-uniform per-voxel gradient), SSIM is local and blind to a DC offset,
# and LPIPS is deliberately contrast-invariant — so nothing forced the background to the
# exact 0 the official [0, 1] contract requires. That is why the earlier run reconstructed
# anatomy but left the background floating. L1 is the plan's Etapa-1 reconstruction term.
DEFAULT_VAE_LOSS_WEIGHTS: dict[str, float] = {
    "l1": 1.0,
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
    # Intensity range the range-normalized losses assume. 1.0 on the official [0, 1]
    # contract; must stay equal to evaluation/stage1_report.py's _DATA_RANGE.
    data_range: float = 1.0
    lpips_num_slices: int = 8
    # Per-epoch validation (runs only when a val loader is passed and steps_per_epoch>0).
    # `val_max_batches`=0 uses the whole val loader; a positive cap bounds the per-epoch
    # cost. History (per-term train+val losses) is appended to `history_filename` in the
    # checkpoint dir, and the best-by-validation-total checkpoint is saved alongside — this
    # is the model-selection signal, unlike the train-EMA early-stop (a GPU-saver only).
    val_max_batches: int = 0
    history_filename: str = "history.jsonl"
    # Run validation (and best-checkpoint selection) every N epochs. 1 = every epoch
    # (unchanged). Raise it to spend fewer GPU-minutes on validation in a long run.
    val_every_epochs: int = 1
    # Per-epoch latent posterior-collapse diagnostics (std/KL per latent dim + active-unit
    # count), computed over the same validation pass. A latent channel counts as "active"
    # when its per-dim KL exceeds this threshold. Diagnostic only — never affects the loss.
    latent_active_kl_threshold: float = 0.01
    # Dump an input/reconstruction/error panel over a FIXED validation batch every N epochs
    # (0 disables). The batch is captured once before training so the same slices are shown
    # across epochs and runs. No-op without a checkpoint_dir, a val_loader, or matplotlib.
    recon_dump_every_epochs: int = 5
    grad_clip_norm: float = 1.0
    # Steps that make up one full pass over the manifest, = ceil(num_volumes *
    # patches_per_volume / batch_size). 0 means unknown (the loader is an IterableDataset
    # with no __len__); the CLI computes it from the manifest and injects it so the loop
    # can log epoch/step-in-epoch. Purely cosmetic — does not gate training length.
    steps_per_epoch: int = 0
    # Training-loss EMA early stopping. A safety net against burning GPU-hours on a
    # plateaued run, NOT a precise convergence detector (training loss can flatten while
    # the model still improves, since each step samples different volumes). Checked once
    # per checkpoint; state persists in the checkpoint so resume_from does not reset it.
    early_stopping: bool = False
    early_stopping_patience: int = 5  # checkpoints without improvement before stopping
    early_stopping_min_delta: float = 0.005  # relative improvement (0.5%) that counts
    early_stopping_ema_decay: float = 0.98  # EMA smoothing of the noisy per-step loss
    # Learning-rate schedule. "constant" (default) keeps the current fixed-lr behavior
    # exactly. "cosine" ramps lr linearly from 0 over `lr_warmup_steps`, then cosine-decays
    # to `lr * lr_min_factor` by `lr_cosine_total_steps` (0 => start_step + steps, i.e. the
    # length of this run — set it explicitly to the full-schedule length when resuming so
    # the curve is invariant across resumes). Applied per-step, stateless (a pure function
    # of the global step), so resume needs no scheduler state persisted.
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_min_factor: float = 0.0
    lr_cosine_total_steps: int = 0
    # Foreground-weighted reconstruction L1 (item 7 — OFF by default, measure before use).
    # When True the plain L1 term is replaced by foreground_weighted_l1_loss so brain voxels
    # (target > foreground_threshold) count `foreground_weight`x a background voxel. With the
    # flag off the loss math is byte-for-byte unchanged.
    foreground_loss_weighting: bool = False
    foreground_weight: float = 5.0
    foreground_threshold: float = 0.0
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
            data_range=float(training.get("data_range", data.get("data_range", defaults.data_range))),
            lpips_num_slices=int(
                training.get("lpips_num_slices", data.get("lpips_num_slices", defaults.lpips_num_slices))
            ),
            val_max_batches=int(
                training.get("val_max_batches", data.get("val_max_batches", defaults.val_max_batches))
            ),
            history_filename=str(
                training.get("history_filename", data.get("history_filename", defaults.history_filename))
            ),
            val_every_epochs=int(
                training.get("val_every_epochs", data.get("val_every_epochs", defaults.val_every_epochs))
            ),
            latent_active_kl_threshold=float(
                training.get(
                    "latent_active_kl_threshold",
                    data.get("latent_active_kl_threshold", defaults.latent_active_kl_threshold),
                )
            ),
            recon_dump_every_epochs=int(
                training.get(
                    "recon_dump_every_epochs", data.get("recon_dump_every_epochs", defaults.recon_dump_every_epochs)
                )
            ),
            grad_clip_norm=float(
                training.get("grad_clip_norm", data.get("grad_clip_norm", defaults.grad_clip_norm))
            ),
            lr_schedule=str(training.get("lr_schedule", data.get("lr_schedule", defaults.lr_schedule))),
            lr_warmup_steps=int(
                training.get("lr_warmup_steps", data.get("lr_warmup_steps", defaults.lr_warmup_steps))
            ),
            lr_min_factor=float(
                training.get("lr_min_factor", data.get("lr_min_factor", defaults.lr_min_factor))
            ),
            lr_cosine_total_steps=int(
                training.get(
                    "lr_cosine_total_steps", data.get("lr_cosine_total_steps", defaults.lr_cosine_total_steps)
                )
            ),
            foreground_loss_weighting=bool(
                training.get(
                    "foreground_loss_weighting",
                    data.get("foreground_loss_weighting", defaults.foreground_loss_weighting),
                )
            ),
            foreground_weight=float(
                training.get("foreground_weight", data.get("foreground_weight", defaults.foreground_weight))
            ),
            foreground_threshold=float(
                training.get(
                    "foreground_threshold", data.get("foreground_threshold", defaults.foreground_threshold)
                )
            ),
            steps_per_epoch=int(
                training.get("steps_per_epoch", data.get("steps_per_epoch", defaults.steps_per_epoch))
            ),
            early_stopping=bool(
                training.get("early_stopping", data.get("early_stopping", defaults.early_stopping))
            ),
            early_stopping_patience=int(
                training.get(
                    "early_stopping_patience", data.get("early_stopping_patience", defaults.early_stopping_patience)
                )
            ),
            early_stopping_min_delta=float(
                training.get(
                    "early_stopping_min_delta",
                    data.get("early_stopping_min_delta", defaults.early_stopping_min_delta),
                )
            ),
            early_stopping_ema_decay=float(
                training.get(
                    "early_stopping_ema_decay",
                    data.get("early_stopping_ema_decay", defaults.early_stopping_ema_decay),
                )
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
            "data_range": self.data_range,
            "val_max_batches": self.val_max_batches,
            "history_filename": self.history_filename,
            "val_every_epochs": self.val_every_epochs,
            "latent_active_kl_threshold": self.latent_active_kl_threshold,
            "recon_dump_every_epochs": self.recon_dump_every_epochs,
            "lpips_num_slices": self.lpips_num_slices,
            "grad_clip_norm": self.grad_clip_norm,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_min_factor": self.lr_min_factor,
            "lr_cosine_total_steps": self.lr_cosine_total_steps,
            "foreground_loss_weighting": self.foreground_loss_weighting,
            "foreground_weight": self.foreground_weight,
            "foreground_threshold": self.foreground_threshold,
            "steps_per_epoch": self.steps_per_epoch,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_ema_decay": self.early_stopping_ema_decay,
            "checkpoint_at_end": self.checkpoint_at_end,
            "log_every_steps": self.log_every_steps,
        }


@dataclass(frozen=True, slots=True)
class Stage1VAETrainResult:
    steps: int
    losses: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    stopped_early: bool = False

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
            "stopped_early": self.stopped_early,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_step": self.seconds_per_step,
        }


@dataclass
class _EarlyStopTracker:
    """EMA of the (noisy) per-step training loss + patience-based stop decision.

    The per-step loss is very noisy here — each step samples a different volume/domain —
    so a raw-loss stop criterion would fire on noise. We smooth with an EMA and only judge
    at checkpoint cadence. State is (de)serializable so resume_from continues the patience
    count instead of restarting it every Colab session.
    """

    decay: float
    min_delta: float
    patience: int
    ema: float | None = None
    best: float | None = None
    num_bad_checkpoints: int = 0

    def update_step(self, loss: float) -> None:
        self.ema = loss if self.ema is None else self.decay * self.ema + (1.0 - self.decay) * loss

    def should_stop(self) -> bool:
        """Call once per checkpoint. True once the EMA has failed to improve by min_delta
        for `patience` consecutive checkpoints."""
        if self.ema is None:
            return False
        if self.best is None or self.ema < self.best * (1.0 - self.min_delta):
            self.best = self.ema
            self.num_bad_checkpoints = 0
            return False
        self.num_bad_checkpoints += 1
        return self.num_bad_checkpoints >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {"ema": self.ema, "best": self.best, "num_bad_checkpoints": self.num_bad_checkpoints}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.ema = state.get("ema")
        self.best = state.get("best")
        self.num_bad_checkpoints = int(state.get("num_bad_checkpoints", 0))


def run_stage1_vae_train(
    config: Stage1VAEConfig | Mapping[str, Any] | None,
    *,
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    loader: DataLoader[RawBatch],
    val_loader: DataLoader[RawBatch] | None = None,
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

    tracker: _EarlyStopTracker | None = None
    if cfg.early_stopping:
        tracker = _EarlyStopTracker(
            decay=cfg.early_stopping_ema_decay,
            min_delta=cfg.early_stopping_min_delta,
            patience=cfg.early_stopping_patience,
        )

    start_step = 0
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        if tracker is not None and isinstance(state.get("early_stop"), Mapping):
            tracker.load_state_dict(state["early_stop"])

    lpips_net: nn.Module | None = None
    if cfg.loss_weights.get("lpips", 0.0) > 0:
        lpips_net = build_lpips_net(device)

    # Cosine schedule length: default to this run's span (start_step..start_step+steps).
    # For a resumed run, set lr_cosine_total_steps to the full-schedule length so the curve
    # is continuous instead of restarting each resume.
    lr_total_steps = cfg.lr_cosine_total_steps if cfg.lr_cosine_total_steps > 0 else start_step + cfg.steps

    iterator = iter(loader)
    autocast_ctx = _autocast_context(device, cfg.precision)
    if cfg.log_every_steps > 0:
        epochs_label = (
            f"~{(start_step + cfg.steps) / cfg.steps_per_epoch:.2f} epochs (steps_per_epoch={cfg.steps_per_epoch})"
            if cfg.steps_per_epoch > 0
            else "epoch size unknown"
        )
        _log_training_progress(
            f"stage1_vae train start: steps={cfg.steps} batch_size={cfg.batch_size} "
            f"device={device.type} {epochs_label} early_stopping={cfg.early_stopping}"
        )

    do_validation = val_loader is not None and cfg.steps_per_epoch > 0
    history_path = (
        cfg.checkpoint_dir / cfg.history_filename if cfg.checkpoint_dir is not None and do_validation else None
    )
    best_val_total = float("inf")

    # Fixed reconstruction-panel batch: captured once (on CPU) so the same slices are shown
    # every dump across epochs and runs. Only when the hook is on and there's somewhere to
    # write; failure to grab a batch (empty val loader) just disables the hook.
    dump_recon = do_validation and cfg.recon_dump_every_epochs > 0 and cfg.checkpoint_dir is not None
    fixed_recon_batch: RawBatch | None = None
    if dump_recon:
        assert val_loader is not None
        fixed_recon_batch = _capture_fixed_batch(val_loader)
        dump_recon = fixed_recon_batch is not None

    losses: list[float] = []
    epoch_sums: dict[str, float] = {}
    epoch_batches = 0
    last_checkpoint_step: int | None = None
    stopped_early = False
    train_start = time.perf_counter()
    for step in range(start_step, start_step + cfg.steps):
        step_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_raw_batch(batch, device)

        current_lr = _lr_at_step(step + 1, cfg, lr_total_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            components = _compute_vae_loss_components(encoder, decoder, batch, cfg, lpips_net=lpips_net)
        total_loss = components["total"]
        total_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
        optimizer.step()
        _sync_if_cuda(device)
        # One detach->cpu read per term, reused for both the epoch means and the per-step
        # console line — no extra syncs beyond the accumulation the loop already did.
        component_values = {name: float(value.detach().cpu()) for name, value in components.items()}
        loss_value = component_values["total"]
        losses.append(loss_value)
        # Accumulate per-term train means for the current epoch's history line.
        for name, value in component_values.items():
            epoch_sums[name] = epoch_sums.get(name, 0.0) + value
        epoch_batches += 1
        if tracker is not None:
            tracker.update_step(loss_value)

        current_step = step + 1
        is_checkpoint_step = bool(cfg.checkpoint_every_steps) and current_step % cfg.checkpoint_every_steps == 0
        is_epoch_end = do_validation and current_step % cfg.steps_per_epoch == 0

        # Evaluate the stop decision *before* saving, so the checkpoint persists the
        # post-decision tracker state (patience count) — resume_from must not be a step behind.
        should_stop = tracker is not None and is_checkpoint_step and tracker.should_stop()

        if cfg.checkpoint_dir is not None and is_checkpoint_step:
            _save_step_checkpoint(
                cfg, encoder, decoder, optimizer, current_step,
                early_stop_state=tracker.state_dict() if tracker is not None else None,
            )
            last_checkpoint_step = current_step

        if cfg.log_every_steps > 0 and (
            len(losses) == 1 or current_step % cfg.log_every_steps == 0 or len(losses) == cfg.steps
        ):
            elapsed = time.perf_counter() - train_start
            step_elapsed = time.perf_counter() - step_start
            ema_str = f" ema={tracker.ema:.6f}" if tracker is not None and tracker.ema is not None else ""
            # Per-term raw losses so posterior collapse vs. real convergence is legible from
            # the per-step log, not just the opaque total (weighted breakdown is in history).
            terms_str = " ".join(
                f"{name}={component_values[name]:.4f}" for name in _LOSS_TERM_ORDER if name in component_values
            )
            lr_str = f" lr={current_lr:.2e}" if cfg.lr_schedule != "constant" else ""
            _log_training_progress(
                f"stage1_vae {_epoch_label(current_step, cfg.steps_per_epoch)} "
                f"step={current_step}/{start_step + cfg.steps} loss={loss_value:.6f}{ema_str} "
                f"[{terms_str}]{lr_str} "
                f"step_sec={step_elapsed:.3f} avg_sec_per_step={elapsed / len(losses):.3f}"
            )

        if is_epoch_end:
            assert val_loader is not None  # do_validation guarantees this
            epoch_index = current_step // cfg.steps_per_epoch
            train_means = {name: total / max(1, epoch_batches) for name, total in epoch_sums.items()}
            epoch_sums, epoch_batches = {}, 0
            run_val = epoch_index % max(1, cfg.val_every_epochs) == 0

            if run_val:
                val_means, latent_stats = _run_validation(
                    encoder, decoder, val_loader, cfg, device, lpips_net, autocast_ctx
                )
                val_total = val_means.get("total", float("nan"))
                is_best = val_total == val_total and val_total < best_val_total  # NaN-safe
                if is_best:
                    best_val_total = val_total
                if history_path is not None:
                    _append_history(
                        history_path,
                        {
                            "epoch": epoch_index,
                            "step": current_step,
                            "train": train_means,
                            "train_weighted": _weighted_terms(train_means, cfg.loss_weights),
                            "validation": val_means,
                            "validation_weighted": _weighted_terms(val_means, cfg.loss_weights),
                            "latent": latent_stats,
                            "lr": current_lr,
                            "best": is_best,
                            "seconds": time.perf_counter() - train_start,
                        },
                    )
                if is_best and cfg.checkpoint_dir is not None:
                    _save_best_checkpoint(cfg, encoder, decoder, optimizer, current_step)
                latent_str = f" | latent: {summarize_latent_stats(latent_stats)}" if latent_stats else ""
                _log_training_progress(
                    f"stage1_vae epoch={epoch_index} val_total={val_total:.6f} "
                    f"train_total={train_means.get('total', float('nan')):.6f}"
                    f"{' [best]' if is_best else ''}{latent_str}"
                )

            if dump_recon and epoch_index % cfg.recon_dump_every_epochs == 0:
                assert fixed_recon_batch is not None and cfg.checkpoint_dir is not None
                _dump_reconstruction_panel(
                    encoder, decoder, fixed_recon_batch, cfg, device, autocast_ctx, epoch_index
                )

            encoder.train()
            decoder.train()

        if should_stop:
            _log_training_progress(
                f"stage1_vae early-stop at step={current_step}: EMA loss {tracker.ema:.6f} did not improve "
                f">{cfg.early_stopping_min_delta:.1%} for {cfg.early_stopping_patience} checkpoints."
            )
            stopped_early = True
            break

    final_step = start_step + len(losses)
    if (
        cfg.checkpoint_dir is not None
        and cfg.checkpoint_at_end
        and losses
        and last_checkpoint_step != final_step
    ):
        _save_step_checkpoint(
            cfg, encoder, decoder, optimizer, final_step,
            early_stop_state=tracker.state_dict() if tracker is not None else None,
        )

    elapsed_seconds = time.perf_counter() - train_start
    return Stage1VAETrainResult(
        steps=len(losses), losses=losses, elapsed_seconds=elapsed_seconds, stopped_early=stopped_early
    )


def _compute_vae_loss(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    batch: RawBatch,
    cfg: Stage1VAEConfig,
    *,
    lpips_net: nn.Module | None,
) -> torch.Tensor:
    """Weighted total loss (backward target). See `_compute_vae_loss_components` for the
    per-term breakdown used by logging/validation."""

    return _compute_vae_loss_components(encoder, decoder, batch, cfg, lpips_net=lpips_net)["total"]


def _compute_vae_loss_components(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    batch: RawBatch,
    cfg: Stage1VAEConfig,
    *,
    lpips_net: nn.Module | None,
) -> dict[str, torch.Tensor]:
    """Weighted `total` plus each *unweighted* term, so logging can show where the loss
    lives (l1/ssim/nrmse/lpips/kl) rather than one opaque number.

    Only the terms with positive weight are computed (LPIPS especially is expensive), so a
    swept-off term is simply absent from the returned dict — callers read with `.get`.
    """

    # Reparameterize once, here, rather than calling encode() and then encode_dist()
    # separately — that would run the encoder twice with two different random samples,
    # silently decorrelating the KL term from the reconstruction term.
    weights = cfg.loss_weights
    mean, logvar = encoder.encode_dist(batch.image, batch.source_domain)
    eps = torch.randn_like(mean)
    z = mean + eps * torch.exp(0.5 * logvar)
    reconstructed = decoder.decode(z, batch.source_domain)

    components: dict[str, torch.Tensor] = {}
    # data_range must match evaluation's (_DATA_RANGE in evaluation/stage1_report.py).
    # It was previously left at the 1.0 default here while eval used 2.0, so the SSIM
    # being optimized had different c1/c2 stabilizers than the SSIM being reported.
    components["nrmse"] = nrmse_loss(reconstructed, batch.image, data_range=cfg.data_range)
    if weights.get("l1", 0.0) > 0:
        # Plain image-space L1 (no mask): the flat per-voxel absolute-error term that
        # anchors the DC level and drives the background to exactly 0. With the (default-off)
        # foreground-weighting flag, brain voxels are up-weighted instead — see _l1_term.
        components["l1"] = _l1_term(reconstructed, batch.image, cfg)
    if weights.get("ssim", 0.0) > 0:
        # ssim_loss dispatches on rank: 5D volumes -> ssim3d (avg_pool3d), 4D -> 2D ssim.
        components["ssim"] = ssim_loss(
            reconstructed, batch.image, window_size=cfg.ssim_window_size, data_range=cfg.data_range
        )
    if weights.get("lpips", 0.0) > 0:
        # LPIPS wraps a 2D VGG net: for 5D volumes use the slice-averaged variant, for 4D
        # (2D slices) the plain one.
        if reconstructed.ndim == 5:
            components["lpips"] = lpips_loss_3d(
                reconstructed, batch.image, net=lpips_net, num_slices=cfg.lpips_num_slices
            )
        else:
            components["lpips"] = lpips_loss(reconstructed, batch.image, net=lpips_net)
    components["kl"] = kl_divergence(mean, logvar)

    total = reconstructed.sum() * 0.0
    for name, value in components.items():
        total = total + weights.get(name, 0.0) * value
    components["total"] = total
    return components


@torch.no_grad()
def _run_validation(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    val_loader: DataLoader[RawBatch],
    cfg: Stage1VAEConfig,
    device: torch.device,
    lpips_net: nn.Module | None,
    autocast_ctx: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Mean per-term loss + latent posterior-collapse stats over the validation loader.

    Deterministic (latent mean, no sampling). `val_max_batches`>0 caps the pass.
    `ssim3d`/`nrmse`/`lpips` are logged as the challenge metrics too (ssim3d = 1 - ssim
    loss). The latent stats reuse the very `mean`/`logvar` already encoded for the
    reconstruction, so they cost one online accumulation and no extra forward pass.
    Returns `(per_term_means, latent_stats)`."""

    encoder.eval()
    decoder.eval()
    sums: dict[str, float] = {}
    latent_acc = LatentStatsAccumulator(encoder.latent_channels)
    count = 0
    for batch in val_loader:
        if cfg.val_max_batches > 0 and count >= cfg.val_max_batches:
            break
        batch = move_raw_batch(batch, device)
        with autocast_ctx:
            # Deterministic: reconstruct from the latent mean, matching eval — not the
            # sampled path, whose noise would make val loss jump epoch to epoch.
            mean, logvar = encoder.encode_dist(batch.image, batch.source_domain)
            reconstructed = decoder.decode(mean, batch.source_domain)
            components = _reconstruction_components(reconstructed, batch.image, mean, logvar, cfg, lpips_net)
        latent_acc.update(mean, logvar)
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        count += 1
    means = {name: total / max(1, count) for name, total in sums.items()}
    if "ssim" in means:
        means["ssim3d"] = 1.0 - means["ssim"]  # challenge metric (higher better)
    means["num_batches"] = count
    latent_stats = latent_acc.compute(active_threshold=cfg.latent_active_kl_threshold) if count else {}
    return means, latent_stats


def _reconstruction_components(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    cfg: Stage1VAEConfig,
    lpips_net: nn.Module | None,
) -> dict[str, torch.Tensor]:
    """Per-term losses from an already-decoded reconstruction (validation path shares the
    exact term definitions with training's `_compute_vae_loss_components`)."""

    weights = cfg.loss_weights
    components: dict[str, torch.Tensor] = {"nrmse": nrmse_loss(reconstructed, target, data_range=cfg.data_range)}
    if weights.get("l1", 0.0) > 0:
        components["l1"] = _l1_term(reconstructed, target, cfg)
    if weights.get("ssim", 0.0) > 0:
        components["ssim"] = ssim_loss(reconstructed, target, window_size=cfg.ssim_window_size, data_range=cfg.data_range)
    if weights.get("lpips", 0.0) > 0:
        components["lpips"] = (
            lpips_loss_3d(reconstructed, target, net=lpips_net, num_slices=cfg.lpips_num_slices)
            if reconstructed.ndim == 5
            else lpips_loss(reconstructed, target, net=lpips_net)
        )
    components["kl"] = kl_divergence(mean, logvar)
    total = reconstructed.sum() * 0.0
    for name, value in components.items():
        total = total + weights.get(name, 0.0) * value
    components["total"] = total
    return components


def _l1_term(reconstructed: torch.Tensor, target: torch.Tensor, cfg: Stage1VAEConfig) -> torch.Tensor:
    """L1 reconstruction term: plain mean L1 (default) or the foreground-weighted variant.

    With `foreground_loss_weighting=False` (default) this is exactly `masked_l1_loss` with
    no mask — the loss math is unchanged. The flag is an opt-in experiment (item 7), kept
    off until its effect is measured."""

    if cfg.foreground_loss_weighting:
        return foreground_weighted_l1_loss(
            reconstructed,
            target,
            threshold=cfg.foreground_threshold,
            foreground_weight=cfg.foreground_weight,
        )
    return masked_l1_loss(reconstructed, target)


def _lr_at_step(current_step: int, cfg: Stage1VAEConfig, total_steps: int) -> float:
    """Learning rate for a 1-indexed global step. Stateless (pure function of the step) so a
    resumed run lands on the same curve without persisting scheduler state.

    "constant" returns cfg.lr unchanged. "cosine": linear warmup 0 -> lr over
    `lr_warmup_steps`, then a cosine decay to `lr * lr_min_factor` at `total_steps`."""

    base_lr = cfg.lr
    if cfg.lr_schedule == "constant":
        return base_lr
    if cfg.lr_schedule != "cosine":
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {cfg.lr_schedule!r}.")
    warmup = cfg.lr_warmup_steps
    if warmup > 0 and current_step <= warmup:
        return base_lr * current_step / warmup
    min_lr = base_lr * cfg.lr_min_factor
    denominator = max(1, total_steps - warmup)
    progress = min(max((current_step - warmup) / denominator, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _weighted_terms(means: Mapping[str, float], weights: Mapping[str, float]) -> dict[str, float]:
    """Per-term means multiplied by their loss weight, so history shows both the raw term
    and its actual contribution to the total. Non-term keys (num_batches, ssim3d, total,
    latent) are skipped — only weighted terms belong here."""

    return {
        name: float(means[name]) * float(weights[name])
        for name in _LOSS_TERM_ORDER
        if name in means and name in weights
    }


def _capture_fixed_batch(loader: DataLoader[RawBatch]) -> RawBatch | None:
    """Grab the first batch of a loader, kept on CPU, for the fixed reconstruction panel.

    Returns None if the loader yields nothing (an empty validation set disables the hook)."""

    for batch in loader:
        return batch
    return None


@torch.no_grad()
def _dump_reconstruction_panel(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    batch: RawBatch,
    cfg: Stage1VAEConfig,
    device: torch.device,
    autocast_ctx: Any,
    epoch_index: int,
) -> None:
    """Render input/reconstruction/|error| for the fixed batch to checkpoint_dir/recon_epochN.png.

    Deterministic mean reconstruction (no sampling), matching eval. Best-effort: a missing
    matplotlib (the optional 'evaluation' extra) just skips the dump with a log line rather
    than killing a training run."""

    assert cfg.checkpoint_dir is not None
    encoder.eval()
    decoder.eval()
    moved = move_raw_batch(batch, device)
    with autocast_ctx:
        mean, _ = encoder.encode_dist(moved.image, moved.source_domain)
        reconstructed = decoder.decode(mean, moved.source_domain)
    try:
        from fieldbridge.evaluation.stage1_report import render_reconstruction_panel
    except ImportError:  # pragma: no cover - matplotlib-optional path
        _log_training_progress(
            f"stage1_vae epoch={epoch_index}: recon dump skipped (matplotlib unavailable)."
        )
        return
    path = cfg.checkpoint_dir / f"recon_epoch{epoch_index:04d}.png"
    render_reconstruction_panel(
        moved.image.float().cpu(),
        reconstructed.float().cpu(),
        labels=_domain_labels(moved.source_domain),
        path=path,
    )
    _log_training_progress(f"stage1_vae epoch={epoch_index}: wrote reconstruction panel {path.name}")


def _domain_labels(domain: Any) -> list[str]:
    """Per-sample display labels from a (possibly batched) domain."""

    if isinstance(domain, (list, tuple)):
        return [getattr(d, "label", str(d)) for d in domain]
    return [getattr(domain, "label", str(domain))]


def _append_history(history_path: Path, entry: Mapping[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _save_best_checkpoint(
    cfg: Stage1VAEConfig,
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    """Overwrite the single stable-named best checkpoint (model selection by val total).

    Fixed name (not step-stamped) so automation/eval always has one path to the best
    model; `overwrite=True` because replacing the previous best is the intent here — this
    is the one place a checkpoint is deliberately overwritten."""

    assert cfg.checkpoint_dir is not None
    save_checkpoint(
        cfg.checkpoint_dir / "vae_kl_vae_best.pt",
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=True,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _save_step_checkpoint(
    cfg: Stage1VAEConfig,
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    early_stop_state: Mapping[str, Any] | None = None,
) -> None:
    assert cfg.checkpoint_dir is not None
    filename = checkpoint_filename("vae", "kl_vae", step)
    payload: dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if early_stop_state is not None:
        # Persist so resume_from continues the patience count instead of restarting it.
        payload["early_stop"] = dict(early_stop_state)
    save_checkpoint(
        cfg.checkpoint_dir / filename,
        payload,
        max_bytes=cfg.checkpoint_max_bytes,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _epoch_label(current_step: int, steps_per_epoch: int) -> str:
    """Human-readable 'epoch E step s/steps_per_epoch' tag, or a bare step if the epoch
    size is unknown (the streaming loader has no __len__ so the CLI injects it)."""
    if steps_per_epoch <= 0:
        return "epoch=?"
    epoch = (current_step - 1) // steps_per_epoch + 1
    step_in_epoch = (current_step - 1) % steps_per_epoch + 1
    return f"epoch={epoch} [{step_in_epoch}/{steps_per_epoch}]"


def _coerce_config(config: Stage1VAEConfig | Mapping[str, Any] | None) -> Stage1VAEConfig:
    if config is None:
        return Stage1VAEConfig()
    if isinstance(config, Stage1VAEConfig):
        return config
    return Stage1VAEConfig.from_mapping(config)


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
