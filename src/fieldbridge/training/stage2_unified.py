"""Complete retrospective Stage-2 experiment over photometry-factored latents.

This module intentionally does not import the legacy latent-bank-v1 reader.  One shared
conditional velocity field serves all 15 field/contrast domains.  Structural-descriptor
coupling is not used: that artifact remains unauthorized for nearest-neighbour coupling.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentRecord,
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import sha256_json
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256
from fieldbridge.models.discriminators import (
    DomainProjectionDiscriminator,
    domain_labels,
    supported_critic_input,
)
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.checkpoints import load_checkpoint, resolve_git_commit, save_checkpoint
from fieldbridge.training.losses import (
    adversarial_hinge_loss_discriminator,
    adversarial_hinge_loss_generator,
    gradient_loss,
    masked_l1_loss,
)
from fieldbridge.training.train_loop import assert_frozen
from fieldbridge.utils.seeding import seed_everything

UNIFIED_STAGE2_CONTRACT = "stage2-unified-retrospective-full-model-v1"
UNIFIED_RESUME_CONTRACT = "stage2-unified-exact-resume-v1"
UNIFIED_HISTORY_CONTRACT = "stage2-unified-term-history-v1"

CriticSpace = Literal["latent", "image"]
Precision = Literal["fp32", "bf16"]
Solver = Literal["euler", "heun"]
Bridge = Literal["schrodinger", "ot_cfm"]

DEFAULT_UNIFIED_WEIGHTS: dict[str, float] = {
    "sb": 1.0,
    "identity": 0.1,
    "anatomy": 0.02,
    "graph": 0.01,
    "adversarial": 0.05,
    "domain": 0.1,
}


@dataclass(frozen=True, slots=True)
class UnifiedStage2Config:
    steps: int = 2
    batch_size: int = 2
    seed: int = 13
    lr_generator: float = 2.0e-4
    lr_critic: float = 2.0e-4
    weight_decay: float = 0.0
    bridge: Bridge = "schrodinger"
    sigma: float = 0.1
    time_eps: float = 1.0e-3
    integration_steps: int = 4
    integration_solver: Solver = "heun"
    critic_space: CriticSpace = "latent"
    critic_channels: tuple[int, ...] = (32, 64, 128)
    anatomy_pool_scales: tuple[int, ...] = (1, 2, 4)
    anatomy_support_erosion: int = 1
    loss_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_UNIFIED_WEIGHTS)
    )
    device: str = "auto"
    precision: Precision = "bf16"
    grad_clip_norm: float = 1.0
    scheduler_t_max: int = 100_000
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 1000
    checkpoint_max_bytes: int = 1_500_000_000
    history_jsonl: Path | None = None
    resume_from: Path | None = None
    sanity_steps: int = 0
    sanity_max_aux_to_flow_ratio: float = 1.0
    variant: str = "full"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UnifiedStage2Config":
        defaults = cls()
        training = data.get("training", data)
        if not isinstance(training, Mapping):
            raise ValueError("Unified Stage-2 training config must be a mapping.")
        weights = dict(DEFAULT_UNIFIED_WEIGHTS)
        weights.update(dict(training.get("loss_weights", {})))
        critic = training.get("critic", {})
        critic = critic if isinstance(critic, Mapping) else {}
        anatomy = training.get("anatomy", {})
        anatomy = anatomy if isinstance(anatomy, Mapping) else {}
        checkpoint = training.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        sanity = training.get("sanity", {})
        sanity = sanity if isinstance(sanity, Mapping) else {}
        value = cls(
            steps=int(training.get("steps", defaults.steps)),
            batch_size=int(training.get("batch_size", defaults.batch_size)),
            seed=int(training.get("seed", defaults.seed)),
            lr_generator=float(training.get("lr_generator", defaults.lr_generator)),
            lr_critic=float(training.get("lr_critic", defaults.lr_critic)),
            weight_decay=float(training.get("weight_decay", defaults.weight_decay)),
            bridge=str(training.get("bridge", defaults.bridge)),  # type: ignore[arg-type]
            sigma=float(training.get("sigma", defaults.sigma)),
            time_eps=float(training.get("time_eps", defaults.time_eps)),
            integration_steps=int(training.get("integration_steps", defaults.integration_steps)),
            integration_solver=str(training.get("integration_solver", defaults.integration_solver)),  # type: ignore[arg-type]
            critic_space=str(critic.get("space", defaults.critic_space)),  # type: ignore[arg-type]
            critic_channels=tuple(int(v) for v in critic.get("channels", defaults.critic_channels)),
            anatomy_pool_scales=tuple(
                int(v) for v in anatomy.get("pool_scales", defaults.anatomy_pool_scales)
            ),
            anatomy_support_erosion=int(
                anatomy.get("support_erosion", defaults.anatomy_support_erosion)
            ),
            loss_weights=weights,
            device=str(training.get("device", defaults.device)),
            precision=str(training.get("precision", defaults.precision)),  # type: ignore[arg-type]
            grad_clip_norm=float(training.get("grad_clip_norm", defaults.grad_clip_norm)),
            scheduler_t_max=int(training.get("scheduler_t_max", defaults.scheduler_t_max)),
            checkpoint_dir=(
                Path(checkpoint["dir"]) if checkpoint.get("dir") else defaults.checkpoint_dir
            ),
            checkpoint_every_steps=int(
                checkpoint.get("every_steps", defaults.checkpoint_every_steps)
            ),
            checkpoint_max_bytes=int(
                checkpoint.get("max_bytes", defaults.checkpoint_max_bytes)
            ),
            history_jsonl=(
                Path(training["history_jsonl"]) if training.get("history_jsonl") else None
            ),
            resume_from=(Path(training["resume_from"]) if training.get("resume_from") else None),
            sanity_steps=int(sanity.get("steps", defaults.sanity_steps)),
            sanity_max_aux_to_flow_ratio=float(
                sanity.get("max_aux_to_flow_ratio", defaults.sanity_max_aux_to_flow_ratio)
            ),
            variant=str(training.get("variant", defaults.variant)),
        )
        validate_unified_config(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": UNIFIED_STAGE2_CONTRACT,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "lr_generator": self.lr_generator,
            "lr_critic": self.lr_critic,
            "weight_decay": self.weight_decay,
            "bridge": self.bridge,
            "sigma": self.sigma,
            "time_eps": self.time_eps,
            "integration_steps": self.integration_steps,
            "integration_solver": self.integration_solver,
            "critic_space": self.critic_space,
            "critic_channels": list(self.critic_channels),
            "anatomy_pool_scales": list(self.anatomy_pool_scales),
            "anatomy_support_erosion": self.anatomy_support_erosion,
            "loss_weights": dict(self.loss_weights),
            "device": self.device,
            "precision": self.precision,
            "grad_clip_norm": self.grad_clip_norm,
            "scheduler_t_max": self.scheduler_t_max,
            "sanity_steps": self.sanity_steps,
            "sanity_max_aux_to_flow_ratio": self.sanity_max_aux_to_flow_ratio,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class UnifiedStage2Result:
    completed_steps: int
    checkpoint: str | None
    history_jsonl: str
    sanity_report: dict[str, Any]
    run_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": UNIFIED_STAGE2_CONTRACT,
            "completed_steps": self.completed_steps,
            "checkpoint": self.checkpoint,
            "history_jsonl": self.history_jsonl,
            "sanity_report": self.sanity_report,
            "run_fingerprint": self.run_fingerprint,
        }


def integrate_transport(
    translator: BaseTranslator,
    z: torch.Tensor,
    source_domains: Sequence[Domain],
    target_domains: Sequence[Domain],
    *,
    steps: int,
    solver: Solver = "heun",
) -> torch.Tensor:
    """Differentiable fixed-grid ODE integration used by identity and graph losses."""

    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("Transport integration requires steps>=1 and euler/heun.")
    h = 1.0 / float(steps)
    current = z
    for index in range(steps):
        t0 = torch.full((z.shape[0],), index * h, device=z.device, dtype=z.dtype)
        velocity = translator(current, source_domains, target_domains, t0)
        proposal = current + h * velocity
        if solver == "euler":
            current = proposal
        else:
            t1 = torch.full((z.shape[0],), (index + 1) * h, device=z.device, dtype=z.dtype)
            correction = translator(proposal, source_domains, target_domains, t1)
            current = current + 0.5 * h * (velocity + correction)
    return current


def graph_consistency_loss(
    translator: BaseTranslator,
    z: torch.Tensor,
    source_domains: Sequence[Domain],
    intermediate_domains: Sequence[Domain],
    target_domains: Sequence[Domain],
    support: torch.Tensor,
    *,
    steps: int,
    solver: Solver,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direct = integrate_transport(
        translator, z, source_domains, target_domains, steps=steps, solver=solver
    )
    first = integrate_transport(
        translator, z, source_domains, intermediate_domains, steps=steps, solver=solver
    )
    composed = integrate_transport(
        translator, first, intermediate_domains, target_domains, steps=steps, solver=solver
    )
    return masked_l1_loss(direct, composed, support), direct, composed


def anatomy_preservation_components(
    source_image: torch.Tensor,
    translated_image: torch.Tensor,
    latent_support: torch.Tensor,
    *,
    pool_scales: Sequence[int] = (1, 2, 4),
    support_erosion: int = 1,
) -> dict[str, torch.Tensor]:
    """Low/mid-frequency and smoothed edge/gradient consistency on valid anatomy.

    There is deliberately no raw-image or high-pass equality term: the objective cannot
    improve by copying/hallucinating unsupported fine texture.
    """

    if source_image.shape != translated_image.shape or source_image.ndim != 5:
        raise ValueError("Anatomy loss requires aligned 3-D image batches.")
    support = F.interpolate(latent_support.float(), size=source_image.shape[2:], mode="nearest")
    for _ in range(max(0, int(support_erosion))):
        unsupported = 1.0 - support
        support = 1.0 - F.max_pool3d(unsupported, 3, stride=1, padding=1)
    support = support > 0.5
    if not bool(support.any()):
        raise ValueError("Anatomy support eroded to empty.")
    low_mid: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    for raw_scale in pool_scales:
        scale = int(raw_scale)
        if scale < 1:
            raise ValueError("Anatomy pooling scales must be positive.")
        if scale == 1:
            # Even scale=1 uses an explicit 3x3 low-pass; raw high-frequency equality
            # is intentionally absent.
            src = F.avg_pool3d(source_image, 3, stride=1, padding=1)
            pred = F.avg_pool3d(translated_image, 3, stride=1, padding=1)
            mask = support
        else:
            src = F.avg_pool3d(source_image, scale, stride=scale)
            pred = F.avg_pool3d(translated_image, scale, stride=scale)
            mask = F.avg_pool3d(support.float(), scale, stride=scale) >= 1.0
        if not bool(mask.any()):
            continue
        low_mid.append(masked_l1_loss(pred, src, mask))
        gradients.append(gradient_loss(pred, src, mask))
        if min(src.shape[2:]) >= 2:
            src_edge = _gradient_magnitude(src)
            pred_edge = _gradient_magnitude(pred)
            edge_mask = _edge_mask(mask)
            if bool(edge_mask.any()):
                edges.append(masked_l1_loss(pred_edge, src_edge, edge_mask))
    if not low_mid:
        raise ValueError("No supported anatomy scale remained.")
    zero = source_image.sum() * 0.0
    result = {
        "low_mid": torch.stack(low_mid).mean(),
        "gradient": torch.stack(gradients).mean() if gradients else zero,
        "edge": torch.stack(edges).mean() if edges else zero,
    }
    result["total"] = (result["low_mid"] + result["gradient"] + result["edge"]) / 3.0
    return result


def run_stage2_unified_train(
    config: UnifiedStage2Config | Mapping[str, Any],
    *,
    translator: BaseTranslator,
    decoder: nn.Module,
    train_index: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    critic: DomainProjectionDiscriminator | None = None,
) -> UnifiedStage2Result:
    cfg = config if isinstance(config, UnifiedStage2Config) else UnifiedStage2Config.from_mapping(config)
    validate_unified_config(cfg)
    if train_index.split != "train":
        raise ValueError("Unified model fitting is restricted to R/train.")
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    translator = translator.to(device)
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    assert_frozen(decoder)
    latent_channels = int(stats.mean.numel())
    critic_input_channels = latent_channels + 1 if cfg.critic_space == "latent" else 2
    critic = critic or DomainProjectionDiscriminator(
        critic_input_channels, cfg.critic_channels
    )
    critic = critic.to(device)
    generator_optimizer = torch.optim.AdamW(
        translator.parameters(), lr=cfg.lr_generator, weight_decay=cfg.weight_decay
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay
    )
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        generator_optimizer, T_max=max(1, cfg.scheduler_t_max)
    )
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer, T_max=max(1, cfg.scheduler_t_max)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.precision == "bf16")
    sampler = torch.Generator().manual_seed(cfg.seed)
    pools = _DomainPools.from_index(train_index)
    pools.require_all_domains()
    history_path = cfg.history_jsonl or (
        (cfg.checkpoint_dir or Path.cwd()) / "stage2_unified_history.jsonl"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "contract_version": UNIFIED_STAGE2_CONTRACT,
        "config": cfg.to_dict(),
        "bank_artifact_sha256": train_index.artifact_sha256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "bank_vae_provenance": dict(getattr(train_index, "manifest", {}).get("vae", {})),
        "frozen_decoder_state_sha256": _module_state_sha256(decoder),
        "git_commit": resolve_git_commit(),
    }
    run_fingerprint = sha256_json(run_identity)
    cursor = 0
    history_generation = 0
    if cfg.resume_from is not None:
        cursor = _restore_exact(
            cfg.resume_from,
            translator=translator,
            critic=critic,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            generator_scheduler=generator_scheduler,
            critic_scheduler=critic_scheduler,
            scaler=scaler,
            sampler=sampler,
            expected_run_fingerprint=run_fingerprint,
            history_path=history_path,
        )
        history_generation = _prepare_history_resume(
            history_path, cursor=cursor, run_fingerprint=run_fingerprint
        )
        if cursor > cfg.steps:
            raise ValueError("Exact-resume cursor is beyond the configured terminal step.")
    elif history_path.exists() and history_path.stat().st_size:
        raise FileExistsError("Non-empty history exists but no exact-resume checkpoint was supplied.")

    sanity_rows: list[dict[str, float]] = []
    last_checkpoint: Path | None = cfg.resume_from
    start = time.perf_counter()
    for step in range(cursor, cfg.steps):
        step_start = time.perf_counter()
        try:
            batch = _sample_training_batch(train_index, pools, stats, cfg, device, sampler)
            row = _train_step(
                cfg,
                translator,
                critic,
                decoder,
                generator_optimizer,
                critic_optimizer,
                scaler,
                batch,
                sampler,
            )
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _append_jsonl(
                history_path,
                {
                    "contract_version": UNIFIED_HISTORY_CONTRACT,
                    "step": step,
                    "event": "oom_hard_stop",
                    "fallback": "forbidden",
                    "run_fingerprint": run_fingerprint,
                    "history_generation": history_generation,
                },
            )
            raise
        generator_scheduler.step()
        if bool(row["critic/updated"]):
            critic_scheduler.step()
        row.update(
            {
                "contract_version": UNIFIED_HISTORY_CONTRACT,
                "step": step + 1,
                "elapsed_seconds": time.perf_counter() - start,
                "step_seconds": time.perf_counter() - step_start,
                "examples_per_second": cfg.batch_size / max(1e-9, time.perf_counter() - step_start),
                "peak_cuda_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "generator_lr": generator_scheduler.get_last_lr()[0],
                "critic_lr": critic_scheduler.get_last_lr()[0],
                "run_fingerprint": run_fingerprint,
                "history_generation": history_generation,
            }
        )
        _append_jsonl(history_path, row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        if step < cfg.sanity_steps:
            sanity_rows.append(
                {
                    "flow": float(row["weighted/sb"]),
                    "aux": float(row["weighted/auxiliary_total"]),
                }
            )
        if cfg.checkpoint_dir is not None and cfg.checkpoint_every_steps > 0:
            if (step + 1) % cfg.checkpoint_every_steps == 0:
                last_checkpoint = _save_exact_checkpoint(
                    cfg,
                    step + 1,
                    translator,
                    critic,
                    generator_optimizer,
                    critic_optimizer,
                    generator_scheduler,
                    critic_scheduler,
                    scaler,
                    sampler,
                    run_fingerprint,
                    history_path,
                )
    sanity = _sanity_report(sanity_rows, cfg.sanity_max_aux_to_flow_ratio)
    if sanity["status"] == "fail":
        raise RuntimeError(
            "Unified Stage-2 sanity failed: weighted auxiliary objectives dominate flow."
        )
    if cfg.checkpoint_dir is not None and cfg.steps > cursor and (
        last_checkpoint is None
        or not last_checkpoint.name.endswith(f"step{cfg.steps:09d}.pt")
    ):
        last_checkpoint = _save_exact_checkpoint(
            cfg,
            cfg.steps,
            translator,
            critic,
            generator_optimizer,
            critic_optimizer,
            generator_scheduler,
            critic_scheduler,
            scaler,
            sampler,
            run_fingerprint,
            history_path,
        )
    return UnifiedStage2Result(
        completed_steps=cfg.steps,
        checkpoint=str(last_checkpoint) if last_checkpoint else None,
        history_jsonl=str(history_path),
        sanity_report=sanity,
        run_fingerprint=run_fingerprint,
    )


@dataclass(slots=True)
class _TrainingBatch:
    source: torch.Tensor
    source_support: torch.Tensor
    source_domains: list[Domain]
    target: torch.Tensor
    target_support: torch.Tensor
    target_domains: list[Domain]
    source_records: list[FactoredLatentRecord]
    target_records: list[FactoredLatentRecord]


def _train_step(
    cfg: UnifiedStage2Config,
    translator: BaseTranslator,
    critic: DomainProjectionDiscriminator,
    decoder: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: _TrainingBatch,
    sampler: torch.Generator,
) -> dict[str, Any]:
    translator.train()
    critic.train()
    support = batch.source_support & batch.target_support
    if not bool(support.any(dim=(1, 2, 3, 4)).all()):
        raise ValueError("Source/target operational support intersection is empty.")
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if batch.source.device.type == "cuda" and cfg.precision == "bf16"
        else nullcontext()
    )
    zero = batch.source.sum() * 0.0
    adversarial_active = cfg.loss_weights["adversarial"] > 0
    domain_active = cfg.loss_weights["domain"] > 0
    critic_active = adversarial_active or domain_active
    critic_adv = zero
    critic_domain = zero
    critic_total = zero
    critic_grad = 0.0
    if critic_active:
        # Critic update uses a detached, fully integrated generator sample.
        critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast:
            fake_detached = integrate_transport(
                translator,
                batch.source,
                batch.source_domains,
                batch.target_domains,
                steps=cfg.integration_steps,
                solver=cfg.integration_solver,
            )
        real_view = _critic_view(
            batch.target, batch.target_support, batch.target_domains, decoder, cfg.critic_space
        )
        fake_view = _critic_view(
            fake_detached, support, batch.target_domains, decoder, cfg.critic_space
        )
        with autocast:
            real_score, real_domain_logits = critic(real_view, batch.target_domains)
            fake_score, _ = critic(fake_view, batch.target_domains)
            critic_adv = adversarial_hinge_loss_discriminator(real_score, fake_score)
            critic_domain = F.cross_entropy(
                real_domain_logits,
                domain_labels(
                    batch.target_domains, len(batch.target_domains), batch.source.device
                ),
            )
            critic_total = (
                critic_adv * float(adversarial_active)
                + cfg.loss_weights["domain"] * critic_domain
            )
        scaler.scale(critic_total).backward()
        scaler.unscale_(critic_optimizer)
        critic_grad = float(
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip_norm)
        )
        scaler.step(critic_optimizer)

    generator_optimizer.zero_grad(set_to_none=True)
    if critic_active:
        critic.requires_grad_(False)
    with autocast:
        time_values = _rand((cfg.batch_size,), sampler, batch.source.device).clamp(
            cfg.time_eps, 1.0 - cfg.time_eps
        )
        z_t, flow_target = _bridge_sample(batch.source, batch.target, time_values, cfg, sampler)
        predicted_velocity = translator(
            z_t, batch.source_domains, batch.target_domains, time_values
        )
        flow = _masked_mse(predicted_velocity, flow_target, support)
        identity = zero
        if cfg.loss_weights["identity"] > 0:
            identity_generated = integrate_transport(
                translator,
                batch.source,
                batch.source_domains,
                batch.source_domains,
                steps=cfg.integration_steps,
                solver=cfg.integration_solver,
            )
            identity = masked_l1_loss(
                identity_generated, batch.source, batch.source_support
            )
        graph = zero
        graph_discrepancy = zero
        intermediate: list[Domain] | None = None
        generated: torch.Tensor | None = None
        if cfg.loss_weights["graph"] > 0:
            intermediate = _intermediate_domains(
                batch.source_domains, batch.target_domains, sampler
            )
            graph, generated, composed = graph_consistency_loss(
                translator,
                batch.source,
                batch.source_domains,
                intermediate,
                batch.target_domains,
                support,
                steps=cfg.integration_steps,
                solver=cfg.integration_solver,
            )
            graph_discrepancy = _masked_mse(generated, composed, support)
        needs_generated = (
            cfg.loss_weights["anatomy"] > 0 or adversarial_active or domain_active
        )
        if generated is None and needs_generated:
            generated = integrate_transport(
                translator,
                batch.source,
                batch.source_domains,
                batch.target_domains,
                steps=cfg.integration_steps,
                solver=cfg.integration_solver,
            )
        anatomy_parts = {"low_mid": zero, "edge": zero, "gradient": zero, "total": zero}
        if cfg.loss_weights["anatomy"] > 0:
            assert generated is not None
            source_image = _decode(decoder, batch.source, batch.source_domains)
            generated_image = _decode(decoder, generated, batch.target_domains)
            anatomy_parts = anatomy_preservation_components(
                source_image,
                generated_image,
                batch.source_support,
                pool_scales=cfg.anatomy_pool_scales,
                support_erosion=cfg.anatomy_support_erosion,
            )
        adversarial = zero
        domain = zero
        if critic_active:
            assert generated is not None
            generated_view = _critic_view(
                generated, support, batch.target_domains, decoder, cfg.critic_space
            )
            fake_score_for_generator, fake_domain_logits = critic(
                generated_view, batch.target_domains
            )
            if adversarial_active:
                adversarial = adversarial_hinge_loss_generator(fake_score_for_generator)
            if domain_active:
                domain = F.cross_entropy(
                    fake_domain_logits,
                    domain_labels(
                        batch.target_domains, len(batch.target_domains), batch.source.device
                    ),
                )
        raw_terms = {
            "sb": flow,
            "identity": identity,
            "anatomy": anatomy_parts["total"],
            "graph": graph,
            "adversarial": adversarial,
            "domain": domain,
        }
        weighted = {name: cfg.loss_weights[name] * value for name, value in raw_terms.items()}
        generator_total = sum(weighted.values())
    if not bool(torch.isfinite(generator_total)):
        raise FloatingPointError("Unified generator loss is non-finite.")
    scaler.scale(generator_total).backward()
    scaler.unscale_(generator_optimizer)
    generator_grad = float(
        torch.nn.utils.clip_grad_norm_(translator.parameters(), cfg.grad_clip_norm)
    )
    scaler.step(generator_optimizer)
    scaler.update()
    if critic_active:
        critic.requires_grad_(True)
    weighted_aux = sum(weighted[name].abs() for name in weighted if name != "sb")
    row: dict[str, Any] = {
        **{f"raw/{key}": float(value.detach().float().cpu()) for key, value in raw_terms.items()},
        **{f"weighted/{key}": float(value.detach().float().cpu()) for key, value in weighted.items()},
        "weighted/auxiliary_total": float(weighted_aux.detach().float().cpu()),
        "weighted/aux_to_flow_ratio": float(
            (weighted_aux / weighted["sb"].abs().clamp_min(1e-12)).detach().float().cpu()
        ),
        "raw/anatomy_low_mid": float(anatomy_parts["low_mid"].detach().float().cpu()),
        "raw/anatomy_edge": float(anatomy_parts["edge"].detach().float().cpu()),
        "raw/anatomy_gradient": float(anatomy_parts["gradient"].detach().float().cpu()),
        "graph/direct_vs_composed_l1": float(graph.detach().float().cpu()),
        "graph/direct_vs_composed_mse": float(graph_discrepancy.detach().float().cpu()),
        "critic/total": float(critic_total.detach().float().cpu()),
        "critic/adversarial": float(critic_adv.detach().float().cpu()),
        "critic/domain": float(critic_domain.detach().float().cpu()),
        "critic/updated": critic_active,
        "gradient/generator_norm": generator_grad,
        "gradient/critic_norm": critic_grad,
        "transition": f"{batch.source_domains[0].label}->{batch.target_domains[0].label}",
        "graph_path": (
            f"{batch.source_domains[0].label}->{intermediate[0].label}"
            f"->{batch.target_domains[0].label}"
            if intermediate is not None
            else "disabled"
        ),
        "source_records": [item.case_id for item in batch.source_records],
        "target_records": [item.case_id for item in batch.target_records],
        "prospective_records_loaded": 0,
        "descriptor_coupling_used": False,
    }
    return row


def _decode(decoder: nn.Module, latent: torch.Tensor, domains: Sequence[Domain]) -> torch.Tensor:
    if hasattr(decoder, "decode"):
        return decoder.decode(latent, domains)  # type: ignore[attr-defined,no-any-return]
    return decoder(latent)  # type: ignore[no-any-return]


def _critic_view(
    latent: torch.Tensor,
    support: torch.Tensor,
    domains: Sequence[Domain],
    decoder: nn.Module,
    space: CriticSpace,
) -> torch.Tensor:
    if space == "latent":
        return supported_critic_input(latent, support)
    image = _decode(decoder, latent, domains)
    image_support = F.interpolate(support.float(), size=image.shape[2:], mode="nearest") > 0.5
    return supported_critic_input(image, image_support)


def _bridge_sample(
    source: torch.Tensor,
    target: torch.Tensor,
    time_values: torch.Tensor,
    cfg: UnifiedStage2Config,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (source.shape[0],) + (1,) * (source.ndim - 1)
    t = time_values.reshape(shape)
    if cfg.bridge == "ot_cfm":
        return (1.0 - t) * source + t * target, target - source
    noise = _randn(tuple(source.shape), sampler, source.device, source.dtype)
    z_t = (1.0 - t) * source + t * target + cfg.sigma * torch.sqrt(t * (1.0 - t)) * noise
    return z_t, (target - z_t) / (1.0 - t).clamp_min(cfg.time_eps)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    mask = support.to(prediction.dtype).expand_as(prediction)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def _gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    components: list[torch.Tensor] = []
    common = [slice(None)] * tensor.ndim
    spatial_shape = [max(0, int(v) - 1) for v in tensor.shape[2:]]
    if any(v == 0 for v in spatial_shape):
        return tensor[..., :0, :0, :0]
    for dim in range(2, tensor.ndim):
        value = tensor.diff(dim=dim).abs()
        slices = list(common)
        for other in range(2, tensor.ndim):
            if other != dim:
                slices[other] = slice(0, spatial_shape[other - 2])
        components.append(value[tuple(slices)])
    return torch.stack(components).mean(0)


def _edge_mask(mask: torch.Tensor) -> torch.Tensor:
    return (
        mask[:, :, :-1, :-1, :-1]
        & mask[:, :, 1:, :-1, :-1]
        & mask[:, :, :-1, 1:, :-1]
        & mask[:, :, :-1, :-1, 1:]
    )


@dataclass(frozen=True, slots=True)
class _DomainPools:
    table: dict[Contrast, dict[float, tuple[int, ...]]]

    @classmethod
    def from_index(cls, index: PhotometryFactoredLatentBankIndex) -> "_DomainPools":
        mutable: dict[Contrast, dict[float, list[int]]] = defaultdict(lambda: defaultdict(list))
        for position, record in enumerate(index.records):
            mutable[Contrast.parse(record.domain.contrast)][record.domain.field_strength_t].append(position)
        return cls(
            {
                contrast: {field: tuple(values) for field, values in fields.items()}
                for contrast, fields in mutable.items()
            }
        )

    def require_all_domains(self) -> None:
        missing = [
            f"{field:g}T/{contrast.value}"
            for contrast in Contrast
            for field in FIELD_STRENGTHS_T
            if not self.table.get(contrast, {}).get(field)
        ]
        if missing:
            raise ValueError(f"Full unified training requires all 15 R/train domains; missing={missing}.")


def _sample_training_batch(
    index: PhotometryFactoredLatentBankIndex,
    pools: _DomainPools,
    stats: FactoredLatentStats,
    cfg: UnifiedStage2Config,
    device: torch.device,
    sampler: torch.Generator,
) -> _TrainingBatch:
    contrast = list(Contrast)[int(torch.randint(0, len(Contrast), (1,), generator=sampler))]
    source_position = int(torch.randint(0, len(FIELD_STRENGTHS_T), (1,), generator=sampler))
    target_position = int(torch.randint(0, len(FIELD_STRENGTHS_T) - 1, (1,), generator=sampler))
    if target_position >= source_position:
        target_position += 1
    source_field = FIELD_STRENGTHS_T[source_position]
    target_field = FIELD_STRENGTHS_T[target_position]
    source_pool = pools.table[contrast][source_field]
    target_pool = pools.table[contrast][target_field]
    source_indices = _draw(source_pool, cfg.batch_size, sampler)
    # Enforce subject-group exclusion before tensor payload loading.
    target_indices: list[int] = []
    for source_index in source_indices:
        candidates = [
            value
            for value in target_pool
            if index.records[value].subject_group_id
            != index.records[source_index].subject_group_id
        ]
        if not candidates:
            raise ValueError("No same-contrast target remains after subject-group exclusion.")
        target_indices.extend(_draw(tuple(candidates), 1, sampler))
    source, source_support, source_domains, source_records = index.load_batch(source_indices)
    target, target_support, target_domains, target_records = index.load_batch(target_indices)
    source = stats.normalize(source.to(device), source_support.to(device))
    target = stats.normalize(target.to(device), target_support.to(device))
    return _TrainingBatch(
        source,
        source_support.to(device),
        source_domains,
        target,
        target_support.to(device),
        target_domains,
        source_records,
        target_records,
    )


def _draw(pool: tuple[int, ...], count: int, sampler: torch.Generator) -> list[int]:
    positions = torch.randint(0, len(pool), (count,), generator=sampler).tolist()
    return [pool[int(position)] for position in positions]


def _intermediate_domains(
    source: Sequence[Domain], target: Sequence[Domain], sampler: torch.Generator
) -> list[Domain]:
    result: list[Domain] = []
    for start, end in zip(source, target):
        fields = [
            field
            for field in FIELD_STRENGTHS_T
            if field not in {start.field_strength_t, end.field_strength_t}
        ]
        chosen = fields[int(torch.randint(0, len(fields), (1,), generator=sampler))]
        result.append(Domain(chosen, start.contrast))
    return result


def _save_exact_checkpoint(
    cfg: UnifiedStage2Config,
    cursor: int,
    translator: nn.Module,
    critic: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    generator_scheduler: torch.optim.lr_scheduler.LRScheduler,
    critic_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: torch.Generator,
    run_fingerprint: str,
    history_path: Path,
) -> Path:
    assert cfg.checkpoint_dir is not None
    path = cfg.checkpoint_dir / f"stage2_unified_{cfg.variant}_step{cursor:09d}.pt"
    history_bytes = history_path.read_bytes()
    state = {
        "contract_version": UNIFIED_RESUME_CONTRACT,
        "run_fingerprint": run_fingerprint,
        "training_cursor": cursor,
        "translator": translator.state_dict(),
        "critic": critic.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "generator_scheduler": generator_scheduler.state_dict(),
        "critic_scheduler": critic_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "sampler_rng": sampler.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": _numpy_rng_state(),
        "history_prefix_bytes": len(history_bytes),
        "history_prefix_sha256": hashlib.sha256(history_bytes).hexdigest(),
    }
    return save_checkpoint(
        path,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=False,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _restore_exact(
    path: Path,
    *,
    translator: nn.Module,
    critic: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    generator_scheduler: torch.optim.lr_scheduler.LRScheduler,
    critic_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: torch.Generator,
    expected_run_fingerprint: str,
    history_path: Path,
) -> int:
    state = load_checkpoint(path)
    if state.get("contract_version") != UNIFIED_RESUME_CONTRACT:
        raise ValueError("Unified checkpoint exact-resume contract mismatch.")
    if state.get("run_fingerprint") != expected_run_fingerprint:
        raise ValueError("Unified checkpoint config/bank/code identity mismatch.")
    if not history_path.is_file():
        raise ValueError("Exact resume requires the checkpoint-bound JSONL history.")
    history_bytes = history_path.read_bytes()
    prefix_bytes = int(state.get("history_prefix_bytes", -1))
    if prefix_bytes < 0 or len(history_bytes) < prefix_bytes or hashlib.sha256(
        history_bytes[:prefix_bytes]
    ).hexdigest() != state.get("history_prefix_sha256"):
        raise ValueError("Unified checkpoint history prefix changed or is incomplete.")
    translator.load_state_dict(state["translator"], strict=True)
    critic.load_state_dict(state["critic"], strict=True)
    generator_optimizer.load_state_dict(state["generator_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    generator_scheduler.load_state_dict(state["generator_scheduler"])
    critic_scheduler.load_state_dict(state["critic_scheduler"])
    scaler.load_state_dict(state["scaler"])
    sampler.set_state(state["sampler_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng"):
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state(
        (
            str(numpy_rng["bit_generator"]),
            numpy_rng["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_rng["position"]),
            int(numpy_rng["has_gauss"]),
            float(numpy_rng["cached_gaussian"]),
        )
    )
    cursor = int(state["training_cursor"])
    if cursor < 0:
        raise ValueError("Unified checkpoint has an invalid training cursor.")
    return cursor


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(encoded)
        handle.flush()


def _prepare_history_resume(path: Path, *, cursor: int, run_fingerprint: str) -> int:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Unified JSONL history belongs to another run.")
    generations = [int(row.get("history_generation", 0)) for row in rows]
    generation = (max(generations) + 1) if generations else 1
    orphan_steps = [
        int(row["step"])
        for row in rows
        if isinstance(row.get("step"), int) and int(row["step"]) > cursor
    ]
    if orphan_steps:
        _append_jsonl(
            path,
            {
                "contract_version": UNIFIED_HISTORY_CONTRACT,
                "event": "exact_resume_rollback",
                "checkpoint_cursor": cursor,
                "preserved_orphan_step_min": min(orphan_steps),
                "preserved_orphan_step_max": max(orphan_steps),
                "authoritative_generation": generation,
                "history_generation": generation,
                "run_fingerprint": run_fingerprint,
            },
        )
    return generation


def _numpy_rng_state() -> dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "keys": torch.from_numpy(keys.astype(np.int64, copy=True)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _module_state_sha256(module: nn.Module) -> str:
    return sha256_json(
        {
            name: storage_tensor_sha256(value.detach().cpu().contiguous())
            for name, value in sorted(module.state_dict().items())
        }
    )


def _sanity_report(rows: Sequence[Mapping[str, float]], threshold: float) -> dict[str, Any]:
    if not rows:
        return {"status": "not_requested", "steps": 0, "max_aux_to_flow_ratio": None}
    ratios = [item["aux"] / max(abs(item["flow"]), 1e-12) for item in rows]
    maximum = max(ratios)
    return {
        "status": "pass" if maximum <= threshold else "fail",
        "steps": len(rows),
        "max_aux_to_flow_ratio": maximum,
        "threshold": threshold,
        "weighted_terms_only": True,
    }


def _rand(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device="cpu").to(device)


def _randn(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def validate_unified_config(cfg: UnifiedStage2Config) -> None:
    if cfg.steps < 1 or cfg.batch_size < 1 or cfg.integration_steps < 1:
        raise ValueError("steps, batch_size, and integration_steps must be positive.")
    if cfg.bridge not in {"schrodinger", "ot_cfm"}:
        raise ValueError("bridge must be schrodinger or ot_cfm.")
    if cfg.integration_solver not in {"euler", "heun"}:
        raise ValueError("integration_solver must be euler or heun.")
    if cfg.critic_space not in {"latent", "image"}:
        raise ValueError("critic.space must be latent or image.")
    if cfg.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16.")
    if set(cfg.loss_weights) != set(DEFAULT_UNIFIED_WEIGHTS):
        raise ValueError("Unified loss weights must specify exactly all six objectives.")
    if any(not np.isfinite(value) or value < 0 for value in cfg.loss_weights.values()):
        raise ValueError("Unified loss weights must be finite and non-negative.")
    if cfg.loss_weights["sb"] <= 0:
        raise ValueError("The SB objective must remain active.")
    if not 0.0 < cfg.sanity_max_aux_to_flow_ratio:
        raise ValueError("Sanity auxiliary/flow threshold must be positive.")


__all__ = [
    "DEFAULT_UNIFIED_WEIGHTS",
    "UNIFIED_HISTORY_CONTRACT",
    "UNIFIED_RESUME_CONTRACT",
    "UNIFIED_STAGE2_CONTRACT",
    "UnifiedStage2Config",
    "UnifiedStage2Result",
    "anatomy_preservation_components",
    "graph_consistency_loss",
    "integrate_transport",
    "run_stage2_unified_train",
    "validate_unified_config",
]
