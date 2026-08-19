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
from fieldbridge.data.photometry_factorization import sha256_json, write_json_atomic
from fieldbridge.data.photometry_factorization import sha256_file
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

UNIFIED_STAGE2_CONTRACT = "stage2-unified-retrospective-full-model-v3"
UNIFIED_RESUME_CONTRACT = "stage2-unified-exact-resume-v3"
UNIFIED_HISTORY_CONTRACT = "stage2-unified-term-history-v3"
UNIFIED_VALIDATION_PLAN_CONTRACT = "stage2-unified-validation-plan-v1"
UNIFIED_SELECTION_CONTRACT = "stage2-unified-unpaired-validation-selection-v2"
UNIFIED_SELECTION_RULE_CONTRACT = "stage2-unified-critic-independent-selection-rule-v1"
UNIFIED_SELECTION_RECEIPT_CONTRACT = "stage2-unified-selection-receipt-v2"

VALIDATION_PLAN_BRIDGE_EPS = 1.0e-3
UNIFIED_SELECTION_RULE: dict[str, Any] = {
    "contract_version": UNIFIED_SELECTION_RULE_CONTRACT,
    "objective": "minimize",
    "expression": "val_sb+0.1*val_identity+0.02*val_anatomy+0.01*val_graph",
    "coefficients": {
        "sb": 1.0,
        "identity": 0.1,
        "anatomy": 0.02,
        "graph": 0.01,
    },
    "paired_endpoint_assumption": False,
    "training_critic_inputs": [],
    "critic_and_domain_accuracy_diagnostic_only": True,
    "applies_to_all_variants_including_sb_only": True,
    "provenance": (
        "fixed engineering rule over frozen-plan complete-R/validation diagnostics; "
        "independent of training loss weights and the jointly trained critic"
    ),
}
UNIFIED_SELECTION_RULE_SHA256 = sha256_json(UNIFIED_SELECTION_RULE)

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
    critic_spectral_normalization: bool = False
    critic_lazy_r1: bool = False
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
    pilot_steps: int = 200
    pilot_smoothing_window: int = 20
    pilot_max_aux_to_flow_ratio: float = 1.0
    pilot_min_term_gradient_norm: float = 1.0e-12
    pilot_max_smoothed_loss_growth: float = 10.0
    pilot_score_saturation_threshold: float = 20.0
    pilot_max_saturation_fraction: float = 0.95
    projected_steps: int = 100_000
    gpu_hourly_cost_usd: float | None = None
    validation_every_steps: int = 1000
    validation_complete_inventory: bool = True
    validation_plan_seed: int = 20_260_818
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
        if "sanity" in training:
            raise ValueError("The v1 sanity block is obsolete; use the v2 pilot contract.")
        pilot = training.get("pilot", {})
        pilot = pilot if isinstance(pilot, Mapping) else {}
        validation = training.get("validation", {})
        validation = validation if isinstance(validation, Mapping) else {}
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
            critic_spectral_normalization=bool(
                critic.get("spectral_normalization", defaults.critic_spectral_normalization)
            ),
            critic_lazy_r1=bool(critic.get("lazy_r1", defaults.critic_lazy_r1)),
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
            pilot_steps=int(pilot.get("steps", defaults.pilot_steps)),
            pilot_smoothing_window=int(
                pilot.get("smoothing_window", defaults.pilot_smoothing_window)
            ),
            pilot_max_aux_to_flow_ratio=float(
                pilot.get("max_aux_to_flow_ratio", defaults.pilot_max_aux_to_flow_ratio)
            ),
            pilot_min_term_gradient_norm=float(
                pilot.get("min_term_gradient_norm", defaults.pilot_min_term_gradient_norm)
            ),
            pilot_max_smoothed_loss_growth=float(
                pilot.get("max_smoothed_loss_growth", defaults.pilot_max_smoothed_loss_growth)
            ),
            pilot_score_saturation_threshold=float(
                pilot.get("score_saturation_threshold", defaults.pilot_score_saturation_threshold)
            ),
            pilot_max_saturation_fraction=float(
                pilot.get("max_saturation_fraction", defaults.pilot_max_saturation_fraction)
            ),
            projected_steps=int(pilot.get("projected_steps", defaults.projected_steps)),
            gpu_hourly_cost_usd=(
                float(pilot["gpu_hourly_cost_usd"])
                if pilot.get("gpu_hourly_cost_usd") is not None
                else None
            ),
            validation_every_steps=int(
                validation.get("every_steps", defaults.validation_every_steps)
            ),
            validation_complete_inventory=bool(
                validation.get("complete_inventory", defaults.validation_complete_inventory)
            ),
            validation_plan_seed=int(
                validation.get("seed", defaults.validation_plan_seed)
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
            "critic_spectral_normalization": self.critic_spectral_normalization,
            "critic_lazy_r1": self.critic_lazy_r1,
            "anatomy_pool_scales": list(self.anatomy_pool_scales),
            "anatomy_support_erosion": self.anatomy_support_erosion,
            "loss_weights": dict(self.loss_weights),
            "device": self.device,
            "precision": self.precision,
            "grad_clip_norm": self.grad_clip_norm,
            "scheduler_t_max": self.scheduler_t_max,
            "pilot_steps": self.pilot_steps,
            "pilot_smoothing_window": self.pilot_smoothing_window,
            "pilot_max_aux_to_flow_ratio": self.pilot_max_aux_to_flow_ratio,
            "pilot_min_term_gradient_norm": self.pilot_min_term_gradient_norm,
            "pilot_max_smoothed_loss_growth": self.pilot_max_smoothed_loss_growth,
            "pilot_score_saturation_threshold": self.pilot_score_saturation_threshold,
            "pilot_max_saturation_fraction": self.pilot_max_saturation_fraction,
            "projected_steps": self.projected_steps,
            "gpu_hourly_cost_usd": self.gpu_hourly_cost_usd,
            "validation_every_steps": self.validation_every_steps,
            "validation_complete_inventory": self.validation_complete_inventory,
            "validation_plan_seed": self.validation_plan_seed,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class UnifiedStage2Result:
    completed_steps: int
    checkpoint: str | None
    history_jsonl: str
    pilot_report: dict[str, Any]
    selection: dict[str, Any]
    run_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": UNIFIED_STAGE2_CONTRACT,
            "completed_steps": self.completed_steps,
            "checkpoint": self.checkpoint,
            "history_jsonl": self.history_jsonl,
            "pilot_report": self.pilot_report,
            "selection": self.selection,
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


def build_unified_validation_plan(
    index: PhotometryFactoredLatentBankIndex,
    *,
    validation_seed: int,
) -> dict[str, Any]:
    """Freeze complete unpaired R/validation draws without opening latent arrays.

    Target assignment, bridge time, and stochastic-noise seed are functions only of
    the reviewed validation seed and source case identity.  No training-step, model,
    variant, checkpoint, or training RNG state enters this contract.
    """

    if index.split != "validation":
        raise ValueError("A unified validation plan requires the R/validation bank role.")
    if validation_seed < 0:
        raise ValueError("The frozen validation-plan seed must be non-negative.")
    records_by_case: dict[str, FactoredLatentRecord] = {}
    for record in index.records:
        if record.case_id in records_by_case:
            raise ValueError(f"Validation case identity is not unique: {record.case_id}.")
        records_by_case[record.case_id] = record
    pools = _DomainPools.from_index(index)
    pools.require_all_domains()
    entries: list[dict[str, Any]] = []
    for source_record in sorted(index.records, key=lambda item: item.case_id):
        target_field = _next_field(source_record.domain.field_strength_t)
        candidates = sorted(
            (
                index.records[position]
                for position in pools.table[
                    Contrast.parse(source_record.domain.contrast)
                ][target_field]
                if index.records[position].subject_group_id
                != source_record.subject_group_id
            ),
            key=lambda item: item.case_id,
        )
        if not candidates:
            raise ValueError(
                "Complete R/validation diagnostics have no subject-excluded target."
            )
        target_position = _validation_u64(
            validation_seed, source_record.case_id, "independent_target"
        ) % len(candidates)
        target_record = candidates[target_position]
        raw_time = _validation_u64(
            validation_seed, source_record.case_id, "bridge_time"
        )
        unit_time = (raw_time + 0.5) / float(1 << 64)
        bridge_time = VALIDATION_PLAN_BRIDGE_EPS + (
            1.0 - 2.0 * VALIDATION_PLAN_BRIDGE_EPS
        ) * unit_time
        noise_seed = _validation_u64(
            validation_seed, source_record.case_id, "stochastic_noise"
        ) % ((1 << 63) - 1)
        entries.append(
            {
                "source_case_identity": source_record.case_id,
                "source_resume_key": source_record.resume_key,
                "source_subject_group_identity": source_record.subject_group_id,
                "source_domain": source_record.domain.to_dict(),
                "target_case_identity": target_record.case_id,
                "target_resume_key": target_record.resume_key,
                "target_subject_group_identity": target_record.subject_group_id,
                "target_domain": target_record.domain.to_dict(),
                "bridge_t": bridge_time,
                "noise_seed": noise_seed,
            }
        )
    inventory = [
        {
            "case_identity": record.case_id,
            "resume_key": record.resume_key,
            "subject_group_identity": record.subject_group_id,
            "domain": record.domain.to_dict(),
        }
        for record in sorted(index.records, key=lambda item: item.case_id)
    ]
    body: dict[str, Any] = {
        "contract_version": UNIFIED_VALIDATION_PLAN_CONTRACT,
        "validation_seed": validation_seed,
        "scope": "complete_R_validation_inventory",
        "bank_artifact_sha256": index.artifact_sha256,
        "inventory_record_count": len(inventory),
        "inventory_sha256": sha256_json(inventory),
        "derivation": {
            "target_assignment": "sha256(seed,source_case_identity,independent_target)_mod_sorted_subject_excluded_pool",
            "bridge_t": (
                "open_interval_affine_sha256_u64(seed,source_case_identity,bridge_time)"
            ),
            "bridge_t_eps": VALIDATION_PLAN_BRIDGE_EPS,
            "stochastic_noise": (
                "torch_cpu_float32_randn_seeded_by_"
                "sha256_u63(seed,source_case_identity,stochastic_noise)"
            ),
            "training_step_dependency": False,
            "model_or_variant_dependency": False,
            "array_payloads_opened": 0,
        },
        "entries": entries,
    }
    body["validation_plan_sha256"] = sha256_json(body)
    return body


def unified_validation_selection_score(means: Mapping[str, float]) -> float:
    """Apply the frozen critic-independent rule to complete validation means."""

    score = sum(
        float(coefficient) * float(means[name])
        for name, coefficient in UNIFIED_SELECTION_RULE["coefficients"].items()
    )
    if not np.isfinite(score):
        raise FloatingPointError("Unpaired R/validation selection score is non-finite.")
    return float(score)


def _validation_u64(seed: int, case_identity: str, purpose: str) -> int:
    return int(sha256_json([int(seed), str(case_identity), str(purpose)])[:16], 16)


def _seal_or_verify_validation_plan(path: Path, plan: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(plan):
            raise ValueError(
                "Existing validation plan differs from the frozen seed/inventory contract."
            )
        return
    write_json_atomic(path, plan, refuse_existing=True)


def run_stage2_unified_train(
    config: UnifiedStage2Config | Mapping[str, Any],
    *,
    translator: BaseTranslator,
    decoder: nn.Module,
    train_index: PhotometryFactoredLatentBankIndex,
    validation_index: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    critic: DomainProjectionDiscriminator | None = None,
) -> UnifiedStage2Result:
    cfg = config if isinstance(config, UnifiedStage2Config) else UnifiedStage2Config.from_mapping(config)
    validate_unified_config(cfg)
    if train_index.split != "train":
        raise ValueError("Unified model fitting is restricted to R/train.")
    if validation_index.split != "validation":
        raise ValueError("Unified model selection requires the complete R/validation bank.")
    train_subjects = {item.subject_group_id for item in train_index.records}
    validation_subjects = {item.subject_group_id for item in validation_index.records}
    overlap = sorted(train_subjects & validation_subjects)
    if overlap:
        raise ValueError(f"R/train and R/validation subject groups overlap: {overlap[:3]}.")
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
    validation_plan = build_unified_validation_plan(
        validation_index, validation_seed=cfg.validation_plan_seed
    )
    validation_plan_sha256 = str(validation_plan["validation_plan_sha256"])
    if cfg.checkpoint_dir is not None:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _seal_or_verify_validation_plan(
            cfg.checkpoint_dir / "stage2_unified_validation_plan_v1.json",
            validation_plan,
        )
    history_path = cfg.history_jsonl or (
        (cfg.checkpoint_dir or Path.cwd()) / "stage2_unified_history.jsonl"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "contract_version": UNIFIED_STAGE2_CONTRACT,
        "config": cfg.to_dict(),
        "bank_artifact_sha256": train_index.artifact_sha256,
        "validation_bank_artifact_sha256": validation_index.artifact_sha256,
        "validation_inventory_sha256": sha256_json(
            [item.resume_key for item in validation_index.records]
        ),
        "validation_plan_sha256": validation_plan_sha256,
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "bank_vae_provenance": dict(getattr(train_index, "manifest", {}).get("vae", {})),
        "frozen_decoder_state_sha256": _module_state_sha256(decoder),
        "git_commit": resolve_git_commit(),
    }
    run_fingerprint = sha256_json(run_identity)
    cursor = 0
    history_generation = 0
    selection: dict[str, Any] = {
        "contract_version": UNIFIED_SELECTION_CONTRACT,
        "validation_plan_sha256": validation_plan_sha256,
        "selection_rule": dict(UNIFIED_SELECTION_RULE),
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "paired_targets_used": False,
        "complete_r_validation_inventory": True,
        "latest_step": None,
        "latest_checkpoint": None,
        "best_step": None,
        "best_checkpoint": None,
        "best_score": None,
    }
    pilot_report: dict[str, Any] = {"status": "not_requested", "steps": 0}
    if cfg.resume_from is not None:
        cursor, restored_selection, pilot_report = _restore_exact(
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
            expected_validation_plan_sha256=validation_plan_sha256,
            history_path=history_path,
        )
        selection.update(restored_selection)
        history_generation = _prepare_history_resume(
            history_path, cursor=cursor, run_fingerprint=run_fingerprint
        )
        if cursor > cfg.steps:
            raise ValueError("Exact-resume cursor is beyond the configured terminal step.")
    elif history_path.exists() and history_path.stat().st_size:
        raise FileExistsError("Non-empty history exists but no exact-resume checkpoint was supplied.")

    pilot_rows: list[dict[str, Any]] = []
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
                stats,
                qualify_term_gradients=step < cfg.pilot_steps,
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
        if step < cfg.pilot_steps:
            pilot_rows.append(dict(row))
        if cfg.pilot_steps > 0 and step + 1 == cfg.pilot_steps:
            pilot_report = _pilot_report(pilot_rows, cfg)
            _append_jsonl(
                history_path,
                {
                    "contract_version": UNIFIED_HISTORY_CONTRACT,
                    "event": "full_objective_pilot",
                    "step": step + 1,
                    "pilot": pilot_report,
                    "run_fingerprint": run_fingerprint,
                    "history_generation": history_generation,
                },
            )
            if pilot_report["status"] != "pass":
                raise RuntimeError(
                    "Unified Stage-2 full-objective pilot failed: "
                    + ", ".join(pilot_report["failures"])
                )
        validation_due = (
            (step + 1) % cfg.validation_every_steps == 0
            or step + 1 == cfg.steps
            or (cfg.pilot_steps > 0 and step + 1 == cfg.pilot_steps)
        )
        if validation_due:
            validation = _evaluate_unpaired_validation(
                cfg,
                translator,
                critic,
                decoder,
                validation_index,
                validation_plan,
                stats,
                device,
                step=step + 1,
            )
            _append_jsonl(
                history_path,
                {
                    "contract_version": UNIFIED_HISTORY_CONTRACT,
                    "event": "unpaired_validation",
                    "step": step + 1,
                    "validation": validation,
                    "run_fingerprint": run_fingerprint,
                    "history_generation": history_generation,
                },
            )
            selection["latest_step"] = step + 1
            current_score = float(validation["selection_score"])
            if selection["best_score"] is None or current_score < float(selection["best_score"]):
                selection["best_score"] = current_score
                selection["best_step"] = step + 1
        if cfg.checkpoint_dir is not None and cfg.checkpoint_every_steps > 0:
            if (step + 1) % cfg.checkpoint_every_steps == 0 or validation_due:
                checkpoint_identity = str(
                    (
                        cfg.checkpoint_dir
                        / f"stage2_unified_{cfg.variant}_step{step + 1:09d}.pt"
                    ).resolve()
                )
                selection["latest_checkpoint"] = checkpoint_identity
                if selection["best_step"] == step + 1:
                    selection["best_checkpoint"] = checkpoint_identity
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
                    validation_plan_sha256,
                    history_path,
                    selection,
                    pilot_report,
                )
                if validation_due:
                    receipt_path = (
                        cfg.checkpoint_dir
                        / f"stage2_unified_{cfg.variant}_selection_step{step + 1:09d}.json"
                    )
                    _write_selection_receipt(
                        receipt_path,
                        cfg=cfg,
                        step=step + 1,
                        selection=selection,
                        validation_plan_sha256=validation_plan_sha256,
                    )
                    selection["latest_selection_receipt"] = str(receipt_path)
                    selection["latest_selection_receipt_sha256"] = sha256_file(receipt_path)
    if cfg.checkpoint_dir is not None and cfg.steps > cursor and (
        last_checkpoint is None
        or not last_checkpoint.name.endswith(f"step{cfg.steps:09d}.pt")
    ):
        checkpoint_identity = str(
            (
                cfg.checkpoint_dir
                / f"stage2_unified_{cfg.variant}_step{cfg.steps:09d}.pt"
            ).resolve()
        )
        selection["latest_checkpoint"] = checkpoint_identity
        if selection["best_step"] == cfg.steps:
            selection["best_checkpoint"] = checkpoint_identity
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
            validation_plan_sha256,
            history_path,
            selection,
            pilot_report,
        )
    if cfg.pilot_steps > 0 and cfg.steps < cfg.pilot_steps:
        pilot_report = {
            "status": "incomplete",
            "steps": len(pilot_rows),
            "required_steps": cfg.pilot_steps,
            "failures": ["configured run ended before the full-objective pilot completed"],
        }
    return UnifiedStage2Result(
        completed_steps=cfg.steps,
        checkpoint=str(last_checkpoint) if last_checkpoint else None,
        history_jsonl=str(history_path),
        pilot_report=pilot_report,
        selection=selection,
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
    stats: FactoredLatentStats,
    *,
    qualify_term_gradients: bool,
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
        # Both branches use the identical frozen intersection.  In particular, the
        # appended support channel cannot reveal whether a sample is real or generated.
        real_view = _critic_view(
            batch.target, support, batch.target_domains, decoder, stats, cfg.critic_space
        )
        fake_view = _critic_view(
            fake_detached, support, batch.target_domains, decoder, stats, cfg.critic_space
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
            source_image = _decode(
                decoder, stats.denormalize(batch.source), batch.source_domains
            )
            generated_image = _decode(
                decoder, stats.denormalize(generated), batch.target_domains
            )
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
                generated, support, batch.target_domains, decoder, stats, cfg.critic_space
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
        term_gradient_norms = (
            generator_term_gradient_norms(raw_terms, translator, cfg.loss_weights)
            if qualify_term_gradients
            else {name: None for name in raw_terms}
        )
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
        "weighted/generator_total": float(generator_total.detach().float().cpu()),
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
        "critic/real_score_mean": (
            float(real_score.detach().float().mean().cpu()) if critic_active else 0.0
        ),
        "critic/real_score_std": (
            float(real_score.detach().float().std(unbiased=False).cpu()) if critic_active else 0.0
        ),
        "critic/fake_score_mean": (
            float(fake_score.detach().float().mean().cpu()) if critic_active else 0.0
        ),
        "critic/fake_score_std": (
            float(fake_score.detach().float().std(unbiased=False).cpu()) if critic_active else 0.0
        ),
        **{
            f"critic/real_score_{name}": value
            for name, value in _score_quantiles(real_score if critic_active else None).items()
        },
        **{
            f"critic/fake_score_{name}": value
            for name, value in _score_quantiles(fake_score if critic_active else None).items()
        },
        "critic/score_separation": (
            float((real_score.mean() - fake_score.mean()).detach().float().cpu())
            if critic_active
            else 0.0
        ),
        "critic/score_saturation_fraction": (
            float(
                torch.cat((real_score, fake_score))
                .detach()
                .abs()
                .ge(cfg.pilot_score_saturation_threshold)
                .float()
                .mean()
                .cpu()
            )
            if critic_active
            else 0.0
        ),
        "critic/real_domain_accuracy": (
            float(
                real_domain_logits.detach()
                .argmax(1)
                .eq(domain_labels(batch.target_domains, len(batch.target_domains), batch.source.device))
                .float()
                .mean()
                .cpu()
            )
            if critic_active
            else 0.0
        ),
        "critic/generated_domain_accuracy": (
            float(
                fake_domain_logits.detach()
                .argmax(1)
                .eq(domain_labels(batch.target_domains, len(batch.target_domains), batch.source.device))
                .float()
                .mean()
                .cpu()
            )
            if critic_active
            else 0.0
        ),
        "gradient/generator_norm": generator_grad,
        "gradient/critic_norm": critic_grad,
        **{
            f"gradient/term_{name}": value
            for name, value in term_gradient_norms.items()
        },
        "critic/support_identical_real_fake": True,
        "critic/support_cell_count": int(support.sum().detach().cpu()),
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


def _score_quantiles(score: torch.Tensor | None) -> dict[str, float]:
    if score is None:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    values = score.detach().float().reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95], device=values.device))
    return {
        "p05": float(quantiles[0].cpu()),
        "p50": float(quantiles[1].cpu()),
        "p95": float(quantiles[2].cpu()),
    }


def _critic_view(
    latent: torch.Tensor,
    support: torch.Tensor,
    domains: Sequence[Domain],
    decoder: nn.Module,
    stats: FactoredLatentStats,
    space: CriticSpace,
) -> torch.Tensor:
    if space == "latent":
        return supported_critic_input(latent, support)
    image = _decode(decoder, stats.denormalize(latent), domains)
    image_support = F.interpolate(support.float(), size=image.shape[2:], mode="nearest") > 0.5
    return supported_critic_input(image, image_support)


def generator_term_gradient_norms(
    raw_terms: Mapping[str, torch.Tensor],
    translator: nn.Module,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Measure each enabled objective's translator gradient on the intact graph.

    The raw term, rather than the sum, is differentiated so a configured weight cannot
    hide a disconnected objective.  Disabled objectives are sealed as exactly zero.
    """

    parameters = tuple(value for value in translator.parameters() if value.requires_grad)
    if not parameters:
        raise ValueError("Translator has no trainable parameters for gradient qualification.")
    result: dict[str, float] = {}
    for name in DEFAULT_UNIFIED_WEIGHTS:
        weight = float(weights[name])
        if weight == 0.0:
            result[name] = 0.0
            continue
        term = raw_terms[name]
        if not bool(torch.isfinite(term)):
            raise FloatingPointError(f"Generator term {name!r} is non-finite.")
        gradients = torch.autograd.grad(
            term,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = term.new_zeros(())
        for gradient in gradients:
            if gradient is not None:
                squared = squared + gradient.float().square().sum()
        norm = float(squared.sqrt().detach().cpu())
        if not np.isfinite(norm):
            raise FloatingPointError(f"Generator term {name!r} has a non-finite gradient.")
        result[name] = norm
    return result


def _bridge_sample(
    source: torch.Tensor,
    target: torch.Tensor,
    time_values: torch.Tensor,
    cfg: UnifiedStage2Config,
    sampler: torch.Generator | None = None,
    *,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (source.shape[0],) + (1,) * (source.ndim - 1)
    t = time_values.reshape(shape)
    if cfg.bridge == "ot_cfm":
        return (1.0 - t) * source + t * target, target - source
    if noise is None:
        if sampler is None:
            raise ValueError("Schrodinger bridge sampling requires a sampler or frozen noise.")
        noise = _randn(tuple(source.shape), sampler, source.device, source.dtype)
    if noise.shape != source.shape or noise.device != source.device:
        raise ValueError("Frozen validation noise is shape/device inconsistent.")
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


@torch.inference_mode()
def _evaluate_unpaired_validation(
    cfg: UnifiedStage2Config,
    translator: BaseTranslator,
    critic: DomainProjectionDiscriminator,
    decoder: nn.Module,
    index: PhotometryFactoredLatentBankIndex,
    validation_plan: Mapping[str, Any],
    stats: FactoredLatentStats,
    device: torch.device,
    *,
    step: int,
) -> dict[str, Any]:
    """Evaluate every R/validation source against deterministic independent targets.

    Targets share contrast but are from a different subject group.  They provide
    distribution/flow diagnostics only and are never represented as paired endpoints.
    """

    stored_plan_sha256 = validation_plan.get("validation_plan_sha256")
    plan_body = dict(validation_plan)
    plan_body.pop("validation_plan_sha256", None)
    if (
        validation_plan.get("contract_version") != UNIFIED_VALIDATION_PLAN_CONTRACT
        or stored_plan_sha256 != sha256_json(plan_body)
    ):
        raise ValueError("Unified validation-plan identity mismatch.")
    current_plan = build_unified_validation_plan(
        index, validation_seed=cfg.validation_plan_seed
    )
    if current_plan != dict(validation_plan):
        raise ValueError("Unified validation inventory or frozen draw plan changed.")
    by_case = {record.case_id: position for position, record in enumerate(index.records)}
    translator.eval()
    critic.eval()
    rows: list[dict[str, float]] = []
    for plan_entry in validation_plan["entries"]:
        source_index = by_case[str(plan_entry["source_case_identity"])]
        target_index = by_case[str(plan_entry["target_case_identity"])]
        source, source_support, source_domains, _ = index.load_batch([source_index])
        target, target_support, target_domains, _ = index.load_batch([target_index])
        source = stats.normalize(source.to(device), source_support.to(device))
        target = stats.normalize(target.to(device), target_support.to(device))
        source_support = source_support.to(device)
        target_support = target_support.to(device)
        support = source_support & target_support
        if not bool(support.any()):
            raise ValueError("R/validation source/target support intersection is empty.")
        time_values = torch.tensor(
            [float(plan_entry["bridge_t"])], device=device, dtype=source.dtype
        )
        noise_generator = torch.Generator().manual_seed(int(plan_entry["noise_seed"]))
        fixed_noise = _randn(
            tuple(source.shape), noise_generator, device, source.dtype
        )
        z_t, flow_target = _bridge_sample(
            source,
            target,
            time_values,
            cfg,
            noise=fixed_noise,
        )
        predicted_velocity = translator(z_t, source_domains, target_domains, time_values)
        sb = _masked_mse(predicted_velocity, flow_target, support)
        identity_generated = integrate_transport(
            translator,
            source,
            source_domains,
            source_domains,
            steps=cfg.integration_steps,
            solver=cfg.integration_solver,
        )
        identity = masked_l1_loss(identity_generated, source, source_support)
        intermediate = [
            Domain(
                next(
                    field
                    for field in FIELD_STRENGTHS_T
                    if field
                    not in {
                        source_domains[0].field_strength_t,
                        target_domains[0].field_strength_t,
                    }
                ),
                source_domains[0].contrast,
            )
        ]
        graph, generated, _ = graph_consistency_loss(
            translator,
            source,
            source_domains,
            intermediate,
            target_domains,
            support,
            steps=cfg.integration_steps,
            solver=cfg.integration_solver,
        )
        anatomy = anatomy_preservation_components(
            _decode(decoder, stats.denormalize(source), source_domains),
            _decode(decoder, stats.denormalize(generated), target_domains),
            source_support,
            pool_scales=cfg.anatomy_pool_scales,
            support_erosion=cfg.anatomy_support_erosion,
        )["total"]
        real_view = _critic_view(target, support, target_domains, decoder, stats, cfg.critic_space)
        fake_view = _critic_view(generated, support, target_domains, decoder, stats, cfg.critic_space)
        real_score, real_logits = critic(real_view, target_domains)
        fake_score, fake_logits = critic(fake_view, target_domains)
        labels = domain_labels(target_domains, 1, device)
        rows.append(
            {
                "sb": float(sb.float().cpu()),
                "identity": float(identity.float().cpu()),
                "graph": float(graph.float().cpu()),
                "anatomy": float(anatomy.float().cpu()),
                "real_score": float(real_score.float().mean().cpu()),
                "generated_score": float(fake_score.float().mean().cpu()),
                "real_domain_correct": float(real_logits.argmax(1).eq(labels).float().cpu()),
                "generated_domain_correct": float(fake_logits.argmax(1).eq(labels).float().cpu()),
            }
        )
    means = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    selection_score = unified_validation_selection_score(means)
    return {
        "contract_version": UNIFIED_SELECTION_CONTRACT,
        "step": step,
        "validation_plan_sha256": stored_plan_sha256,
        "inventory_record_count": len(index.records),
        "inventory_sha256": validation_plan["inventory_sha256"],
        "complete_inventory_used": True,
        "paired_endpoint_assumption": False,
        "subject_group_exclusion": True,
        "means": means,
        "selection_score": selection_score,
        "selection_rule": dict(UNIFIED_SELECTION_RULE),
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "critic_and_domain_metrics_role": "diagnostic_only_excluded_from_selection",
    }


def _next_field(value: float) -> float:
    position = FIELD_STRENGTHS_T.index(value)
    return FIELD_STRENGTHS_T[(position + 1) % len(FIELD_STRENGTHS_T)]


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
    validation_plan_sha256: str,
    history_path: Path,
    selection: Mapping[str, Any],
    pilot_report: Mapping[str, Any],
) -> Path:
    assert cfg.checkpoint_dir is not None
    path = cfg.checkpoint_dir / f"stage2_unified_{cfg.variant}_step{cursor:09d}.pt"
    history_bytes = history_path.read_bytes()
    state = {
        "contract_version": UNIFIED_RESUME_CONTRACT,
        "run_fingerprint": run_fingerprint,
        "validation_plan_sha256": validation_plan_sha256,
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
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
        "validation_selection": dict(selection),
        "pilot_report": dict(pilot_report),
    }
    return save_checkpoint(
        path,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=False,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _write_selection_receipt(
    path: Path,
    *,
    cfg: UnifiedStage2Config,
    step: int,
    selection: Mapping[str, Any],
    validation_plan_sha256: str,
) -> dict[str, Any]:
    latest_path = Path(str(selection["latest_checkpoint"]))
    best_path = Path(str(selection["best_checkpoint"]))
    if not latest_path.is_file() or not best_path.is_file():
        raise ValueError("Selection receipt cannot resolve latest/best checkpoints.")
    checkpoints: dict[str, Any] = {
        "latest": {
            "path": str(latest_path),
            "file_sha256": sha256_file(latest_path),
        },
        "best": {
            "path": str(best_path),
            "file_sha256": sha256_file(best_path),
        },
    }
    run_complete = step == cfg.steps
    if run_complete:
        checkpoints["final"] = dict(checkpoints["latest"])
    body: dict[str, Any] = {
        **dict(selection),
        "receipt_contract_version": UNIFIED_SELECTION_RECEIPT_CONTRACT,
        "variant": cfg.variant,
        "receipt_step": step,
        "terminal_step": cfg.steps,
        "run_complete": run_complete,
        "validation_plan_sha256": validation_plan_sha256,
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "checkpoint_hashes": checkpoints,
    }
    body["selection_sha256"] = sha256_json(body)
    write_json_atomic(path, body, refuse_existing=True)
    return body


def load_stage2_selection_receipt(
    path: str | Path,
    *,
    expected_variant: str | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Verify a receipt and every referenced best/final checkpoint identity."""

    receipt_path = Path(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("Unified selection receipt root must be an object.")
    normalized = dict(receipt)
    stored = normalized.pop("selection_sha256", None)
    if stored != sha256_json(normalized):
        raise ValueError("Unified selection receipt SHA-256 mismatch.")
    normalized["selection_sha256"] = stored
    if (
        normalized.get("receipt_contract_version")
        != UNIFIED_SELECTION_RECEIPT_CONTRACT
        or normalized.get("contract_version") != UNIFIED_SELECTION_CONTRACT
    ):
        raise ValueError("Unified selection receipt contract mismatch.")
    if expected_variant is not None and normalized.get("variant") != expected_variant:
        raise ValueError("Unified selection receipt variant mismatch.")
    if require_complete and normalized.get("run_complete") is not True:
        raise ValueError("Unified selection receipt does not represent a completed run.")
    if normalized.get("selection_rule_sha256") != UNIFIED_SELECTION_RULE_SHA256:
        raise ValueError("Unified selection receipt rule provenance mismatch.")
    plan_sha = normalized.get("validation_plan_sha256")
    if not isinstance(plan_sha, str) or len(plan_sha) != 64:
        raise ValueError("Unified selection receipt validation-plan identity is invalid.")
    checkpoint_hashes = normalized.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, Mapping):
        raise ValueError("Unified selection receipt checkpoint hashes are missing.")
    required = {"latest", "best"} | ({"final"} if require_complete else set())
    if not required <= set(checkpoint_hashes):
        raise ValueError("Unified selection receipt omits a required checkpoint identity.")
    for role, raw in checkpoint_hashes.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Unified selection checkpoint {role!r} is malformed.")
        checkpoint_path = Path(str(raw.get("path", "")))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != raw.get(
            "file_sha256"
        ):
            raise ValueError(f"Unified selection checkpoint {role!r} hash mismatch.")
        state = load_checkpoint(checkpoint_path)
        if state.get("contract_version") != UNIFIED_RESUME_CONTRACT:
            raise ValueError(f"Unified selection checkpoint {role!r} contract mismatch.")
        if state.get("validation_plan_sha256") != plan_sha:
            raise ValueError(
                f"Unified selection checkpoint {role!r} uses another validation plan."
            )
        if state.get("selection_rule_sha256") != UNIFIED_SELECTION_RULE_SHA256:
            raise ValueError(
                f"Unified selection checkpoint {role!r} uses another selection rule."
            )
    if str(normalized.get("best_checkpoint")) != str(
        checkpoint_hashes["best"]["path"]
    ):
        raise ValueError("Unified selection receipt best-checkpoint pointer mismatch.")
    if require_complete and checkpoint_hashes["final"] != checkpoint_hashes["latest"]:
        raise ValueError("Unified final/latest checkpoint identities disagree.")
    return normalized


def find_latest_stage2_selection_receipt(
    checkpoint_dir: str | Path,
    *,
    variant: str,
    require_complete: bool = True,
) -> tuple[Path, dict[str, Any]]:
    root = Path(checkpoint_dir)
    candidates = sorted(root.glob(f"stage2_unified_{variant}_selection_step*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No immutable selection receipt exists for variant {variant!r}."
        )
    latest = candidates[-1]
    receipt = load_stage2_selection_receipt(
        latest, expected_variant=variant, require_complete=require_complete
    )
    if int(receipt["receipt_step"]) != max(
        int(candidate.stem.rsplit("step", 1)[1]) for candidate in candidates
    ):
        raise ValueError("Latest unified selection receipt filename/cursor mismatch.")
    return latest, receipt


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
    expected_validation_plan_sha256: str,
    history_path: Path,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    state = load_checkpoint(path)
    if state.get("contract_version") != UNIFIED_RESUME_CONTRACT:
        raise ValueError("Unified checkpoint exact-resume contract mismatch.")
    if state.get("run_fingerprint") != expected_run_fingerprint:
        raise ValueError("Unified checkpoint config/bank/code identity mismatch.")
    if state.get("validation_plan_sha256") != expected_validation_plan_sha256:
        raise ValueError("Unified checkpoint validation-plan identity mismatch.")
    if state.get("selection_rule_sha256") != UNIFIED_SELECTION_RULE_SHA256:
        raise ValueError("Unified checkpoint selection-rule identity mismatch.")
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
    selection = state.get("validation_selection")
    if not isinstance(selection, Mapping) or selection.get(
        "contract_version"
    ) != UNIFIED_SELECTION_CONTRACT:
        raise ValueError("Unified checkpoint validation-selection contract mismatch.")
    if selection.get("validation_plan_sha256") != expected_validation_plan_sha256:
        raise ValueError("Unified checkpoint selection uses another validation plan.")
    if selection.get("selection_rule_sha256") != UNIFIED_SELECTION_RULE_SHA256:
        raise ValueError("Unified checkpoint selection uses another selection rule.")
    pilot_report = state.get("pilot_report")
    if not isinstance(pilot_report, Mapping):
        raise ValueError("Unified checkpoint pilot-report contract mismatch.")
    return cursor, dict(selection), dict(pilot_report)


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


def _pilot_report(
    rows: Sequence[Mapping[str, Any]], cfg: UnifiedStage2Config
) -> dict[str, Any]:
    if not rows:
        return {"status": "not_requested", "steps": 0, "failures": []}
    numeric_keys = [
        "weighted/generator_total",
        "weighted/aux_to_flow_ratio",
        "gradient/generator_norm",
        "gradient/critic_norm",
        "critic/score_saturation_fraction",
        "critic/real_score_mean",
        "critic/fake_score_mean",
        "critic/real_domain_accuracy",
        "critic/generated_domain_accuracy",
        "step_seconds",
    ]
    for branch in ("real", "fake"):
        numeric_keys.extend(f"critic/{branch}_score_{name}" for name in ("p05", "p50", "p95"))
    for term in DEFAULT_UNIFIED_WEIGHTS:
        numeric_keys.extend((f"raw/{term}", f"weighted/{term}", f"gradient/term_{term}"))
    failures: list[str] = []
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        if not all(np.isfinite(value) for value in values):
            failures.append(f"nonfinite:{key}")
    gradient_summary: dict[str, Any] = {}
    for term, weight in cfg.loss_weights.items():
        values = [float(row[f"gradient/term_{term}"]) for row in rows]
        gradient_summary[term] = {
            "enabled": weight > 0,
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        if weight > 0 and max(values) <= cfg.pilot_min_term_gradient_norm:
            failures.append(f"missing_translator_gradient:{term}")
        if weight == 0 and any(value != 0.0 for value in values):
            failures.append(f"disabled_term_gradient_nonzero:{term}")
        if weight == 0 and any(float(row[f"weighted/{term}"]) != 0.0 for row in rows):
            failures.append(f"disabled_term_contribution_nonzero:{term}")
    ratios = [float(row["weighted/aux_to_flow_ratio"]) for row in rows]
    if max(ratios) > cfg.pilot_max_aux_to_flow_ratio:
        failures.append("auxiliary_objectives_dominate_flow")
    saturation = [float(row["critic/score_saturation_fraction"]) for row in rows]
    if max(saturation) > cfg.pilot_max_saturation_fraction:
        failures.append("critic_saturation")
    if cfg.loss_weights["adversarial"] > 0 or cfg.loss_weights["domain"] > 0:
        if max(float(row["gradient/critic_norm"]) for row in rows) <= 0:
            failures.append("missing_critic_gradient")
    window = min(cfg.pilot_smoothing_window, len(rows))
    losses = [float(row["weighted/generator_total"]) for row in rows]
    first_smoothed = float(np.mean(losses[:window]))
    last_smoothed = float(np.mean(losses[-window:]))
    if abs(last_smoothed) > max(abs(first_smoothed), 1.0e-12) * cfg.pilot_max_smoothed_loss_growth:
        failures.append("uncontrolled_smoothed_loss_growth")
    step_seconds = [float(row["step_seconds"]) for row in rows]
    mean_step_seconds = float(np.mean(step_seconds))
    projected_seconds = mean_step_seconds * cfg.projected_steps
    projected_cost = (
        projected_seconds / 3600.0 * cfg.gpu_hourly_cost_usd
        if cfg.gpu_hourly_cost_usd is not None
        else None
    )
    return {
        "status": "pass" if not failures else "fail",
        "failures": sorted(set(failures)),
        "steps": len(rows),
        "full_objective": all(cfg.loss_weights[name] > 0 for name in DEFAULT_UNIFIED_WEIGHTS),
        "loss_behavior": {
            "first": losses[0],
            "last": losses[-1],
            "first_smoothed": first_smoothed,
            "last_smoothed": last_smoothed,
            "smoothing_window": window,
        },
        "raw_term_means": {
            term: float(np.mean([float(row[f"raw/{term}"]) for row in rows]))
            for term in DEFAULT_UNIFIED_WEIGHTS
        },
        "weighted_term_means": {
            term: float(np.mean([float(row[f"weighted/{term}"]) for row in rows]))
            for term in DEFAULT_UNIFIED_WEIGHTS
        },
        "term_gradient_norms": gradient_summary,
        "generator_gradient_norm_mean": float(
            np.mean([float(row["gradient/generator_norm"]) for row in rows])
        ),
        "critic_gradient_norm_mean": float(
            np.mean([float(row["gradient/critic_norm"]) for row in rows])
        ),
        "aux_to_flow_ratio": {"maximum": max(ratios), "mean": float(np.mean(ratios))},
        "critic": {
            "real_score_mean": float(np.mean([float(row["critic/real_score_mean"]) for row in rows])),
            "fake_score_mean": float(np.mean([float(row["critic/fake_score_mean"]) for row in rows])),
            "real_score_distribution": {
                name: float(
                    np.mean([float(row[f"critic/real_score_{name}"]) for row in rows])
                )
                for name in ("p05", "p50", "p95")
            },
            "fake_score_distribution": {
                name: float(
                    np.mean([float(row[f"critic/fake_score_{name}"]) for row in rows])
                )
                for name in ("p05", "p50", "p95")
            },
            "score_separation_mean": float(
                np.mean([float(row["critic/score_separation"]) for row in rows])
            ),
            "saturation_fraction_max": max(saturation),
            "real_domain_accuracy": float(
                np.mean([float(row["critic/real_domain_accuracy"]) for row in rows])
            ),
            "generated_domain_accuracy": float(
                np.mean([float(row["critic/generated_domain_accuracy"]) for row in rows])
            ),
        },
        "runtime": {
            "mean_step_seconds": mean_step_seconds,
            "examples_per_second": cfg.batch_size / max(mean_step_seconds, 1.0e-12),
            "peak_cuda_bytes": max(int(row["peak_cuda_bytes"]) for row in rows),
            "projected_steps": cfg.projected_steps,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600.0,
            "gpu_hourly_cost_usd": cfg.gpu_hourly_cost_usd,
            "projected_cost_usd": projected_cost,
            "cost_status": "measured_rate" if projected_cost is not None else "operator_rate_required",
        },
        "hard_stop_conditions": [
            "nonfinite",
            "missing_gradient",
            "critic_saturation",
            "auxiliary_dominance",
            "uncontrolled_loss_growth",
            "oom",
        ],
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
    if cfg.critic_spectral_normalization or cfg.critic_lazy_r1:
        raise ValueError(
            "Spectral normalization/lazy R1 are evidence-gated ablations and are not "
            "enabled by the primary v3 contract."
        )
    if cfg.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16.")
    if set(cfg.loss_weights) != set(DEFAULT_UNIFIED_WEIGHTS):
        raise ValueError("Unified loss weights must specify exactly all six objectives.")
    if any(not np.isfinite(value) or value < 0 for value in cfg.loss_weights.values()):
        raise ValueError("Unified loss weights must be finite and non-negative.")
    if cfg.loss_weights["sb"] <= 0:
        raise ValueError("The SB objective must remain active.")
    if cfg.pilot_steps < 0 or cfg.pilot_smoothing_window < 1:
        raise ValueError("Pilot steps must be non-negative and its smoothing window positive.")
    if not 0.0 < cfg.pilot_max_aux_to_flow_ratio:
        raise ValueError("Pilot auxiliary/flow threshold must be positive.")
    if cfg.pilot_min_term_gradient_norm < 0 or cfg.pilot_max_smoothed_loss_growth <= 0:
        raise ValueError("Pilot gradient/growth thresholds are invalid.")
    if cfg.pilot_score_saturation_threshold <= 0 or not 0 <= cfg.pilot_max_saturation_fraction <= 1:
        raise ValueError("Pilot critic-saturation thresholds are invalid.")
    if cfg.projected_steps < 1 or (
        cfg.gpu_hourly_cost_usd is not None and cfg.gpu_hourly_cost_usd < 0
    ):
        raise ValueError("Pilot projection steps/cost are invalid.")
    if cfg.validation_every_steps < 1 or not cfg.validation_complete_inventory:
        raise ValueError("Validation must use the complete R/validation inventory.")
    if cfg.validation_plan_seed < 0:
        raise ValueError("The frozen validation-plan seed must be non-negative.")


__all__ = [
    "DEFAULT_UNIFIED_WEIGHTS",
    "UNIFIED_HISTORY_CONTRACT",
    "UNIFIED_RESUME_CONTRACT",
    "UNIFIED_SELECTION_CONTRACT",
    "UNIFIED_SELECTION_RECEIPT_CONTRACT",
    "UNIFIED_SELECTION_RULE",
    "UNIFIED_SELECTION_RULE_CONTRACT",
    "UNIFIED_SELECTION_RULE_SHA256",
    "UNIFIED_STAGE2_CONTRACT",
    "UNIFIED_VALIDATION_PLAN_CONTRACT",
    "UnifiedStage2Config",
    "UnifiedStage2Result",
    "anatomy_preservation_components",
    "build_unified_validation_plan",
    "find_latest_stage2_selection_receipt",
    "graph_consistency_loss",
    "generator_term_gradient_norms",
    "integrate_transport",
    "load_stage2_selection_receipt",
    "run_stage2_unified_train",
    "unified_validation_selection_score",
    "validate_unified_config",
]
