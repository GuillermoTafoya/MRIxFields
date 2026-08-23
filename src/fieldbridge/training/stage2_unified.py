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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentRecord,
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import sha256_json, write_json_atomic
from fieldbridge.data.photometry_factorization import sha256_file
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256
from fieldbridge.models.autoencoders.kl_vae import (
    KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT,
    KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE,
)
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

UNIFIED_STAGE2_CONTRACT = 'stage2-unified-retrospective-full-model-v7'
UNIFIED_STAGE2_CONFIG_CONTRACT = 'stage2-unified-retrospective-full-model-config-v7'
UNIFIED_RESUME_CONTRACT = 'stage2-unified-exact-resume-v7'
UNIFIED_HISTORY_CONTRACT = 'stage2-unified-term-history-v7'
UNIFIED_ANATOMY_MEMORY_CONTRACT = 'stage2-unified-anatomy-memory-qualification-v1'
UNIFIED_GENERATOR_ACCUMULATION_CONTRACT = (
    'stage2-unified-term-wise-recomputation-v6'
)
UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT = (
    'stage2-unified-nonreentrant-rng-preserving-translator-checkpoint-v1'
)
UNIFIED_A100_GATE_CONTRACT = 'stage2-unified-a100-one-step-memory-gate-v1'
UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES = 72 * 1024**3
UNIFIED_PILOT_RUNTIME_PROJECTION_CONTRACT = (
    "stage2-unified-pilot-training-plus-validation-runtime-projection-v1"
)
UNIFIED_VALIDATION_PLAN_CONTRACT = "stage2-unified-validation-plan-v2"
UNIFIED_SELECTION_CONTRACT = "stage2-unified-unpaired-validation-selection-v3"
UNIFIED_SELECTION_RULE_CONTRACT = "stage2-unified-critic-independent-selection-rule-v1"
UNIFIED_SELECTION_RECEIPT_CONTRACT = "stage2-unified-selection-receipt-v3"

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
        "fixed engineering rule over equal-weighted frozen-plan directed-domain macro "
        "means from complete-R/validation diagnostics; "
        "independent of training loss weights and the jointly trained critic"
    ),
}
UNIFIED_SELECTION_RULE_SHA256 = sha256_json(UNIFIED_SELECTION_RULE)

CriticSpace = Literal["latent", "image"]
Precision = Literal["fp32", "bf16"]
Solver = Literal["euler", "heun"]
Bridge = Literal["schrodinger", "ot_cfm"]
DecoderCheckpointMode = Literal['disabled', 'fine_grained_full_volume_v1']

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
    decoder_activation_checkpoint_mode: DecoderCheckpointMode = (
        KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE
    )
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
    pilot_a100_peak_allocated_limit_bytes: int = UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
    projected_steps: int = 100_000
    gpu_hourly_cost_usd: float | None = None
    validation_every_steps: int = 1000
    validation_complete_inventory: bool = True
    validation_plan_seed: int = 20_260_818
    variant: str = "full"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UnifiedStage2Config":
        declared_contract = data.get("contract")
        if declared_contract is not None and declared_contract != UNIFIED_STAGE2_CONFIG_CONTRACT:
            raise ValueError(
                "Unified Stage-2 configuration contract is obsolete or unsupported."
            )
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
        decoder_checkpointing = training.get('decoder_activation_checkpoint', {})
        if not isinstance(decoder_checkpointing, Mapping):
            raise ValueError('decoder_activation_checkpoint must be a mapping.')
        checkpointing_contract = decoder_checkpointing.get('contract')
        if (
            checkpointing_contract is not None
            and checkpointing_contract != KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT
        ):
            raise ValueError('Decoder activation-checkpoint contract is unsupported.')
        accumulation = training.get('generator_gradient_accumulation', {})
        if not isinstance(accumulation, Mapping):
            raise ValueError('generator_gradient_accumulation must be a mapping.')
        if declared_contract == UNIFIED_STAGE2_CONFIG_CONTRACT:
            required_checkpointing = {
                'contract': KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT,
                'mode': KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE,
                'full_volume': 'required',
                'group_norm_scope': 'complete_spatial_volume',
                'upsample_regions': ['up1', 'up2'],
                'residual_branch_regions': 'each_norm_activation_conv',
                'outer_full_decoder_checkpoint': 'forbidden',
                'source_no_grad_decode': 'ordinary_uncheckpointed',
                'spatial_crop_or_tile': 'forbidden',
                'allocator_fallback': 'forbidden',
            }
            if any(
                decoder_checkpointing.get(key) != value
                for key, value in required_checkpointing.items()
            ):
                raise ValueError('The v7 decoder checkpoint block is incomplete or changed.')
            required_accumulation = {
                'contract': UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
                'term_order': list(DEFAULT_UNIFIED_WEIGHTS),
                'graph_construction': 'one_term_at_a_time',
                'gradient_measurement': 'inline_during_term_backward',
                'backward': 'immediate_without_retain_graph',
                'graph_release': 'before_next_term',
                'translator_checkpoint': (
                    'non_reentrant_rng_preserving_every_differentiable_call'
                ),
                'saved_tensor_policy': 'save_on_cpu',
                'optimizer_updates_per_step': 1,
            }
            if any(
                accumulation.get(key) != value
                for key, value in required_accumulation.items()
            ):
                raise ValueError('The v7 generator accumulation block is incomplete or changed.')
        checkpoint = training.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        if "sanity" in training:
            raise ValueError("The v1 sanity block is obsolete; use the v2 pilot contract.")
        pilot = training.get("pilot", {})
        pilot = pilot if isinstance(pilot, Mapping) else {}
        if declared_contract == UNIFIED_STAGE2_CONFIG_CONTRACT:
            required_memory_qualification = {
                'anatomy_memory_contract': UNIFIED_ANATOMY_MEMORY_CONTRACT,
                'anatomy_memory_qualification_steps': 1,
                'a100_memory_gate_contract': UNIFIED_A100_GATE_CONTRACT,
                'a100_memory_gate_steps': 1,
                'a100_peak_allocated_limit_bytes': (
                    UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
                ),
            }
            if any(
                pilot.get(key) != value
                for key, value in required_memory_qualification.items()
            ):
                raise ValueError('The v7 one-step anatomy memory qualification changed.')
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
            decoder_activation_checkpoint_mode=str(
                decoder_checkpointing.get('mode', defaults.decoder_activation_checkpoint_mode)
            ),  # type: ignore[arg-type]
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
            pilot_a100_peak_allocated_limit_bytes=int(
                pilot.get(
                    'a100_peak_allocated_limit_bytes',
                    defaults.pilot_a100_peak_allocated_limit_bytes,
                )
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
        if declared_contract == UNIFIED_STAGE2_CONFIG_CONTRACT:
            primary_invariants = {
                'batch_size': value.batch_size == 1,
                'precision': value.precision == 'bf16',
                'integration_steps': value.integration_steps == 4,
                'integration_solver': value.integration_solver == 'heun',
                'six_weights': (
                    value.variant != 'full'
                    or value.loss_weights == DEFAULT_UNIFIED_WEIGHTS
                ),
                'checkpoint_mode': (
                    value.decoder_activation_checkpoint_mode
                    == KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE
                ),
            }
            failed = [name for name, satisfied in primary_invariants.items() if not satisfied]
            if failed:
                raise ValueError(f'Primary v7 execution invariants changed: {failed}.')
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
            'decoder_activation_checkpoint': {
                'contract': KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT,
                'mode': self.decoder_activation_checkpoint_mode,
            },
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
            'pilot_a100_peak_allocated_limit_bytes': (
                self.pilot_a100_peak_allocated_limit_bytes
            ),
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


def _translator_call(
    translator: BaseTranslator,
    latent: torch.Tensor,
    source_domains: Sequence[Domain],
    target_domains: Sequence[Domain],
    time_values: torch.Tensor,
    *,
    checkpoint_differentiable: bool,
) -> torch.Tensor:
    if not checkpoint_differentiable or not torch.is_grad_enabled():
        return translator(latent, source_domains, target_domains, time_values)

    def forward(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return translator(value, source_domains, target_domains, time)

    return checkpoint(
        forward,
        latent,
        time_values,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def integrate_transport(
    translator: BaseTranslator,
    z: torch.Tensor,
    source_domains: Sequence[Domain],
    target_domains: Sequence[Domain],
    *,
    steps: int,
    solver: Solver = "heun",
    checkpoint_differentiable: bool = False,
) -> torch.Tensor:
    """Differentiable fixed-grid ODE integration used by identity and graph losses."""

    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("Transport integration requires steps>=1 and euler/heun.")
    h = 1.0 / float(steps)
    current = z
    for index in range(steps):
        t0 = torch.full((z.shape[0],), index * h, device=z.device, dtype=z.dtype)
        velocity = _translator_call(
            translator,
            current,
            source_domains,
            target_domains,
            t0,
            checkpoint_differentiable=checkpoint_differentiable,
        )
        proposal = current + h * velocity
        if solver == "euler":
            current = proposal
        else:
            t1 = torch.full((z.shape[0],), (index + 1) * h, device=z.device, dtype=z.dtype)
            correction = _translator_call(
                translator,
                proposal,
                source_domains,
                target_domains,
                t1,
                checkpoint_differentiable=checkpoint_differentiable,
            )
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
    checkpoint_differentiable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direct = integrate_transport(
        translator,
        z,
        source_domains,
        target_domains,
        steps=steps,
        solver=solver,
        checkpoint_differentiable=checkpoint_differentiable,
    )
    first = integrate_transport(
        translator,
        z,
        source_domains,
        intermediate_domains,
        steps=steps,
        solver=solver,
        checkpoint_differentiable=checkpoint_differentiable,
    )
    composed = integrate_transport(
        translator,
        first,
        intermediate_domains,
        target_domains,
        steps=steps,
        solver=solver,
        checkpoint_differentiable=checkpoint_differentiable,
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
    directed_cell_counts: dict[str, int] = {
        _directed_domain_cell(contrast, source_field, target_field): 0
        for contrast in Contrast
        for source_field in FIELD_STRENGTHS_T
        for target_field in FIELD_STRENGTHS_T
        if source_field != target_field
    }
    source_usage_counts: dict[str, int] = defaultdict(int)
    for source_record in sorted(index.records, key=lambda item: item.case_id):
        contrast = Contrast.parse(source_record.domain.contrast)
        for target_field in FIELD_STRENGTHS_T:
            if target_field == source_record.domain.field_strength_t:
                continue
            cell = _directed_domain_cell(
                contrast, source_record.domain.field_strength_t, target_field
            )
            candidates = sorted(
                (
                    index.records[position]
                    for position in pools.table[contrast][target_field]
                    if index.records[position].subject_group_id
                    != source_record.subject_group_id
                ),
                key=lambda item: item.case_id,
            )
            if not candidates:
                raise ValueError(
                    "Frozen R/validation plan cannot represent required directed-domain "
                    f"cell {cell!r} with a subject-excluded target."
                )
            edge_identity = f"{source_record.case_id}|{cell}"
            target_position = _validation_u64(
                validation_seed, edge_identity, "independent_target"
            ) % len(candidates)
            target_record = candidates[target_position]
            raw_time = _validation_u64(
                validation_seed, edge_identity, "bridge_time"
            )
            unit_time = (raw_time + 0.5) / float(1 << 64)
            bridge_time = VALIDATION_PLAN_BRIDGE_EPS + (
                1.0 - 2.0 * VALIDATION_PLAN_BRIDGE_EPS
            ) * unit_time
            noise_seed = _validation_u64(
                validation_seed, edge_identity, "stochastic_noise"
            ) % ((1 << 63) - 1)
            entries.append(
                {
                    "directed_domain_cell": cell,
                    "edge_identity_sha256": sha256_json(
                        [validation_seed, source_record.case_id, cell]
                    ),
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
            directed_cell_counts[cell] += 1
            source_usage_counts[source_record.case_id] += 1
    missing_cells = sorted(cell for cell, count in directed_cell_counts.items() if count < 1)
    if missing_cells:
        raise ValueError(
            "Frozen R/validation plan is missing required directed-domain cells: "
            f"{missing_cells}."
        )
    if set(source_usage_counts) != set(records_by_case) or set(source_usage_counts.values()) != {4}:
        raise ValueError("Frozen validation plan must use every source once per other field.")
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
        "edge_count": len(entries),
        "required_directed_domain_cell_count": 60,
        "directed_domain_cell_counts": dict(sorted(directed_cell_counts.items())),
        "source_usage_counts": dict(sorted(source_usage_counts.items())),
        "derivation": {
            "target_assignment": (
                "for_each_source_and_each_other_field_sha256(seed,source_case_identity,"
                "directed_domain_cell,independent_target)_mod_sorted_subject_excluded_pool"
            ),
            "bridge_t": (
                "edge_specific_open_interval_affine_sha256_u64(seed,source_case_identity,"
                "directed_domain_cell,bridge_time)"
            ),
            "bridge_t_eps": VALIDATION_PLAN_BRIDGE_EPS,
            "stochastic_noise": (
                "torch_cpu_float32_randn_seeded_by_"
                "sha256_u63(seed,source_case_identity,directed_domain_cell,stochastic_noise)"
            ),
            "source_edges": "exactly_one_edge_to_each_of_four_other_fields",
            "target_subject_exclusion": True,
            "required_cells": "all_3_contrasts_x_20_directed_field_pairs",
            "aggregation": "equal_weighted_directed_domain_macro_mean",
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


def directed_domain_macro_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_cells: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    """Aggregate validation metrics with one equal vote per directed-domain cell."""

    if not rows:
        raise ValueError("Directed-domain macro aggregation requires validation rows.")
    expected = set(required_cells)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cell = str(row.get("directed_domain_cell", ""))
        if cell not in expected:
            raise ValueError(f"Unexpected directed-domain validation cell {cell!r}.")
        grouped[cell].append(row)
    if set(grouped) != expected:
        raise ValueError(
            "Directed-domain macro aggregation is missing required cells: "
            f"{sorted(expected - set(grouped))}."
        )
    metric_names = sorted(set(rows[0]) - {"directed_domain_cell"})
    if any(set(row) - {"directed_domain_cell"} != set(metric_names) for row in rows):
        raise ValueError("Validation rows do not expose an identical metric schema.")
    cell_means = {
        cell: {
            name: float(np.mean([float(row[name]) for row in cell_rows]))
            for name in metric_names
        }
        for cell, cell_rows in sorted(grouped.items())
    }
    macro = {
        name: float(np.mean([values[name] for values in cell_means.values()]))
        for name in metric_names
    }
    record_weighted = {
        name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names
    }
    if not all(np.isfinite(value) for value in (*macro.values(), *record_weighted.values())):
        raise FloatingPointError("Validation aggregation produced a non-finite value.")
    return macro, cell_means, record_weighted


def _directed_domain_cell(
    contrast: Contrast, source_field: float, target_field: float
) -> str:
    return f"{contrast.value}:{source_field:g}T->{target_field:g}T"


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
    decoder_checkpoint_evidence = _decoder_checkpoint_evidence(cfg, decoder)
    frozen_decoder_state_sha256 = _module_state_sha256(decoder)
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
            cfg.checkpoint_dir / "stage2_unified_validation_plan_v2.json",
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
        "frozen_decoder_state_sha256": frozen_decoder_state_sha256,
        'decoder_activation_checkpoint': decoder_checkpoint_evidence,
        'decoder_activation_checkpoint_sha256': sha256_json(decoder_checkpoint_evidence),
        'generator_gradient_accumulation': {
            'contract_version': UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
            'term_order': list(DEFAULT_UNIFIED_WEIGHTS),
            'graph_construction': 'one_term_at_a_time',
            'forward_backward_interleaved': True,
            'retain_graph': False,
            'graph_release': 'before_next_term',
            'gradient_measurement': 'inline_during_term_backward',
            'saved_tensor_policy': 'save_on_cpu',
            'translator_checkpoint_contract': UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT,
            'translator_checkpoint_use_reentrant': False,
            'translator_checkpoint_preserve_rng_state': True,
            'frozen_step_plan_replayed_per_term': True,
            'generator_optimizer_updates_per_step': 1,
        },
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
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
        if step == 0:
            decoder_state_after_qualification = _module_state_sha256(decoder)
            if decoder_state_after_qualification != frozen_decoder_state_sha256:
                raise RuntimeError('Frozen decoder state changed during anatomy qualification.')
            row['memory/anatomy_qualification'].update(
                {
                    'step': 1,
                    'decoder_state_sha256_before': frozen_decoder_state_sha256,
                    'decoder_state_sha256_after': decoder_state_after_qualification,
                    'decoder_state_unchanged': True,
                    'checkpoint_evidence_sha256': sha256_json(
                        decoder_checkpoint_evidence
                    ),
                    'checkpoint_evidence': decoder_checkpoint_evidence,
                }
            )
        generator_scheduler.step()
        if bool(row["critic/updated"]):
            critic_scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_start
        row.update(
            {
                "contract_version": UNIFIED_HISTORY_CONTRACT,
                "step": step + 1,
                "elapsed_seconds": time.perf_counter() - start,
                "step_seconds": step_seconds,
                "examples_per_second": cfg.batch_size / max(1e-9, step_seconds),
                "peak_cuda_bytes": (
                    max(
                        int(row['memory/training_peak_cuda_allocated_bytes']),
                        int(torch.cuda.max_memory_allocated(device)),
                    )
                    if device.type == "cuda"
                    else 0
                ),
                'peak_cuda_reserved_bytes': (
                    max(
                        int(row['memory/training_peak_cuda_reserved_bytes']),
                        int(torch.cuda.max_memory_reserved(device)),
                    )
                    if device.type == 'cuda'
                    else 0
                ),
                "generator_lr": generator_scheduler.get_last_lr()[0],
                "critic_lr": critic_scheduler.get_last_lr()[0],
                "run_fingerprint": run_fingerprint,
                "history_generation": history_generation,
            }
        )
        if step == 0:
            row['memory/a100_one_step_gate'] = _a100_one_step_gate(
                row,
                device=device,
                limit_bytes=cfg.pilot_a100_peak_allocated_limit_bytes,
            )
        _append_jsonl(history_path, row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        if step == 0:
            _append_jsonl(
                history_path,
                {
                    'contract_version': UNIFIED_HISTORY_CONTRACT,
                    'event': 'one_step_anatomy_memory_qualification',
                    'step': 1,
                    'qualification': row['memory/anatomy_qualification'],
                    'run_fingerprint': run_fingerprint,
                    'history_generation': history_generation,
                },
            )
            _append_jsonl(
                history_path,
                {
                    'contract_version': UNIFIED_HISTORY_CONTRACT,
                    'event': 'one_step_a100_memory_gate',
                    'step': 1,
                    'gate': row['memory/a100_one_step_gate'],
                    'run_fingerprint': run_fingerprint,
                    'history_generation': history_generation,
                },
            )
            if (
                device.type == 'cuda'
                and row['memory/a100_one_step_gate']['status'] != 'pass'
            ):
                raise RuntimeError(
                    'The dedicated one-step A100 <=72 GiB allocated-memory gate failed.'
                )
        if step < cfg.pilot_steps:
            pilot_rows.append(dict(row))
        pilot_complete = cfg.pilot_steps > 0 and step + 1 == cfg.pilot_steps
        validation_due = (
            (step + 1) % cfg.validation_every_steps == 0
            or step + 1 == cfg.steps
            or pilot_complete
        )
        validation: dict[str, Any] | None = None
        if validation_due:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            validation_start = time.perf_counter()
            try:
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
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _append_jsonl(
                    history_path,
                    {
                        "contract_version": UNIFIED_HISTORY_CONTRACT,
                        "step": step + 1,
                        "event": "oom_hard_stop",
                        "phase": "complete_validation",
                        "fallback": "forbidden",
                        "run_fingerprint": run_fingerprint,
                        "history_generation": history_generation,
                    },
                )
                raise
            complete_validation_seconds = time.perf_counter() - validation_start
            validation_peak_cuda_bytes = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            validation["runtime"] = {
                "contract_version": UNIFIED_PILOT_RUNTIME_PROJECTION_CONTRACT,
                "measured_complete_validation_seconds": complete_validation_seconds,
                "complete_validation_edge_count": int(validation["edge_count"]),
                "complete_validation_directed_domain_cell_count": len(
                    validation["directed_domain_cell_counts"]
                ),
                "peak_cuda_bytes": validation_peak_cuda_bytes,
            }
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
        if pilot_complete:
            if validation is None:
                raise RuntimeError("Pilot completion requires one complete frozen-plan validation.")
            validation_runtime = validation["runtime"]
            pilot_report = _pilot_report(
                pilot_rows,
                cfg,
                complete_validation_seconds=float(
                    validation_runtime["measured_complete_validation_seconds"]
                ),
                validation_peak_cuda_bytes=int(validation_runtime["peak_cuda_bytes"]),
                complete_validation_directed_domain_cell_count=int(
                    validation_runtime["complete_validation_directed_domain_cell_count"]
                ),
            )
            _append_jsonl(
                history_path,
                {
                    "contract_version": UNIFIED_HISTORY_CONTRACT,
                    "event": "full_objective_pilot",
                    "step": step + 1,
                    "pilot": pilot_report,
                    "validation_plan_sha256": validation_plan_sha256,
                    "run_fingerprint": run_fingerprint,
                    "history_generation": history_generation,
                },
            )
            if pilot_report["status"] != "pass":
                raise RuntimeError(
                    "Unified Stage-2 full-objective pilot failed: "
                    + ", ".join(pilot_report["failures"])
                )
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


@dataclass(frozen=True, slots=True)
class _FrozenTrainingStepPlan:
    time_values: torch.Tensor
    bridge_noise: torch.Tensor | None
    bridge_noise_seed: int | None
    intermediate_domains: tuple[Domain, ...] | None
    differentiable_replay_seed: int
    plan_sha256: str


@dataclass(slots=True)
class _TermForward:
    loss: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _freeze_training_step_plan(
    cfg: UnifiedStage2Config,
    batch: _TrainingBatch,
    sampler: torch.Generator,
) -> _FrozenTrainingStepPlan:
    time_values = _rand((cfg.batch_size,), sampler, batch.source.device).clamp(
        cfg.time_eps, 1.0 - cfg.time_eps
    )
    bridge_noise_seed: int | None = None
    bridge_noise: torch.Tensor | None = None
    if cfg.bridge == 'schrodinger':
        bridge_noise_seed = int(
            torch.randint(0, (1 << 63) - 1, (1,), generator=sampler, dtype=torch.int64)
        )
        noise_generator = torch.Generator().manual_seed(bridge_noise_seed)
        bridge_noise = _randn(
            tuple(batch.source.shape),
            noise_generator,
            batch.source.device,
            batch.source.dtype,
        )
    intermediate = (
        tuple(_intermediate_domains(batch.source_domains, batch.target_domains, sampler))
        if cfg.loss_weights['graph'] > 0
        else None
    )
    replay_seed = int(
        torch.randint(0, (1 << 63) - 1, (1,), generator=sampler, dtype=torch.int64)
    )
    body = {
        'source_records': [item.case_id for item in batch.source_records],
        'target_records': [item.case_id for item in batch.target_records],
        'time_values': [float(value) for value in time_values.detach().cpu()],
        'bridge': cfg.bridge,
        'bridge_noise_seed': bridge_noise_seed,
        'bridge_noise_shape': list(bridge_noise.shape) if bridge_noise is not None else None,
        'intermediate_domains': (
            [domain.to_dict() for domain in intermediate] if intermediate is not None else None
        ),
        'differentiable_replay_seed': replay_seed,
    }
    return _FrozenTrainingStepPlan(
        time_values=time_values,
        bridge_noise=bridge_noise,
        bridge_noise_seed=bridge_noise_seed,
        intermediate_domains=intermediate,
        differentiable_replay_seed=replay_seed,
        plan_sha256=sha256_json(body),
    )


@contextmanager
def _replay_step_rng(plan: _FrozenTrainingStepPlan, device: torch.device):
    cuda_devices: list[int] = []
    if device.type == 'cuda':
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(plan.differentiable_replay_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(plan.differentiable_replay_seed)
        yield


def _autocast_context(cfg: UnifiedStage2Config, device: torch.device):
    if device.type == 'cuda' and cfg.precision == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def _saved_tensor_offload_context(device: torch.device):
    return torch.autograd.graph.save_on_cpu(
        pin_memory=device.type == 'cuda',
        device_type=device.type,
    )


def _backward_isolated_generator_term(
    loss: torch.Tensor,
    translator: nn.Module,
    *,
    weight: float,
    scaler: torch.amp.GradScaler,
) -> float:
    if weight <= 0:
        raise ValueError('Only enabled generator terms may enter isolated backward.')
    parameters = tuple(value for value in translator.parameters() if value.requires_grad)
    if not parameters:
        raise ValueError('Translator has no trainable parameters for gradient qualification.')
    squared_parts: list[torch.Tensor] = []

    def capture(gradient: torch.Tensor) -> None:
        squared_parts.append(gradient.detach().float().square().sum())

    handles = [parameter.register_hook(capture) for parameter in parameters]
    scale = float(scaler.get_scale())
    try:
        scaler.scale(loss * weight).backward()
    finally:
        for handle in handles:
            handle.remove()
    if not squared_parts:
        return 0.0
    scaled_weighted_norm = torch.stack(squared_parts).sum().sqrt()
    raw_norm = float(
        (scaled_weighted_norm / max(abs(weight) * scale, 1.0e-30)).detach().cpu()
    )
    if not np.isfinite(raw_norm):
        raise FloatingPointError('Generator term has a non-finite translator gradient.')
    return raw_norm


def _forward_generator_term(
    name: str,
    cfg: UnifiedStage2Config,
    translator: BaseTranslator,
    critic: DomainProjectionDiscriminator,
    decoder: nn.Module,
    batch: _TrainingBatch,
    support: torch.Tensor,
    stats: FactoredLatentStats,
    plan: _FrozenTrainingStepPlan,
) -> _TermForward:
    if name == 'sb':
        z_t, flow_target = _bridge_sample(
            batch.source,
            batch.target,
            plan.time_values,
            cfg,
            noise=plan.bridge_noise,
        )
        predicted_velocity = _translator_call(
            translator,
            z_t,
            batch.source_domains,
            batch.target_domains,
            plan.time_values,
            checkpoint_differentiable=True,
        )
        return _TermForward(_masked_mse(predicted_velocity, flow_target, support))

    if name == 'identity':
        generated = integrate_transport(
            translator,
            batch.source,
            batch.source_domains,
            batch.source_domains,
            steps=cfg.integration_steps,
            solver=cfg.integration_solver,
            checkpoint_differentiable=True,
        )
        return _TermForward(
            masked_l1_loss(generated, batch.source, batch.source_support)
        )

    if name == 'anatomy':
        generated = integrate_transport(
            translator,
            batch.source,
            batch.source_domains,
            batch.target_domains,
            steps=cfg.integration_steps,
            solver=cfg.integration_solver,
            checkpoint_differentiable=True,
        )
        with torch.no_grad():
            source_image = _decode(
                decoder,
                stats.denormalize(batch.source),
                batch.source_domains,
                checkpoint_mode='disabled',
            )
        generated_image = _decode(
            decoder,
            stats.denormalize(generated),
            batch.target_domains,
            checkpoint_mode=cfg.decoder_activation_checkpoint_mode,
        )
        parts = anatomy_preservation_components(
            source_image,
            generated_image,
            batch.source_support,
            pool_scales=cfg.anatomy_pool_scales,
            support_erosion=cfg.anatomy_support_erosion,
        )
        return _TermForward(
            parts['total'],
            {
                'anatomy_low_mid': parts['low_mid'].detach(),
                'anatomy_edge': parts['edge'].detach(),
                'anatomy_gradient': parts['gradient'].detach(),
            },
        )

    if name == 'graph':
        if plan.intermediate_domains is None:
            raise RuntimeError('Enabled graph loss has no frozen intermediate domains.')
        graph, direct, composed = graph_consistency_loss(
            translator,
            batch.source,
            batch.source_domains,
            plan.intermediate_domains,
            batch.target_domains,
            support,
            steps=cfg.integration_steps,
            solver=cfg.integration_solver,
            checkpoint_differentiable=True,
        )
        return _TermForward(
            graph,
            {'graph_mse': _masked_mse(direct, composed, support).detach()},
        )

    if name in {'adversarial', 'domain'}:
        generated = integrate_transport(
            translator,
            batch.source,
            batch.source_domains,
            batch.target_domains,
            steps=cfg.integration_steps,
            solver=cfg.integration_solver,
            checkpoint_differentiable=True,
        )
        generated_view = _critic_view(
            generated,
            support,
            batch.target_domains,
            decoder,
            stats,
            cfg.critic_space,
            checkpoint_mode=cfg.decoder_activation_checkpoint_mode,
        )
        fake_score, fake_domain_logits = critic(generated_view, batch.target_domains)
        diagnostics = {
            'generator_fake_score': fake_score.detach(),
            'generator_fake_domain_logits': fake_domain_logits.detach(),
        }
        if name == 'adversarial':
            return _TermForward(
                adversarial_hinge_loss_generator(fake_score),
                diagnostics,
            )
        return _TermForward(
            F.cross_entropy(
                fake_domain_logits,
                domain_labels(
                    batch.target_domains,
                    len(batch.target_domains),
                    batch.source.device,
                ),
            ),
            diagnostics,
        )

    raise ValueError(f'Unsupported unified generator term: {name!r}.')


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
    zero = batch.source.sum() * 0.0
    cuda_peak_segments: list[dict[str, Any]] = []
    anatomy_forward_peak = {'allocated_bytes': 0, 'reserved_bytes': 0}
    anatomy_backward_peak = {'allocated_bytes': 0, 'reserved_bytes': 0}
    plan = _freeze_training_step_plan(cfg, batch, sampler)
    adversarial_active = cfg.loss_weights["adversarial"] > 0
    domain_active = cfg.loss_weights["domain"] > 0
    critic_active = adversarial_active or domain_active
    critic_adv = zero
    critic_domain = zero
    critic_total = zero
    critic_grad = 0.0
    real_score: torch.Tensor | None = None
    fake_score: torch.Tensor | None = None
    real_domain_logits: torch.Tensor | None = None
    if critic_active:
        # Critic update uses a detached, fully integrated generator sample.
        critic_optimizer.zero_grad(set_to_none=True)
        with _replay_step_rng(plan, batch.source.device), torch.no_grad(), _autocast_context(
            cfg, batch.source.device
        ):
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
        with _autocast_context(cfg, batch.source.device):
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
    active_terms = [
        name for name in DEFAULT_UNIFIED_WEIGHTS if cfg.loss_weights[name] > 0
    ]
    if not active_terms:
        raise ValueError('At least one generator objective must be active.')

    raw_values = {name: 0.0 for name in DEFAULT_UNIFIED_WEIGHTS}
    weighted_values = {name: 0.0 for name in DEFAULT_UNIFIED_WEIGHTS}
    term_gradient_norms = {name: 0.0 for name in DEFAULT_UNIFIED_WEIGHTS}
    term_execution: list[dict[str, Any]] = []
    anatomy_diagnostics = {
        'anatomy_low_mid': 0.0,
        'anatomy_edge': 0.0,
        'anatomy_gradient': 0.0,
    }
    graph_mse = 0.0
    generator_fake_domain_logits: torch.Tensor | None = None

    for name in active_terms:
        if name == 'anatomy':
            _capture_and_reset_cuda_peak(
                batch.source.device,
                cuda_peak_segments,
                phase='before_anatomy_term_forward',
            )
        with (
            _replay_step_rng(plan, batch.source.device),
            _saved_tensor_offload_context(batch.source.device),
            _autocast_context(cfg, batch.source.device),
        ):
            term_forward = _forward_generator_term(
                name,
                cfg,
                translator,
                critic,
                decoder,
                batch,
                support,
                stats,
                plan,
            )
        raw_loss = term_forward.loss
        if not bool(torch.isfinite(raw_loss)):
            raise FloatingPointError(f'Unified generator term {name!r} is non-finite.')
        raw_values[name] = float(raw_loss.detach().float().cpu())
        weighted_values[name] = cfg.loss_weights[name] * raw_values[name]
        if name == 'anatomy':
            anatomy_diagnostics = {
                key: float(value.float().cpu())
                for key, value in term_forward.diagnostics.items()
            }
            anatomy_forward_peak = _capture_and_reset_cuda_peak(
                batch.source.device,
                cuda_peak_segments,
                phase='anatomy_term_forward',
            )
            _capture_and_reset_cuda_peak(
                batch.source.device,
                cuda_peak_segments,
                phase='before_anatomy_term_backward',
            )
        elif name == 'graph':
            graph_mse = float(term_forward.diagnostics['graph_mse'].float().cpu())
        elif name in {'adversarial', 'domain'}:
            generator_fake_domain_logits = term_forward.diagnostics[
                'generator_fake_domain_logits'
            ]

        forward_sequence = len(term_execution)
        term_execution.append(
            {
                'sequence': forward_sequence,
                'event': 'forward_complete',
                'term': name,
                'isolated_graph': True,
                'saved_tensor_policy': 'save_on_cpu',
                'frozen_step_plan_sha256': plan.plan_sha256,
            }
        )
        term_gradient_norms[name] = _backward_isolated_generator_term(
            raw_loss,
            translator,
            weight=cfg.loss_weights[name],
            scaler=scaler,
        )
        term_execution.append(
            {
                'sequence': len(term_execution),
                'event': 'backward_complete',
                'term': name,
                'retain_graph': False,
                'graph_released_before_next_term': True,
            }
        )
        if name == 'anatomy':
            anatomy_backward_peak = _capture_and_reset_cuda_peak(
                batch.source.device,
                cuda_peak_segments,
                phase='anatomy_term_backward',
            )
        del raw_loss, term_forward

    scaler.unscale_(generator_optimizer)
    generator_grad = float(
        torch.nn.utils.clip_grad_norm_(translator.parameters(), cfg.grad_clip_norm)
    )
    scaler.step(generator_optimizer)
    scaler.update()
    _capture_and_reset_cuda_peak(
        batch.source.device, cuda_peak_segments, phase='after_generator_update'
    )
    if critic_active:
        critic.requires_grad_(True)
    generator_total = sum(weighted_values.values())
    if not np.isfinite(generator_total):
        raise FloatingPointError('Unified weighted generator loss is non-finite.')
    weighted_aux = sum(
        abs(weighted_values[name]) for name in weighted_values if name != 'sb'
    )
    training_peak_allocated = max(
        (segment['allocated_bytes'] for segment in cuda_peak_segments), default=0
    )
    training_peak_reserved = max(
        (segment['reserved_bytes'] for segment in cuda_peak_segments), default=0
    )
    anatomy_memory_status = (
        'disabled_for_ablation'
        if cfg.loss_weights['anatomy'] == 0
        else 'pass'
        if batch.source.device.type == 'cuda'
        and anatomy_forward_peak['allocated_bytes'] > 0
        and anatomy_backward_peak['allocated_bytes'] > 0
        else 'not_applicable_cpu'
        if batch.source.device.type != 'cuda'
        else 'fail'
    )
    if anatomy_memory_status == 'fail':
        raise RuntimeError('CUDA anatomy memory qualification did not record both phases.')
    row: dict[str, Any] = {
        **{f'raw/{key}': value for key, value in raw_values.items()},
        **{f'weighted/{key}': value for key, value in weighted_values.items()},
        'weighted/generator_total': generator_total,
        'weighted/auxiliary_total': weighted_aux,
        'weighted/aux_to_flow_ratio': (
            weighted_aux / max(abs(weighted_values['sb']), 1.0e-12)
        ),
        'raw/anatomy_low_mid': anatomy_diagnostics['anatomy_low_mid'],
        'raw/anatomy_edge': anatomy_diagnostics['anatomy_edge'],
        'raw/anatomy_gradient': anatomy_diagnostics['anatomy_gradient'],
        'graph/direct_vs_composed_l1': raw_values['graph'],
        'graph/direct_vs_composed_mse': graph_mse,
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
                generator_fake_domain_logits
                .argmax(1)
                .eq(domain_labels(batch.target_domains, len(batch.target_domains), batch.source.device))
                .float()
                .mean()
                .cpu()
            )
            if generator_fake_domain_logits is not None
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
            f"{batch.source_domains[0].label}->{plan.intermediate_domains[0].label}"
            f"->{batch.target_domains[0].label}"
            if plan.intermediate_domains is not None
            else "disabled"
        ),
        "source_records": [item.case_id for item in batch.source_records],
        "target_records": [item.case_id for item in batch.target_records],
        "prospective_records_loaded": 0,
        "descriptor_coupling_used": False,
        'generator/accumulation_contract': UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
        'generator/weighted_term_order': active_terms,
        'generator/optimizer_updates': 1,
        'generator/term_execution': term_execution,
        'generator/forward_backward_interleaved': True,
        'generator/retain_graph_used': False,
        'generator/save_on_cpu': True,
        'generator/translator_checkpoint_contract': (
            UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT
        ),
        'generator/translator_checkpoint_use_reentrant': False,
        'generator/translator_checkpoint_preserve_rng_state': True,
        'generator/term_gradient_probe_reconstructed_joint_graph': False,
        'generator/term_gradients_qualified': qualify_term_gradients,
        'generator/frozen_step_plan_sha256': plan.plan_sha256,
        'generator/frozen_step_plan': {
            'time_values': [
                float(value) for value in plan.time_values.detach().cpu()
            ],
            'bridge_noise_seed': plan.bridge_noise_seed,
            'bridge_noise_shape': (
                list(plan.bridge_noise.shape) if plan.bridge_noise is not None else None
            ),
            'intermediate_domains': (
                [domain.to_dict() for domain in plan.intermediate_domains]
                if plan.intermediate_domains is not None
                else None
            ),
            'differentiable_replay_seed': plan.differentiable_replay_seed,
            'rng_replayed_for_each_term': True,
        },
        'memory/training_peak_cuda_allocated_bytes': training_peak_allocated,
        'memory/training_peak_cuda_reserved_bytes': training_peak_reserved,
        'memory/anatomy_forward_peak_cuda_allocated_bytes': anatomy_forward_peak[
            'allocated_bytes'
        ],
        'memory/anatomy_forward_peak_cuda_reserved_bytes': anatomy_forward_peak[
            'reserved_bytes'
        ],
        'memory/anatomy_backward_peak_cuda_allocated_bytes': anatomy_backward_peak[
            'allocated_bytes'
        ],
        'memory/anatomy_backward_peak_cuda_reserved_bytes': anatomy_backward_peak[
            'reserved_bytes'
        ],
        'memory/anatomy_qualification': {
            'contract_version': UNIFIED_ANATOMY_MEMORY_CONTRACT,
            'status': anatomy_memory_status,
            'full_volume': True,
            'source_decode_checkpointed': False,
            'generated_decode_checkpoint_mode': cfg.decoder_activation_checkpoint_mode,
            'group_norm_scope': 'complete_spatial_volume',
            'spatial_crop_or_tile': False,
            'allocator_fallback': False,
        },
    }
    return row


def _decoder_checkpoint_evidence(
    cfg: UnifiedStage2Config, decoder: nn.Module
) -> dict[str, Any]:
    if cfg.decoder_activation_checkpoint_mode == 'disabled':
        return {
            'contract_version': KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT,
            'mode': 'disabled',
            'outer_full_decoder_checkpoint': False,
        }
    evidence_method = getattr(decoder, 'activation_checkpoint_evidence', None)
    decode_method = getattr(decoder, 'decode_fine_grained_checkpointed', None)
    if not callable(evidence_method) or not callable(decode_method):
        raise TypeError(
            'Fine-grained checkpointing requires the reviewed 3-D KL-VAE decoder API.'
        )
    evidence = dict(evidence_method())
    required = {
        'contract_version': KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT,
        'mode': KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE,
        'spatial_dims': 3,
        'full_volume': True,
        'group_norm_scope': 'complete_spatial_volume',
        'upsample_regions': ['up1', 'up2'],
        'residual_skip_checkpointed': False,
        'outer_full_decoder_checkpoint': False,
        'checkpoint_use_reentrant': False,
        'checkpoint_preserve_rng_state': True,
        'source_no_grad_decode_checkpointed': False,
        'state_dict_schema_changed': False,
    }
    if any(evidence.get(key) != value for key, value in required.items()):
        raise ValueError('Frozen decoder checkpoint evidence does not satisfy the v7 contract.')
    if not evidence.get('residual_branch_regions'):
        raise ValueError('Frozen decoder has no sealed residual checkpoint regions.')
    return evidence


def _decode(
    decoder: nn.Module,
    latent: torch.Tensor,
    domains: Sequence[Domain],
    *,
    checkpoint_mode: DecoderCheckpointMode = 'disabled',
) -> torch.Tensor:
    if checkpoint_mode == KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE:
        method = getattr(decoder, 'decode_fine_grained_checkpointed', None)
        if not callable(method):
            raise TypeError('Decoder does not implement fine-grained full-volume checkpointing.')
        return method(latent, domains)  # type: ignore[no-any-return]
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


def _a100_one_step_gate(
    row: Mapping[str, Any],
    *,
    device: torch.device,
    limit_bytes: int,
) -> dict[str, Any]:
    if limit_bytes != UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES:
        raise ValueError('The reviewed A100 allocated-memory limit changed.')
    if device.type != 'cuda':
        return {
            'contract_version': UNIFIED_A100_GATE_CONTRACT,
            'status': 'not_applicable_cpu',
            'required_gpu': 'NVIDIA A100',
            'peak_allocated_limit_bytes': limit_bytes,
            'peak_allocated_bytes': 0,
            'before_pilot_steps': [20, 200],
        }
    gpu_name = torch.cuda.get_device_name(device)
    peak_allocated = int(row['peak_cuda_bytes'])
    is_a100 = 'A100' in gpu_name.upper()
    within_limit = peak_allocated <= limit_bytes
    return {
        'contract_version': UNIFIED_A100_GATE_CONTRACT,
        'status': 'pass' if is_a100 and within_limit else 'fail',
        'required_gpu': 'NVIDIA A100',
        'gpu_name': gpu_name,
        'gpu_identity_matches': is_a100,
        'peak_allocated_limit_bytes': limit_bytes,
        'peak_allocated_bytes': peak_allocated,
        'within_allocated_limit': within_limit,
        'before_pilot_steps': [20, 200],
        'full_objective': True,
        'batch_size': 1,
        'precision': 'bf16',
        'integration_steps': 4,
        'integration_solver': 'heun',
    }


def _capture_and_reset_cuda_peak(
    device: torch.device,
    segments: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[str, int]:
    if device.type != 'cuda':
        return {'allocated_bytes': 0, 'reserved_bytes': 0}
    torch.cuda.synchronize(device)
    value = {
        'phase': phase,
        'allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
    }
    segments.append(value)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        'allocated_bytes': value['allocated_bytes'],
        'reserved_bytes': value['reserved_bytes'],
    }


def _critic_view(
    latent: torch.Tensor,
    support: torch.Tensor,
    domains: Sequence[Domain],
    decoder: nn.Module,
    stats: FactoredLatentStats,
    space: CriticSpace,
    *,
    checkpoint_mode: DecoderCheckpointMode = 'disabled',
) -> torch.Tensor:
    if space == "latent":
        return supported_critic_input(latent, support)
    image = _decode(
        decoder,
        stats.denormalize(latent),
        domains,
        checkpoint_mode=checkpoint_mode,
    )
    image_support = F.interpolate(support.float(), size=image.shape[2:], mode="nearest") > 0.5
    return supported_critic_input(image, image_support)


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
    rows: list[dict[str, Any]] = []
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
                "directed_domain_cell": str(plan_entry["directed_domain_cell"]),
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
    required_cells = sorted(validation_plan["directed_domain_cell_counts"])
    means, cell_means, record_weighted_means = directed_domain_macro_means(
        rows, required_cells=required_cells
    )
    selection_score = unified_validation_selection_score(means)
    return {
        "contract_version": UNIFIED_SELECTION_CONTRACT,
        "step": step,
        "validation_plan_sha256": stored_plan_sha256,
        "inventory_record_count": len(index.records),
        "edge_count": len(rows),
        "inventory_sha256": validation_plan["inventory_sha256"],
        "complete_inventory_used": True,
        "paired_endpoint_assumption": False,
        "subject_group_exclusion": True,
        "means": means,
        "aggregation": "equal_weighted_directed_domain_macro_mean",
        "directed_domain_cell_counts": dict(
            validation_plan["directed_domain_cell_counts"]
        ),
        "directed_domain_cell_means": cell_means,
        "record_weighted_means_diagnostic_only": record_weighted_means,
        "selection_score": selection_score,
        "selection_rule": dict(UNIFIED_SELECTION_RULE),
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "critic_and_domain_metrics_role": "diagnostic_only_excluded_from_selection",
    }


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
    rows: Sequence[Mapping[str, Any]],
    cfg: UnifiedStage2Config,
    *,
    complete_validation_seconds: float,
    validation_peak_cuda_bytes: int,
    complete_validation_directed_domain_cell_count: int,
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
        'peak_cuda_reserved_bytes',
        'memory/anatomy_forward_peak_cuda_allocated_bytes',
        'memory/anatomy_forward_peak_cuda_reserved_bytes',
        'memory/anatomy_backward_peak_cuda_allocated_bytes',
        'memory/anatomy_backward_peak_cuda_reserved_bytes',
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
    if not np.isfinite(complete_validation_seconds) or complete_validation_seconds <= 0:
        failures.append("invalid_complete_validation_runtime")
    if validation_peak_cuda_bytes < 0:
        failures.append("invalid_validation_peak_memory")
    runtime = _pilot_runtime_projection(
        mean_training_step_seconds=mean_step_seconds,
        complete_validation_seconds=complete_validation_seconds,
        training_peak_cuda_bytes=max(int(row["peak_cuda_bytes"]) for row in rows),
        validation_peak_cuda_bytes=validation_peak_cuda_bytes,
        complete_validation_directed_domain_cell_count=(
            complete_validation_directed_domain_cell_count
        ),
        cfg=cfg,
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
        "runtime": runtime,
        'decoder_activation_checkpoint': {
            'contract_version': KLVAE_DECODER_ACTIVATION_CHECKPOINT_CONTRACT,
            'mode': cfg.decoder_activation_checkpoint_mode,
            'outer_full_decoder_checkpoint': False,
        },
        'generator_gradient_accumulation': {
            'contract_version': UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
            'term_order': list(DEFAULT_UNIFIED_WEIGHTS),
            'graph_construction': 'one_term_at_a_time',
            'forward_backward_interleaved': True,
            'retain_graph': False,
            'graph_release': 'before_next_term',
            'gradient_measurement': 'inline_during_term_backward',
            'joint_six_term_gradient_probe': False,
            'saved_tensor_policy': 'save_on_cpu',
            'translator_checkpoint_contract': UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT,
            'translator_checkpoint_use_reentrant': False,
            'translator_checkpoint_preserve_rng_state': True,
            'frozen_step_plan_replayed_per_term': True,
            'generator_optimizer_updates_per_step': 1,
        },
        'one_step_a100_memory_gate': rows[0]['memory/a100_one_step_gate'],
        'one_step_anatomy_memory_qualification': rows[0][
            'memory/anatomy_qualification'
        ],
        'anatomy_cuda_peak_memory': {
            'forward_allocated_bytes_max': max(
                int(row['memory/anatomy_forward_peak_cuda_allocated_bytes'])
                for row in rows
            ),
            'forward_reserved_bytes_max': max(
                int(row['memory/anatomy_forward_peak_cuda_reserved_bytes'])
                for row in rows
            ),
            'backward_allocated_bytes_max': max(
                int(row['memory/anatomy_backward_peak_cuda_allocated_bytes'])
                for row in rows
            ),
            'backward_reserved_bytes_max': max(
                int(row['memory/anatomy_backward_peak_cuda_reserved_bytes'])
                for row in rows
            ),
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


def _pilot_runtime_projection(
    *,
    mean_training_step_seconds: float,
    complete_validation_seconds: float,
    training_peak_cuda_bytes: int,
    validation_peak_cuda_bytes: int,
    complete_validation_directed_domain_cell_count: int,
    cfg: UnifiedStage2Config,
) -> dict[str, Any]:
    """Project the configured run from measured training and complete validation phases."""

    if not np.isfinite(mean_training_step_seconds) or mean_training_step_seconds <= 0:
        raise ValueError("Mean training-step runtime must be finite and positive.")
    if not np.isfinite(complete_validation_seconds) or complete_validation_seconds <= 0:
        raise ValueError("Complete-validation runtime must be finite and positive.")
    if training_peak_cuda_bytes < 0 or validation_peak_cuda_bytes < 0:
        raise ValueError("Peak memory measurements must be non-negative.")
    if complete_validation_directed_domain_cell_count != 60:
        raise ValueError("Pilot projection requires the complete 60-cell validation plan.")
    schedule = _planned_validation_schedule(
        projected_steps=cfg.projected_steps,
        validation_every_steps=cfg.validation_every_steps,
        pilot_steps=cfg.pilot_steps,
    )
    projected_training_seconds = mean_training_step_seconds * cfg.projected_steps
    projected_validation_seconds = (
        complete_validation_seconds * schedule["planned_validation_run_count"]
    )
    projected_total_seconds = projected_training_seconds + projected_validation_seconds
    projected_total_hours = projected_total_seconds / 3600.0
    projected_total_gpu_cost = (
        projected_total_hours * cfg.gpu_hourly_cost_usd
        if cfg.gpu_hourly_cost_usd is not None
        else None
    )
    return {
        "contract_version": UNIFIED_PILOT_RUNTIME_PROJECTION_CONTRACT,
        "authorization_estimate_scope": "projected_training_plus_complete_validation",
        "measured_mean_training_step_seconds": mean_training_step_seconds,
        "measured_complete_validation_seconds": complete_validation_seconds,
        "measured_complete_validation_directed_domain_cell_count": (
            complete_validation_directed_domain_cell_count
        ),
        "examples_per_training_second": cfg.batch_size
        / max(mean_training_step_seconds, 1.0e-12),
        "projected_steps": cfg.projected_steps,
        "validation_every_steps": cfg.validation_every_steps,
        **schedule,
        "projected_training_seconds": projected_training_seconds,
        "projected_validation_seconds": projected_validation_seconds,
        "projected_total_seconds": projected_total_seconds,
        "projected_total_hours": projected_total_hours,
        "gpu_hourly_cost_usd": cfg.gpu_hourly_cost_usd,
        "projected_total_gpu_cost_usd": projected_total_gpu_cost,
        "cost_status": (
            "measured_training_and_validation_with_operator_rate"
            if projected_total_gpu_cost is not None
            else "operator_rate_required"
        ),
        "training_peak_cuda_bytes": training_peak_cuda_bytes,
        "validation_peak_cuda_bytes": validation_peak_cuda_bytes,
        "peak_cuda_bytes_across_training_and_validation": max(
            training_peak_cuda_bytes, validation_peak_cuda_bytes
        ),
    }


def _planned_validation_schedule(
    *, projected_steps: int, validation_every_steps: int, pilot_steps: int
) -> dict[str, Any]:
    """Count actual validation events, de-duplicating cadence, pilot, and terminal steps."""

    if projected_steps < 1 or validation_every_steps < 1 or pilot_steps < 0:
        raise ValueError("Projected steps/cadence must be positive and pilot steps non-negative.")
    cadence_run_count = projected_steps // validation_every_steps
    special_steps = {projected_steps}
    pilot_in_projected_run = 0 < pilot_steps <= projected_steps
    if pilot_in_projected_run:
        special_steps.add(pilot_steps)
    additional_steps = sorted(
        step for step in special_steps if step % validation_every_steps != 0
    )
    return {
        "cadence_validation_run_count": cadence_run_count,
        "pilot_boundary_validation_in_projected_run": pilot_in_projected_run,
        "terminal_validation_count": 1,
        "terminal_validation_already_on_cadence": (
            projected_steps % validation_every_steps == 0
        ),
        "additional_non_cadence_validation_steps": additional_steps,
        "planned_validation_run_count": cadence_run_count + len(additional_steps),
        "terminal_validation_counting_rule": (
            "terminal step is the union of cadence, pilot-boundary, and terminal events "
            "and is counted exactly once"
        ),
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
            "enabled by the primary v4 contract."
        )
    if cfg.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16.")
    if cfg.decoder_activation_checkpoint_mode not in {
        'disabled',
        KLVAE_DECODER_ACTIVATION_CHECKPOINT_MODE,
    }:
        raise ValueError('Decoder activation-checkpoint mode is unsupported.')
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
    if (
        cfg.pilot_a100_peak_allocated_limit_bytes
        != UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
    ):
        raise ValueError('The one-step A100 peak-allocated gate must remain exactly 72 GiB.')
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
    'UNIFIED_ANATOMY_MEMORY_CONTRACT',
    'UNIFIED_A100_GATE_CONTRACT',
    'UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES',
    'UNIFIED_GENERATOR_ACCUMULATION_CONTRACT',
    "UNIFIED_HISTORY_CONTRACT",
    "UNIFIED_RESUME_CONTRACT",
    "UNIFIED_SELECTION_CONTRACT",
    "UNIFIED_SELECTION_RECEIPT_CONTRACT",
    "UNIFIED_SELECTION_RULE",
    "UNIFIED_SELECTION_RULE_CONTRACT",
    "UNIFIED_SELECTION_RULE_SHA256",
    "UNIFIED_STAGE2_CONTRACT",
    'UNIFIED_STAGE2_CONFIG_CONTRACT',
    'UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT',
    "UNIFIED_VALIDATION_PLAN_CONTRACT",
    "UnifiedStage2Config",
    "UnifiedStage2Result",
    "anatomy_preservation_components",
    "build_unified_validation_plan",
    "directed_domain_macro_means",
    "find_latest_stage2_selection_receipt",
    "graph_consistency_loss",
    "integrate_transport",
    "load_stage2_selection_receipt",
    "run_stage2_unified_train",
    "unified_validation_selection_score",
    "validate_unified_config",
]
