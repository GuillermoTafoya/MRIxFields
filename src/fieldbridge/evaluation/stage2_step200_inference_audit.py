"""Streaming, inference-only visual audit for the verified Stage-2 step-200 pilot.

This module deliberately does not construct optimizers, call backward, select a
checkpoint, or authorize training.  P:0006 remains development/model-assessment
evidence from one traveller protocol.  All tensor-bearing cases are explicitly
released before the streaming protocol iterator may advance.
"""

from __future__ import annotations

import base64
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Contrast, Domain
from fieldbridge.data.latent_bank import encode_latent
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factored_latent_bank import (
    propagate_encoder_local_valid_core_support,
)
from fieldbridge.data.photometry_factorization import (
    FrozenPhotometryArtifact,
    canonical_tensor_sha256,
    sha256_file,
    sha256_json,
)
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256
from fieldbridge.evaluation.metrics import gradient_mae
from fieldbridge.evaluation.stage2_gate01 import (
    Gate01Case,
    fixed_montage_specifications,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    STAGE1_RUN_C_CHECKPOINT_SHA256,
    PosthocTargetCalibrator,
)
from fieldbridge.evaluation.stage2_gate01_montage import (
    Gate01MontageCollector,
    _encode_grayscale_png,
    _normalize_panel,
)
from fieldbridge.evaluation.stage2_step200_lpips_audit import forbid_network_access
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    GATE01_P0006_EVALUATION_PROTOCOL,
    P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
    P0006_EVIDENCE_LIMITATION,
    P0009_CONFIRMATION_STATUS,
    iter_gate01_p0006_evaluation_cases,
)
from fieldbridge.models.factory import build_decoder, build_encoder, build_translator
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.stage2_unified import (
    UNIFIED_RESUME_CONTRACT,
    UnifiedStage2Config,
    anatomy_preservation_components,
    graph_consistency_loss,
    integrate_transport,
)


INFERENCE_AUDIT_CONTRACT = "stage2-step200-p0006-inference-audit-v2"
INFERENCE_PLAN_CONTRACT = "stage2-step200-p0006-frozen-inference-plan-v1"
MONTAGE_SPECIFICATION_CONTRACT = "stage2-step200-p0006-frozen-montage-v1"
CASE_RECEIPT_CONTRACT = "stage2-step200-p0006-inference-case-receipt-v1"
MEMORY_GATE_CONTRACT = "stage2-step200-p0006-a100-one-case-gate-v1"
ARTIFACT_MANIFEST_CONTRACT = "stage2-step200-p0006-audit-artifact-manifest-v2"
METRIC_CONTRACT = "stage2-step200-p0006-descriptive-official-task3-v1"
TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
CHECKPOINT_SHA256 = "09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021"
RUN_FINGERPRINT = "c814c948a5b85bd3a694db7c8e074894e97c16a96a36acbfa6f370faf2dac0aa"
P0006_PROTOCOL_FILE_SHA256 = "3c11092a4a5e5342726947d705eca8fd8c52a70b82b96892529fe564c5f5f809"
P0006_PROTOCOL_SHA256 = "2cd8e17207175f6a8f1f11f8afd748beca11ae0f47a08a0b2e1529a2272274e4"
EVALUATION_READINESS_FILE_SHA256 = "dc6695d3a9d9f69749af1421e92a3b008f240e147a59619957cd4888af71d7d2"
EVALUATION_READINESS_SHA256 = "ff5d4e8e80f48fecdf0e320fd56e9fd9431145798908b56d7e363b5760d8e0ec"
INFERENCE_SEED = 20_260_825
A100_MAX_PEAK_ALLOCATED_BYTES = 72 * 1024**3
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_METHOD_ORDER = (
    "raw_identity",
    "calibrated_identity",
    "raw_original_sb_v2",
    "calibrated_original_sb_v2",
    "raw_unified_step200",
    "calibrated_unified_step200",
    "stage1_reconstruction_ceiling",
)
_METRICS = ("nrmse", "ssim", "lpips")
_LOWER_IS_BETTER = {"nrmse": True, "ssim": False, "lpips": True, "edge_mae": True}


@dataclass(frozen=True, slots=True)
class FrozenStep200InferencePlan:
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.payload["inference_plan_sha256"])

    @property
    def montage_specification(self) -> dict[str, Any]:
        return dict(self.payload["montage_specification"])


@dataclass(frozen=True, slots=True)
class InferenceCaseOutputs:
    methods: dict[str, torch.Tensor]
    anatomy: dict[str, float]
    graph: dict[str, Any]
    sweep_slices: dict[str, np.ndarray]
    decoded_canonical_sha256: str


class Step200CaseInferenceRuntime(Protocol):
    device: torch.device
    gpu_identity: Mapping[str, Any]

    def infer_case(
        self,
        case: Gate01Case,
        calibrator: PosthocTargetCalibrator,
        *,
        plan: FrozenStep200InferencePlan,
    ) -> InferenceCaseOutputs: ...

    def state_identity(self) -> Mapping[str, str]: ...


@dataclass
class UnifiedStep200InferenceRuntime:
    """Generator plus frozen VAE inference components; no critic or optimizer."""

    translator: torch.nn.Module
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    artifact: FrozenPhotometryArtifact
    stats: FactoredLatentStats
    support_rule_sha256: str
    device: torch.device
    gpu_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("The production step-200 inference runtime requires CUDA.")
        for module in (self.translator, self.encoder, self.decoder):
            module.to(self.device).eval().requires_grad_(False)
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError("Inference components must be frozen.")

    def state_identity(self) -> Mapping[str, str]:
        return {
            "translator": _module_state_sha256(self.translator),
            "encoder": _module_state_sha256(self.encoder),
            "decoder": _module_state_sha256(self.decoder),
        }

    @torch.inference_mode()
    def infer_case(
        self,
        case: Gate01Case,
        calibrator: PosthocTargetCalibrator,
        *,
        plan: FrozenStep200InferencePlan,
    ) -> InferenceCaseOutputs:
        if torch.is_grad_enabled():
            raise RuntimeError("Step-200 inference unexpectedly enabled gradients.")
        if case.source_image is None:
            raise ValueError("P:0006 inference case lacks its verified source image.")
        case_seed = _case_seed(plan, case)
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(case_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(case_seed)
            canonical_context = self.artifact.normalize_source(
                case.source_image, case.source_domain
            )
            canonical = canonical_context.values
            if canonical.ndim == 3:
                canonical_batch = canonical[None, None]
            elif canonical.ndim == 4:
                canonical_batch = canonical[None]
            else:
                raise ValueError("P:0006 canonical source is not one full 3-D volume.")
            support_image = canonical_context.support_mask.detach().to(torch.bool)
            while support_image.ndim > 3 and int(support_image.shape[0]) == 1:
                support_image = support_image[0]
            latent, path_used = encode_latent(
                self.encoder,
                canonical_batch.to(self.device),
                case.source_domain,
                strategy="full",
                block_size=(1, 1, 1),
                halo=(0, 0, 0),
                precision="fp32",
            )
            if path_used != "full":
                raise ValueError("Step-200 P:0006 inference requires full-volume VAE encoding.")
            support = propagate_encoder_local_valid_core_support(
                support_image,
                self.encoder,
                expected_rule_sha256=self.support_rule_sha256,
            )
            support_batch = support[None, None].to(self.device)
            z = self.stats.normalize(latent, support_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_z = integrate_transport(
                    self.translator,
                    z,
                    [case.source_domain],
                    [case.target_domain],
                    steps=4,
                    solver="heun",
                )
                decoded = _decode(
                    self.decoder,
                    self.stats.denormalize(generated_z),
                    [case.target_domain],
                )[0].float().cpu()
            unified = self.artifact.render_target(
                canonical_context.with_values(decoded), case.target_domain
            ).float().cpu()
            calibrated_identity = calibrator.apply(
                case.raw_identity,
                case.target_domain,
                support_mask=case.support_mask,
                mode="histogram",
            )
            calibrated_sb = calibrator.apply(
                case.raw_sb_v2,
                case.target_domain,
                support_mask=case.support_mask,
                mode="histogram",
            )
            # The reviewed calibrator is fitted only from retrospective training target
            # CDFs and consumes the prediction/support, never the paired P:0006 target.
            calibrated_unified = calibrator.apply(
                unified,
                case.target_domain,
                support_mask=case.support_mask,
                mode="histogram",
            )
            methods = {
                "raw_identity": case.raw_identity,
                "calibrated_identity": calibrated_identity,
                "raw_original_sb_v2": case.raw_sb_v2,
                "calibrated_original_sb_v2": calibrated_sb,
                "raw_unified_step200": unified,
                "calibrated_unified_step200": calibrated_unified,
                "stage1_reconstruction_ceiling": case.stage1_reconstruction_ceiling,
            }
            image_support = canonical_context.support_mask
            if image_support.ndim == 3:
                image_support_batch = image_support[None, None].to(self.device)
            else:
                image_support_batch = image_support[None].to(self.device)
            anatomy_values = anatomy_preservation_components(
                canonical_batch.to(self.device), decoded[None].to(self.device), image_support_batch
            )
            anatomy = {key: float(value.detach().float().cpu()) for key, value in anatomy_values.items()}
            graph = self._graph_diagnostic(case, z, support_batch, canonical_context, plan)
            sweep_slices = self._field_sweep(case, z, canonical_context, plan)
            decoded_sha = canonical_tensor_sha256(decoded)
            del generated_z, decoded, latent, z, support, support_batch, canonical_batch
            del image_support_batch, anatomy_values
        return InferenceCaseOutputs(
            methods=methods,
            anatomy=anatomy,
            graph=graph,
            sweep_slices=sweep_slices,
            decoded_canonical_sha256=decoded_sha,
        )

    def _graph_diagnostic(
        self,
        case: Gate01Case,
        z: torch.Tensor,
        support: torch.Tensor,
        canonical_context: Any,
        plan: FrozenStep200InferencePlan,
    ) -> dict[str, Any]:
        selected = _graph_intermediate(plan, case)
        if selected is None:
            return {"selected": False}
        intermediate = Domain(selected, case.source_domain.contrast)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l1, direct, composed = graph_consistency_loss(
                self.translator,
                z,
                [case.source_domain],
                [intermediate],
                [case.target_domain],
                support,
                steps=4,
                solver="heun",
            )
            direct_decoded = _decode(
                self.decoder, self.stats.denormalize(direct), [case.target_domain]
            )[0].float().cpu()
            composed_decoded = _decode(
                self.decoder, self.stats.denormalize(composed), [case.target_domain]
            )[0].float().cpu()
        direct_image = self.artifact.render_target(
            canonical_context.with_values(direct_decoded), case.target_domain
        )
        composed_image = self.artifact.render_target(
            canonical_context.with_values(composed_decoded), case.target_domain
        )
        result = {
            "selected": True,
            "intermediate_field_t": selected,
            "direct_vs_composed_l1": float(l1.detach().float().cpu()),
            "direct_vs_composed_mse": float(
                torch.mean((direct - composed).float().pow(2)).detach().cpu()
            ),
            "direct_slice": _middle_slice(direct_image),
            "composed_slice": _middle_slice(composed_image),
            "absolute_difference_slice": _middle_slice((direct_image - composed_image).abs()),
        }
        del direct, composed, direct_decoded, composed_decoded, direct_image, composed_image
        return result

    def _field_sweep(
        self,
        case: Gate01Case,
        z: torch.Tensor,
        canonical_context: Any,
        plan: FrozenStep200InferencePlan,
    ) -> dict[str, np.ndarray]:
        if float(case.source_domain.field_strength_t) != float(
            plan.payload["field_sweep"]["source_field_t"]
        ):
            return {}
        slices: dict[str, np.ndarray] = {}
        for field in plan.payload["field_sweep"]["target_fields_t"]:
            target = Domain(float(field), case.source_domain.contrast)
            if target == case.source_domain:
                target_z = z
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    target_z = integrate_transport(
                        self.translator,
                        z,
                        [case.source_domain],
                        [target],
                        steps=4,
                        solver="heun",
                    )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded = _decode(
                    self.decoder, self.stats.denormalize(target_z), [target]
                )[0].float().cpu()
            rendered = self.artifact.render_target(
                canonical_context.with_values(decoded), target
            )
            slices[f"{float(field):g}T"] = _middle_slice(rendered)
            if target_z is not z:
                del target_z
            del decoded, rendered
        return slices


def build_frozen_step200_inference_plan(protocol: Mapping[str, Any]) -> FrozenStep200InferencePlan:
    """Freeze target-array-independent seeds, display selection, paths, and ordering."""

    if (
        protocol.get("contract_version") != GATE01_P0006_EVALUATION_PROTOCOL
        or protocol.get("protocol_sha256") != P0006_PROTOCOL_SHA256
        or protocol.get("data_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
        or protocol.get("training_or_model_selection_use") is not False
        or protocol.get("population_or_generalization_claims_authorized") is not False
    ):
        raise ValueError("P:0006 protocol is incompatible with the inference audit.")
    receipts = protocol.get("case_receipts")
    if not isinstance(receipts, list) or len(receipts) != 60:
        raise ValueError("Inference plan requires exactly 60 P:0006 receipts.")
    seeds = []
    for index, receipt in enumerate(receipts, 1):
        if not isinstance(receipt, Mapping):
            raise ValueError("P:0006 receipt inventory is malformed.")
        source = Domain.from_dict(dict(receipt["source_domain"]))
        target = Domain.from_dict(dict(receipt["target_domain"]))
        source_sha = str(receipt.get("source_image_sha256", ""))
        if _SHA256_RE.fullmatch(source_sha) is None:
            raise ValueError("Inference plan source identity is malformed.")
        material = f"{INFERENCE_SEED}|{index}|{source_sha}|{source.label}|{target.label}"
        seeds.append(
            {
                "index": index,
                "source_domain": source.to_dict(),
                "target_domain": target.to_dict(),
                "source_image_sha256": source_sha,
                "seed": int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
                % (2**31),
            }
        )
    base_montage = fixed_montage_specifications()
    montage = {
        "contract_version": MONTAGE_SPECIFICATION_CONTRACT,
        "selection_frozen_before_inference": True,
        "selection_basis": "contrast and directed field pair only; never target values or metrics",
        "canonical_orientation_contract": "nib.as_closest_canonical RAS array inherited by reviewed Gate01 producer",
        "planes": {"sagittal": 0, "coronal": 1, "axial": 2},
        "relative_slice_positions": [0.35, 0.5, 0.65],
        "directed_pairs_per_contrast": base_montage["directed_pairs_per_contrast"],
        "contrasts": [contrast.value for contrast in CONTRASTS],
        "display_order": ["source", "target", *_METHOD_ORDER, "absolute_difference", "edge_difference"],
        "shared_display_range_within_case": True,
        "normalization": "shared finite min/max for intensity panels; independent nonnegative max for differences",
        "interpolation": "none",
        "post_result_selection": False,
    }
    montage["montage_specification_sha256"] = sha256_json(montage)
    body: dict[str, Any] = {
        "contract_version": INFERENCE_PLAN_CONTRACT,
        "protocol_sha256": P0006_PROTOCOL_SHA256,
        "global_seed": INFERENCE_SEED,
        "seed_derivation": "global_seed + ordinal + source-array SHA + source/target domain labels; target tensor never read",
        "case_seeds": seeds,
        "preprocessing": {
            "source_only": True,
            "photometry": "frozen R-train photometry artifact",
            "encode": "posterior_mean_full_volume_fp32",
            "target_tensor_influence": False,
        },
        "transport": {"steps": 4, "solver": "heun", "precision": "bf16", "batch_size": 1},
        "memory_gate_case": {
            "selection_frozen_before_inference": True,
            "contrast": CONTRASTS[0].value,
            "source_field_t": 3.0,
            "target_field_t": 7.0,
            "selection_uses_target_tensor": False,
            "rationale": "exercises the five-field sweep in one full-volume case",
        },
        "field_sweep": {
            "source_field_t": 3.0,
            "target_fields_t": [0.1, 1.5, 3.0, 5.0, 7.0],
            "one_source_acquisition_per_contrast_selected_by_source_domain_only": True,
        },
        "graph_paths": [
            {"contrast": contrast.value, "source_field_t": source, "intermediate_field_t": 3.0, "target_field_t": target}
            for contrast in CONTRASTS
            for source, target in ((0.1, 7.0), (7.0, 0.1))
        ],
        "montage_specification": montage,
        "P0006_target_used_for_seed_preprocessing_or_selection": False,
        "P0006_training_or_model_selection_use": False,
        "P0007_access": False,
        "P0009_access": False,
    }
    body["inference_plan_sha256"] = sha256_json(body)
    return FrozenStep200InferencePlan(body)


def load_unified_step200_inference_runtime(
    *,
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    vae_config_path: str | Path,
    vae_checkpoint_path: str | Path,
    photometry_artifact_path: str | Path,
    bank_dir: str | Path,
    device: str | torch.device = "cuda",
) -> UnifiedStep200InferenceRuntime:
    """Verify the complete checkpoint, then extract only generator/frozen VAE state."""

    checkpoint_path = Path(checkpoint_path)
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("Step-200 checkpoint file SHA-256 mismatch.")
    config = _load_json(Path(resolved_config_path))
    training = config.get("training")
    if (
        not isinstance(training, Mapping)
        or training.get("device") != "auto"
        or training.get("batch_size") != 1
        or training.get("precision") != "bf16"
        or training.get("integration_steps") != 4
        or training.get("integration_solver") != "heun"
    ):
        raise ValueError("Resolved step-200 inference configuration changed.")
    declared_config = UnifiedStage2Config.from_mapping(config).to_dict()
    if declared_config.get("device") != "auto":
        raise ValueError("Resolved step-200 configuration must declare device=auto.")
    effective_payload = copy.deepcopy(config)
    effective_training = effective_payload.get("training")
    if not isinstance(effective_training, dict):
        raise ValueError("Resolved step-200 training mapping is not mutable JSON metadata.")
    effective_training["device"] = "cuda"
    effective_config = UnifiedStage2Config.from_mapping(effective_payload).to_dict()
    changed_config_fields = sorted(
        key
        for key in set(declared_config) | set(effective_config)
        if declared_config.get(key) != effective_config.get(key)
    )
    if changed_config_fields != ["device"] or effective_config.get("device") != "cuda":
        raise ValueError("Historical inference configuration replay changed more than auto->cuda.")
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    expected_keys = {
        "contract_version", "run_fingerprint", "validation_plan_sha256",
        "selection_rule_sha256", "training_cursor", "translator", "critic",
        "generator_optimizer", "critic_optimizer", "generator_scheduler",
        "critic_scheduler", "scaler", "sampler_rng", "torch_rng", "cuda_rng",
        "python_rng", "numpy_rng", "history_prefix_bytes", "history_prefix_sha256",
        "validation_selection", "pilot_report", "in_progress_pilot_rows", "_meta",
    }
    if set(state) != expected_keys:
        raise ValueError("Complete step-200 checkpoint key inventory changed.")
    meta = state.get("_meta")
    if (
        state.get("contract_version") != UNIFIED_RESUME_CONTRACT
        or state.get("training_cursor") != 200
        or state.get("run_fingerprint") != RUN_FINGERPRINT
        or not isinstance(meta, Mapping)
        or meta.get("git_commit") != TRAINING_EVIDENCE_COMMIT
        or meta.get("config") != effective_config
    ):
        raise ValueError("Step-200 checkpoint metadata identity changed.")
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Resolved step-200 configuration lacks the translator model mapping.")
    translator_config = dict(model_config)
    translator_name = str(translator_config.pop("name", "flow_matching_latent"))
    translator = build_translator(translator_name, **translator_config)
    translator_state = state["translator"]
    if not isinstance(translator_state, Mapping):
        raise ValueError("Step-200 checkpoint translator state is malformed.")
    translator.load_state_dict(translator_state, strict=True)
    # Optimizer/critic dictionaries were verified as members of the complete container,
    # but no critic or optimizer object is constructed or loaded.
    del state
    gc.collect()

    bank = PhotometryFactoredLatentBankIndex(bank_dir, "validation")
    stats = FactoredLatentStats.from_bank(bank_dir)
    vae_path = Path(vae_checkpoint_path)
    if sha256_file(vae_path) != STAGE1_RUN_C_CHECKPOINT_SHA256:
        raise ValueError("Frozen Stage-1 VAE checkpoint identity changed.")
    if bank.manifest.get("vae", {}).get("checkpoint_sha256") != STAGE1_RUN_C_CHECKPOINT_SHA256:
        raise ValueError("Restored bank and frozen VAE checkpoint disagree.")
    vae_config = load_yaml_config(Path(vae_config_path))
    if sha256_json(vae_config) != bank.manifest.get("vae", {}).get("config_sha256"):
        raise ValueError("Frozen VAE configuration and restored bank disagree.")
    vae_state = load_checkpoint(vae_path, map_location="cpu")
    vae_model = vae_config.get("model") if isinstance(vae_config, Mapping) else None
    if not isinstance(vae_model, Mapping):
        raise ValueError("Frozen VAE checkpoint lacks its complete model configuration.")
    encoder = build_encoder("kl_vae", **_kl_vae_kwargs(vae_model, "encoder"))
    decoder = build_decoder("kl_vae", **_kl_vae_kwargs(vae_model, "decoder"))
    encoder.load_state_dict(vae_state["encoder"], strict=True)
    decoder.load_state_dict(vae_state["decoder"], strict=True)
    del vae_state
    gc.collect()

    artifact_path = Path(photometry_artifact_path)
    if sha256_file(artifact_path) != bank.manifest.get("photometry", {}).get(
        "artifact_file_sha256"
    ):
        raise ValueError("Frozen photometry artifact and restored bank disagree.")
    artifact = FrozenPhotometryArtifact.load(artifact_path)
    if artifact.artifact_sha256 != bank.manifest.get("photometry", {}).get("artifact_sha256"):
        raise ValueError("Frozen photometry artifact identity changed.")
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Step-200 inference audit requires an NVIDIA A100 CUDA runtime.")
    properties = torch.cuda.get_device_properties(device_obj)
    gpu_identity = {
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "cuda_runtime": str(torch.version.cuda),
        "torch": str(torch.__version__),
    }
    if "A100" not in gpu_identity["name"] or gpu_identity["total_memory_bytes"] < 79 * 1024**3:
        raise RuntimeError("First step-200 inference qualification requires an NVIDIA A100 80 GB.")
    support_rule = bank.manifest.get("operational_support_rule")
    if not isinstance(support_rule, Mapping) or _SHA256_RE.fullmatch(
        str(support_rule.get("rule_sha256", ""))
    ) is None:
        raise ValueError("Restored bank support-rule identity is missing.")
    return UnifiedStep200InferenceRuntime(
        translator=translator,
        encoder=encoder,
        decoder=decoder,
        artifact=artifact,
        stats=stats,
        support_rule_sha256=str(support_rule["rule_sha256"]),
        device=device_obj,
        gpu_identity=gpu_identity,
    )


def run_step200_p0006_inference_audit(
    *,
    protocol_path: str | Path,
    evaluation_readiness_path: str | Path,
    runtime: Step200CaseInferenceRuntime,
    output_dir: str | Path,
    audit_implementation_commit: str,
    require_a100: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    metric_fn: Callable[
        [torch.Tensor, torch.Tensor, Sequence[str], str], Mapping[str, float]
    ] | None = None,
    dependency_provenance: Mapping[str, Any] | None = None,
    lpips_provenance: Mapping[str, Any] | None = None,
    lpips_integrity_verifier: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume the bounded-memory 60-case descriptive audit."""

    _require_git(audit_implementation_commit, "audit implementation commit")
    if metric_fn is None:
        raise ValueError("Inference audit requires one explicitly injected metric evaluator.")
    if require_a100 and (
        dependency_provenance is None
        or lpips_provenance is None
        or lpips_integrity_verifier is None
    ):
        raise ValueError("Production inference audit lacks sealed dependency/LPIPS provenance.")
    protocol_path = Path(protocol_path)
    readiness_path = Path(evaluation_readiness_path)
    if sha256_file(protocol_path) != P0006_PROTOCOL_FILE_SHA256:
        raise ValueError("P:0006 protocol file SHA-256 mismatch.")
    protocol = _load_self_hashed(protocol_path, "protocol_sha256")
    if protocol.get("protocol_sha256") != P0006_PROTOCOL_SHA256:
        raise ValueError("P:0006 internal protocol SHA-256 mismatch.")
    if sha256_file(readiness_path) != EVALUATION_READINESS_FILE_SHA256:
        raise ValueError("Evaluation-readiness file SHA-256 mismatch.")
    readiness = _load_self_hashed(readiness_path, "readiness_sha256")
    if readiness.get("readiness_sha256") != EVALUATION_READINESS_SHA256:
        raise ValueError("Evaluation-readiness internal SHA-256 mismatch.")
    _validate_scientific_role(protocol, readiness)
    plan = build_frozen_step200_inference_plan(protocol)
    if require_a100:
        _validate_a100_identity(runtime)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dependency_receipt = _seal_or_reuse_runtime_receipt(
        root / "dependency_environment_receipt.json",
        dependency_provenance or {"synthetic_test_environment": True},
        label="dependency environment",
    )
    lpips_receipt = _seal_or_reuse_runtime_receipt(
        root / "lpips_provenance_receipt.json",
        lpips_provenance or {"synthetic_test_evaluator": True},
        label="LPIPS provenance",
    )
    receipt_dir = root / "stage2_step200_p0006_case_receipts"
    montage_dir = root / "montages"
    slice_dir = root / "predeclared_2d_slices"
    for directory in (receipt_dir, montage_dir, slice_dir):
        directory.mkdir(exist_ok=True)
    plan_path = root / "frozen_inference_plan.json"
    _seal_or_verify_json(plan_path, plan.payload, "inference_plan_sha256")
    run_contract = _run_contract(
        plan,
        runtime,
        audit_implementation_commit=audit_implementation_commit,
        dependency_receipt=dependency_receipt,
        lpips_receipt=lpips_receipt,
    )
    _seal_or_verify_json(root / "run_contract.json", run_contract, "run_contract_sha256")
    initial_state = dict(runtime.state_identity())
    with forbid_network_access():
        gate = _run_or_verify_one_case_gate(
            protocol_path,
            runtime,
            plan,
            root / "one_case_inference_memory_gate.json",
            initial_state=initial_state,
            require_a100=require_a100,
        )
        if dict(runtime.state_identity()) != initial_state:
            raise RuntimeError("Model state changed during the one-case inference gate.")

        emitted = 0
        for streamed in iter_gate01_p0006_evaluation_cases(
            protocol_path, progress_callback=progress_callback
        ):
            case = streamed.case
            case_hash = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
            receipt_path = receipt_dir / f"case_{case_hash}.json"
            if receipt_path.exists():
                receipt = _load_self_hashed(receipt_path, "case_receipt_sha256")
                _validate_case_receipt(
                    receipt,
                    expected_case_hash=case_hash,
                    expected_protocol_receipt=streamed.case_receipt,
                    run_contract_sha256=run_contract["run_contract_sha256"],
                    root=root,
                )
                streamed.release()
                del case
                emitted += 1
                _emit_count(progress_callback, emitted, resumed=True)
                continue
            started = time.perf_counter()
            outputs: InferenceCaseOutputs | None = None
            try:
                outputs = runtime.infer_case(case, streamed.calibrator, plan=plan)
                metrics = _score_case(
                    outputs, case, metric_fn=metric_fn, device=str(runtime.device)
                )
                slice_artifacts = _render_case_artifacts(
                    case,
                    outputs,
                    plan,
                    case_hash=case_hash,
                    montage_dir=montage_dir,
                    slice_dir=slice_dir,
                )
                receipt_body: dict[str, Any] = {
                "contract_version": CASE_RECEIPT_CONTRACT,
                "run_contract_sha256": run_contract["run_contract_sha256"],
                "case_ordinal": streamed.index,
                "case_identity_sha256": case_hash,
                "protocol_case_receipt_sha256": sha256_json(streamed.case_receipt),
                "contrast": case.source_domain.contrast.value,
                "source_field_t": float(case.source_domain.field_strength_t),
                "target_field_t": float(case.target_domain.field_strength_t),
                "directed_field_pair": (
                    f"{case.source_domain.field_strength_t:g}T->"
                    f"{case.target_domain.field_strength_t:g}T"
                ),
                "metrics": metrics,
                "anatomy": outputs.anatomy,
                "graph": _graph_receipt(outputs.graph),
                "decoded_canonical_sha256": outputs.decoded_canonical_sha256,
                "slice_artifacts": slice_artifacts,
                "inference_seconds": time.perf_counter() - started,
                "training_invoked": False,
                "gradients_enabled": False,
                "optimizer_loaded": False,
                "P0006_training_or_model_selection_use": False,
                }
                receipt_body["case_receipt_sha256"] = sha256_json(receipt_body)
                _write_json_atomic(receipt_path, receipt_body)
            finally:
                if outputs is not None:
                    _release_case_outputs(outputs)
                del outputs
                streamed.release()
                del case
                gc.collect()
            emitted += 1
            _emit_count(progress_callback, emitted, resumed=False)
    if emitted != 60:
        raise ValueError("Inference audit did not process exactly 60 P:0006 cases.")
    if dict(runtime.state_identity()) != initial_state:
        raise RuntimeError("Model state changed during the 60-case inference audit.")
    lpips_post_audit = (
        dict(lpips_integrity_verifier())
        if lpips_integrity_verifier is not None
        else {"unchanged": True, "synthetic_test_evaluator": True}
    )
    if lpips_post_audit.get("unchanged") is not True:
        raise RuntimeError("LPIPS integrity verifier did not seal an unchanged evaluator.")

    receipts = _load_complete_case_receipts(
        receipt_dir, run_contract_sha256=run_contract["run_contract_sha256"], root=root
    )
    metrics_path = root / "stage2_step200_p0006_metrics.csv"
    _write_metrics_csv_or_verify(metrics_path, receipts)
    montage_manifest = _render_aggregate_montages(
        receipts, plan, montage_dir=montage_dir, slice_dir=slice_dir
    )
    summary = _build_inference_summary(
        receipts,
        plan,
        gate,
        runtime,
        audit_implementation_commit=audit_implementation_commit,
        montage_manifest=montage_manifest,
        dependency_receipt=dependency_receipt,
        lpips_receipt=lpips_receipt,
        lpips_post_audit=lpips_post_audit,
    )
    summary_path = root / "stage2_step200_p0006_summary.json"
    _seal_or_verify_json(summary_path, summary, "summary_sha256")
    pdf_path = root / "stage2_step200_p0006_montages.pdf"
    html_path = root / "stage2_step200_p0006_audit.html"
    _render_pdf_or_verify(pdf_path, montage_dir, montage_manifest)
    _render_html_or_verify(html_path, summary, montage_dir, montage_manifest)
    manifest = _seal_artifact_manifest(
        root,
        audit_implementation_commit=audit_implementation_commit,
        plan=plan,
        runtime=runtime,
        gate=gate,
        run_contract=run_contract,
        dependency_receipt=dependency_receipt,
        lpips_receipt=lpips_receipt,
        lpips_post_audit=lpips_post_audit,
    )
    return {
        **summary,
        "artifact_manifest_file_sha256": sha256_file(root / "artifact_manifest.json"),
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "final_stop": "STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION",
    }


def _score_case(
    outputs: InferenceCaseOutputs,
    case: Gate01Case,
    *,
    metric_fn: Callable[[torch.Tensor, torch.Tensor, Sequence[str], str], Mapping[str, float]],
    device: str,
) -> dict[str, dict[str, float]]:
    if tuple(outputs.methods) != _METHOD_ORDER:
        raise ValueError("Inference method inventory or deterministic ordering changed.")
    target = case.target
    scored: dict[str, dict[str, float]] = {}
    for method, prediction in outputs.methods.items():
        values = dict(metric_fn(prediction, target, _METRICS, device))
        if set(values) != set(_METRICS) or any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Official P:0006 metric result is incomplete or nonfinite.")
        values["edge_mae"] = float(gradient_mae(prediction[None], target[None]).cpu())
        scored[method] = {key: float(value) for key, value in values.items()}
    return scored


def _render_case_artifacts(
    case: Gate01Case,
    outputs: InferenceCaseOutputs,
    plan: FrozenStep200InferencePlan,
    *,
    case_hash: str,
    montage_dir: Path,
    slice_dir: Path,
) -> dict[str, Any]:
    spec = plan.montage_specification
    key = (
        case.source_domain.contrast.value,
        float(case.source_domain.field_strength_t),
        float(case.target_domain.field_strength_t),
    )
    selected = {
        (contrast, float(pair["source_field_t"]), float(pair["target_field_t"]))
        for contrast in spec["contrasts"]
        for pair in spec["directed_pairs_per_contrast"]
    }
    artifacts: dict[str, Any] = {"montage_selected": key in selected, "files": []}
    if key in selected:
        methods = {"source": case.source_image, **outputs.methods}
        target_diff = (outputs.methods["raw_unified_step200"] - case.target).abs()
        edge_diff = _edge_difference_volume(outputs.methods["raw_unified_step200"], case.target)
        methods["absolute_difference"] = target_diff
        methods["edge_difference"] = edge_diff
        gate_spec = dict(fixed_montage_specifications())
        gate_spec["display_order"] = list(spec["display_order"])
        collector = Gate01MontageCollector(gate_spec)
        collector.observe(case, methods)
        # The reviewed collector supplies the exact frozen axial positions/range.
        if len(collector.selected) != 1:
            raise ValueError("Frozen montage collector did not select the predeclared case.")
        pngs = _render_orthogonal_case_pngs(case, methods, spec, case_hash, montage_dir)
        artifacts["files"].extend(pngs)
        del target_diff, edge_diff, collector
    if outputs.sweep_slices:
        sweep_path = slice_dir / f"sweep_{case_hash}.npy"
        ordered = np.stack(
            [outputs.sweep_slices[f"{field:g}T"] for field in FIELD_STRENGTHS_T]
        ).astype("<f4", copy=False)
        _write_npy_atomic(sweep_path, ordered)
        artifacts["files"].append(_artifact_identity(sweep_path, slice_dir.parent))
    if outputs.graph.get("selected") is True:
        graph_path = slice_dir / f"graph_{case_hash}.npy"
        ordered = np.stack(
            [
                outputs.graph["direct_slice"],
                outputs.graph["composed_slice"],
                outputs.graph["absolute_difference_slice"],
            ]
        ).astype("<f4", copy=False)
        _write_npy_atomic(graph_path, ordered)
        artifacts["files"].append(_artifact_identity(graph_path, slice_dir.parent))
    return artifacts


def _render_orthogonal_case_pngs(
    case: Gate01Case,
    methods: Mapping[str, torch.Tensor | None],
    spec: Mapping[str, Any],
    case_hash: str,
    montage_dir: Path,
) -> list[dict[str, Any]]:
    arrays = {
        name: _volume_array(case.target if name == "target" else methods[name], name)
        for name in spec["display_order"]
    }
    intensity_names = [name for name in spec["display_order"] if "difference" not in name]
    low = min(float(arrays[name].min()) for name in intensity_names)
    high = max(float(arrays[name].max()) for name in intensity_names)
    artifacts = []
    for plane, axis in spec["planes"].items():
        rows = []
        for position in spec["relative_slice_positions"]:
            panels = []
            for name in spec["display_order"]:
                array = arrays[name]
                index = int(round(float(position) * (array.shape[int(axis)] - 1)))
                panel = np.take(array, index, axis=int(axis))
                if "difference" in name:
                    panel_low, panel_high = 0.0, float(max(panel.max(), 1.0e-12))
                else:
                    panel_low, panel_high = low, high
                panels.append(_normalize_panel(panel, panel_low, panel_high))
            row_height = max(panel.shape[0] for panel in panels)
            row_width = sum(panel.shape[1] for panel in panels) + len(panels) - 1
            row = np.zeros((row_height, row_width), dtype=np.uint8)
            offset = 0
            for panel in panels:
                row[: panel.shape[0], offset : offset + panel.shape[1]] = panel
                offset += panel.shape[1] + 1
            rows.append(row)
        height = sum(row.shape[0] for row in rows) + len(rows) - 1
        width = max(row.shape[1] for row in rows)
        canvas = np.zeros((height, width), dtype=np.uint8)
        offset = 0
        for row in rows:
            canvas[offset : offset + row.shape[0], : row.shape[1]] = row
            offset += row.shape[0] + 1
        path = montage_dir / f"case_{case_hash}_{plane}.png"
        _write_bytes_atomic(path, _encode_grayscale_png(canvas))
        artifacts.append(_artifact_identity(path, montage_dir.parent))
    return artifacts


def _run_or_verify_one_case_gate(
    protocol_path: Path,
    runtime: Step200CaseInferenceRuntime,
    plan: FrozenStep200InferencePlan,
    path: Path,
    *,
    initial_state: Mapping[str, str],
    require_a100: bool,
) -> dict[str, Any]:
    if path.exists():
        gate = _load_self_hashed(path, "gate_sha256")
        if (
            gate.get("inference_plan_sha256") != plan.sha256
            or gate.get("memory_gate_case") != plan.payload["memory_gate_case"]
            or gate.get("gpu_identity") != dict(runtime.gpu_identity)
            or gate.get("model_state_before") != dict(initial_state)
            or gate.get("model_state_after") != dict(initial_state)
            or gate.get("status") != "pass"
        ):
            raise ValueError("Existing one-case inference gate is incompatible.")
        return gate
    if require_a100:
        torch.cuda.reset_peak_memory_stats(runtime.device)
        torch.cuda.synchronize(runtime.device)
    started = time.perf_counter()
    iterator = iter_gate01_p0006_evaluation_cases(protocol_path)
    streamed = None
    for candidate in iterator:
        candidate_case = candidate.case
        selector = plan.payload["memory_gate_case"]
        selected = (
            candidate_case.source_domain.contrast.value == selector["contrast"]
            and float(candidate_case.source_domain.field_strength_t)
            == float(selector["source_field_t"])
            and float(candidate_case.target_domain.field_strength_t)
            == float(selector["target_field_t"])
        )
        del candidate_case
        if selected:
            streamed = candidate
            break
        candidate.release()
    if streamed is None:
        iterator.close()
        raise ValueError("Frozen one-case memory-gate domain cell is absent from P:0006.")
    outputs: InferenceCaseOutputs | None = None
    try:
        outputs = runtime.infer_case(streamed.case, streamed.calibrator, plan=plan)
    finally:
        if outputs is not None:
            _release_case_outputs(outputs)
        del outputs
        streamed.release()
        iterator.close()
    if require_a100:
        torch.cuda.synchronize(runtime.device)
        peak_allocated = int(torch.cuda.max_memory_allocated(runtime.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(runtime.device))
    else:
        peak_allocated = 0
        peak_reserved = 0
    state_after = dict(runtime.state_identity())
    if state_after != dict(initial_state):
        raise RuntimeError("Model state changed during one-case inference gate.")
    if require_a100 and peak_allocated > A100_MAX_PEAK_ALLOCATED_BYTES:
        raise RuntimeError("One-case A100 inference gate exceeded the 72 GiB limit.")
    body: dict[str, Any] = {
        "contract_version": MEMORY_GATE_CONTRACT,
        "status": "pass",
        "case_count": 1,
        "memory_gate_case": dict(plan.payload["memory_gate_case"]),
        "inference_plan_sha256": plan.sha256,
        "gpu_identity": dict(runtime.gpu_identity),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_limit_bytes": A100_MAX_PEAK_ALLOCATED_BYTES,
        "elapsed_seconds": time.perf_counter() - started,
        "model_state_before": dict(initial_state),
        "model_state_after": state_after,
        "gradients_enabled": False,
        "optimizer_loaded": False,
        "training_invoked": False,
    }
    body["gate_sha256"] = sha256_json(body)
    _write_json_atomic(path, body)
    return body


def _load_complete_case_receipts(
    receipt_dir: Path, *, run_contract_sha256: str, root: Path
) -> list[dict[str, Any]]:
    paths = sorted(receipt_dir.glob("case_*.json"))
    if len(paths) != 60:
        raise ValueError("Inference audit requires exactly 60 immutable case receipts.")
    receipts = []
    identities = set()
    cells = set()
    for path in paths:
        receipt = _load_self_hashed(path, "case_receipt_sha256")
        _validate_case_receipt(
            receipt,
            expected_case_hash=str(receipt.get("case_identity_sha256", "")),
            expected_protocol_receipt=None,
            run_contract_sha256=run_contract_sha256,
            root=root,
        )
        if receipt["case_identity_sha256"] in identities:
            raise ValueError("Duplicate inference case receipt identity.")
        identities.add(receipt["case_identity_sha256"])
        cells.add(
            (
                receipt["contrast"],
                float(receipt["source_field_t"]),
                float(receipt["target_field_t"]),
            )
        )
        receipts.append(receipt)
    if len(identities) != 60 or len(cells) != 60:
        raise ValueError("Inference receipt graph is incomplete or ambiguous.")
    return sorted(receipts, key=lambda item: int(item["case_ordinal"]))


def _validate_case_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_case_hash: str,
    expected_protocol_receipt: Mapping[str, Any] | None,
    run_contract_sha256: str,
    root: Path,
) -> None:
    if (
        receipt.get("contract_version") != CASE_RECEIPT_CONTRACT
        or receipt.get("run_contract_sha256") != run_contract_sha256
        or receipt.get("case_identity_sha256") != expected_case_hash
        or _SHA256_RE.fullmatch(expected_case_hash) is None
        or receipt.get("training_invoked") is not False
        or receipt.get("gradients_enabled") is not False
        or receipt.get("optimizer_loaded") is not False
        or receipt.get("P0006_training_or_model_selection_use") is not False
    ):
        raise ValueError("Inference case receipt identity or safety contract changed.")
    if expected_protocol_receipt is not None and receipt.get(
        "protocol_case_receipt_sha256"
    ) != sha256_json(expected_protocol_receipt):
        raise ValueError("Inference case receipt belongs to different P:0006 arrays.")
    metrics = receipt.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METHOD_ORDER):
        raise ValueError("Inference case receipt method inventory changed.")
    for values in metrics.values():
        if not isinstance(values, Mapping) or set(values) != {*_METRICS, "edge_mae"}:
            raise ValueError("Inference case receipt metric inventory changed.")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Inference case receipt contains a nonfinite metric.")
    artifacts = receipt.get("slice_artifacts", {}).get("files", [])
    if not isinstance(artifacts, list):
        raise ValueError("Inference case slice-artifact inventory is malformed.")
    for item in artifacts:
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Inference case slice artifact path is unsafe.")
        path = (root / relative).resolve(strict=True)
        path.relative_to(root.resolve())
        if sha256_file(path) != item.get("file_sha256") or path.stat().st_size != item.get(
            "size_bytes"
        ):
            raise ValueError("Inference case slice artifact changed.")


def _write_metrics_csv_or_verify(path: Path, receipts: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for receipt in receipts:
        for method in _METHOD_ORDER:
            rows.append(
                {
                    "case_identity_sha256": receipt["case_identity_sha256"],
                    "contrast": receipt["contrast"],
                    "source_field_t": receipt["source_field_t"],
                    "target_field_t": receipt["target_field_t"],
                    "directed_field_pair": receipt["directed_field_pair"],
                    "method": method,
                    **receipt["metrics"][method],
                }
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _write_or_verify_bytes(path, buffer.getvalue().encode("utf-8"))


def _build_inference_summary(
    receipts: Sequence[Mapping[str, Any]],
    plan: FrozenStep200InferencePlan,
    gate: Mapping[str, Any],
    runtime: Step200CaseInferenceRuntime,
    *,
    audit_implementation_commit: str,
    montage_manifest: Mapping[str, Any],
    dependency_receipt: Mapping[str, Any],
    lpips_receipt: Mapping[str, Any],
    lpips_post_audit: Mapping[str, Any],
) -> dict[str, Any]:
    def reduce(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            method: {
                metric: float(np.mean([item["metrics"][method][metric] for item in items]))
                for metric in (*_METRICS, "edge_mae")
            }
            for method in _METHOD_ORDER
        }
    by_contrast = {
        contrast.value: reduce([item for item in receipts if item["contrast"] == contrast.value])
        for contrast in CONTRASTS
    }
    pair_labels = sorted({str(item["directed_field_pair"]) for item in receipts})
    by_pair = {
        label: reduce([item for item in receipts if item["directed_field_pair"] == label])
        for label in pair_labels
    }
    comparisons = {}
    for baseline in ("raw_identity", "raw_original_sb_v2"):
        comparisons[baseline] = {}
        for metric, lower in _LOWER_IS_BETTER.items():
            differences = [
                float(item["metrics"]["raw_unified_step200"][metric])
                - float(item["metrics"][baseline][metric])
                for item in receipts
            ]
            wins = sum((value < -1.0e-12 if lower else value > 1.0e-12) for value in differences)
            losses = sum((value > 1.0e-12 if lower else value < -1.0e-12) for value in differences)
            comparisons[baseline][metric] = {
                "mean_unified_minus_baseline": float(np.mean(differences)),
                "wins": wins,
                "ties": len(differences) - wins - losses,
                "losses": losses,
            }
    graph_rows = [item["graph"] for item in receipts if item["graph"].get("selected") is True]
    body: dict[str, Any] = {
        "contract_version": INFERENCE_AUDIT_CONTRACT,
        "scientific_role": P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
        "evidence_interpretation": P0006_EVIDENCE_LIMITATION,
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "audit_implementation_commit": audit_implementation_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "run_fingerprint": RUN_FINGERPRINT,
        "P0006_protocol_file_sha256": P0006_PROTOCOL_FILE_SHA256,
        "P0006_protocol_sha256": P0006_PROTOCOL_SHA256,
        "evaluation_readiness_file_sha256": EVALUATION_READINESS_FILE_SHA256,
        "evaluation_readiness_sha256": EVALUATION_READINESS_SHA256,
        "inference_plan_sha256": plan.sha256,
        "montage_specification_sha256": plan.montage_specification["montage_specification_sha256"],
        "metric_contract_version": METRIC_CONTRACT,
        "case_count": 60,
        "acquisition_count": 15,
        "directed_pair_count": 60,
        "overall": reduce(receipts),
        "by_contrast": by_contrast,
        "by_directed_field_pair": by_pair,
        "paired_descriptive_differences": comparisons,
        "graph_path_count": len(graph_rows),
        "graph_consistency": {
            "mean_direct_vs_composed_l1": float(np.mean([item["direct_vs_composed_l1"] for item in graph_rows])),
            "mean_direct_vs_composed_mse": float(np.mean([item["direct_vs_composed_mse"] for item in graph_rows])),
        },
        "montage_manifest": dict(montage_manifest),
        "GPU_identity": dict(runtime.gpu_identity),
        "peak_allocated_bytes": gate["peak_allocated_bytes"],
        "peak_reserved_bytes": gate["peak_reserved_bytes"],
        "one_case_inference_seconds": gate["elapsed_seconds"],
        "projected_60_case_inference_seconds": float(gate["elapsed_seconds"]) * 60.0,
        "lpips_initialization_seconds": lpips_receipt.get("initialization_seconds"),
        "dependency_download_observed": dependency_receipt.get(
            "dependency_download_observed", False
        ),
        "alexnet_weight_downloaded": lpips_receipt.get(
            "alexnet_weight_downloaded", False
        ),
        "lpips_linear_weight_downloaded": lpips_receipt.get(
            "lpips_linear_weight_downloaded", False
        ),
        "dependency_environment_sha256": sha256_json(dependency_receipt),
        "lpips_provenance_sha256": sha256_json(lpips_receipt),
        "lpips_post_audit": dict(lpips_post_audit),
        "training_invoked": False,
        "gradients_enabled": False,
        "optimizer_loaded": False,
        "P0006_training_or_model_selection_use": False,
        "population_or_generalization_claims_authorized": False,
        "long_run_training_authorized": False,
        "P0007_access": False,
        "P0009_status": P0009_CONFIRMATION_STATUS,
        "P0009_access": False,
        "inferential_p_values_or_confidence_claims_reported": False,
        "model_status": "step200_full_objective_pilot_not_converged",
        "final_stop": "STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION",
    }
    body["summary_sha256"] = sha256_json(body)
    return body


def _render_aggregate_montages(
    receipts: Sequence[Mapping[str, Any]],
    plan: FrozenStep200InferencePlan,
    *,
    montage_dir: Path,
    slice_dir: Path,
) -> dict[str, Any]:
    entries = []
    for path in sorted(montage_dir.glob("case_*.png")):
        entries.append(_artifact_identity(path, montage_dir.parent))
    for contrast in CONTRASTS:
        matches = [
            item for item in receipts
            if item["contrast"] == contrast.value and float(item["source_field_t"]) == 3.0
        ]
        sweep_files = []
        for item in matches:
            for artifact in item["slice_artifacts"]["files"]:
                if Path(artifact["relative_path"]).name.startswith("sweep_"):
                    sweep_files.append(montage_dir.parent / artifact["relative_path"])
        if len(sweep_files) != 4:
            raise ValueError("Frozen field sweep requires four directed cases from the 3T source.")
        arrays = [np.load(path, allow_pickle=False) for path in sorted(sweep_files)]
        # Every directed case from the same source must reproduce the exact five sweep slices.
        if any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
            raise ValueError("Deterministic field sweep differs across the same source acquisition.")
        canvas = _row_canvas(arrays[0])
        output = montage_dir / f"field_sweep_{contrast.value.replace('-', '_')}.png"
        _write_or_verify_bytes(output, _encode_grayscale_png(canvas))
        entries.append(_artifact_identity(output, montage_dir.parent))
    graph_files = sorted(slice_dir.glob("graph_*.npy"))
    if len(graph_files) != 6:
        raise ValueError("Frozen graph panel set requires exactly six paths.")
    for path in graph_files:
        array = np.load(path, allow_pickle=False)
        output = montage_dir / f"{path.stem}.png"
        _write_or_verify_bytes(output, _encode_grayscale_png(_row_canvas(array)))
        entries.append(_artifact_identity(output, montage_dir.parent))
    manifest = {
        "contract_version": "stage2-step200-p0006-montage-render-v1",
        "montage_specification_sha256": plan.montage_specification["montage_specification_sha256"],
        "entry_count": len(entries),
        "entries": sorted(entries, key=lambda item: item["relative_path"]),
    }
    manifest["montage_manifest_sha256"] = sha256_json(manifest)
    _seal_or_verify_json(montage_dir / "montage_manifest.json", manifest, "montage_manifest_sha256")
    return manifest


def _render_pdf_or_verify(
    path: Path, montage_dir: Path, montage_manifest: Mapping[str, Any]
) -> None:
    if path.exists():
        if path.stat().st_size <= 0:
            raise ValueError("Existing montage PDF is empty.")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with PdfPages(temporary) as pdf:
            for entry in montage_manifest["entries"]:
                if not str(entry["relative_path"]).endswith(".png"):
                    continue
                image = plt.imread(montage_dir.parent / entry["relative_path"])
                fig, axis = plt.subplots(figsize=(12, 7))
                axis.imshow(image, cmap="gray", interpolation="none")
                axis.axis("off")
                axis.set_title(Path(entry["relative_path"]).stem)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_html_or_verify(
    path: Path,
    summary: Mapping[str, Any],
    montage_dir: Path,
    montage_manifest: Mapping[str, Any],
) -> None:
    images = []
    for entry in montage_manifest["entries"]:
        if not str(entry["relative_path"]).endswith(".png"):
            continue
        payload = (montage_dir.parent / entry["relative_path"]).read_bytes()
        images.append(
            f'<figure><img alt="predeclared montage" src="data:image/png;base64,{base64.b64encode(payload).decode()}"><figcaption>{Path(entry["relative_path"]).stem}</figcaption></figure>'
        )
    summary_json = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    escaped = summary_json.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Stage-2 step-200 P:0006 audit</title><style>:root{{color-scheme:light dark}}body{{font:16px system-ui;max-width:1500px;margin:auto;padding:2rem;background:Canvas;color:CanvasText}}img{{max-width:100%;image-rendering:auto;border:1px solid GrayText}}figure{{margin:2rem 0}}pre{{white-space:pre-wrap}}.warning{{border-left:.4rem solid #c77d00;padding:1rem}}</style></head><body><h1>Stage-2 step-200 P:0006 inference-only audit</h1><p class="warning">Development/model-assessment evidence only. One traveller protocol; no population/generalization claim. The checkpoint is a pilot, not a converged model.</p>{''.join(images)}<h2>Summary</h2><pre>{escaped}</pre><p><strong>STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION</strong></p></body></html>"""
    _write_or_verify_bytes(path, html.encode("utf-8"))


def _seal_artifact_manifest(
    root: Path,
    *,
    audit_implementation_commit: str,
    plan: FrozenStep200InferencePlan,
    runtime: Step200CaseInferenceRuntime,
    gate: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    dependency_receipt: Mapping[str, Any],
    lpips_receipt: Mapping[str, Any],
    lpips_post_audit: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        outputs[relative] = {"file_sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    body: dict[str, Any] = {
        "contract_version": ARTIFACT_MANIFEST_CONTRACT,
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "audit_implementation_commit": audit_implementation_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "run_fingerprint": RUN_FINGERPRINT,
        "P0006_protocol_file_sha256": P0006_PROTOCOL_FILE_SHA256,
        "P0006_protocol_sha256": P0006_PROTOCOL_SHA256,
        "inference_plan_sha256": plan.sha256,
        "montage_specification_sha256": plan.montage_specification["montage_specification_sha256"],
        "metric_contract_version": METRIC_CONTRACT,
        "GPU_identity": dict(runtime.gpu_identity),
        "peak_allocated_bytes": gate["peak_allocated_bytes"],
        "peak_reserved_bytes": gate["peak_reserved_bytes"],
        "one_case_inference_seconds": gate["elapsed_seconds"],
        "projected_60_case_inference_seconds": float(gate["elapsed_seconds"]) * 60.0,
        "dependency_environment": dict(dependency_receipt),
        "lpips_provenance": dict(lpips_receipt),
        "lpips_post_audit": dict(lpips_post_audit),
        "run_contract_sha256": run_contract["run_contract_sha256"],
        "render_library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": _distribution_version("matplotlib"),
            "nibabel": _distribution_version("nibabel"),
            "scikit-image": _distribution_version("scikit-image"),
            "lpips": _distribution_version("lpips"),
        },
        "outputs": outputs,
        "training_invoked": False,
        "gradients_enabled": False,
        "optimizer_loaded": False,
        "P0006_training_or_model_selection_use": False,
        "population_or_generalization_claims_authorized": False,
        "long_run_training_authorized": False,
    }
    body["manifest_sha256"] = sha256_json(body)
    _seal_or_verify_json(root / "artifact_manifest.json", body, "manifest_sha256")
    return body


def _run_contract(
    plan: FrozenStep200InferencePlan,
    runtime: Step200CaseInferenceRuntime,
    *,
    audit_implementation_commit: str,
    dependency_receipt: Mapping[str, Any],
    lpips_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "contract_version": "stage2-step200-p0006-inference-run-v2",
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "audit_implementation_commit": audit_implementation_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "run_fingerprint": RUN_FINGERPRINT,
        "protocol_sha256": P0006_PROTOCOL_SHA256,
        "inference_plan_sha256": plan.sha256,
        "model_state": dict(runtime.state_identity()),
        "GPU_identity": dict(runtime.gpu_identity),
        "dependency_environment_sha256": sha256_json(dependency_receipt),
        "lpips_provenance_sha256": sha256_json(lpips_receipt),
        "training_invoked": False,
        "gradients_enabled": False,
        "optimizer_loaded": False,
    }
    body["run_contract_sha256"] = sha256_json(body)
    return body


def _seal_or_reuse_runtime_receipt(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    current = copy.deepcopy(dict(provenance))
    if path.exists():
        stored = _load_self_hashed(path, "receipt_sha256")
        if (
            stored.get("contract_version")
            != "stage2-step200-inference-runtime-provenance-receipt-v1"
            or stored.get("label") != label
            or not isinstance(stored.get("provenance"), Mapping)
            or _stable_runtime_provenance(stored["provenance"])
            != _stable_runtime_provenance(current)
        ):
            raise ValueError(f"Existing {label} receipt is incompatible.")
        return copy.deepcopy(dict(stored["provenance"]))
    body: dict[str, Any] = {
        "contract_version": "stage2-step200-inference-runtime-provenance-receipt-v1",
        "label": label,
        "provenance": current,
    }
    body["receipt_sha256"] = sha256_json(body)
    _write_json_atomic(path, body)
    return current


def _stable_runtime_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for key in (
        "initialization_seconds",
        "alexnet_weight_downloaded",
        "dependency_download_observed",
        "pip_install_invoked",
    ):
        result.pop(key, None)
    return result


def _validate_scientific_role(protocol: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    if (
        protocol.get("data_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
        or protocol.get("training_or_model_selection_use") is not False
        or protocol.get("population_or_generalization_claims_authorized") is not False
        or protocol.get("P0009_executed") is not False
        or readiness.get("long_run_training_authorized") is not False
        or readiness.get("population_or_generalization_claims_authorized") is not False
    ):
        raise ValueError("P:0006 scientific role or readiness safety boundary changed.")


def _validate_a100_identity(runtime: Step200CaseInferenceRuntime) -> None:
    identity = runtime.gpu_identity
    if "A100" not in str(identity.get("name", "")) or int(identity.get("total_memory_bytes", 0)) < 79 * 1024**3:
        raise RuntimeError("First inference qualification requires NVIDIA A100 80 GB.")


def _case_seed(plan: FrozenStep200InferencePlan, case: Gate01Case) -> int:
    matches = [
        item
        for item in plan.payload["case_seeds"]
        if item["source_image_sha256"] == case.array_sha256["source_image"]
        and item["source_domain"] == case.source_domain.to_dict()
        and item["target_domain"] == case.target_domain.to_dict()
    ]
    if len(matches) != 1:
        raise ValueError("Frozen inference seed plan does not uniquely cover the case.")
    return int(matches[0]["seed"])


def _graph_intermediate(plan: FrozenStep200InferencePlan, case: Gate01Case) -> float | None:
    matches = [
        item for item in plan.payload["graph_paths"]
        if item["contrast"] == case.source_domain.contrast.value
        and float(item["source_field_t"]) == float(case.source_domain.field_strength_t)
        and float(item["target_field_t"]) == float(case.target_domain.field_strength_t)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Frozen graph path selection is ambiguous.")
    return float(matches[0]["intermediate_field_t"])


def _graph_receipt(graph: Mapping[str, Any]) -> dict[str, Any]:
    if graph.get("selected") is not True:
        return {"selected": False}
    return {
        "selected": True,
        "intermediate_field_t": graph["intermediate_field_t"],
        "direct_vs_composed_l1": graph["direct_vs_composed_l1"],
        "direct_vs_composed_mse": graph["direct_vs_composed_mse"],
    }


def _release_case_outputs(outputs: InferenceCaseOutputs) -> None:
    outputs.methods.clear()
    outputs.sweep_slices.clear()
    outputs.graph.clear()
    outputs.anatomy.clear()


def _emit_count(
    callback: Callable[[dict[str, Any]], None] | None, count: int, *, resumed: bool
) -> None:
    if callback is not None:
        callback(
            {
                "stage": "stage2_step200_p0006_inference_audit",
                "status": "periodic",
                "case_count": count,
                "expected_case_count": 60,
                "case_receipt_reused": int(resumed),
            }
        )


def _edge_difference_volume(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.float()
    tgt = target.float()
    result = torch.zeros_like(pred)
    for dim in range(max(0, pred.ndim - 3), pred.ndim):
        if pred.shape[dim] < 2:
            continue
        diff = (pred.diff(dim=dim) - tgt.diff(dim=dim)).abs()
        slices = [slice(None)] * pred.ndim
        slices[dim] = slice(0, -1)
        result[tuple(slices)] += diff
    return result


def _middle_slice(value: torch.Tensor) -> np.ndarray:
    array = _volume_array(value, "slice source")
    return np.ascontiguousarray(array[..., array.shape[-1] // 2], dtype=np.float32)


def _volume_array(value: torch.Tensor | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Montage volume {name!r} is missing.")
    array = value.detach().cpu().float().numpy().squeeze()
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Montage volume {name!r} is not one finite 3-D array.")
    return np.asarray(array, dtype=np.float32)


def _row_canvas(panels: np.ndarray) -> np.ndarray:
    normalized = []
    for panel in panels:
        normalized.append(_normalize_panel(panel, float(panel.min()), float(panel.max())))
    height = max(panel.shape[0] for panel in normalized)
    width = sum(panel.shape[1] for panel in normalized) + len(normalized) - 1
    canvas = np.zeros((height, width), dtype=np.uint8)
    offset = 0
    for panel in normalized:
        canvas[: panel.shape[0], offset : offset + panel.shape[1]] = panel
        offset += panel.shape[1] + 1
    return canvas


def _artifact_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _module_state_sha256(module: torch.nn.Module) -> str:
    return sha256_json(
        {
            name: storage_tensor_sha256(value.detach().cpu().contiguous())
            for name, value in sorted(module.state_dict().items())
        }
    )


def _decode(decoder: torch.nn.Module, latent: torch.Tensor, domains: Sequence[Domain]) -> torch.Tensor:
    if hasattr(decoder, "decode"):
        return decoder.decode(latent, domains)  # type: ignore[attr-defined,no-any-return]
    return decoder(latent)  # type: ignore[no-any-return]


def _kl_vae_kwargs(model_config: Mapping[str, Any], component: str) -> dict[str, Any]:
    shared = {"base_channels", "latent_channels", "spatial_dims", "activation", "use_norm", "num_res_blocks"}
    kwargs = {key: value for key, value in model_config.items() if key in shared}
    if component == "encoder" and "in_channels" in model_config:
        kwargs["in_channels"] = model_config["in_channels"]
    if component == "decoder" and "out_channels" in model_config:
        kwargs["out_channels"] = model_config["out_channels"]
    if component == "decoder" and "output_activation" in model_config:
        kwargs["output_activation"] = model_config["output_activation"]
    if component == "decoder" and "domain_conditioning_dim" in model_config:
        kwargs["domain_conditioning_dim"] = model_config["domain_conditioning_dim"]
    return kwargs


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes().decode("utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return dict(payload)


def _load_self_hashed(path: Path, hash_key: str) -> dict[str, Any]:
    payload = _load_json(path)
    body = dict(payload)
    stored = body.pop(hash_key, None)
    if stored != sha256_json(body):
        raise ValueError(f"Self-hash mismatch: {path.name}")
    return payload


def _seal_or_verify_json(path: Path, payload: Mapping[str, Any], hash_key: str) -> None:
    if payload.get(hash_key) != sha256_json({key: value for key, value in payload.items() if key != hash_key}):
        raise ValueError(f"Invalid self-hashed payload for {path.name}.")
    if path.exists():
        if _load_json(path) != dict(payload):
            raise ValueError(f"Existing immutable audit JSON changed: {path.name}")
        return
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _write_or_verify_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing immutable audit artifact changed: {path.name}")
        return
    _write_bytes_atomic(path, payload)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite audit artifact: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        with temporary.open("ab") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if not np.array_equal(existing, array):
            raise ValueError(f"Existing predeclared slice changed: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_git(value: str, label: str) -> None:
    if _GIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git identity.")


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


__all__ = [
    "ARTIFACT_MANIFEST_CONTRACT",
    "CASE_RECEIPT_CONTRACT",
    "CHECKPOINT_SHA256",
    "FrozenStep200InferencePlan",
    "INFERENCE_AUDIT_CONTRACT",
    "INFERENCE_PLAN_CONTRACT",
    "InferenceCaseOutputs",
    "MEMORY_GATE_CONTRACT",
    "MONTAGE_SPECIFICATION_CONTRACT",
    "P0006_PROTOCOL_FILE_SHA256",
    "P0006_PROTOCOL_SHA256",
    "RUN_FINGERPRINT",
    "Step200CaseInferenceRuntime",
    "TRAINING_EVIDENCE_COMMIT",
    "UnifiedStep200InferenceRuntime",
    "build_frozen_step200_inference_plan",
    "load_unified_step200_inference_runtime",
    "run_step200_p0006_inference_audit",
]
