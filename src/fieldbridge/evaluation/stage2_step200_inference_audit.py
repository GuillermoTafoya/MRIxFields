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
import inspect
import io
import json
import math
import os
import platform
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data import photometry_factorization as _photometry
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
    STAGE1_RUN_C_CONFIG_SHA256,
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
    P0006_IDENTITY_SHA256,
    P0006_SUBJECT_GROUP,
    P0009_CONFIRMATION_STATUS,
    iter_gate01_p0006_evaluation_cases,
)
from fieldbridge.evaluation.stage2_unified_preflight import (
    LONG_RUN_EVALUATION_READINESS_CONTRACT,
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


INFERENCE_AUDIT_CONTRACT = "stage2-step200-p0006-inference-audit-v7"
INFERENCE_PLAN_CONTRACT = "stage2-step200-p0006-frozen-inference-plan-v1"
MONTAGE_SPECIFICATION_CONTRACT = "stage2-step200-p0006-frozen-montage-v1"
CASE_RECEIPT_CONTRACT = "stage2-step200-p0006-inference-case-receipt-v2"
MEMORY_GATE_CONTRACT = "stage2-step200-p0006-a100-one-case-gate-v3"
ARTIFACT_MANIFEST_CONTRACT = "stage2-step200-p0006-audit-artifact-manifest-v7"
METRIC_CONTRACT = "stage2-step200-p0006-descriptive-official-task3-v1"
FROZEN_STAGE1_VAE_PROVENANCE_CONTRACT = "stage2-step200-frozen-stage1-vae-provenance-v1"
REVIEWED_PHOTOMETRY_NAMESPACE_PROVENANCE_CONTRACT = (
    "stage2-step200-reviewed-photometry-namespace-provenance-v1"
)
FROZEN_P0006_SCIENTIFIC_ROLE_PREFLIGHT_CONTRACT = (
    "stage2-step200-frozen-p0006-scientific-role-preflight-v1"
)
FULL_VOLUME_LAYOUT_ADAPTER_CONTRACT = (
    "stage2-step200-full-volume-layout-adapter-v1"
)
DECODER_EVALUATION_RANGE_CONTRACT = (
    "stage2-step200-decoder-evaluation-range-v1"
)
TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
CHECKPOINT_SHA256 = "09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021"
RUN_FINGERPRINT = "c814c948a5b85bd3a694db7c8e074894e97c16a96a36acbfa6f370faf2dac0aa"
P0006_PROTOCOL_FILE_SHA256 = "3c11092a4a5e5342726947d705eca8fd8c52a70b82b96892529fe564c5f5f809"
P0006_PROTOCOL_SHA256 = "2cd8e17207175f6a8f1f11f8afd748beca11ae0f47a08a0b2e1529a2272274e4"
EVALUATION_READINESS_FILE_SHA256 = "dc6695d3a9d9f69749af1421e92a3b008f240e147a59619957cd4888af71d7d2"
EVALUATION_READINESS_SHA256 = "ff5d4e8e80f48fecdf0e320fd56e9fd9431145798908b56d7e363b5760d8e0ec"
INFERENCE_SEED = 20_260_825
A100_MAX_PEAK_ALLOCATED_BYTES = 72 * 1024**3
STAGE1_RUN_C_CONFIG_SIZE_BYTES = 4_290
STAGE1_RUN_C_CONFIG_BASENAME = "stage1-run-c.yaml"
REVIEWED_PHOTOMETRY_ROLE = "frozen_stage2_photometry_factorization_v1"
REVIEWED_PHOTOMETRY_BASENAME = "stage2_photometry_factorization_v1.json"
REVIEWED_PHOTOMETRY_FILE_SHA256 = (
    "de5bd993f34056873a5bc176c9320ff55040c80fa888c224e037a520478009ca"
)
REVIEWED_PHOTOMETRY_ARTIFACT_SHA256 = (
    "076baade3b1f4250124071dc572c40d012b0345f6a62c3e7c1de4283eb2ee923"
)
REVIEWED_PHOTOMETRY_PRODUCTION_COMMIT = "1ca2b4a170ad8186b02d44a814e279c9c0e02cb5"
PROTECTED_PHOTOMETRY_MODULE_SHA256 = (
    "e4be9c63f4e2b5044678ea1204fee5fe1dfbeda4508a71cf91af159d7e4c6f5e"
)
HISTORICAL_PHOTOMETRY_OPERATOR_OVERLAY_SHA256 = (
    "a43dfbfa8febb89daf1308fd8b945957cc200e62d8ca286555634c9437aef5d4"
)
REVIEWED_NAMESPACE_PREDICATE_SOURCE_SHA256 = (
    "3d5e4dc687aecd2b330f573f6d97b342327477db6f1a12783b43c94e8818b81f"
)
REVIEWED_PHOTOMETRY_ACCEPTED_RECORD_COUNT = 1_560
REVIEWED_PHOTOMETRY_PROSPECTIVE_EXCLUDED_COUNT = 30
REVIEWED_PHOTOMETRY_RETROSPECTIVE_COLLISION_COUNT = 6
REVIEWED_PHOTOMETRY_COLLISION_GROUP_COUNTS = {
    "0f8f0768d3538fb451949ace1532421724771ccf3d18de3f6ac778a033d20d64": 3,
    "85c0a97a21f7b9a87027508520bd930a0195d267a78f42ed13a1e0e84d989786": 3,
}
_PINNED_PHOTOMETRY_OPERATOR_OVERLAY_SHA256 = (
    "a43dfbfa8febb89daf1308fd8b945957cc200e62d8ca286555634c9437aef5d4"
)
_PINNED_NAMESPACE_PREDICATE_SOURCE_SHA256 = (
    "3d5e4dc687aecd2b330f573f6d97b342327477db6f1a12783b43c94e8818b81f"
)
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


def _namespace_aware_forbidden_traveller(record_identity, subject_identity):
    upper = str(record_identity).upper()
    return any(
        f"P_{traveller}" in upper or f"P:{traveller}" in upper
        for traveller in _photometry.FORBIDDEN_TRAVELLER_IDS
    )


_BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER = _photometry._is_forbidden_traveller
_REVIEWED_NAMESPACE_PREDICATE = _namespace_aware_forbidden_traveller
_REVIEWED_NAMESPACE_PREDICATE_SOURCE = inspect.getsource(
    _namespace_aware_forbidden_traveller
)
_REVIEWED_NAMESPACE_PREDICATE_CODE = (
    _namespace_aware_forbidden_traveller.__code__.co_code,
    _namespace_aware_forbidden_traveller.__code__.co_consts,
    _namespace_aware_forbidden_traveller.__code__.co_names,
)
_PHOTOMETRY_NAMESPACE_LOCK = threading.Lock()
_PHOTOMETRY_NAMESPACE_SCOPE_ACTIVE = False


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
class FrozenP0006ScientificRolePreflight:
    """Pinned small-file P:0006 protocol/readiness authentication."""

    protocol_path: Path
    readiness_path: Path
    protocol: dict[str, Any]
    readiness: dict[str, Any]
    protocol_file_sha256: str
    protocol_sha256: str
    readiness_file_sha256: str
    readiness_sha256: str

    def sanitized_provenance(self) -> dict[str, Any]:
        return {
            "contract_version": FROZEN_P0006_SCIENTIFIC_ROLE_PREFLIGHT_CONTRACT,
            "protocol_contract_version": self.protocol["contract_version"],
            "protocol_file_sha256": self.protocol_file_sha256,
            "protocol_sha256": self.protocol_sha256,
            "readiness_contract_version": self.readiness["contract_version"],
            "readiness_file_sha256": self.readiness_file_sha256,
            "readiness_sha256": self.readiness_sha256,
            "evaluation_role": self.readiness["evaluation_role"],
            "evidence_interpretation": self.readiness["evidence_interpretation"],
            "traveller_identity_sha256": self.protocol["traveller_identity_sha256"],
            "acquisition_count": self.protocol["acquisition_count"],
            "directed_pair_count": self.protocol["directed_pair_count"],
            "wrong_target_reference_count": self.protocol[
                "wrong_target_reference_count"
            ],
            "factored_bank_P_record_count": self.readiness[
                "factored_bank_P_record_count"
            ],
            "unpaired_validation_P_endpoint_count": self.readiness[
                "unpaired_validation_P_endpoint_count"
            ],
            "long_run_authorized_by_evaluation_path": True,
            "long_run_training_authorized": False,
            "prospective_training_or_model_selection_use": False,
            "population_or_generalization_claims_authorized": False,
            "P0009_confirmation_status": self.readiness[
                "P0009_confirmation_status"
            ],
            "P0009_executed": False,
        }


@dataclass(frozen=True, slots=True)
class AdaptedFullVolume:
    """Representation-only view of one 3-D volume with singleton leading axes."""

    tensor: torch.Tensor
    role: str
    authenticated_shape: tuple[int, ...]
    spatial_shape: tuple[int, int, int]

    @property
    def authenticated_rank(self) -> int:
        return len(self.authenticated_shape)

    @property
    def leading_singleton_axis_count(self) -> int:
        return self.authenticated_rank - 3

    def model_batch(self) -> torch.Tensor:
        expected = (1, 1, *self.spatial_shape)
        value = (
            self.tensor
            if self.authenticated_shape == expected
            else self.tensor.reshape(expected)
        )
        if tuple(value.shape) != expected:
            raise RuntimeError(f"{self.role} model-batch adaptation changed shape.")
        if not torch.equal(value.reshape(self.authenticated_shape), self.tensor):
            raise RuntimeError(f"{self.role} model-batch adaptation changed values.")
        return value

    def spatial_volume(self) -> torch.Tensor:
        value = self.tensor.reshape(self.spatial_shape)
        if not torch.equal(value.reshape(self.authenticated_shape), self.tensor):
            raise RuntimeError(f"{self.role} spatial adaptation changed values.")
        return value

    def restore_decoder_output(
        self, decoded_model_batch: torch.Tensor, *, role: str
    ) -> torch.Tensor:
        expected = (1, 1, *self.spatial_shape)
        if not isinstance(decoded_model_batch, torch.Tensor):
            raise TypeError(f"{role} decoder output must be a torch.Tensor.")
        if tuple(decoded_model_batch.shape) != expected:
            raise ValueError(
                f"{role} decoder output must be one channel with unchanged spatial axes: "
                f"{tuple(decoded_model_batch.shape)} != {expected}."
            )
        if not bool(torch.isfinite(decoded_model_batch).all()):
            raise ValueError(f"{role} decoder output contains non-finite values.")
        restored = decoded_model_batch.reshape(self.authenticated_shape)
        if not torch.equal(restored.reshape(expected), decoded_model_batch):
            raise RuntimeError(f"{role} decoder-output restoration changed values.")
        return restored

    def sanitized_provenance(self, *, source_rank: int) -> dict[str, Any]:
        return {
            "contract_version": FULL_VOLUME_LAYOUT_ADAPTER_CONTRACT,
            "authenticated_source_rank": source_rank,
            "authenticated_canonical_rank": self.authenticated_rank,
            "leading_singleton_axis_count": self.leading_singleton_axis_count,
            "model_input_rank": 5,
            "representation_only_reshape": True,
            "spatial_axes_preserved": True,
            "resampling_performed": False,
            "cropping_performed": False,
            "padding_performed": False,
            "transposition_performed": False,
            "output_restored_to_authenticated_case_representation": True,
            "training_invoked": False,
        }


def adapt_full_volume_layout(
    tensor: torch.Tensor,
    *,
    role: str,
    expected_shape: tuple[int, ...] | None = None,
) -> AdaptedFullVolume:
    """Accept only 3-D, CXYZ, or BCXYZ singleton-leading full volumes."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{role} must be a torch.Tensor.")
    if tensor.ndim < 3 or tensor.ndim > 5:
        raise ValueError(f"{role} rank must be between 3 and 5.")
    shape = tuple(int(value) for value in tensor.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{role} shape changed: {shape} != {expected_shape}.")
    leading = shape[:-3]
    if any(value != 1 for value in leading):
        raise ValueError(f"{role} has a non-singleton batch/channel axis.")
    spatial = shape[-3:]
    if any(value <= 0 for value in spatial):
        raise ValueError(f"{role} has a malformed spatial dimension.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{role} contains non-finite values.")
    adapted = AdaptedFullVolume(
        tensor=tensor,
        role=role,
        authenticated_shape=shape,
        spatial_shape=(spatial[0], spatial[1], spatial[2]),
    )
    adapted.model_batch()
    return adapted


def _validate_authenticated_case_layout(case: Gate01Case) -> AdaptedFullVolume:
    if case.source_image is None:
        raise ValueError("P:0006 inference case lacks its verified source image.")
    source = adapt_full_volume_layout(case.source_image, role="P:0006 source image")
    expected = source.authenticated_shape
    for role, tensor in (
        ("P:0006 target", case.target),
        ("P:0006 raw identity", case.raw_identity),
        ("P:0006 raw original SB-v2", case.raw_sb_v2),
        ("P:0006 Stage-1 reconstruction ceiling", case.stage1_reconstruction_ceiling),
    ):
        adapt_full_volume_layout(tensor, role=role, expected_shape=expected)
    if case.support_mask.dtype != torch.bool:
        raise ValueError("P:0006 support mask must be boolean.")
    adapt_full_volume_layout(
        case.support_mask,
        role="P:0006 source-derived support mask",
        expected_shape=expected,
    )
    return source


def _apply_shape_bound_calibrator(
    calibrator: PosthocTargetCalibrator,
    prediction: torch.Tensor,
    target_domain: Domain,
    support_mask: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    role: str,
) -> torch.Tensor:
    adapt_full_volume_layout(prediction, role=role, expected_shape=expected_shape)
    if support_mask.dtype != torch.bool:
        raise ValueError(f"{role} calibrator support must be boolean.")
    adapt_full_volume_layout(
        support_mask,
        role=f"{role} calibrator support",
        expected_shape=expected_shape,
    )
    calibrated = calibrator.apply(
        prediction,
        target_domain,
        support_mask=support_mask,
        mode="histogram",
    )
    adapt_full_volume_layout(
        calibrated,
        role=f"{role} calibrated output",
        expected_shape=expected_shape,
    )
    return calibrated


def _validate_full_volume_layout_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Full-volume layout provenance is missing or malformed.")
    provenance = dict(value)
    expected_keys = {
        "contract_version",
        "authenticated_source_rank",
        "authenticated_canonical_rank",
        "leading_singleton_axis_count",
        "model_input_rank",
        "representation_only_reshape",
        "spatial_axes_preserved",
        "resampling_performed",
        "cropping_performed",
        "padding_performed",
        "transposition_performed",
        "output_restored_to_authenticated_case_representation",
        "training_invoked",
    }
    if set(provenance) != expected_keys:
        raise ValueError("Full-volume layout provenance key inventory changed.")
    source_rank = provenance.get("authenticated_source_rank")
    canonical_rank = provenance.get("authenticated_canonical_rank")
    leading_count = provenance.get("leading_singleton_axis_count")
    if (
        isinstance(source_rank, bool)
        or not isinstance(source_rank, int)
        or source_rank not in {3, 4, 5}
        or canonical_rank != source_rank
        or isinstance(leading_count, bool)
        or not isinstance(leading_count, int)
        or leading_count != source_rank - 3
    ):
        raise ValueError("Full-volume layout rank provenance changed.")
    if (
        provenance.get("contract_version") != FULL_VOLUME_LAYOUT_ADAPTER_CONTRACT
        or provenance.get("model_input_rank") != 5
        or provenance.get("representation_only_reshape") is not True
        or provenance.get("spatial_axes_preserved") is not True
        or provenance.get("resampling_performed") is not False
        or provenance.get("cropping_performed") is not False
        or provenance.get("padding_performed") is not False
        or provenance.get("transposition_performed") is not False
        or provenance.get("output_restored_to_authenticated_case_representation")
        is not True
        or provenance.get("training_invoked") is not False
    ):
        raise ValueError("Full-volume layout safety provenance changed.")
    return provenance


_DECODER_RANGE_OBSERVATION_ROLES = frozenset(
    {
        "primary_unified_output",
        "graph_direct_output",
        "graph_composed_output",
        *(f"field_sweep_{float(field):g}T" for field in FIELD_STRENGTHS_T),
    }
)
_DECODER_RANGE_GRAPH_ROLES = frozenset(
    {"graph_direct_output", "graph_composed_output"}
)
_DECODER_RANGE_SWEEP_ROLES = frozenset(
    f"field_sweep_{float(field):g}T" for field in FIELD_STRENGTHS_T
)


def decoder_evaluation_range_policy() -> dict[str, Any]:
    """Return the closed official evaluation-only range policy."""

    return {
        "contract_version": DECODER_EVALUATION_RANGE_CONTRACT,
        "effective_decoder_output_activation": "none",
        "raw_output_retained_for_diagnostics": True,
        "photometry_input_rule": "clamp(raw_decoder_output,0,1)",
        "transformation": "pointwise_hard_clamp",
        "sigmoid_applied": False,
        "tanh_applied": False,
        "normalization_applied": False,
        "percentile_scaling_applied": False,
        "rescaling_applied": False,
        "cropping_performed": False,
        "padding_performed": False,
        "interpolation_performed": False,
        "transposition_performed": False,
        "source_support_unchanged": True,
        "training_invoked": False,
    }


def _validate_decoder_evaluation_range_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Decoder evaluation-range policy is missing or malformed.")
    policy = dict(value)
    expected = decoder_evaluation_range_policy()
    if set(policy) != set(expected) or policy != expected:
        raise ValueError("Decoder evaluation-range policy changed.")
    return policy


def _require_linear_decoder_output_activation(decoder: torch.nn.Module) -> None:
    if getattr(decoder, "output_activation", None) != "none":
        raise ValueError(
            "Frozen Stage-1 decoder effective output_activation must be exactly none."
        )


@dataclass(frozen=True, slots=True)
class DecoderEvaluationRange:
    """Raw linear decoder output plus its sole authorized evaluation view."""

    raw: torch.Tensor
    bounded: torch.Tensor
    observation: dict[str, Any]


def adapt_decoder_evaluation_range(
    raw: torch.Tensor,
    source_support: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    role: str,
) -> DecoderEvaluationRange:
    """Apply only the official pointwise hard clamp at a photometry boundary."""

    if role not in _DECODER_RANGE_OBSERVATION_ROLES:
        raise ValueError("Decoder evaluation-range role is not predeclared.")
    raw_layout = adapt_full_volume_layout(
        raw, role=f"{role} raw decoder output", expected_shape=expected_shape
    )
    if raw.dtype != torch.float32:
        raise ValueError(f"{role} raw decoder output must be float32.")
    if source_support.dtype != torch.bool:
        raise ValueError(f"{role} source support must be boolean.")
    support_layout = adapt_full_volume_layout(
        source_support,
        role=f"{role} immutable source support",
        expected_shape=expected_shape,
    )
    raw_spatial = raw_layout.spatial_volume()
    support_spatial = support_layout.spatial_volume()
    if raw_spatial.shape != support_spatial.shape:
        raise ValueError(f"{role} raw decoder output and source support disagree.")
    supported_voxel_count = int(support_spatial.sum().item())
    if supported_voxel_count <= 0:
        raise ValueError(f"{role} source support is empty.")

    below = raw_spatial < 0.0
    above = raw_spatial > 1.0
    voxel_count = int(raw_spatial.numel())
    below_count = int(below.sum().item())
    above_count = int(above.sum().item())
    supported_below_count = int((below & support_spatial).sum().item())
    supported_above_count = int((above & support_spatial).sum().item())

    bounded = raw.clamp(0.0, 1.0)
    if bounded.shape != raw.shape or bounded.dtype != raw.dtype:
        raise RuntimeError(f"{role} official hard clamp changed shape or dtype.")
    bounded_layout = adapt_full_volume_layout(
        bounded,
        role=f"{role} bounded photometry input",
        expected_shape=expected_shape,
    )
    bounded_spatial = bounded_layout.spatial_volume()
    bounded_min = float(bounded_spatial.min().item())
    bounded_max = float(bounded_spatial.max().item())
    if bounded_min < 0.0 or bounded_max > 1.0:
        raise RuntimeError(f"{role} official hard clamp violated [0,1].")
    if below_count == 0 and above_count == 0 and not torch.equal(bounded, raw):
        raise RuntimeError(f"{role} in-range decoder output changed under hard clamp.")

    observation = {
        "role": role,
        "voxel_count": voxel_count,
        "supported_voxel_count": supported_voxel_count,
        "raw_min": float(raw_spatial.min().item()),
        "raw_max": float(raw_spatial.max().item()),
        "raw_below_zero_count": below_count,
        "raw_below_zero_fraction": below_count / voxel_count,
        "raw_above_one_count": above_count,
        "raw_above_one_fraction": above_count / voxel_count,
        "supported_raw_below_zero_count": supported_below_count,
        "supported_raw_below_zero_fraction": (
            supported_below_count / supported_voxel_count
        ),
        "supported_raw_above_one_count": supported_above_count,
        "supported_raw_above_one_fraction": (
            supported_above_count / supported_voxel_count
        ),
        "bounded_min": bounded_min,
        "bounded_max": bounded_max,
        "raw_canonical_tensor_sha256": canonical_tensor_sha256(raw_spatial),
        "bounded_canonical_tensor_sha256": canonical_tensor_sha256(bounded_spatial),
    }
    return DecoderEvaluationRange(
        raw=raw,
        bounded=bounded,
        observation=_validate_decoder_range_observation(observation),
    )


def _validate_decoder_range_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Decoder range observation is missing or malformed.")
    observation = dict(value)
    expected_keys = {
        "role",
        "voxel_count",
        "supported_voxel_count",
        "raw_min",
        "raw_max",
        "raw_below_zero_count",
        "raw_below_zero_fraction",
        "raw_above_one_count",
        "raw_above_one_fraction",
        "supported_raw_below_zero_count",
        "supported_raw_below_zero_fraction",
        "supported_raw_above_one_count",
        "supported_raw_above_one_fraction",
        "bounded_min",
        "bounded_max",
        "raw_canonical_tensor_sha256",
        "bounded_canonical_tensor_sha256",
    }
    if set(observation) != expected_keys:
        raise ValueError("Decoder range observation key inventory changed.")
    if observation.get("role") not in _DECODER_RANGE_OBSERVATION_ROLES:
        raise ValueError("Decoder range observation role changed.")
    voxel_count = observation.get("voxel_count")
    supported_count = observation.get("supported_voxel_count")
    count_keys = (
        "raw_below_zero_count",
        "raw_above_one_count",
        "supported_raw_below_zero_count",
        "supported_raw_above_one_count",
    )
    if (
        isinstance(voxel_count, bool)
        or not isinstance(voxel_count, int)
        or voxel_count <= 0
        or isinstance(supported_count, bool)
        or not isinstance(supported_count, int)
        or supported_count <= 0
        or supported_count > voxel_count
    ):
        raise ValueError("Decoder range observation voxel counts changed.")
    for key in count_keys:
        count = observation.get(key)
        limit = supported_count if key.startswith("supported_") else voxel_count
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= limit:
            raise ValueError("Decoder range observation overshoot counts changed.")
    fraction_pairs = (
        ("raw_below_zero_fraction", "raw_below_zero_count", voxel_count),
        ("raw_above_one_fraction", "raw_above_one_count", voxel_count),
        (
            "supported_raw_below_zero_fraction",
            "supported_raw_below_zero_count",
            supported_count,
        ),
        (
            "supported_raw_above_one_fraction",
            "supported_raw_above_one_count",
            supported_count,
        ),
    )
    for fraction_key, count_key, denominator in fraction_pairs:
        fraction = observation.get(fraction_key)
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not math.isclose(
                float(fraction), int(observation[count_key]) / denominator, abs_tol=1e-15
            )
        ):
            raise ValueError("Decoder range observation overshoot fractions changed.")
    for key in ("raw_min", "raw_max", "bounded_min", "bounded_max"):
        number = observation.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise ValueError("Decoder range observation extrema changed.")
    if (
        float(observation["raw_min"]) > float(observation["raw_max"])
        or float(observation["bounded_min"]) < 0.0
        or float(observation["bounded_max"]) > 1.0
        or float(observation["bounded_min"]) > float(observation["bounded_max"])
    ):
        raise ValueError("Decoder range observation bounds changed.")
    for key in ("raw_canonical_tensor_sha256", "bounded_canonical_tensor_sha256"):
        if _SHA256_RE.fullmatch(str(observation.get(key, ""))) is None:
            raise ValueError("Decoder range observation tensor identity changed.")
    return observation


def _validate_decoder_range_observations(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("Decoder range observations are missing or malformed.")
    observations = {
        str(role): _validate_decoder_range_observation(observation)
        for role, observation in value.items()
    }
    if set(observations) - _DECODER_RANGE_OBSERVATION_ROLES:
        raise ValueError("Decoder range observations contain an unknown role.")
    if "primary_unified_output" not in observations:
        raise ValueError("Decoder range observations lack the primary output.")
    graph_roles = set(observations) & _DECODER_RANGE_GRAPH_ROLES
    if graph_roles and graph_roles != _DECODER_RANGE_GRAPH_ROLES:
        raise ValueError("Decoder graph range observations are incomplete.")
    sweep_roles = set(observations) & _DECODER_RANGE_SWEEP_ROLES
    if sweep_roles and sweep_roles != _DECODER_RANGE_SWEEP_ROLES:
        raise ValueError("Decoder sweep range observations are incomplete.")
    return observations


@dataclass(frozen=True, slots=True)
class InferenceCaseOutputs:
    methods: dict[str, torch.Tensor]
    anatomy: dict[str, float]
    graph: dict[str, Any]
    sweep_slices: dict[str, np.ndarray]
    decoded_canonical_sha256: str
    bounded_decoded_canonical_sha256: str
    full_volume_layout_provenance: dict[str, Any]
    decoder_evaluation_range_policy: dict[str, Any]
    decoder_range_observations: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class FrozenStage1VAEConfigPreflight:
    """Verified raw owner YAML plus separately derived parsed provenance."""

    path: Path
    raw_file_sha256: str
    size_bytes: int
    parsed_canonical_sha256: str
    parsed_config: dict[str, Any]

    def sanitized_provenance(self) -> dict[str, Any]:
        return {
            "config_role": "frozen_stage1_run_c",
            "raw_config_file_sha256": self.raw_file_sha256,
            "raw_config_file_size_bytes": self.size_bytes,
            "parsed_canonical_config_sha256": self.parsed_canonical_sha256,
            "raw_config_identity_match": True,
        }


@dataclass(frozen=True, slots=True)
class FrozenStage1VAEProvenance:
    """Raw-file identities cross-checked against the restored bank manifest."""

    config: FrozenStage1VAEConfigPreflight
    checkpoint_path: Path
    bank_dir: Path
    checkpoint_file_sha256: str
    bank_manifest_config_sha256: str
    bank_manifest_checkpoint_sha256: str

    def sanitized_provenance(self) -> dict[str, Any]:
        return {
            "contract_version": FROZEN_STAGE1_VAE_PROVENANCE_CONTRACT,
            **self.config.sanitized_provenance(),
            "bank_manifest_config_sha256": self.bank_manifest_config_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "bank_manifest_checkpoint_sha256": self.bank_manifest_checkpoint_sha256,
            "checkpoint_identity_match": True,
        }


@dataclass(frozen=True, slots=True)
class ReviewedPhotometryPreflight:
    """Fully validated frozen artifact under the sealed historical namespace rule."""

    path: Path
    artifact: FrozenPhotometryArtifact
    artifact_file_sha256: str
    artifact_internal_sha256: str
    accepted_records_sha256: str
    excluded_records_sha256: str
    accepted_record_count: int
    prospective_accepted_count: int
    prospective_excluded_count: int
    retrospective_numeric_collision_count: int
    collision_group_counts: Mapping[str, int]

    def sanitized_provenance(self) -> dict[str, Any]:
        return {
            "contract_version": REVIEWED_PHOTOMETRY_NAMESPACE_PROVENANCE_CONTRACT,
            "artifact_role": REVIEWED_PHOTOMETRY_ROLE,
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_internal_sha256": self.artifact_internal_sha256,
            "artifact_production_commit": REVIEWED_PHOTOMETRY_PRODUCTION_COMMIT,
            "protected_base_module_sha256": PROTECTED_PHOTOMETRY_MODULE_SHA256,
            "historical_operator_overlay_sha256": (
                HISTORICAL_PHOTOMETRY_OPERATOR_OVERLAY_SHA256
            ),
            "namespace_predicate_source_sha256": (
                REVIEWED_NAMESPACE_PREDICATE_SOURCE_SHA256
            ),
            "accepted_records_sha256": self.accepted_records_sha256,
            "excluded_records_sha256": self.excluded_records_sha256,
            "accepted_record_count": self.accepted_record_count,
            "prospective_accepted_count": self.prospective_accepted_count,
            "prospective_excluded_count": self.prospective_excluded_count,
            "retrospective_numeric_collision_count": (
                self.retrospective_numeric_collision_count
            ),
            "retrospective_collision_group_counts": dict(
                sorted(self.collision_group_counts.items())
            ),
            "compatibility_scope_restored": True,
        }


@dataclass(frozen=True, slots=True)
class ReviewedPhotometryBankProvenance:
    """Preflighted artifact cross-checked against the restored bank manifest."""

    preflight: ReviewedPhotometryPreflight
    bank_dir: Path
    bank_manifest_artifact_file_sha256: str
    bank_manifest_artifact_sha256: str

    def sanitized_provenance(self) -> dict[str, Any]:
        return {
            **self.preflight.sanitized_provenance(),
            "bank_manifest_artifact_file_sha256": (
                self.bank_manifest_artifact_file_sha256
            ),
            "bank_manifest_artifact_sha256": self.bank_manifest_artifact_sha256,
            "bank_photometry_identity_match": True,
        }


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
    frozen_stage1_vae_provenance: Mapping[str, Any]
    reviewed_photometry_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if photometry_namespace_compatibility_active():
            raise RuntimeError(
                "Photometry compatibility scope leaked into model construction."
            )
        if self.device.type != "cuda":
            raise ValueError("The production step-200 inference runtime requires CUDA.")
        _require_linear_decoder_output_activation(self.decoder)
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
        _require_linear_decoder_output_activation(self.decoder)
        range_policy = _validate_decoder_evaluation_range_policy(
            decoder_evaluation_range_policy()
        )
        source_layout = _validate_authenticated_case_layout(case)
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
            canonical_layout = adapt_full_volume_layout(
                canonical,
                role="P:0006 authenticated canonical source",
                expected_shape=source_layout.authenticated_shape,
            )
            canonical_batch = canonical_layout.model_batch()
            canonical_support_layout = adapt_full_volume_layout(
                canonical_context.support_mask,
                role="P:0006 canonical source support",
                expected_shape=source_layout.authenticated_shape,
            )
            if canonical_context.support_mask.dtype != torch.bool:
                raise ValueError("P:0006 canonical source support must be boolean.")
            support_image = canonical_support_layout.spatial_volume().detach().to(
                torch.bool
            )
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
            support_batch = adapt_full_volume_layout(
                support,
                role="P:0006 propagated latent support",
            ).model_batch().to(self.device)
            z = self.stats.normalize(latent, support_batch)
            with _bf16_inference_autocast(self.device):
                generated_z = integrate_transport(
                    self.translator,
                    z,
                    [case.source_domain],
                    [case.target_domain],
                    steps=4,
                    solver="heun",
                )
                decoded_model_batch = _decode(
                    self.decoder,
                    self.stats.denormalize(generated_z),
                    [case.target_domain],
                ).float()
            decoded = canonical_layout.restore_decoder_output(
                decoded_model_batch,
                role="primary unified output",
            ).cpu()
            primary_range = adapt_decoder_evaluation_range(
                decoded,
                canonical_context.support_mask,
                expected_shape=canonical_layout.authenticated_shape,
                role="primary_unified_output",
            )
            unified = self.artifact.render_target(
                canonical_context.with_values(primary_range.bounded),
                case.target_domain,
            ).float().cpu()
            adapt_full_volume_layout(
                unified,
                role="rendered primary unified output",
                expected_shape=source_layout.authenticated_shape,
            )
            calibrated_identity = _apply_shape_bound_calibrator(
                calibrator,
                case.raw_identity,
                case.target_domain,
                case.support_mask,
                expected_shape=source_layout.authenticated_shape,
                role="raw identity",
            )
            calibrated_sb = _apply_shape_bound_calibrator(
                calibrator,
                case.raw_sb_v2,
                case.target_domain,
                case.support_mask,
                expected_shape=source_layout.authenticated_shape,
                role="raw original SB-v2",
            )
            # The reviewed calibrator is fitted only from retrospective training target
            # CDFs and consumes the prediction/support, never the paired P:0006 target.
            calibrated_unified = _apply_shape_bound_calibrator(
                calibrator,
                unified,
                case.target_domain,
                case.support_mask,
                expected_shape=source_layout.authenticated_shape,
                role="raw unified step-200 output",
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
            for method, prediction in methods.items():
                adapt_full_volume_layout(
                    prediction,
                    role=f"P:0006 method {method}",
                    expected_shape=source_layout.authenticated_shape,
                )
            image_support_batch = canonical_support_layout.model_batch().to(self.device)
            anatomy_values = anatomy_preservation_components(
                canonical_batch.to(self.device),
                decoded_model_batch.to(self.device),
                image_support_batch,
            )
            anatomy = {key: float(value.detach().float().cpu()) for key, value in anatomy_values.items()}
            graph, graph_range_observations = self._graph_diagnostic(
                case,
                z,
                support_batch,
                canonical_context,
                canonical_layout,
                canonical_context.support_mask,
                plan,
            )
            sweep_slices, sweep_range_observations = self._field_sweep(
                case,
                z,
                canonical_context,
                canonical_layout,
                canonical_context.support_mask,
                plan,
            )
            range_observations = _validate_decoder_range_observations(
                {
                    "primary_unified_output": primary_range.observation,
                    **graph_range_observations,
                    **sweep_range_observations,
                }
            )
            decoded_sha = primary_range.observation["raw_canonical_tensor_sha256"]
            bounded_decoded_sha = primary_range.observation[
                "bounded_canonical_tensor_sha256"
            ]
            layout_provenance = canonical_layout.sanitized_provenance(
                source_rank=source_layout.authenticated_rank
            )
            del generated_z, decoded, decoded_model_batch, latent, z, support
            del support_batch, canonical_batch
            del image_support_batch, anatomy_values
            del primary_range
        return InferenceCaseOutputs(
            methods=methods,
            anatomy=anatomy,
            graph=graph,
            sweep_slices=sweep_slices,
            decoded_canonical_sha256=decoded_sha,
            bounded_decoded_canonical_sha256=bounded_decoded_sha,
            full_volume_layout_provenance=layout_provenance,
            decoder_evaluation_range_policy=range_policy,
            decoder_range_observations=range_observations,
        )

    def _graph_diagnostic(
        self,
        case: Gate01Case,
        z: torch.Tensor,
        support: torch.Tensor,
        canonical_context: Any,
        canonical_layout: AdaptedFullVolume,
        source_support: torch.Tensor,
        plan: FrozenStep200InferencePlan,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        selected = _graph_intermediate(plan, case)
        if selected is None:
            return {"selected": False}, {}
        intermediate = Domain(selected, case.source_domain.contrast)
        with _bf16_inference_autocast(self.device):
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
            direct_decoded_batch = _decode(
                self.decoder, self.stats.denormalize(direct), [case.target_domain]
            ).float()
            composed_decoded_batch = _decode(
                self.decoder, self.stats.denormalize(composed), [case.target_domain]
            ).float()
        direct_decoded = canonical_layout.restore_decoder_output(
            direct_decoded_batch, role="graph direct output"
        ).cpu()
        composed_decoded = canonical_layout.restore_decoder_output(
            composed_decoded_batch, role="graph composed output"
        ).cpu()
        direct_range = adapt_decoder_evaluation_range(
            direct_decoded,
            source_support,
            expected_shape=canonical_layout.authenticated_shape,
            role="graph_direct_output",
        )
        composed_range = adapt_decoder_evaluation_range(
            composed_decoded,
            source_support,
            expected_shape=canonical_layout.authenticated_shape,
            role="graph_composed_output",
        )
        direct_image = self.artifact.render_target(
            canonical_context.with_values(direct_range.bounded), case.target_domain
        )
        composed_image = self.artifact.render_target(
            canonical_context.with_values(composed_range.bounded), case.target_domain
        )
        adapt_full_volume_layout(
            direct_image,
            role="rendered graph direct output",
            expected_shape=canonical_layout.authenticated_shape,
        )
        adapt_full_volume_layout(
            composed_image,
            role="rendered graph composed output",
            expected_shape=canonical_layout.authenticated_shape,
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
        del direct, composed, direct_decoded_batch, composed_decoded_batch
        del direct_decoded, composed_decoded, direct_image, composed_image
        observations = {
            "graph_direct_output": direct_range.observation,
            "graph_composed_output": composed_range.observation,
        }
        del direct_range, composed_range
        return result, observations

    def _field_sweep(
        self,
        case: Gate01Case,
        z: torch.Tensor,
        canonical_context: Any,
        canonical_layout: AdaptedFullVolume,
        source_support: torch.Tensor,
        plan: FrozenStep200InferencePlan,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if float(case.source_domain.field_strength_t) != float(
            plan.payload["field_sweep"]["source_field_t"]
        ):
            return {}, {}
        slices: dict[str, np.ndarray] = {}
        observations: dict[str, dict[str, Any]] = {}
        for field in plan.payload["field_sweep"]["target_fields_t"]:
            target = Domain(float(field), case.source_domain.contrast)
            if target == case.source_domain:
                target_z = z
            else:
                with _bf16_inference_autocast(self.device):
                    target_z = integrate_transport(
                        self.translator,
                        z,
                        [case.source_domain],
                        [target],
                        steps=4,
                        solver="heun",
                    )
            with _bf16_inference_autocast(self.device):
                decoded_batch = _decode(
                    self.decoder, self.stats.denormalize(target_z), [target]
                ).float()
            decoded = canonical_layout.restore_decoder_output(
                decoded_batch, role=f"field-sweep {float(field):g}T output"
            ).cpu()
            range_role = f"field_sweep_{float(field):g}T"
            sweep_range = adapt_decoder_evaluation_range(
                decoded,
                source_support,
                expected_shape=canonical_layout.authenticated_shape,
                role=range_role,
            )
            rendered = self.artifact.render_target(
                canonical_context.with_values(sweep_range.bounded), target
            )
            adapt_full_volume_layout(
                rendered,
                role=f"rendered field-sweep {float(field):g}T output",
                expected_shape=canonical_layout.authenticated_shape,
            )
            slices[f"{float(field):g}T"] = _middle_slice(rendered)
            observations[range_role] = sweep_range.observation
            if target_z is not z:
                del target_z
            del decoded_batch, decoded, rendered, sweep_range
        return slices, observations


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


def photometry_namespace_compatibility_active() -> bool:
    """Expose only the compatibility-scope state for containment assertions."""

    return _PHOTOMETRY_NAMESPACE_SCOPE_ACTIVE


def _verify_reviewed_photometry_code_boundary() -> None:
    module_path = Path(str(_photometry.__file__)).resolve(strict=True)
    checkout_root = Path(__file__).resolve().parents[3]
    expected_path = (
        checkout_root / "src/fieldbridge/data/photometry_factorization.py"
    ).resolve(strict=True)
    if module_path != expected_path:
        raise ValueError(
            "Active photometry module is outside the detached implementation checkout."
        )
    if sha256_file(module_path) != PROTECTED_PHOTOMETRY_MODULE_SHA256:
        raise ValueError("Protected base photometry module SHA-256 mismatch.")
    if _photometry._is_forbidden_traveller is not _BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER:
        raise RuntimeError("Protected photometry namespace helper was substituted.")
    if (
        _namespace_aware_forbidden_traveller is not _REVIEWED_NAMESPACE_PREDICATE
        or inspect.getsource(_namespace_aware_forbidden_traveller)
        != _REVIEWED_NAMESPACE_PREDICATE_SOURCE
        or (
            _namespace_aware_forbidden_traveller.__code__.co_code,
            _namespace_aware_forbidden_traveller.__code__.co_consts,
            _namespace_aware_forbidden_traveller.__code__.co_names,
        )
        != _REVIEWED_NAMESPACE_PREDICATE_CODE
    ):
        raise RuntimeError("Reviewed photometry namespace predicate was substituted.")
    if (
        HISTORICAL_PHOTOMETRY_OPERATOR_OVERLAY_SHA256
        != _PINNED_PHOTOMETRY_OPERATOR_OVERLAY_SHA256
    ):
        raise ValueError("Historical photometry operator-overlay identity changed.")
    if (
        REVIEWED_NAMESPACE_PREDICATE_SOURCE_SHA256
        != _PINNED_NAMESPACE_PREDICATE_SOURCE_SHA256
    ):
        raise ValueError("Reviewed photometry namespace-predicate identity changed.")


@contextmanager
def _reviewed_photometry_namespace_scope():
    """Temporarily replay exactly the reviewed traveller-namespace predicate."""

    global _PHOTOMETRY_NAMESPACE_SCOPE_ACTIVE
    if not _PHOTOMETRY_NAMESPACE_LOCK.acquire(blocking=False):
        raise RuntimeError("Nested or concurrent photometry compatibility scope is forbidden.")
    installed = False
    try:
        if _PHOTOMETRY_NAMESPACE_SCOPE_ACTIVE:
            raise RuntimeError("Nested photometry compatibility scope is forbidden.")
        _verify_reviewed_photometry_code_boundary()
        _PHOTOMETRY_NAMESPACE_SCOPE_ACTIVE = True
        _photometry._is_forbidden_traveller = _REVIEWED_NAMESPACE_PREDICATE
        installed = True
        yield
        if _photometry._is_forbidden_traveller is not _REVIEWED_NAMESPACE_PREDICATE:
            raise RuntimeError("Photometry compatibility helper changed inside its scope.")
    finally:
        if installed:
            _photometry._is_forbidden_traveller = _BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER
        _PHOTOMETRY_NAMESPACE_SCOPE_ACTIVE = False
        restored = (
            _photometry._is_forbidden_traveller
            is _BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER
        )
        _PHOTOMETRY_NAMESPACE_LOCK.release()
        if installed and not restored:
            raise RuntimeError("Protected photometry namespace helper was not restored.")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in frozen audit evidence: {key!r}.")
        result[key] = value
    return result


def _parse_reviewed_photometry_snapshot(raw_bytes: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw_bytes.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Frozen photometry artifact JSON is malformed.") from error
    if not isinstance(payload, dict):
        raise ValueError("Frozen photometry artifact root must be a JSON object.")
    return payload


def _validate_reviewed_photometry_membership(
    artifact: FrozenPhotometryArtifact,
) -> tuple[str, str, dict[str, int]]:
    provenance = artifact.provenance
    accepted_raw = provenance.get("accepted_records")
    excluded_raw = provenance.get("excluded_prospective_records")
    if (
        not isinstance(accepted_raw, Sequence)
        or isinstance(accepted_raw, (str, bytes))
        or not isinstance(excluded_raw, Sequence)
        or isinstance(excluded_raw, (str, bytes))
    ):
        raise ValueError("Reviewed photometry membership evidence is malformed.")
    accepted = [dict(item) for item in accepted_raw if isinstance(item, Mapping)]
    excluded = [dict(item) for item in excluded_raw if isinstance(item, Mapping)]
    if len(accepted) != len(accepted_raw) or len(excluded) != len(excluded_raw):
        raise ValueError("Reviewed photometry membership evidence contains malformed rows.")
    if len(accepted) != REVIEWED_PHOTOMETRY_ACCEPTED_RECORD_COUNT:
        raise ValueError("Reviewed photometry accepted-record count changed.")
    if len(excluded) != REVIEWED_PHOTOMETRY_PROSPECTIVE_EXCLUDED_COUNT:
        raise ValueError("Reviewed photometry prospective-exclusion count changed.")
    accepted_sha = str(provenance.get("accepted_records_sha256", ""))
    excluded_sha = str(provenance.get("excluded_prospective_records_sha256", ""))
    if sha256_json(accepted) != accepted_sha:
        raise ValueError("Reviewed photometry accepted-record content hash mismatch.")
    if sha256_json(excluded) != excluded_sha:
        raise ValueError("Reviewed photometry excluded-record content hash mismatch.")

    collision_counts: dict[str, int] = defaultdict(int)
    for item in accepted:
        identity = _photometry.classify_variant_a_cohort(
            case_identity=str(item.get("record_identity", "")),
            metadata_prefix=item.get("metadata_prefix"),
            supplied_cohort=item.get("cohort"),
            subject_identity=item.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if item.get("split") != "train":
            raise ValueError("Reviewed photometry accepted a non-R/train record.")
        if item.get("subject_group_identity") != identity.subject_group_identity:
            raise ValueError("Reviewed photometry subject grouping changed.")
        if _REVIEWED_NAMESPACE_PREDICATE(
            identity.case_identity, identity.subject_identity
        ):
            raise ValueError("Reviewed photometry accepted a prospective traveller token.")
        if str(identity.subject_identity).zfill(4) in _photometry.FORBIDDEN_TRAVELLER_IDS:
            group_hash = hashlib.sha256(
                identity.subject_group_identity.encode("utf-8")
            ).hexdigest()
            collision_counts[group_hash] += 1

    normalized_excluded = _photometry.normalize_variant_a_prospective_exclusions(
        excluded, expected_split="train"
    )
    if normalized_excluded != excluded:
        raise ValueError("Reviewed photometry prospective exclusions are not canonical.")
    proof = provenance.get("eligibility_proof")
    if (
        not isinstance(proof, Mapping)
        or proof.get("accepted_count") != REVIEWED_PHOTOMETRY_ACCEPTED_RECORD_COUNT
        or proof.get("all_cohort_R") is not True
        or proof.get("all_split_train") is not True
        or proof.get("prospective_accepted_count") != 0
        or proof.get("prospective_excluded_count")
        != REVIEWED_PHOTOMETRY_PROSPECTIVE_EXCLUDED_COUNT
        or proof.get("forbidden_traveller_accepted_count") != 0
    ):
        raise ValueError("Reviewed photometry eligibility proof changed.")
    if dict(sorted(collision_counts.items())) != dict(
        sorted(REVIEWED_PHOTOMETRY_COLLISION_GROUP_COUNTS.items())
    ):
        raise ValueError("Reviewed photometry retrospective collision membership changed.")
    return accepted_sha, excluded_sha, dict(collision_counts)


def preflight_reviewed_photometry_namespace_artifact(
    photometry_artifact_path: str | Path,
) -> ReviewedPhotometryPreflight:
    """Validate the small frozen artifact before bank, checkpoint, model, or CUDA I/O."""

    path = Path(photometry_artifact_path)
    if not path.exists():
        raise FileNotFoundError("Frozen reviewed photometry artifact is missing.")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Frozen reviewed photometry artifact is not a regular file.")
    if path.name != REVIEWED_PHOTOMETRY_BASENAME:
        raise ValueError("Frozen reviewed photometry artifact path role is incorrect.")
    raw_bytes = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha256 != REVIEWED_PHOTOMETRY_FILE_SHA256:
        raise ValueError("Frozen reviewed photometry artifact raw-file SHA-256 mismatch.")
    payload = _parse_reviewed_photometry_snapshot(raw_bytes)
    stored_internal_sha256 = str(payload.get("artifact_sha256", ""))
    internal_body = dict(payload)
    internal_body.pop("artifact_sha256", None)
    if (
        stored_internal_sha256 != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256
        or sha256_json(internal_body) != stored_internal_sha256
    ):
        raise ValueError("Frozen reviewed photometry artifact internal SHA-256 mismatch.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Frozen reviewed photometry artifact provenance is malformed.")
    if provenance.get("code_commit") != REVIEWED_PHOTOMETRY_PRODUCTION_COMMIT:
        raise ValueError("Frozen reviewed photometry artifact production commit changed.")
    code_provenance = provenance.get("code_provenance")
    module_hashes = (
        code_provenance.get("module_sha256")
        if isinstance(code_provenance, Mapping)
        else None
    )
    if (
        not isinstance(module_hashes, Mapping)
        or module_hashes.get("src/fieldbridge/data/photometry_factorization.py")
        != PROTECTED_PHOTOMETRY_MODULE_SHA256
    ):
        raise ValueError("Frozen photometry code-provenance module hash changed.")
    _verify_reviewed_photometry_code_boundary()
    with _reviewed_photometry_namespace_scope():
        artifact = FrozenPhotometryArtifact.load(
            path,
            expected_artifact_sha256=REVIEWED_PHOTOMETRY_ARTIFACT_SHA256,
        )
    if photometry_namespace_compatibility_active():
        raise RuntimeError("Photometry compatibility scope leaked after artifact loading.")
    if (
        path.read_bytes() != raw_bytes
        or sha256_file(path) != REVIEWED_PHOTOMETRY_FILE_SHA256
    ):
        raise ValueError("Frozen reviewed photometry artifact changed during preflight.")
    if artifact.artifact_sha256 != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256:
        raise ValueError("Frozen reviewed photometry in-memory identity changed.")
    accepted_sha, excluded_sha, collision_counts = (
        _validate_reviewed_photometry_membership(artifact)
    )
    return ReviewedPhotometryPreflight(
        path=path,
        artifact=artifact,
        artifact_file_sha256=raw_sha256,
        artifact_internal_sha256=artifact.artifact_sha256,
        accepted_records_sha256=accepted_sha,
        excluded_records_sha256=excluded_sha,
        accepted_record_count=REVIEWED_PHOTOMETRY_ACCEPTED_RECORD_COUNT,
        prospective_accepted_count=0,
        prospective_excluded_count=REVIEWED_PHOTOMETRY_PROSPECTIVE_EXCLUDED_COUNT,
        retrospective_numeric_collision_count=sum(collision_counts.values()),
        collision_group_counts=collision_counts,
    )


def verify_reviewed_photometry_bank_provenance(
    preflight: ReviewedPhotometryPreflight,
    *,
    bank_dir: str | Path,
) -> ReviewedPhotometryBankProvenance:
    """Cross-check the preflighted immutable artifact against the restored bank."""

    if (
        sha256_file(preflight.path) != REVIEWED_PHOTOMETRY_FILE_SHA256
        or preflight.artifact_file_sha256 != REVIEWED_PHOTOMETRY_FILE_SHA256
    ):
        raise ValueError("Frozen reviewed photometry artifact raw-file SHA-256 mismatch.")
    if (
        preflight.artifact.artifact_sha256 != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256
        or preflight.artifact_internal_sha256
        != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256
    ):
        raise ValueError("Frozen reviewed photometry in-memory identity changed.")
    if photometry_namespace_compatibility_active():
        raise RuntimeError("Photometry compatibility scope leaked into bank verification.")
    bank_path = Path(bank_dir)
    bank = PhotometryFactoredLatentBankIndex(bank_path, "validation")
    manifest_photometry = bank.manifest.get("photometry")
    if not isinstance(manifest_photometry, Mapping):
        raise ValueError("Restored bank manifest lacks photometry provenance.")
    manifest_file_sha256 = str(
        manifest_photometry.get("artifact_file_sha256", "")
    )
    if manifest_file_sha256 != REVIEWED_PHOTOMETRY_FILE_SHA256:
        raise ValueError("Restored bank manifest photometry file SHA-256 mismatch.")
    manifest_artifact_sha256 = str(manifest_photometry.get("artifact_sha256", ""))
    if manifest_artifact_sha256 != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256:
        raise ValueError("Restored bank manifest photometry internal SHA-256 mismatch.")
    return ReviewedPhotometryBankProvenance(
        preflight=preflight,
        bank_dir=bank_path,
        bank_manifest_artifact_file_sha256=manifest_file_sha256,
        bank_manifest_artifact_sha256=manifest_artifact_sha256,
    )


def preflight_frozen_p0006_scientific_role(
    protocol_path: str | Path,
    readiness_path: str | Path,
) -> FrozenP0006ScientificRolePreflight:
    """Authenticate the two small sealed P:0006 role artifacts before bank I/O."""

    protocol_file = Path(protocol_path)
    readiness_file = Path(readiness_path)
    protocol = _load_pinned_self_hashed_snapshot(
        protocol_file,
        role="P:0006 protocol",
        expected_file_sha256=P0006_PROTOCOL_FILE_SHA256,
        hash_key="protocol_sha256",
        expected_internal_sha256=P0006_PROTOCOL_SHA256,
    )
    readiness = _load_pinned_self_hashed_snapshot(
        readiness_file,
        role="evaluation readiness",
        expected_file_sha256=EVALUATION_READINESS_FILE_SHA256,
        hash_key="readiness_sha256",
        expected_internal_sha256=EVALUATION_READINESS_SHA256,
    )
    _validate_scientific_role(protocol, readiness)
    return FrozenP0006ScientificRolePreflight(
        protocol_path=protocol_file,
        readiness_path=readiness_file,
        protocol=copy.deepcopy(protocol),
        readiness=copy.deepcopy(readiness),
        protocol_file_sha256=P0006_PROTOCOL_FILE_SHA256,
        protocol_sha256=P0006_PROTOCOL_SHA256,
        readiness_file_sha256=EVALUATION_READINESS_FILE_SHA256,
        readiness_sha256=EVALUATION_READINESS_SHA256,
    )


def verify_frozen_p0006_scientific_role_preflight(
    preflight: FrozenP0006ScientificRolePreflight,
) -> FrozenP0006ScientificRolePreflight:
    """Rehash and semantically revalidate both artifacts immediately before use."""

    current = preflight_frozen_p0006_scientific_role(
        preflight.protocol_path,
        preflight.readiness_path,
    )
    if current.protocol != preflight.protocol:
        raise ValueError("P:0006 protocol changed after scientific-role preflight.")
    if current.readiness != preflight.readiness:
        raise ValueError("Evaluation readiness changed after scientific-role preflight.")
    if current.sanitized_provenance() != preflight.sanitized_provenance():
        raise ValueError("P:0006 scientific-role provenance changed after preflight.")
    return current


def preflight_frozen_stage1_run_c_config(
    vae_config_path: str | Path,
) -> FrozenStage1VAEConfigPreflight:
    """Authenticate the exact owner YAML bytes before any restored-bank I/O."""

    path = Path(vae_config_path)
    if not path.exists():
        raise FileNotFoundError("Frozen Stage-1 Run C VAE config is missing.")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Frozen Stage-1 Run C VAE config is not a regular file.")
    if path.name != STAGE1_RUN_C_CONFIG_BASENAME:
        raise ValueError("Frozen Stage-1 Run C VAE config path role is incorrect.")
    size_bytes = path.stat().st_size
    raw_file_sha256 = sha256_file(path)
    if (
        size_bytes != STAGE1_RUN_C_CONFIG_SIZE_BYTES
        or raw_file_sha256 != STAGE1_RUN_C_CONFIG_SHA256
    ):
        raise ValueError("Frozen Stage-1 Run C VAE config raw-file SHA-256 mismatch.")
    try:
        parsed = load_yaml_config(path)
    except Exception as error:
        raise ValueError("Frozen Stage-1 Run C VAE model configuration is malformed.") from error
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("model"), Mapping):
        raise ValueError("Frozen Stage-1 Run C VAE model configuration is malformed.")
    # Detect a replacement between raw authentication and parsing.
    if path.stat().st_size != size_bytes or sha256_file(path) != raw_file_sha256:
        raise ValueError("Frozen Stage-1 Run C VAE config changed while being parsed.")
    parsed_config = copy.deepcopy(dict(parsed))
    return FrozenStage1VAEConfigPreflight(
        path=path,
        raw_file_sha256=raw_file_sha256,
        size_bytes=size_bytes,
        parsed_canonical_sha256=sha256_json(parsed_config),
        parsed_config=parsed_config,
    )


def verify_frozen_stage1_vae_bank_provenance(
    config_preflight: FrozenStage1VAEConfigPreflight,
    *,
    vae_checkpoint_path: str | Path,
    bank_dir: str | Path,
) -> FrozenStage1VAEProvenance:
    """Cross-check raw frozen-VAE identities with a verified restored bank."""

    if (
        config_preflight.raw_file_sha256 != STAGE1_RUN_C_CONFIG_SHA256
        or config_preflight.size_bytes != STAGE1_RUN_C_CONFIG_SIZE_BYTES
        or sha256_file(config_preflight.path) != STAGE1_RUN_C_CONFIG_SHA256
    ):
        raise ValueError("Frozen Stage-1 Run C VAE config raw-file SHA-256 mismatch.")
    if (
        not isinstance(config_preflight.parsed_config.get("model"), Mapping)
        or sha256_json(config_preflight.parsed_config)
        != config_preflight.parsed_canonical_sha256
    ):
        raise ValueError("Frozen Stage-1 Run C VAE model configuration is malformed.")
    bank_path = Path(bank_dir)
    bank = PhotometryFactoredLatentBankIndex(bank_path, "validation")
    manifest_vae = bank.manifest.get("vae")
    if not isinstance(manifest_vae, Mapping):
        raise ValueError("Restored bank manifest lacks frozen VAE provenance.")
    bank_config_sha256 = str(manifest_vae.get("config_sha256", ""))
    if bank_config_sha256 != config_preflight.raw_file_sha256:
        raise ValueError("Restored bank manifest VAE config SHA-256 mismatch.")

    checkpoint_path = Path(vae_checkpoint_path)
    if (
        not checkpoint_path.exists()
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
    ):
        raise ValueError("Frozen Stage-1 Run C VAE checkpoint is missing or non-regular.")
    checkpoint_file_sha256 = sha256_file(checkpoint_path)
    if checkpoint_file_sha256 != STAGE1_RUN_C_CHECKPOINT_SHA256:
        raise ValueError("Frozen Stage-1 Run C VAE checkpoint raw-file SHA-256 mismatch.")
    bank_checkpoint_sha256 = str(manifest_vae.get("checkpoint_sha256", ""))
    if bank_checkpoint_sha256 != checkpoint_file_sha256:
        raise ValueError("Restored bank manifest VAE checkpoint SHA-256 mismatch.")
    return FrozenStage1VAEProvenance(
        config=config_preflight,
        checkpoint_path=checkpoint_path,
        bank_dir=bank_path,
        checkpoint_file_sha256=checkpoint_file_sha256,
        bank_manifest_config_sha256=bank_config_sha256,
        bank_manifest_checkpoint_sha256=bank_checkpoint_sha256,
    )


def load_unified_step200_inference_runtime(
    *,
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    vae_config_path: str | Path,
    vae_checkpoint_path: str | Path,
    photometry_artifact_path: str | Path,
    bank_dir: str | Path,
    device: str | torch.device = "cuda",
    verified_vae_provenance: FrozenStage1VAEProvenance | None = None,
    verified_photometry_provenance: ReviewedPhotometryBankProvenance | None = None,
) -> UnifiedStep200InferenceRuntime:
    """Authenticate frozen inputs before extracting only verified inference state."""

    vae_config_path = Path(vae_config_path)
    vae_checkpoint_path = Path(vae_checkpoint_path)
    bank_dir = Path(bank_dir)
    config_preflight = preflight_frozen_stage1_run_c_config(vae_config_path)
    artifact_path = Path(photometry_artifact_path)
    if verified_photometry_provenance is None:
        photometry_preflight = preflight_reviewed_photometry_namespace_artifact(
            artifact_path
        )
    else:
        photometry_preflight = verified_photometry_provenance.preflight
        if (
            photometry_preflight.path.resolve(strict=True)
            != artifact_path.resolve(strict=True)
            or verified_photometry_provenance.bank_dir.resolve(strict=True)
            != bank_dir.resolve(strict=True)
        ):
            raise ValueError(
                "Preflighted reviewed photometry operational role changed before loading."
            )
    current_photometry_provenance = verify_reviewed_photometry_bank_provenance(
        photometry_preflight,
        bank_dir=bank_dir,
    )
    if (
        verified_photometry_provenance is not None
        and verified_photometry_provenance.sanitized_provenance()
        != current_photometry_provenance.sanitized_provenance()
    ):
        raise ValueError("Preflighted reviewed photometry provenance changed before loading.")
    if photometry_namespace_compatibility_active():
        raise RuntimeError("Photometry compatibility scope leaked before model loading.")

    current_vae_provenance = verify_frozen_stage1_vae_bank_provenance(
        config_preflight,
        vae_checkpoint_path=vae_checkpoint_path,
        bank_dir=bank_dir,
    )
    if verified_vae_provenance is not None:
        if (
            verified_vae_provenance.config.path.resolve(strict=True)
            != vae_config_path.resolve(strict=True)
            or verified_vae_provenance.checkpoint_path.resolve(strict=True)
            != vae_checkpoint_path.resolve(strict=True)
            or verified_vae_provenance.bank_dir.resolve(strict=True)
            != bank_dir.resolve(strict=True)
            or verified_vae_provenance.sanitized_provenance()
            != current_vae_provenance.sanitized_provenance()
        ):
            raise ValueError("Preflighted frozen Stage-1 VAE provenance changed before loading.")

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
    runtime_manifest_vae = bank.manifest.get("vae")
    if not isinstance(runtime_manifest_vae, Mapping):
        raise ValueError("Restored bank manifest lacks frozen VAE provenance.")
    if (
        runtime_manifest_vae.get("config_sha256")
        != current_vae_provenance.bank_manifest_config_sha256
    ):
        raise ValueError("Restored bank manifest VAE config SHA-256 mismatch.")
    if (
        runtime_manifest_vae.get("checkpoint_sha256")
        != current_vae_provenance.bank_manifest_checkpoint_sha256
    ):
        raise ValueError("Restored bank manifest VAE checkpoint SHA-256 mismatch.")
    if (
        sha256_file(vae_checkpoint_path)
        != current_vae_provenance.checkpoint_file_sha256
    ):
        raise ValueError("Frozen Stage-1 Run C VAE checkpoint raw-file SHA-256 mismatch.")
    vae_config = current_vae_provenance.config.parsed_config
    vae_state = load_checkpoint(vae_checkpoint_path, map_location="cpu")
    vae_model = vae_config.get("model") if isinstance(vae_config, Mapping) else None
    if not isinstance(vae_model, Mapping):
        raise ValueError("Frozen VAE checkpoint lacks its complete model configuration.")
    effective_output_activation = vae_model.get("output_activation", "none")
    if effective_output_activation != "none":
        raise ValueError(
            "Frozen Stage-1 decoder effective output_activation must be exactly none."
        )
    encoder = build_encoder("kl_vae", **_kl_vae_kwargs(vae_model, "encoder"))
    decoder = build_decoder("kl_vae", **_kl_vae_kwargs(vae_model, "decoder"))
    _require_linear_decoder_output_activation(decoder)
    encoder.load_state_dict(vae_state["encoder"], strict=True)
    decoder.load_state_dict(vae_state["decoder"], strict=True)
    del vae_state
    gc.collect()

    if sha256_file(artifact_path) != REVIEWED_PHOTOMETRY_FILE_SHA256:
        raise ValueError("Frozen reviewed photometry artifact changed before runtime use.")
    artifact = current_photometry_provenance.preflight.artifact
    if artifact.artifact_sha256 != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256:
        raise ValueError("Frozen reviewed photometry in-memory identity changed.")
    if photometry_namespace_compatibility_active():
        raise RuntimeError("Photometry compatibility scope leaked into runtime use.")
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
        frozen_stage1_vae_provenance=current_vae_provenance.sanitized_provenance(),
        reviewed_photometry_provenance=(
            current_photometry_provenance.sanitized_provenance()
        ),
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
    verified_scientific_role_preflight: (
        FrozenP0006ScientificRolePreflight | None
    ) = None,
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
    if require_a100 and _runtime_frozen_stage1_vae_provenance(runtime).get(
        "synthetic_test_runtime"
    ) is True:
        raise ValueError("Production inference audit lacks frozen Stage-1 VAE provenance.")
    if require_a100 and _runtime_reviewed_photometry_provenance(runtime).get(
        "synthetic_test_runtime"
    ) is True:
        raise ValueError("Production inference audit lacks reviewed photometry provenance.")
    if photometry_namespace_compatibility_active():
        raise RuntimeError("Photometry compatibility scope leaked into the inference audit.")
    protocol_path = Path(protocol_path)
    readiness_path = Path(evaluation_readiness_path)
    if verified_scientific_role_preflight is None:
        if require_a100:
            raise ValueError(
                "Production inference audit lacks the early P:0006 scientific-role preflight."
            )
        verified_scientific_role_preflight = (
            preflight_frozen_p0006_scientific_role(protocol_path, readiness_path)
        )
    if (
        verified_scientific_role_preflight.protocol_path != protocol_path
        or verified_scientific_role_preflight.readiness_path != readiness_path
    ):
        raise ValueError("P:0006 scientific-role preflight paths changed before use.")
    current_scientific_role = verify_frozen_p0006_scientific_role_preflight(
        verified_scientific_role_preflight
    )
    protocol = current_scientific_role.protocol
    readiness = current_scientific_role.readiness
    scientific_role_provenance = current_scientific_role.sanitized_provenance()
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
        layout_provenance = _validate_full_volume_layout_provenance(
            gate.get("full_volume_layout_provenance")
        )
        range_policy = _validate_decoder_evaluation_range_policy(
            gate.get("decoder_evaluation_range_policy")
        )
        _validate_decoder_range_observations(
            gate.get("decoder_range_observations")
        )
        run_contract = _run_contract(
            plan,
            runtime,
            audit_implementation_commit=audit_implementation_commit,
            dependency_receipt=dependency_receipt,
            lpips_receipt=lpips_receipt,
            scientific_role_provenance=scientific_role_provenance,
            full_volume_layout_provenance=layout_provenance,
            decoder_evaluation_range_policy=range_policy,
        )
        _seal_or_verify_json(
            root / "run_contract.json", run_contract, "run_contract_sha256"
        )

        emitted = 0
        for streamed in iter_gate01_p0006_evaluation_cases(
            protocol_path, progress_callback=progress_callback
        ):
            if photometry_namespace_compatibility_active():
                raise RuntimeError(
                    "Photometry compatibility scope leaked into the case loop."
                )
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
                if (
                    _validate_full_volume_layout_provenance(
                        outputs.full_volume_layout_provenance
                    )
                    != layout_provenance
                ):
                    raise ValueError(
                        "P:0006 case full-volume layout differs from the one-case gate."
                    )
                if (
                    _validate_decoder_evaluation_range_policy(
                        outputs.decoder_evaluation_range_policy
                    )
                    != range_policy
                ):
                    raise ValueError(
                        "P:0006 case decoder evaluation-range policy differs from the gate."
                    )
                range_observations = _validate_decoder_range_observations(
                    outputs.decoder_range_observations
                )
                primary_range = range_observations["primary_unified_output"]
                if (
                    outputs.decoded_canonical_sha256
                    != primary_range["raw_canonical_tensor_sha256"]
                    or outputs.bounded_decoded_canonical_sha256
                    != primary_range["bounded_canonical_tensor_sha256"]
                ):
                    raise ValueError(
                        "P:0006 primary decoder raw/bounded identities changed."
                    )
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
                "bounded_decoded_canonical_sha256": (
                    outputs.bounded_decoded_canonical_sha256
                ),
                "decoder_evaluation_range_policy": range_policy,
                "decoder_range_observations": range_observations,
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
        scientific_role_provenance=scientific_role_provenance,
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
        scientific_role_provenance=scientific_role_provenance,
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
    target_layout = adapt_full_volume_layout(case.target, role="P:0006 metric target")
    target = target_layout.spatial_volume()
    scored: dict[str, dict[str, float]] = {}
    for method, prediction in outputs.methods.items():
        prediction_layout = adapt_full_volume_layout(
            prediction,
            role=f"P:0006 metric method {method}",
            expected_shape=target_layout.authenticated_shape,
        )
        prediction_volume = prediction_layout.spatial_volume()
        values = dict(metric_fn(prediction_volume, target, _METRICS, device))
        if set(values) != set(_METRICS) or any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Official P:0006 metric result is incomplete or nonfinite.")
        values["edge_mae"] = float(
            gradient_mae(prediction_volume[None], target[None]).cpu()
        )
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
            gate.get("contract_version") != MEMORY_GATE_CONTRACT
            or gate.get("inference_plan_sha256") != plan.sha256
            or gate.get("memory_gate_case") != plan.payload["memory_gate_case"]
            or gate.get("gpu_identity") != dict(runtime.gpu_identity)
            or gate.get("model_state_before") != dict(initial_state)
            or gate.get("model_state_after") != dict(initial_state)
            or gate.get("status") != "pass"
        ):
            raise ValueError("Existing one-case inference gate is incompatible.")
        _validate_full_volume_layout_provenance(
            gate.get("full_volume_layout_provenance")
        )
        _validate_decoder_evaluation_range_policy(
            gate.get("decoder_evaluation_range_policy")
        )
        _validate_decoder_range_observations(
            gate.get("decoder_range_observations")
        )
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
    layout_provenance: dict[str, Any] | None = None
    range_policy: dict[str, Any] | None = None
    range_observations: dict[str, dict[str, Any]] | None = None
    try:
        outputs = runtime.infer_case(streamed.case, streamed.calibrator, plan=plan)
        layout_provenance = _validate_full_volume_layout_provenance(
            outputs.full_volume_layout_provenance
        )
        range_policy = _validate_decoder_evaluation_range_policy(
            outputs.decoder_evaluation_range_policy
        )
        range_observations = _validate_decoder_range_observations(
            outputs.decoder_range_observations
        )
        primary_range = range_observations["primary_unified_output"]
        if (
            outputs.decoded_canonical_sha256
            != primary_range["raw_canonical_tensor_sha256"]
            or outputs.bounded_decoded_canonical_sha256
            != primary_range["bounded_canonical_tensor_sha256"]
        ):
            raise ValueError("One-case decoder raw/bounded identities changed.")
    finally:
        if outputs is not None:
            _release_case_outputs(outputs)
        del outputs
        streamed.release()
        iterator.close()
    if layout_provenance is None:
        raise RuntimeError("One-case inference gate did not produce layout provenance.")
    if range_policy is None or range_observations is None:
        raise RuntimeError("One-case inference gate did not produce range provenance.")
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
        "full_volume_layout_provenance": layout_provenance,
        "decoder_evaluation_range_policy": range_policy,
        "decoder_range_observations": range_observations,
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
    _validate_decoder_evaluation_range_policy(
        receipt.get("decoder_evaluation_range_policy")
    )
    range_observations = _validate_decoder_range_observations(
        receipt.get("decoder_range_observations")
    )
    primary_range = range_observations["primary_unified_output"]
    if (
        receipt.get("decoded_canonical_sha256")
        != primary_range["raw_canonical_tensor_sha256"]
        or receipt.get("bounded_decoded_canonical_sha256")
        != primary_range["bounded_canonical_tensor_sha256"]
    ):
        raise ValueError("Inference case decoder raw/bounded identities changed.")
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
    scientific_role_provenance: Mapping[str, Any],
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
        "frozen_p0006_scientific_role_provenance": dict(
            scientific_role_provenance
        ),
        "inference_plan_sha256": plan.sha256,
        "full_volume_layout_provenance": _validate_full_volume_layout_provenance(
            gate.get("full_volume_layout_provenance")
        ),
        "decoder_evaluation_range_policy": (
            _validate_decoder_evaluation_range_policy(
                gate.get("decoder_evaluation_range_policy")
            )
        ),
        "one_case_decoder_range_observations": (
            _validate_decoder_range_observations(
                gate.get("decoder_range_observations")
            )
        ),
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
        "frozen_stage1_vae_provenance": _runtime_frozen_stage1_vae_provenance(runtime),
        "reviewed_photometry_provenance": (
            _runtime_reviewed_photometry_provenance(runtime)
        ),
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
    scientific_role_provenance: Mapping[str, Any],
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
        "evaluation_readiness_file_sha256": EVALUATION_READINESS_FILE_SHA256,
        "evaluation_readiness_sha256": EVALUATION_READINESS_SHA256,
        "frozen_p0006_scientific_role_provenance": dict(
            scientific_role_provenance
        ),
        "inference_plan_sha256": plan.sha256,
        "full_volume_layout_provenance": _validate_full_volume_layout_provenance(
            gate.get("full_volume_layout_provenance")
        ),
        "decoder_evaluation_range_policy": (
            _validate_decoder_evaluation_range_policy(
                gate.get("decoder_evaluation_range_policy")
            )
        ),
        "one_case_decoder_range_observations": (
            _validate_decoder_range_observations(
                gate.get("decoder_range_observations")
            )
        ),
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
        "frozen_stage1_vae_provenance": _runtime_frozen_stage1_vae_provenance(runtime),
        "reviewed_photometry_provenance": (
            _runtime_reviewed_photometry_provenance(runtime)
        ),
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
    scientific_role_provenance: Mapping[str, Any],
    full_volume_layout_provenance: Mapping[str, Any],
    decoder_evaluation_range_policy: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "contract_version": "stage2-step200-p0006-inference-run-v7",
        "training_evidence_commit": TRAINING_EVIDENCE_COMMIT,
        "audit_implementation_commit": audit_implementation_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "run_fingerprint": RUN_FINGERPRINT,
        "protocol_sha256": P0006_PROTOCOL_SHA256,
        "evaluation_readiness_sha256": EVALUATION_READINESS_SHA256,
        "frozen_p0006_scientific_role_provenance": dict(
            scientific_role_provenance
        ),
        "inference_plan_sha256": plan.sha256,
        "full_volume_layout_provenance": _validate_full_volume_layout_provenance(
            full_volume_layout_provenance
        ),
        "decoder_evaluation_range_policy": (
            _validate_decoder_evaluation_range_policy(
                decoder_evaluation_range_policy
            )
        ),
        "model_state": dict(runtime.state_identity()),
        "GPU_identity": dict(runtime.gpu_identity),
        "dependency_environment_sha256": sha256_json(dependency_receipt),
        "lpips_provenance_sha256": sha256_json(lpips_receipt),
        "frozen_stage1_vae_provenance": _runtime_frozen_stage1_vae_provenance(runtime),
        "reviewed_photometry_provenance": (
            _runtime_reviewed_photometry_provenance(runtime)
        ),
        "training_invoked": False,
        "gradients_enabled": False,
        "optimizer_loaded": False,
        "P0006_training_or_model_selection_use": False,
        "population_or_generalization_claims_authorized": False,
        "long_run_training_authorized": False,
    }
    body["run_contract_sha256"] = sha256_json(body)
    return body


def _runtime_frozen_stage1_vae_provenance(
    runtime: Step200CaseInferenceRuntime,
) -> dict[str, Any]:
    provenance = getattr(runtime, "frozen_stage1_vae_provenance", None)
    if provenance is None:
        return {"synthetic_test_runtime": True}
    if not isinstance(provenance, Mapping):
        raise ValueError("Frozen Stage-1 VAE runtime provenance is malformed.")
    body = dict(provenance)
    if set(body) != {
        "contract_version",
        "config_role",
        "raw_config_file_sha256",
        "raw_config_file_size_bytes",
        "parsed_canonical_config_sha256",
        "raw_config_identity_match",
        "bank_manifest_config_sha256",
        "checkpoint_file_sha256",
        "bank_manifest_checkpoint_sha256",
        "checkpoint_identity_match",
    }:
        raise ValueError("Frozen Stage-1 VAE runtime provenance key inventory changed.")
    if (
        body["contract_version"] != FROZEN_STAGE1_VAE_PROVENANCE_CONTRACT
        or body["config_role"] != "frozen_stage1_run_c"
        or body["raw_config_file_sha256"] != STAGE1_RUN_C_CONFIG_SHA256
        or body["raw_config_file_size_bytes"] != STAGE1_RUN_C_CONFIG_SIZE_BYTES
        or body["bank_manifest_config_sha256"] != STAGE1_RUN_C_CONFIG_SHA256
        or body["raw_config_identity_match"] is not True
        or body["checkpoint_file_sha256"] != STAGE1_RUN_C_CHECKPOINT_SHA256
        or body["bank_manifest_checkpoint_sha256"] != STAGE1_RUN_C_CHECKPOINT_SHA256
        or body["checkpoint_identity_match"] is not True
        or _SHA256_RE.fullmatch(str(body["parsed_canonical_config_sha256"])) is None
    ):
        raise ValueError("Frozen Stage-1 VAE runtime provenance identity changed.")
    return copy.deepcopy(body)


def _runtime_reviewed_photometry_provenance(
    runtime: Step200CaseInferenceRuntime,
) -> dict[str, Any]:
    provenance = getattr(runtime, "reviewed_photometry_provenance", None)
    if provenance is None:
        return {"synthetic_test_runtime": True}
    if not isinstance(provenance, Mapping):
        raise ValueError("Reviewed photometry runtime provenance is malformed.")
    body = dict(provenance)
    expected_keys = {
        "contract_version",
        "artifact_role",
        "artifact_file_sha256",
        "artifact_internal_sha256",
        "artifact_production_commit",
        "protected_base_module_sha256",
        "historical_operator_overlay_sha256",
        "namespace_predicate_source_sha256",
        "accepted_records_sha256",
        "excluded_records_sha256",
        "accepted_record_count",
        "prospective_accepted_count",
        "prospective_excluded_count",
        "retrospective_numeric_collision_count",
        "retrospective_collision_group_counts",
        "compatibility_scope_restored",
        "bank_manifest_artifact_file_sha256",
        "bank_manifest_artifact_sha256",
        "bank_photometry_identity_match",
    }
    if set(body) != expected_keys:
        raise ValueError("Reviewed photometry runtime provenance key inventory changed.")
    count_fields = {
        "accepted_record_count": REVIEWED_PHOTOMETRY_ACCEPTED_RECORD_COUNT,
        "prospective_accepted_count": 0,
        "prospective_excluded_count": REVIEWED_PHOTOMETRY_PROSPECTIVE_EXCLUDED_COUNT,
        "retrospective_numeric_collision_count": (
            REVIEWED_PHOTOMETRY_RETROSPECTIVE_COLLISION_COUNT
        ),
    }
    if any(
        isinstance(body[key], bool)
        or not isinstance(body[key], int)
        or body[key] != expected
        for key, expected in count_fields.items()
    ):
        raise ValueError("Reviewed photometry runtime provenance counts changed.")
    collision_groups = body["retrospective_collision_group_counts"]
    if not isinstance(collision_groups, Mapping):
        raise ValueError("Reviewed photometry collision-group provenance is malformed.")
    if (
        body["contract_version"]
        != REVIEWED_PHOTOMETRY_NAMESPACE_PROVENANCE_CONTRACT
        or body["artifact_role"] != REVIEWED_PHOTOMETRY_ROLE
        or body["artifact_file_sha256"] != REVIEWED_PHOTOMETRY_FILE_SHA256
        or body["artifact_internal_sha256"] != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256
        or body["artifact_production_commit"]
        != REVIEWED_PHOTOMETRY_PRODUCTION_COMMIT
        or body["protected_base_module_sha256"]
        != PROTECTED_PHOTOMETRY_MODULE_SHA256
        or body["historical_operator_overlay_sha256"]
        != HISTORICAL_PHOTOMETRY_OPERATOR_OVERLAY_SHA256
        or body["namespace_predicate_source_sha256"]
        != REVIEWED_NAMESPACE_PREDICATE_SOURCE_SHA256
        or body["bank_manifest_artifact_file_sha256"]
        != REVIEWED_PHOTOMETRY_FILE_SHA256
        or body["bank_manifest_artifact_sha256"]
        != REVIEWED_PHOTOMETRY_ARTIFACT_SHA256
        or body["compatibility_scope_restored"] is not True
        or body["bank_photometry_identity_match"] is not True
        or dict(collision_groups)
        != REVIEWED_PHOTOMETRY_COLLISION_GROUP_COUNTS
        or _SHA256_RE.fullmatch(str(body["accepted_records_sha256"])) is None
        or _SHA256_RE.fullmatch(str(body["excluded_records_sha256"])) is None
    ):
        raise ValueError("Reviewed photometry runtime provenance identity changed.")
    return copy.deepcopy(body)


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
        "dependency_download_invoked",
        "dependency_download_observed",
        "lpips_bootstrap_state",
        "lpips_force_reinstall_invoked",
        "notebook_installed_packages",
        "pip_install_invoked",
    ):
        result.pop(key, None)
    return result


def _load_pinned_self_hashed_snapshot(
    path: Path,
    *,
    role: str,
    expected_file_sha256: str,
    hash_key: str,
    expected_internal_sha256: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pinned {role} is missing.")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Pinned {role} is not a non-symlink regular file.")
    raw_bytes = path.read_bytes()
    observed_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if observed_file_sha256 != expected_file_sha256:
        raise ValueError(f"Pinned {role} file SHA-256 mismatch.")
    try:
        parsed = json.loads(
            raw_bytes.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Pinned {role} JSON is malformed.") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Pinned {role} root must be a JSON object.")
    payload = dict(parsed)
    body = dict(payload)
    stored_internal_sha256 = body.pop(hash_key, None)
    if (
        stored_internal_sha256 != sha256_json(body)
        or stored_internal_sha256 != expected_internal_sha256
    ):
        raise ValueError(f"Pinned {role} internal SHA-256 mismatch.")
    if path.read_bytes() != raw_bytes:
        raise ValueError(f"Pinned {role} changed while being authenticated.")
    return payload


def _require_exact_nonnegative_int(value: Any, label: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} must be exactly {expected}.")


def _validate_p0006_protocol_v4(protocol: Mapping[str, Any]) -> None:
    if protocol.get("contract_version") != GATE01_P0006_EVALUATION_PROTOCOL:
        raise ValueError("P:0006 protocol contract version changed.")
    if protocol.get("data_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE:
        raise ValueError("P:0006 protocol data role changed.")
    if protocol.get("evidence_interpretation") != P0006_EVIDENCE_LIMITATION:
        raise ValueError("P:0006 protocol evidence interpretation changed.")
    if protocol.get("subject_group_identity") != P0006_SUBJECT_GROUP:
        raise ValueError("P:0006 protocol subject-group identity changed.")
    if protocol.get("traveller_identity_sha256") != P0006_IDENTITY_SHA256:
        raise ValueError("P:0006 protocol traveller identity changed.")
    if protocol.get("population_or_generalization_claims_authorized") is not False:
        raise ValueError("P:0006 protocol authorized population/generalization claims.")
    if protocol.get("training_or_model_selection_use") is not False:
        raise ValueError("P:0006 protocol authorized training or model selection.")
    if protocol.get("private_arrays_validated") is not True:
        raise ValueError("P:0006 protocol lacks complete private-array validation.")
    if protocol.get("P0009_confirmation_status") != P0009_CONFIRMATION_STATUS:
        raise ValueError("P:0006 protocol changed frozen P:0009 status.")
    if protocol.get("P0009_executed") is not False:
        raise ValueError("P:0006 protocol indicates P:0009 execution.")
    if protocol.get("forbidden_travellers") != ["P:0007", "P:0009"]:
        raise ValueError("P:0006 protocol forbidden-traveller inventory changed.")
    _require_exact_nonnegative_int(
        protocol.get("acquisition_count"), "P:0006 acquisition count", 15
    )
    _require_exact_nonnegative_int(
        protocol.get("directed_pair_count"), "P:0006 directed-pair count", 60
    )
    _require_exact_nonnegative_int(
        protocol.get("wrong_target_reference_count"),
        "P:0006 wrong-target reference count",
        180,
    )
    factored_bank = protocol.get("factored_bank")
    if not isinstance(factored_bank, Mapping):
        raise ValueError("P:0006 protocol factored-bank P-record count changed.")
    _require_exact_nonnegative_int(
        factored_bank.get("P_record_count"),
        "P:0006 protocol factored-bank P-record count",
        0,
    )
    frozen_validation = protocol.get("frozen_unpaired_validation")
    if not isinstance(frozen_validation, Mapping):
        raise ValueError("P:0006 protocol validation P-endpoint count changed.")
    _require_exact_nonnegative_int(
        frozen_validation.get("P_endpoint_count"),
        "P:0006 protocol validation P-endpoint count",
        0,
    )
    receipts = protocol.get("case_receipts")
    if not isinstance(receipts, list) or len(receipts) != 60:
        raise ValueError("P:0006 protocol must contain exactly 60 case receipts.")
    expected_cells = {
        (contrast.value, float(source), float(target))
        for contrast in CONTRASTS
        for source in FIELD_STRENGTHS_T
        for target in FIELD_STRENGTHS_T
        if source != target
    }
    observed_cells: set[tuple[str, float, float]] = set()
    observed_nodes: set[tuple[str, float]] = set()
    observed_case_identities: set[str] = set()
    wrong_target_count = 0
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("P:0006 protocol contains a malformed case receipt.")
        case_identity = receipt.get("case_identity")
        if not isinstance(case_identity, str) or not case_identity:
            raise ValueError("P:0006 protocol contains a malformed case identity.")
        if case_identity in observed_case_identities:
            raise ValueError("P:0006 protocol contains a duplicate case identity.")
        observed_case_identities.add(case_identity)
        try:
            source = Domain.from_dict(dict(receipt["source_domain"]))
            target = Domain.from_dict(dict(receipt["target_domain"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("P:0006 protocol contains malformed case domains.") from error
        if source.contrast != target.contrast or source == target:
            raise ValueError("P:0006 protocol contains an invalid directed domain pair.")
        cell = (
            source.contrast.value,
            float(source.field_strength_t),
            float(target.field_strength_t),
        )
        if cell in observed_cells:
            raise ValueError("P:0006 protocol contains a duplicate directed domain pair.")
        observed_cells.add(cell)
        observed_nodes.add((source.contrast.value, float(source.field_strength_t)))
        observed_nodes.add((target.contrast.value, float(target.field_strength_t)))
        wrong_targets = receipt.get("wrong_target_sb_v2_sha256")
        if not isinstance(wrong_targets, Mapping) or len(wrong_targets) != 3:
            raise ValueError("P:0006 protocol case has an invalid wrong-target inventory.")
        if any(_SHA256_RE.fullmatch(str(value)) is None for value in wrong_targets.values()):
            raise ValueError("P:0006 protocol contains a malformed wrong-target identity.")
        wrong_target_count += len(wrong_targets)
    if observed_cells != expected_cells:
        raise ValueError("P:0006 protocol directed-pair inventory changed.")
    if len(observed_nodes) != 15:
        raise ValueError("P:0006 protocol acquisition-node inventory changed.")
    if wrong_target_count != 180:
        raise ValueError("P:0006 protocol wrong-target inventory changed.")


def _validate_readiness_v3(
    readiness: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    expected_keys = {
        "contract_version",
        "long_run_authorized_by_evaluation_path",
        "evaluation_role",
        "evidence_interpretation",
        "population_or_generalization_claims_authorized",
        "prospective_protocol_used",
        "prospective_training_or_model_selection_use",
        "reviewed_prospective_protocol_available",
        "complete_inventory_no_selection",
        "directed_pair_count",
        "feasibility_result_sha256",
        "retrospective_pair_feasibility",
        "p0006_evaluation_protocol_sha256",
        "p0006_gate01_result_file_sha256",
        "factored_bank_P_record_count",
        "unpaired_validation_P_endpoint_count",
        "P0009_confirmation_status",
        "P0009_executed",
        "readiness_sha256",
    }
    if set(readiness) != expected_keys:
        raise ValueError("Evaluation-readiness v3 key inventory changed.")
    if readiness.get("contract_version") != LONG_RUN_EVALUATION_READINESS_CONTRACT:
        raise ValueError("Evaluation-readiness contract version changed.")
    if readiness.get("long_run_authorized_by_evaluation_path") is not True:
        raise ValueError("Evaluation path is not authorized by the sealed readiness evidence.")
    if readiness.get("evaluation_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE:
        raise ValueError("Evaluation-readiness role changed.")
    if readiness.get("evidence_interpretation") != P0006_EVIDENCE_LIMITATION:
        raise ValueError("Evaluation-readiness evidence interpretation changed.")
    false_fields = (
        "population_or_generalization_claims_authorized",
        "prospective_training_or_model_selection_use",
        "retrospective_pair_feasibility",
        "P0009_executed",
    )
    if any(readiness.get(field) is not False for field in false_fields):
        raise ValueError("Evaluation-readiness safety boundary changed.")
    true_fields = (
        "prospective_protocol_used",
        "reviewed_prospective_protocol_available",
        "complete_inventory_no_selection",
    )
    if any(readiness.get(field) is not True for field in true_fields):
        raise ValueError("Evaluation-readiness prospective inventory is incomplete.")
    _require_exact_nonnegative_int(
        readiness.get("directed_pair_count"), "Readiness directed-pair count", 60
    )
    _require_exact_nonnegative_int(
        readiness.get("factored_bank_P_record_count"),
        "Readiness factored-bank P-record count",
        0,
    )
    _require_exact_nonnegative_int(
        readiness.get("unpaired_validation_P_endpoint_count"),
        "Readiness validation P-endpoint count",
        0,
    )
    if readiness.get("p0006_evaluation_protocol_sha256") != protocol.get(
        "protocol_sha256"
    ):
        raise ValueError("Evaluation readiness references another P:0006 protocol.")
    gate01_result = protocol.get("gate01_result")
    if (
        not isinstance(gate01_result, Mapping)
        or readiness.get("p0006_gate01_result_file_sha256")
        != gate01_result.get("file_sha256")
    ):
        raise ValueError("Evaluation readiness references another Gate 0.1 result.")
    if _SHA256_RE.fullmatch(str(readiness.get("feasibility_result_sha256", ""))) is None:
        raise ValueError("Evaluation-readiness feasibility identity is malformed.")
    if readiness.get("P0009_confirmation_status") != P0009_CONFIRMATION_STATUS:
        raise ValueError("Evaluation-readiness P:0009 status changed.")


def _validate_scientific_role(protocol: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    _validate_p0006_protocol_v4(protocol)
    _validate_readiness_v3(readiness, protocol)


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
    spatial = adapt_full_volume_layout(value, role=f"montage volume {name}").spatial_volume()
    array = spatial.detach().cpu().float().numpy()
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


def _bf16_inference_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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
    "AdaptedFullVolume",
    "ARTIFACT_MANIFEST_CONTRACT",
    "CASE_RECEIPT_CONTRACT",
    "CHECKPOINT_SHA256",
    "DECODER_EVALUATION_RANGE_CONTRACT",
    "DecoderEvaluationRange",
    "FROZEN_STAGE1_VAE_PROVENANCE_CONTRACT",
    "FROZEN_P0006_SCIENTIFIC_ROLE_PREFLIGHT_CONTRACT",
    "FULL_VOLUME_LAYOUT_ADAPTER_CONTRACT",
    "FrozenP0006ScientificRolePreflight",
    "FrozenStage1VAEConfigPreflight",
    "FrozenStage1VAEProvenance",
    "FrozenStep200InferencePlan",
    "INFERENCE_AUDIT_CONTRACT",
    "INFERENCE_PLAN_CONTRACT",
    "InferenceCaseOutputs",
    "MEMORY_GATE_CONTRACT",
    "MONTAGE_SPECIFICATION_CONTRACT",
    "P0006_PROTOCOL_FILE_SHA256",
    "P0006_PROTOCOL_SHA256",
    "PROTECTED_PHOTOMETRY_MODULE_SHA256",
    "REVIEWED_NAMESPACE_PREDICATE_SOURCE_SHA256",
    "REVIEWED_PHOTOMETRY_ARTIFACT_SHA256",
    "REVIEWED_PHOTOMETRY_FILE_SHA256",
    "REVIEWED_PHOTOMETRY_NAMESPACE_PROVENANCE_CONTRACT",
    "ReviewedPhotometryBankProvenance",
    "ReviewedPhotometryPreflight",
    "RUN_FINGERPRINT",
    "STAGE1_RUN_C_CONFIG_SIZE_BYTES",
    "STAGE1_RUN_C_CONFIG_BASENAME",
    "Step200CaseInferenceRuntime",
    "TRAINING_EVIDENCE_COMMIT",
    "UnifiedStep200InferenceRuntime",
    "adapt_decoder_evaluation_range",
    "adapt_full_volume_layout",
    "build_frozen_step200_inference_plan",
    "decoder_evaluation_range_policy",
    "load_unified_step200_inference_runtime",
    "photometry_namespace_compatibility_active",
    "preflight_frozen_stage1_run_c_config",
    "preflight_frozen_p0006_scientific_role",
    "preflight_reviewed_photometry_namespace_artifact",
    "run_step200_p0006_inference_audit",
    "verify_frozen_stage1_vae_bank_provenance",
    "verify_frozen_p0006_scientific_role_preflight",
    "verify_reviewed_photometry_bank_provenance",
]
