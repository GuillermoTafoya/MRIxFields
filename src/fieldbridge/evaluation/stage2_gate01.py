"""Frozen Gate 0.1 equal-photometry full-volume diagnostic.

The central comparison is calibrated SB-v2 versus calibrated identity under the same
method-agnostic target-domain calibrator.  Evaluation targets are visible only to the
source-pinned official metric functions; calibration receives predictions and frozen
training templates only.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Contrast, Domain
from fieldbridge.evaluation.mrixfields2026_official import (
    OFFICIAL_TASK3_METRIC_CONTRACT,
    load_official_nifti,
    official_task3_lpips,
    official_task3_nrmse,
    official_task3_runtime_provenance,
    official_task3_ssim,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    FULL_LATENT_BANK_SOURCE_SPLIT_FILE_SHA256,
    FULL_LATENT_BANK_SOURCE_SPLIT_FINGERPRINT,
    GATE0_DIAGNOSTIC_COMMIT,
    GATE01_CALIBRATION_SEMANTICS,
    GATE01_SUPPORT_THRESHOLD,
    PosthocTargetCalibrator,
    RESPLIT_FINGERPRINT,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
    reject_target_derived_calibration_fields,
)
from fieldbridge.evaluation.stage2_gate01_protocol import (
    GATE01_SCIENTIFIC_MODULES,
    Gate01ProtocolLock,
)

GATE01_CONTRACT_VERSION = "stage2-gate01-equal-photometry-v2"
GATE01_INPUT_CONTRACT_VERSION = "stage2-gate01-input-v5"
GATE01_VERIFIED_PRODUCER_RECEIPT_VERSION = (
    "stage2-gate01-verified-producer-receipt-v2"
)
GATE01_SCIENTIFIC_CASE_COUNT = 60
GATE01_EXECUTION_MODES = ("scientific", "development-incomplete")

OFFICIAL_METRICS = ("nrmse", "ssim", "lpips")
LOWER_IS_BETTER = {"nrmse": True, "ssim": False, "lpips": True}
CORE_METHODS = (
    "raw_identity",
    "calibrated_identity",
    "raw_sb_v2",
    "calibrated_sb_v2",
    "stage1_reconstruction_ceiling",
)
DIAGNOSTIC_METHODS = (
    "diagnostic_robust_affine_identity",
    "diagnostic_robust_affine_sb_v2",
)

MetricFn = Callable[[torch.Tensor, torch.Tensor, Sequence[str], str], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class Gate01Case:
    """One directed, same-contrast full-volume evaluation pair."""

    case_id: str
    source_domain: Domain
    target_domain: Domain
    target: torch.Tensor
    raw_identity: torch.Tensor
    raw_sb_v2: torch.Tensor
    stage1_reconstruction_ceiling: torch.Tensor
    support_mask: torch.Tensor
    traveller_identity_sha256: str = ""
    array_sha256: Mapping[str, str] = field(default_factory=dict)
    wrong_target_sb_v2: Mapping[str, torch.Tensor] = field(default_factory=dict)
    source_image: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate 0.1 case_id must be non-empty.")
        source_contrast = Contrast.parse(self.source_domain.contrast)
        target_contrast = Contrast.parse(self.target_domain.contrast)
        if source_contrast != target_contrast:
            raise ValueError("Gate 0.1 evaluates same-contrast field translation only.")
        if self.source_domain.field_strength_t == self.target_domain.field_strength_t:
            raise ValueError("Gate 0.1 cases must be directed non-identity field pairs.")

        tensors = {
            "target": self.target,
            "raw_identity": self.raw_identity,
            "raw_sb_v2": self.raw_sb_v2,
            "stage1_reconstruction_ceiling": self.stage1_reconstruction_ceiling,
            **{
                f"wrong_target_sb_v2[{label}]": tensor
                for label, tensor in self.wrong_target_sb_v2.items()
            },
        }
        if self.source_image is not None:
            tensors["source_image"] = self.source_image
        target_shape = tuple(self.target.shape)
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Gate 0.1 {name} must be a torch.Tensor.")
            if tensor.ndim < 3:
                raise ValueError(f"Gate 0.1 {name} is not a full volume.")
            if tuple(tensor.shape) != target_shape:
                raise ValueError(
                    f"Gate 0.1 shape mismatch for {name}: "
                    f"{tuple(tensor.shape)} != {target_shape}."
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Gate 0.1 {name} contains non-finite values.")

        if not isinstance(self.support_mask, torch.Tensor):
            raise TypeError("Gate 0.1 support_mask must be a torch.Tensor.")
        if self.support_mask.dtype != torch.bool:
            raise ValueError("Gate 0.1 support_mask must have boolean dtype.")
        if tuple(self.support_mask.shape) != target_shape:
            raise ValueError(
                "Gate 0.1 support-mask shape mismatch: "
                f"{tuple(self.support_mask.shape)} != {target_shape}."
            )
        if not bool(self.support_mask.any()):
            raise ValueError("Gate 0.1 support_mask has no foreground voxels.")

        for label in self.wrong_target_sb_v2:
            wrong = _wrong_target_domain(label, target_contrast)
            if wrong.field_strength_t == self.target_domain.field_strength_t:
                raise ValueError(
                    "wrong_target_sb_v2 must not repeat the requested target field."
                )

        if self.traveller_identity_sha256 and not _is_sha256(
            self.traveller_identity_sha256
        ):
            raise ValueError("Gate 0.1 traveller identity must be a SHA-256 digest.")

    @property
    def case_identity_sha256(self) -> str:
        """Sanitized case identity used in committed/reportable result structures."""

        return hashlib.sha256(self.case_id.encode("utf-8")).hexdigest()


def frozen_artifact_provenance() -> dict[str, str]:
    """Return the exact scientific identities frozen for Gate 0.1."""

    return {
        "stage1_run_c_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
        "full_latent_bank_build_commit": FULL_LATENT_BANK_BUILD_COMMIT,
        "gate0_diagnostic_commit": GATE0_DIAGNOSTIC_COMMIT,
        "sb_v2_checkpoint_sha256": SB_V2_CHECKPOINT_SHA256,
        "resplit_fingerprint": RESPLIT_FINGERPRINT,
    }


def validate_frozen_artifact_provenance(provenance: Mapping[str, Any]) -> None:
    """Reject missing, stale, or incompatible method/checkpoint provenance."""

    expected = frozen_artifact_provenance()
    missing = sorted(set(expected) - set(provenance))
    if missing:
        raise ValueError(f"Gate 0.1 artifact provenance is incomplete; missing {missing}.")
    unexpected = sorted(set(provenance) - set(expected))
    if unexpected:
        raise ValueError(
            f"Gate 0.1 artifact provenance has unexpected fields {unexpected}."
        )
    for key, value in expected.items():
        if str(provenance[key]) != value:
            raise ValueError(
                f"Gate 0.1 frozen artifact mismatch for {key}: "
                f"{provenance[key]!r} != {value!r}."
            )


def official_gate01_metric_fn(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metrics: Sequence[str],
    device: str,
) -> Mapping[str, float]:
    """Use the unchanged source-pinned official metric functions on a full volume."""

    pred = _official_array(prediction, "prediction")
    tgt = _official_array(target, "target")
    result: dict[str, float] = {}
    if "nrmse" in metrics:
        result["nrmse"] = official_task3_nrmse(pred, tgt)
    if "ssim" in metrics:
        result["ssim"] = official_task3_ssim(pred, tgt)
    if "lpips" in metrics:
        result["lpips"] = official_task3_lpips(pred, tgt, device=device)
    return result


@torch.inference_mode()
def evaluate_gate01(
    cases: Iterable[Gate01Case],
    *,
    calibrator: PosthocTargetCalibrator,
    artifact_provenance: Mapping[str, Any],
    code_commit: str,
    evidence_scope: Mapping[str, Any],
    input_manifest_sha256: str,
    producer_receipt: Mapping[str, Any] | None = None,
    split_provenance: Mapping[str, Any] | None = None,
    support_threshold: float = GATE01_SUPPORT_THRESHOLD,
    execution_mode: str = "development-incomplete",
    selection_fingerprint_sha256: str | None = None,
    code_provenance: Mapping[str, Any] | None = None,
    protocol_lock: Gate01ProtocolLock | None = None,
    metrics: Sequence[str] = OFFICIAL_METRICS,
    device: str = "cuda",
    include_robust_affine: bool = False,
    metric_fn: MetricFn = official_gate01_metric_fn,
    case_observer: Callable[[Gate01Case, Mapping[str, torch.Tensor]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate raw/equal-calibrated references and reduce the frozen Gate 0.1 tables."""

    if execution_mode not in GATE01_EXECUTION_MODES:
        raise ValueError(f"Unknown Gate 0.1 execution mode {execution_mode!r}.")
    requested_metrics = tuple(metrics)
    unsupported = sorted(set(requested_metrics) - set(OFFICIAL_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported Gate 0.1 official metrics: {unsupported}.")
    if "nrmse" not in requested_metrics:
        raise ValueError("Gate 0.1 requires official nRMSE to freeze difficulty strata.")
    if execution_mode == "scientific" and set(requested_metrics) != set(OFFICIAL_METRICS):
        raise ValueError(
            "Scientific Gate 0.1 mode requires nRMSE, SSIM, and LPIPS together."
        )
    if not code_commit:
        raise ValueError("Gate 0.1 requires the evaluation code commit.")
    if not input_manifest_sha256:
        raise ValueError("Gate 0.1 requires the input manifest SHA-256.")
    evidence_scope = _validated_evidence_scope(evidence_scope)
    validate_frozen_artifact_provenance(artifact_provenance)
    calibrator.assert_split_fingerprint(RESPLIT_FINGERPRINT)
    if support_threshold != GATE01_SUPPORT_THRESHOLD:
        raise ValueError("Gate 0.1 runtime support threshold is frozen at exactly 0.0.")
    if calibrator.support_threshold != support_threshold:
        raise ValueError("Gate 0.1 runtime/calibrator support threshold mismatch.")
    normalized_code_provenance = dict(code_provenance or {})
    normalized_producer_receipt = (
        validate_gate01_verified_producer_receipt(producer_receipt)
        if producer_receipt is not None
        else None
    )
    if execution_mode == "scientific":
        if protocol_lock is None:
            raise ValueError("Scientific Gate 0.1 mode requires an external protocol lock.")
        if normalized_producer_receipt is None:
            raise ValueError(
                "Scientific Gate 0.1 mode requires a verified full-decode producer receipt."
            )
        producer_protocol = normalized_producer_receipt["producer_receipt"][
            "protocol_lock_artifact_sha256"
        ]
        if producer_protocol != protocol_lock.artifact_sha256:
            raise ValueError("Gate 0.1 producer receipt/protocol-lock identity mismatch.")
        protocol_lock.assert_evaluation_code_provenance(
            code_commit=code_commit,
            code_provenance=normalized_code_provenance,
        )
        if selection_fingerprint_sha256 is None or not _is_sha256(
            selection_fingerprint_sha256
        ):
            raise ValueError(
                "Scientific Gate 0.1 mode requires a predeclared selection fingerprint."
            )
        protocol_lock.assert_calibrator(calibrator)
        protocol_lock.assert_manifest_contract(
            traveller_identity_sha256=evidence_scope["traveller_identity_sha256"],
            selection_fingerprint_sha256=str(selection_fingerprint_sha256),
            split_fingerprint=RESPLIT_FINGERPRINT,
            split_provenance=dict(split_provenance or {}),
            support_threshold=support_threshold,
            artifact_provenance=artifact_provenance,
        )
        protocol_lock.assert_runtime_contract(
            metrics=requested_metrics,
            montage_specification=fixed_montage_specifications(),
        )

    method_names = list(CORE_METHODS)
    if include_robust_affine:
        method_names.extend(DIAGNOSTIC_METHODS)

    rows: list[dict[str, Any]] = []
    selection_descriptors: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_directions: set[tuple[str, float, float]] = set()
    traveller_hashes: set[str] = set()
    graph_specs: list[dict[str, Any]] = []
    for case in cases:
        _validate_case_content_hashes(case)
        if case.case_identity_sha256 in seen_case_ids:
            raise ValueError("Gate 0.1 input contains duplicate case IDs.")
        seen_case_ids.add(case.case_identity_sha256)
        direction = (
            Contrast.parse(case.target_domain.contrast).value,
            float(case.source_domain.field_strength_t),
            float(case.target_domain.field_strength_t),
        )
        if direction in seen_directions:
            raise ValueError(
                "Gate 0.1 input contains a duplicate contrast/directed-field pair."
            )
        seen_directions.add(direction)
        if case.traveller_identity_sha256:
            traveller_hashes.add(case.traveller_identity_sha256)
        selection_descriptors.append(
            {
                "case_identity_sha256": case.case_identity_sha256,
                "traveller_identity_sha256": case.traveller_identity_sha256,
                "contrast": direction[0],
                "source_field_t": direction[1],
                "target_field_t": direction[2],
            }
        )
        graph_specs.append(
            {
                "contrast": direction[0],
                "source_field_t": direction[1],
                "target_field_t": direction[2],
                "arrays": dict(case.array_sha256),
            }
        )
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
        predictions: dict[str, torch.Tensor] = {
            "raw_identity": case.raw_identity,
            "calibrated_identity": calibrated_identity,
            "raw_sb_v2": case.raw_sb_v2,
            "calibrated_sb_v2": calibrated_sb,
            "stage1_reconstruction_ceiling": case.stage1_reconstruction_ceiling,
        }
        if include_robust_affine:
            predictions.update(
                {
                    "diagnostic_robust_affine_identity": calibrator.apply(
                        case.raw_identity,
                        case.target_domain,
                        support_mask=case.support_mask,
                        mode="robust_affine",
                    ),
                    "diagnostic_robust_affine_sb_v2": calibrator.apply(
                        case.raw_sb_v2,
                        case.target_domain,
                        support_mask=case.support_mask,
                        mode="robust_affine",
                    ),
                }
            )

        if case_observer is not None:
            case_observer(case, predictions)

        method_metrics = {
            name: _validated_metric_result(
                metric_fn(prediction, case.target, requested_metrics, device),
                requested_metrics,
                name,
            )
            for name, prediction in predictions.items()
        }
        central = _paired_comparison(
            method_metrics["calibrated_identity"],
            method_metrics["calibrated_sb_v2"],
            requested_metrics,
        )
        row = {
            "case_identity_sha256": case.case_identity_sha256,
            "contrast": Contrast.parse(case.target_domain.contrast).value,
            "source_field_t": float(case.source_domain.field_strength_t),
            "target_field_t": float(case.target_domain.field_strength_t),
            "stratum": (
                "catastrophic_identity"
                if method_metrics["raw_identity"]["nrmse"] > 1.0
                else "ordinary"
            ),
            "methods": method_metrics,
            "central_comparison": central,
            "raw_pre_mask_background_leakage": {
                "raw_identity": _background_leakage(case.raw_identity, case.support_mask),
                "raw_sb_v2": _background_leakage(case.raw_sb_v2, case.support_mask),
                "stage1_reconstruction_ceiling": _background_leakage(
                    case.stage1_reconstruction_ceiling, case.support_mask
                ),
            },
            "array_sha256": dict(sorted(case.array_sha256.items())),
            "requested_vs_wrong_target": _wrong_target_diagnostic(
                case,
                calibrator=calibrator,
                requested_raw_metrics=method_metrics["raw_sb_v2"],
                requested_calibrated_metrics=method_metrics["calibrated_sb_v2"],
                metrics=requested_metrics,
                device=device,
                metric_fn=metric_fn,
            ),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["contrast"],
            row["source_field_t"],
            row["target_field_t"],
            row["case_identity_sha256"],
        )
    )
    if not rows:
        raise ValueError("Gate 0.1 requires at least one full-volume case.")
    computed_selection_fingerprint = gate01_selection_fingerprint(selection_descriptors)
    if len(traveller_hashes) > 1:
        raise ValueError("Gate 0.1 input mixes multiple travellers.")
    if traveller_hashes and evidence_scope["traveller_identity_sha256"] not in traveller_hashes:
        raise ValueError(
            "Gate 0.1 evidence scope and cases identify different travellers."
        )
    if execution_mode == "scientific":
        _validate_scientific_selection(
            rows=rows,
            seen_directions=seen_directions,
            traveller_hashes=traveller_hashes,
            expected_fingerprint=str(selection_fingerprint_sha256),
            computed_fingerprint=computed_selection_fingerprint,
        )
        hash_graph = validate_gate01_scientific_hash_graph(graph_specs)
    else:
        hash_graph = {
            "validated": False,
            "reason": "development-incomplete mode",
        }
    overall = _reduce_rows(rows, method_names, requested_metrics)
    catastrophic = [row for row in rows if row["stratum"] == "catastrophic_identity"]
    ordinary = [row for row in rows if row["stratum"] == "ordinary"]
    by_contrast = {
        contrast.value: _reduce_rows(
            [row for row in rows if row["contrast"] == contrast.value],
            method_names,
            requested_metrics,
        )
        for contrast in CONTRASTS
    }
    directed_pairs = _directed_pair_reductions(rows, method_names, requested_metrics)
    requested_wrong = _reduce_wrong_target(rows, requested_metrics)

    eligibility_reasons: list[str] = []
    if execution_mode != "scientific":
        eligibility_reasons.append("execution_mode_is_not_scientific")
    if evidence_scope["private_data_run"] is not True:
        eligibility_reasons.append("private_data_run_is_not_true")
    if evidence_scope["evidence_kind"] != "private":
        eligibility_reasons.append("evidence_kind_is_not_private")
    if protocol_lock is None:
        eligibility_reasons.append("protocol_lock_not_validated")
    if normalized_producer_receipt is None:
        eligibility_reasons.append("producer_receipt_not_validated")
    eligible = not eligibility_reasons
    promotion_decision = _scientific_promotion_decision(
        private_data_run=evidence_scope["private_data_run"], eligible=eligible
    )

    contract = {
        "contract_version": GATE01_CONTRACT_VERSION,
        "code_commit": code_commit,
        "code_provenance": normalized_code_provenance,
        "input_manifest_sha256": input_manifest_sha256,
        "producer_receipt": normalized_producer_receipt,
        "execution_mode": execution_mode,
        "selection_fingerprint_sha256": computed_selection_fingerprint,
        "support_threshold": support_threshold,
        "protocol_lock": protocol_lock.summary() if protocol_lock is not None else None,
        "scientific_hash_graph": hash_graph,
        "evidence_scope": dict(evidence_scope),
        "scientific_promotion_decision": promotion_decision,
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "split_provenance": dict(split_provenance or {}),
        "training_cohort_identity": calibrator.provenance[
            "training_cohort_identity"
        ],
        "calibrator": {
            "contract_version": calibrator.to_dict()["contract_version"],
            "semantics": GATE01_CALIBRATION_SEMANTICS,
            "artifact_sha256": calibrator.artifact_sha256,
            "template_sha256": calibrator.template_sha256,
            "support_threshold": calibrator.support_threshold,
            "template_hashes": {
                label: template.template_sha256
                for label, template in sorted(calibrator.templates.items())
            },
            "config": dict(calibrator.provenance["config"]),
            "balancing": dict(calibrator.provenance["balancing"]),
            "domain_volume_counts": dict(
                calibrator.provenance["domain_volume_counts"]
            ),
            "training_records_sha256": calibrator.provenance[
                "training_records_sha256"
            ],
            "training_volume_content_set_sha256": calibrator.provenance[
                "training_volume_content_set_sha256"
            ],
            "training_volume_content_identities": list(
                calibrator.provenance["training_volume_content_identities"]
            ),
            "fit_code_commit": calibrator.provenance["code_commit"],
            "fit_code_provenance": dict(calibrator.provenance["code_provenance"]),
        },
        "artifact_provenance": dict(artifact_provenance),
        "verified_loaded_array_sha256": [
            {
                "case_identity_sha256": row["case_identity_sha256"],
                "arrays": row["array_sha256"],
            }
            for row in rows
        ],
        "official_runtime_provenance": official_task3_runtime_provenance(
            metrics=requested_metrics, device=device
        ),
        "method_provenance": {
            "raw_identity": {
                "kind": "Stage-1 reconstruction of the source; no Stage-2 checkpoint"
            },
            "calibrated_identity": {
                "base": "raw_identity",
                "calibrator_template_sha256": calibrator.template_sha256,
            },
            "raw_sb_v2": {
                "checkpoint_sha256": artifact_provenance[
                    "sb_v2_checkpoint_sha256"
                ]
            },
            "calibrated_sb_v2": {
                "base": "raw_sb_v2",
                "checkpoint_sha256": artifact_provenance[
                    "sb_v2_checkpoint_sha256"
                ],
                "calibrator_template_sha256": calibrator.template_sha256,
            },
            "stage1_reconstruction_ceiling": {
                "checkpoint_sha256": artifact_provenance[
                    "stage1_run_c_checkpoint_sha256"
                ]
            },
        },
        "target_independence_guarantee": {
            "calibrator_signature": (
                "prediction + requested target domain + frozen source-derived support mask"
            ),
            "template_fit": "retrospective training records only",
            "paired_target_use": "official metrics and qualitative comparison only",
            "paired_target_for_calibration": False,
            "paired_target_mask_for_calibration": False,
            "paired_target_statistics_for_calibration": False,
            "support_mask_source": "frozen source image only; shared by identity and SB",
            "background": "exact zero outside the frozen source-derived support mask",
            "raw_pre_mask_background_leakage_reported": True,
        },
    }

    return {
        "contract_version": GATE01_CONTRACT_VERSION,
        "evidence_scope": dict(evidence_scope),
        "scientific_status": {
            "execution_mode": execution_mode,
            "private_data_run": evidence_scope["private_data_run"],
            "selection_contract_complete": execution_mode == "scientific",
            "eligible_for_scientific_conclusions": eligible,
            "ineligibility_reasons": eligibility_reasons,
            "evidence": (
                "scientific-contract diagnostic"
                if execution_mode == "scientific"
                else "development-only incomplete diagnostic"
            ),
            "promotion_decision": promotion_decision,
            "population_or_challenge_claim": False,
        },
        "central_question": (
            "calibrated(SB(x, target_condition)) versus calibrated(identity(x))"
        ),
        "metric_roles": {
            "official": {
                "names": list(requested_metrics),
                "contract": OFFICIAL_TASK3_METRIC_CONTRACT,
                "formulas_modified": False,
            },
            "diagnostic": {
                "requested_vs_wrong_target_common_requested_domain_calibration": (
                    "equal-photometry mechanistic comparison: requested and wrong-conditioned "
                    "predictions use the same requested-domain template"
                ),
                "requested_vs_wrong_target_condition_native_calibration": (
                    "endpoint diagnostic only: conditioning and photometric template both "
                    "change, so this does not isolate target control"
                ),
                "robust_affine_methods": (
                    "diagnostic-only methods; never eligible for scientific promotion"
                    if include_robust_affine
                    else "not requested"
                ),
            },
        },
        "method_roles": {
            "raw_identity": "official reference",
            "calibrated_identity": "official equal-photometry reference",
            "raw_sb_v2": "official learned reference",
            "calibrated_sb_v2": "official central learned comparison",
            "stage1_reconstruction_ceiling": "official reconstruction ceiling",
            **(
                {
                    name: "diagnostic-only robust-affine calibration"
                    for name in DIAGNOSTIC_METHODS
                }
                if include_robust_affine
                else {}
            ),
        },
        "num_pairs": len(rows),
        "methods": method_names,
        "overall": overall,
        "strata": {
            "frozen_definition": "raw identity official nRMSE > 1.0",
            "assignment_source": "raw_identity only",
            "catastrophic_identity": _reduce_rows(
                catastrophic, method_names, requested_metrics
            ),
            "ordinary": _reduce_rows(ordinary, method_names, requested_metrics),
        },
        "raw_pre_mask_background_leakage": {
            "overall": _reduce_background_leakage(rows),
            "by_contrast": {
                contrast.value: _reduce_background_leakage(
                    [row for row in rows if row["contrast"] == contrast.value]
                )
                for contrast in CONTRASTS
            },
        },
        "by_contrast": by_contrast,
        "directed_pair_results": directed_pairs["results"],
        "directed_pair_matrices": directed_pairs["matrices"],
        "central_paired_deltas_and_wins": _reduce_central(rows, requested_metrics),
        "requested_vs_wrong_target_diagnostic": requested_wrong,
        "montage_specifications": fixed_montage_specifications(),
        "pairs": rows,
        "contract": contract,
    }


def fixed_montage_specifications() -> dict[str, Any]:
    """Return the predeclared, target-independent qualitative panel selection."""

    return {
        "version": "gate01-montage-v1",
        "selection_frozen_before_private_run": True,
        "selection_basis": "contrast and directed field pair only; never metric rank",
        "tensor_axis_convention": (
            "volume[..., z]; no anatomical plane name is inferred without affine/orientation"
        ),
        "relative_slice_positions": [0.35, 0.50, 0.65],
        "display_order": [
            "target",
            "raw_identity",
            "calibrated_identity",
            "raw_sb_v2",
            "calibrated_sb_v2",
            "stage1_reconstruction_ceiling",
        ],
        "directed_pairs_per_contrast": [
            {"source_field_t": 0.1, "target_field_t": 7.0},
            {"source_field_t": 7.0, "target_field_t": 0.1},
            {"source_field_t": 1.5, "target_field_t": 3.0},
            {"source_field_t": 3.0, "target_field_t": 1.5},
        ],
        "contrasts": [contrast.value for contrast in CONTRASTS],
        "rendering": {
            "shared_display_range_within_pair": True,
            "interpolation": "none",
            "crop": "none unless a separately frozen source-derived crop is supplied",
        },
    }


def render_gate01_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact report while leaving every machine table in JSON."""

    lines = [
        "# Stage 2 Gate 0.1 — Equal-photometry diagnostic",
        "",
        f"Contract: `{result['contract_version']}`",
        "",
        "## Scientific status",
        "",
        _markdown_scientific_status(result["scientific_status"]),
        "",
        "Evidence scope: " + json.dumps(result["evidence_scope"], sort_keys=True),
        "",
        "## Private producer provenance",
        "",
        _markdown_producer_receipt(result["contract"].get("producer_receipt")),
        "",
        "Split roles: "
        + json.dumps(result["contract"].get("split_provenance", {}), sort_keys=True),
        "",
        "## Calibration contract",
        "",
        "Both identity and SB-v2 are projected from their own prediction CDF onto the same "
        "frozen requested-target-domain foreground CDF fitted from retrospective training "
        "records. The paired evaluation target is used only by the official metrics and is "
        "never passed to calibration. One frozen source-derived support mask is shared by "
        "identity and SB; calibrated output is exact zero outside it, and raw pre-mask "
        "background leakage is reported.",
        "",
        "## Official aggregate metrics",
        "",
        _markdown_metrics_table(result["overall"]["methods"]),
        "",
        "## Frozen raw-identity strata",
        "",
        "Catastrophic pairs are assigned once using raw identity official nRMSE > 1.0.",
        "",
    ]
    for name in ("catastrophic_identity", "ordinary"):
        block = result["strata"][name]
        lines.extend(
            [
                f"### {name.replace('_', ' ').title()} ({block['num_pairs']} pairs)",
                "",
                _markdown_metrics_table(block["methods"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Raw pre-mask background leakage",
            "",
            _markdown_background_leakage(
                result["raw_pre_mask_background_leakage"]["overall"]
            ),
            "",
            "## Per contrast",
            "",
        ]
    )
    for contrast, block in result["by_contrast"].items():
        lines.extend(
            [
                f"### {contrast} ({block['num_pairs']} pairs)",
                "",
                _markdown_metrics_table(block["methods"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Central paired comparison",
            "",
            "Positive improvement favors calibrated SB-v2. Raw signed delta is "
            "calibrated SB-v2 minus calibrated identity.",
            "",
            _markdown_central_table(result["central_paired_deltas_and_wins"]["overall"]),
            "",
            "## Requested versus wrong-target diagnostics",
            "",
            "Common requested-domain calibration is the equal-photometry mechanistic "
            "comparison: requested and wrong-conditioned predictions use the same "
            "requested-domain template. Condition-native calibration is a separate "
            "endpoint diagnostic; it changes both condition and template and does not "
            "isolate target control.",
            "",
            _markdown_wrong_target(result["requested_vs_wrong_target_diagnostic"]),
            "",
            "## Directed-pair matrices and qualitative panels",
            "",
            "All 20 directed field pairs for every contrast, their paired deltas/wins, and "
            "the frozen montage specification are included in the machine-readable JSON. "
            + (
                "Deterministic PNG hashes and their provenance manifest are recorded."
                if "montage_rendering" in result
                else "Montage rendering was not requested for this development invocation."
            ),
            "",
            "## Known limitations",
            "",
            "- Gate 0.1 does not retrain or promote a model.",
            "- Development traveller directions are repeated measurements from one subject, "
            "not independent subjects.",
            "- Requested-versus-wrong-target results are present only when those existing "
            "predictions are supplied.",
            "- Robust-affine outputs, when requested, are diagnostic-only.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_producer_receipt(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "No verified producer receipt (development-only invocation)."
    receipt = value["producer_receipt"]
    return (
        "Verified sealed producer spec/state/build-plan handoff: "
        f"spec file `{value['producer_spec_file_sha256']}`, state file "
        f"`{value['producer_state_file_sha256']}`, plan `{value['build_plan_sha256']}`; "
        f"decode strategy `{receipt['decode_strategy']}` with path used "
        f"`{json.dumps(receipt['path_used'])}`; counts "
        f"{receipt['acquisition_count']}/"
        f"{receipt['direction_count']}/"
        f"{receipt['wrong_target_reference_count']} acquisitions/directions/wrong-target "
        "references."
    )


def write_gate01_outputs(
    result: Mapping[str, Any],
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    contract_path: str | Path | None = None,
) -> dict[str, str]:
    """Atomically write any requested Gate 0.1 outputs."""

    written: dict[str, str] = {}
    if json_path is not None:
        path = _write_text_atomic(
            json_path,
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        )
        written["json"] = str(path)
    if markdown_path is not None:
        path = _write_text_atomic(markdown_path, render_gate01_markdown(result))
        written["markdown"] = str(path)
    if contract_path is not None:
        path = _write_text_atomic(
            contract_path,
            json.dumps(result["contract"], indent=2, sort_keys=True, allow_nan=False),
        )
        written["contract"] = str(path)
    return written


_ROOT_INPUT_KEYS = {
    "contract_version",
    "execution_mode",
    "selection_fingerprint_sha256",
    "evidence_scope",
    "split_fingerprint",
    "split_provenance",
    "artifact_provenance",
    "source_support_contract",
    "producer_receipt",
    "cases",
}
_CASE_INPUT_KEYS = {
    "case_id",
    "traveller_identity_sha256",
    "source_domain",
    "target_domain",
    "source_image",
    "source_support_mask",
    "target",
    "raw_identity",
    "raw_sb_v2",
    "stage1_reconstruction_ceiling",
    "wrong_target_sb_v2",
}
_ARRAY_REFERENCE_KEYS = {"path", "sha256"}
_SOURCE_SUPPORT_KEYS = {"derivation", "threshold"}


@dataclass(frozen=True, slots=True)
class Gate01InputManifest:
    """Metadata-only manifest whose iterator loads and releases one case at a time."""

    root: Path
    case_specs: tuple[Mapping[str, Any], ...]
    support_threshold: float

    def __iter__(self) -> Iterator[Gate01Case]:
        for index, item in enumerate(self.case_specs):
            yield _load_gate01_case(
                self.root,
                item,
                index=index,
                support_threshold=self.support_threshold,
            )


def validate_gate01_verified_producer_receipt(value: Any) -> dict[str, Any]:
    """Fail closed on the builder-authenticated producer/spec/state handoff."""

    if not isinstance(value, Mapping):
        raise ValueError("Gate 0.1 verified producer receipt must be a mapping.")
    outer_keys = {
        "contract_version",
        "producer_spec_contract_version",
        "producer_state_contract_version",
        "producer_spec_file_sha256",
        "producer_spec_artifact_sha256",
        "producer_state_file_sha256",
        "build_plan_sha256",
        "producer_receipt",
    }
    _assert_exact_keys(value, outer_keys, "Gate 0.1 verified producer receipt")
    if value["contract_version"] != GATE01_VERIFIED_PRODUCER_RECEIPT_VERSION:
        raise ValueError("Gate 0.1 verified producer-receipt contract is incompatible.")
    if (
        value["producer_spec_contract_version"]
        != "stage2-gate01-private-producer-spec-v4"
        or value["producer_state_contract_version"]
        != "stage2-gate01-private-producer-state-v3"
    ):
        raise ValueError("Gate 0.1 producer spec/state receipt versions are incompatible.")
    for key in (
        "producer_spec_file_sha256",
        "producer_spec_artifact_sha256",
        "producer_state_file_sha256",
        "build_plan_sha256",
    ):
        if not _is_sha256(str(value[key])):
            raise ValueError(f"Gate 0.1 verified producer receipt has invalid {key}.")
    receipt = value["producer_receipt"]
    if not isinstance(receipt, Mapping):
        raise ValueError("Gate 0.1 sealed producer receipt must be a mapping.")
    receipt_keys = {
        "contract_version",
        "producer_spec_artifact_sha256",
        "protocol_lock_artifact_sha256",
        "selection_artifact_sha256",
        "selection_fingerprint_sha256",
        "split_provenance",
        "selected_source_acquisitions_sha256",
        "selected_payload_identity_set_sha256",
        "selected_payload_count",
        "latent_bank",
        "stage1_config_sha256",
        "stage1_checkpoint_sha256",
        "sb_v2_config_sha256",
        "sb_v2_checkpoint_sha256",
        "sampler_specification_sha256",
        "decode_specification_sha256",
        "decode_strategy",
        "path_used",
        "acquisition_count",
        "stage1_inference_count",
        "direction_count",
        "sb_v2_inference_count",
        "wrong_target_reference_count",
    }
    _assert_exact_keys(receipt, receipt_keys, "Gate 0.1 sealed producer receipt")
    if (
        receipt["contract_version"]
        != "stage2-gate01-private-producer-receipt-v2"
        or receipt["producer_spec_artifact_sha256"]
        != value["producer_spec_artifact_sha256"]
        or receipt["decode_strategy"] != "full"
        or receipt["path_used"] != ["full"]
        or receipt["selected_payload_count"] != 15
        or receipt["acquisition_count"] != 15
        or receipt["stage1_inference_count"] != 15
        or receipt["direction_count"] != 60
        or receipt["sb_v2_inference_count"] != 60
        or receipt["wrong_target_reference_count"] != 180
    ):
        raise ValueError("Gate 0.1 producer receipt lacks the exact full-decode proof.")
    sha_keys = receipt_keys - {
        "contract_version",
        "latent_bank",
        "decode_strategy",
        "path_used",
        "selected_payload_count",
        "acquisition_count",
        "stage1_inference_count",
        "direction_count",
        "sb_v2_inference_count",
        "wrong_target_reference_count",
        "split_provenance",
    }
    if any(not _is_sha256(str(receipt[key])) for key in sha_keys):
        raise ValueError("Gate 0.1 producer receipt contains an invalid identity digest.")
    _validated_split_provenance(receipt["split_provenance"])
    bank = receipt["latent_bank"]
    if not isinstance(bank, Mapping) or set(bank) != {
        "artifact_sha256",
        "manifest_sha256",
        "stats_sha256",
        "record_count",
        "build_git_commit",
        "vae_checkpoint_sha256",
        "encode_provenance",
    }:
        raise ValueError("Gate 0.1 producer receipt latent-bank identity is malformed.")
    if (
        not all(
            _is_sha256(str(bank[key]))
            for key in (
                "artifact_sha256",
                "manifest_sha256",
                "stats_sha256",
                "vae_checkpoint_sha256",
            )
        )
        or int(bank["record_count"]) < 15
        or bank["encode_provenance"] != {"strategy": "full", "path_used": ["full"]}
    ):
        raise ValueError("Gate 0.1 producer receipt latent bank is incompatible.")
    return {**dict(value), "producer_receipt": dict(receipt)}


def _validated_split_provenance(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"evaluation", "bank_storage"}:
        raise ValueError("Gate 0.1 split provenance must contain both split roles.")
    expected_fields = {"role", "file_sha256", "membership_fingerprint"}
    evaluation = value["evaluation"]
    bank = value["bank_storage"]
    if (
        not isinstance(evaluation, Mapping)
        or not isinstance(bank, Mapping)
        or set(evaluation) != expected_fields
        or set(bank) != expected_fields
        or evaluation["role"] != "scientific_evaluation_resplit"
        or evaluation["membership_fingerprint"] != RESPLIT_FINGERPRINT
        or not _is_sha256(str(evaluation["file_sha256"]))
        or bank["role"] != "frozen_latent_bank_source_split"
        or bank["file_sha256"] != FULL_LATENT_BANK_SOURCE_SPLIT_FILE_SHA256
        or bank["membership_fingerprint"]
        != FULL_LATENT_BANK_SOURCE_SPLIT_FINGERPRINT
    ):
        raise ValueError("Gate 0.1 split provenance is stale or malformed.")
    return {
        "evaluation": {str(key): str(item) for key, item in evaluation.items()},
        "bank_storage": {str(key): str(item) for key, item in bank.items()},
    }


def load_gate01_input_manifest(
    path: str | Path,
    *,
    protocol_lock: Gate01ProtocolLock | None = None,
    calibrator: PosthocTargetCalibrator | None = None,
) -> tuple[Gate01InputManifest, dict[str, Any]]:
    """Validate metadata now and stream verified loaded arrays during evaluation."""

    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 input manifest root must be a mapping.")
    reject_target_derived_calibration_fields(payload)
    _assert_exact_keys(
        payload,
        _ROOT_INPUT_KEYS,
        "Gate 0.1 input manifest",
        optional={"producer_receipt"},
    )
    if payload["contract_version"] != GATE01_INPUT_CONTRACT_VERSION:
        raise ValueError("Gate 0.1 input manifest contract is incompatible.")
    if payload["split_fingerprint"] != RESPLIT_FINGERPRINT:
        raise ValueError("Gate 0.1 input manifest has a stale split fingerprint.")
    split_provenance = _validated_split_provenance(payload["split_provenance"])
    validate_frozen_artifact_provenance(payload["artifact_provenance"])
    evidence_scope = _validated_evidence_scope(payload["evidence_scope"])
    execution_mode = str(payload["execution_mode"])
    if execution_mode not in GATE01_EXECUTION_MODES:
        raise ValueError(f"Unknown Gate 0.1 execution mode {execution_mode!r}.")
    producer_receipt = (
        validate_gate01_verified_producer_receipt(payload["producer_receipt"])
        if "producer_receipt" in payload
        else None
    )
    if execution_mode == "scientific" and producer_receipt is None:
        raise ValueError(
            "Scientific Gate 0.1 manifest requires verified full-decode producer provenance."
        )
    selection_fingerprint = str(payload["selection_fingerprint_sha256"])
    if not _is_sha256(selection_fingerprint):
        raise ValueError("Gate 0.1 selection fingerprint must be a SHA-256 digest.")
    if producer_receipt is not None:
        sealed = producer_receipt["producer_receipt"]
        if (
            sealed["selection_fingerprint_sha256"] != selection_fingerprint
            or sealed["split_provenance"] != split_provenance
            or (
                protocol_lock is not None
                and sealed["protocol_lock_artifact_sha256"]
                != protocol_lock.artifact_sha256
            )
        ):
            raise ValueError(
                "Gate 0.1 producer receipt/manifest/protocol identities disagree."
            )
    support_contract = payload["source_support_contract"]
    if not isinstance(support_contract, Mapping):
        raise ValueError("Gate 0.1 source_support_contract must be a mapping.")
    _assert_exact_keys(
        support_contract, _SOURCE_SUPPORT_KEYS, "Gate 0.1 source support contract"
    )
    if support_contract["derivation"] != "abs(source_image)>threshold":
        raise ValueError("Gate 0.1 source support derivation is incompatible.")
    support_threshold = float(support_contract["threshold"])
    if support_threshold != GATE01_SUPPORT_THRESHOLD:
        raise ValueError("Gate 0.1 source support threshold is frozen at exactly 0.0.")
    if calibrator is not None and calibrator.support_threshold != support_threshold:
        raise ValueError("Gate 0.1 manifest/calibrator support threshold mismatch.")

    case_specs: list[Mapping[str, Any]] = []
    selection_descriptors: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_directions: set[tuple[str, float, float]] = set()
    traveller_hashes: set[str] = set()
    for index, item in enumerate(payload["cases"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"Gate 0.1 case {index} must be a mapping.")
        _assert_exact_keys(
            item,
            _CASE_INPUT_KEYS,
            f"Gate 0.1 case {index}",
            optional={"wrong_target_sb_v2"},
        )
        case_id = str(item["case_id"])
        case_hash = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
        if case_hash in seen_case_ids:
            raise ValueError("Gate 0.1 input contains duplicate case IDs.")
        seen_case_ids.add(case_hash)
        traveller_hash = str(item["traveller_identity_sha256"])
        if not _is_sha256(traveller_hash):
            raise ValueError(f"Gate 0.1 case {index} has an invalid traveller digest.")
        traveller_hashes.add(traveller_hash)
        source_domain = Domain.from_dict(dict(item["source_domain"]))
        target_domain = Domain.from_dict(dict(item["target_domain"]))
        if Contrast.parse(source_domain.contrast) != Contrast.parse(target_domain.contrast):
            raise ValueError("Gate 0.1 evaluates same-contrast field translation only.")
        direction = (
            Contrast.parse(target_domain.contrast).value,
            float(source_domain.field_strength_t),
            float(target_domain.field_strength_t),
        )
        if direction in seen_directions:
            raise ValueError(
                "Gate 0.1 input contains a duplicate contrast/directed-field pair."
            )
        seen_directions.add(direction)
        for key in (
            "source_image",
            "source_support_mask",
            "target",
            "raw_identity",
            "raw_sb_v2",
            "stage1_reconstruction_ceiling",
        ):
            _validated_array_reference(item[key], f"Gate 0.1 case {index}.{key}")
        wrong = item.get("wrong_target_sb_v2", {})
        if not isinstance(wrong, Mapping):
            raise ValueError(f"Gate 0.1 case {index}.wrong_target_sb_v2 must be a mapping.")
        for label, reference in wrong.items():
            _wrong_target_domain(str(label), Contrast.parse(target_domain.contrast))
            _validated_array_reference(
                reference, f"Gate 0.1 case {index}.wrong_target_sb_v2[{label}]"
            )
        selection_descriptors.append(
            {
                "case_identity_sha256": case_hash,
                "traveller_identity_sha256": traveller_hash,
                "contrast": direction[0],
                "source_field_t": direction[1],
                "target_field_t": direction[2],
            }
        )
        case_specs.append(dict(item))

    if len(traveller_hashes) > 1:
        raise ValueError("Gate 0.1 input mixes multiple travellers.")
    if traveller_hashes and evidence_scope["traveller_identity_sha256"] not in traveller_hashes:
        raise ValueError(
            "Gate 0.1 evidence scope and cases identify different travellers."
        )
    computed_selection = gate01_selection_fingerprint(selection_descriptors)
    if computed_selection != selection_fingerprint:
        raise ValueError(
            "Gate 0.1 predeclared selection fingerprint does not match the cases."
        )
    if execution_mode == "scientific":
        if protocol_lock is None or calibrator is None:
            raise ValueError(
                "Scientific Gate 0.1 manifest validation requires an independent "
                "protocol lock and calibrator."
            )
        protocol_lock.assert_calibrator(calibrator)
        protocol_lock.assert_manifest_contract(
            traveller_identity_sha256=evidence_scope["traveller_identity_sha256"],
            selection_fingerprint_sha256=selection_fingerprint,
            split_fingerprint=str(payload["split_fingerprint"]),
            split_provenance=split_provenance,
            support_threshold=support_threshold,
            artifact_provenance=payload["artifact_provenance"],
        )
        _validate_scientific_selection(
            rows=[{}] * len(case_specs),
            seen_directions=seen_directions,
            traveller_hashes=traveller_hashes,
            expected_fingerprint=selection_fingerprint,
            computed_fingerprint=computed_selection,
        )
        hash_graph = validate_gate01_scientific_hash_graph(case_specs)
    else:
        hash_graph = {
            "validated": False,
            "reason": "development-incomplete mode",
        }
    metadata = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_scope": dict(evidence_scope),
        "artifact_provenance": dict(payload["artifact_provenance"]),
        "split_fingerprint": str(payload["split_fingerprint"]),
        "split_provenance": split_provenance,
        "execution_mode": execution_mode,
        "selection_fingerprint_sha256": selection_fingerprint,
        "support_threshold": support_threshold,
        "scientific_hash_graph": hash_graph,
        "protocol_lock_artifact_sha256": (
            protocol_lock.artifact_sha256 if protocol_lock is not None else None
        ),
        "producer_receipt": producer_receipt,
    }
    return (
        Gate01InputManifest(
            root=source.resolve().parent,
            case_specs=tuple(case_specs),
            support_threshold=support_threshold,
        ),
        metadata,
    )


def _load_gate01_case(
    root: Path,
    item: Mapping[str, Any],
    *,
    index: int,
    support_threshold: float,
) -> Gate01Case:
    arrays: dict[str, torch.Tensor] = {}
    identities: dict[str, str] = {}
    source_image, source_hash = _load_verified_array(
        root, item["source_image"], f"case {index} source_image"
    )
    support_mask, support_hash = _load_verified_array(
        root,
        item["source_support_mask"],
        f"case {index} source_support_mask",
        mask=True,
    )
    if tuple(source_image.shape) != tuple(support_mask.shape):
        raise ValueError("Gate 0.1 source image/support-mask shape mismatch.")
    derived_mask = source_image.abs() > support_threshold
    if not torch.equal(derived_mask, support_mask):
        raise ValueError(
            "Gate 0.1 frozen support mask does not match its source-derived contract."
        )
    identities["source_image"] = source_hash
    identities["source_support_mask"] = support_hash
    for key in (
        "target",
        "raw_identity",
        "raw_sb_v2",
        "stage1_reconstruction_ceiling",
    ):
        arrays[key], identities[key] = _load_verified_array(
            root, item[key], f"case {index} {key}"
        )
    wrong: dict[str, torch.Tensor] = {}
    for label, reference in sorted(dict(item.get("wrong_target_sb_v2", {})).items()):
        wrong[label], identities[f"wrong_target_sb_v2[{label}]"] = _load_verified_array(
            root, reference, f"case {index} wrong_target_sb_v2[{label}]"
        )
    return Gate01Case(
        case_id=str(item["case_id"]),
        traveller_identity_sha256=str(item["traveller_identity_sha256"]),
        source_domain=Domain.from_dict(dict(item["source_domain"])),
        target_domain=Domain.from_dict(dict(item["target_domain"])),
        target=arrays["target"],
        raw_identity=arrays["raw_identity"],
        raw_sb_v2=arrays["raw_sb_v2"],
        stage1_reconstruction_ceiling=arrays["stage1_reconstruction_ceiling"],
        support_mask=support_mask,
        array_sha256=identities,
        wrong_target_sb_v2=wrong,
        source_image=source_image,
    )


def _load_external_volume(path: Path) -> torch.Tensor:
    name = path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        array, _ = load_official_nifti(path)
    elif name.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
    else:
        raise ValueError(
            f"Unsupported Gate 0.1 volume format for {path.name!r}; use NIfTI or .npy."
        )
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _load_verified_array(
    root: Path,
    reference: Any,
    role: str,
    *,
    mask: bool = False,
) -> tuple[torch.Tensor, str]:
    validated = _validated_array_reference(reference, role)
    path = _resolve_manifest_path(root, validated["path"])
    if mask:
        if path.suffix.lower() != ".npy":
            raise ValueError("Gate 0.1 source support masks must use .npy.")
        raw = np.load(path, allow_pickle=False)
        if raw.dtype != np.bool_ and not np.isin(raw, (0, 1)).all():
            raise ValueError("Gate 0.1 source support mask must contain boolean values.")
        tensor = torch.from_numpy(np.asarray(raw, dtype=np.bool_))
    else:
        tensor = _load_external_volume(path)
        if tensor.ndim < 3:
            raise ValueError(f"Gate 0.1 {role} is not a full volume.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Gate 0.1 {role} contains non-finite values.")
    actual = canonical_loaded_array_sha256(tensor)
    if actual != validated["sha256"]:
        raise ValueError(
            f"Gate 0.1 loaded-array SHA-256 mismatch for {role}; file changed "
            "after manifest creation or was loaded incompatibly."
        )
    return tensor, actual


def _validated_array_reference(value: Any, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a path/SHA-256 mapping.")
    _assert_exact_keys(value, _ARRAY_REFERENCE_KEYS, role)
    path = str(value["path"])
    digest = str(value["sha256"])
    if not path or not _is_sha256(digest):
        raise ValueError(f"{role} has an invalid path or SHA-256 digest.")
    return {"path": path, "sha256": digest}


def canonical_loaded_array_sha256(value: torch.Tensor | np.ndarray) -> str:
    """Hash the canonical loaded array identity, independent of container bytes."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        array = tensor.numpy()
    else:
        array = np.asarray(value)
    if array.dtype == np.bool_:
        canonical = np.ascontiguousarray(array, dtype=np.bool_)
        dtype = "bool"
    else:
        canonical = np.ascontiguousarray(array, dtype="<f4")
        dtype = "float32-le"
    header = json.dumps(
        {"dtype": dtype, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _validate_case_content_hashes(case: Gate01Case) -> None:
    """Bind in-memory evaluator tensors to the declared canonical identities."""

    if not case.array_sha256:
        return
    tensors = {
        "target": case.target,
        "raw_identity": case.raw_identity,
        "raw_sb_v2": case.raw_sb_v2,
        "stage1_reconstruction_ceiling": case.stage1_reconstruction_ceiling,
        "source_support_mask": case.support_mask,
        **{
            f"wrong_target_sb_v2[{label}]": value
            for label, value in case.wrong_target_sb_v2.items()
        },
    }
    for role, tensor in tensors.items():
        expected = str(case.array_sha256.get(role, ""))
        if not _is_sha256(expected):
            raise ValueError(f"Gate 0.1 case is missing canonical identity for {role}.")
        if canonical_loaded_array_sha256(tensor) != expected:
            raise ValueError(
                f"Gate 0.1 in-memory array identity mismatch for {role}; "
                "content changed after manifest verification."
            )


def validate_gate01_scientific_hash_graph(
    case_specs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all acquisition/reconstruction/prediction identities by canonical hash."""

    directions: dict[tuple[str, float, float], dict[str, Any]] = {}
    for index, item in enumerate(case_specs):
        if "source_domain" in item:
            source_domain = Domain.from_dict(dict(item["source_domain"]))
            target_domain = Domain.from_dict(dict(item["target_domain"]))
            contrast = Contrast.parse(target_domain.contrast).value
            source = float(source_domain.field_strength_t)
            target = float(target_domain.field_strength_t)
            arrays = {
                key: str(
                    _validated_array_reference(
                        item[key], f"graph case {index}.{key}"
                    )["sha256"]
                )
                for key in (
                    "source_image",
                    "source_support_mask",
                    "target",
                    "raw_identity",
                    "raw_sb_v2",
                    "stage1_reconstruction_ceiling",
                )
            }
            wrong = {
                str(label): str(
                    _validated_array_reference(
                        reference, f"graph case {index}.wrong_target_sb_v2[{label}]"
                    )["sha256"]
                )
                for label, reference in dict(item.get("wrong_target_sb_v2", {})).items()
            }
        else:
            contrast = Contrast.parse(item["contrast"]).value
            source = float(item["source_field_t"])
            target = float(item["target_field_t"])
            supplied = dict(item["arrays"])
            arrays = {
                key: str(supplied.get(key, ""))
                for key in (
                    "source_image",
                    "source_support_mask",
                    "target",
                    "raw_identity",
                    "raw_sb_v2",
                    "stage1_reconstruction_ceiling",
                )
            }
            wrong = {
                key[len("wrong_target_sb_v2[") : -1]: str(value)
                for key, value in supplied.items()
                if key.startswith("wrong_target_sb_v2[") and key.endswith("]")
            }
        if source == target:
            raise ValueError("Gate 0.1 hash graph contains an identity direction.")
        if not all(_is_sha256(value) for value in arrays.values()):
            raise ValueError("Gate 0.1 hash graph has a missing or invalid array identity.")
        key = (contrast, source, target)
        if key in directions:
            raise ValueError("Gate 0.1 hash graph contains a duplicate direction.")
        directions[key] = {"arrays": arrays, "wrong": wrong}

    expected_directions = {
        (contrast.value, float(source), float(target))
        for contrast in CONTRASTS
        for source in FIELD_STRENGTHS_T
        for target in FIELD_STRENGTHS_T
        if source != target
    }
    if set(directions) != expected_directions:
        raise ValueError("Gate 0.1 hash graph does not contain all 60 directions exactly.")

    node_roles: dict[tuple[str, float], dict[str, list[str]]] = {}
    for (contrast, source, target), entry in directions.items():
        source_node = node_roles.setdefault(
            (contrast, source),
            {"source": [], "target": [], "identity": [], "ceiling": [], "support": []},
        )
        target_node = node_roles.setdefault(
            (contrast, target),
            {"source": [], "target": [], "identity": [], "ceiling": [], "support": []},
        )
        arrays = entry["arrays"]
        source_node["source"].append(arrays["source_image"])
        source_node["identity"].append(arrays["raw_identity"])
        source_node["support"].append(arrays["source_support_mask"])
        target_node["target"].append(arrays["target"])
        target_node["ceiling"].append(arrays["stage1_reconstruction_ceiling"])

    if len(node_roles) != 15:
        raise ValueError("Gate 0.1 hash graph must contain exactly 15 acquisitions.")
    node_contract: dict[str, dict[str, str]] = {}
    for (contrast, field_strength), roles in sorted(node_roles.items()):
        if any(len(roles[name]) != 4 for name in roles):
            raise ValueError("Gate 0.1 acquisition roles must each repeat exactly four times.")
        if any(len(set(roles[name])) != 1 for name in roles):
            raise ValueError("Gate 0.1 repeated acquisition roles have inconsistent hashes.")
        source_hash = roles["source"][0]
        target_hash = roles["target"][0]
        identity_hash = roles["identity"][0]
        ceiling_hash = roles["ceiling"][0]
        if source_hash != target_hash:
            raise ValueError(
                "Gate 0.1 source acquisition does not equal its corresponding target hash."
            )
        if identity_hash != ceiling_hash:
            raise ValueError(
                "Gate 0.1 raw identity does not equal its corresponding Stage-1 ceiling hash."
            )
        node_contract[f"{contrast}:{field_strength:g}T"] = {
            "acquisition_sha256": source_hash,
            "stage1_reconstruction_sha256": identity_hash,
            "source_support_mask_sha256": roles["support"][0],
        }

    wrong_comparisons = 0
    for (contrast, source, requested), entry in directions.items():
        expected_wrong_fields = {
            float(value)
            for value in FIELD_STRENGTHS_T
            if value != source and value != requested
        }
        supplied_wrong: dict[float, str] = {}
        for label, digest in entry["wrong"].items():
            wrong_domain = _wrong_target_domain(label, Contrast.parse(contrast))
            supplied_wrong[float(wrong_domain.field_strength_t)] = digest
        if set(supplied_wrong) != expected_wrong_fields:
            raise ValueError(
                "Scientific Gate 0.1 requires all three sibling wrong-target predictions."
            )
        for wrong_field, digest in supplied_wrong.items():
            sibling = directions[(contrast, source, wrong_field)]["arrays"]["raw_sb_v2"]
            if digest != sibling:
                raise ValueError(
                    "Gate 0.1 wrong-target prediction is not the canonical sibling prediction."
                )
            wrong_comparisons += 1

    node_contract_sha256 = _sha256_json(node_contract)
    return {
        "validated": True,
        "acquisition_count": len(node_contract),
        "direction_count": len(directions),
        "wrong_target_comparison_count": wrong_comparisons,
        "node_contract_sha256": node_contract_sha256,
        "nodes": node_contract,
    }


def gate01_selection_fingerprint(descriptors: Iterable[Mapping[str, Any]]) -> str:
    """Hash the external case selection without exposing case or traveller identifiers."""

    normalized = [
        {
            "case_identity_sha256": str(item["case_identity_sha256"]),
            "traveller_identity_sha256": str(item["traveller_identity_sha256"]),
            "contrast": Contrast.parse(item["contrast"]).value,
            "source_field_t": float(item["source_field_t"]),
            "target_field_t": float(item["target_field_t"]),
        }
        for item in descriptors
    ]
    normalized.sort(
        key=lambda item: (
            item["contrast"],
            item["source_field_t"],
            item["target_field_t"],
            item["case_identity_sha256"],
        )
    )
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_scientific_selection(
    *,
    rows: Sequence[Mapping[str, Any]],
    seen_directions: set[tuple[str, float, float]],
    traveller_hashes: set[str],
    expected_fingerprint: str,
    computed_fingerprint: str,
) -> None:
    expected_directions = {
        (contrast.value, float(source), float(target))
        for contrast in CONTRASTS
        for source in FIELD_STRENGTHS_T
        for target in FIELD_STRENGTHS_T
        if source != target
    }
    if len(rows) != GATE01_SCIENTIFIC_CASE_COUNT:
        raise ValueError(
            "Scientific Gate 0.1 mode requires exactly 60 unique full-volume cases."
        )
    if seen_directions != expected_directions:
        missing = sorted(expected_directions - seen_directions)
        unexpected = sorted(seen_directions - expected_directions)
        raise ValueError(
            "Scientific Gate 0.1 directed selection is incomplete or unexpected: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if len(traveller_hashes) != 1:
        raise ValueError("Scientific Gate 0.1 mode requires exactly one traveller.")
    if computed_fingerprint != expected_fingerprint:
        raise ValueError("Scientific Gate 0.1 selection fingerprint mismatch.")


def gate01_code_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Capture clean-checkout and scientific module identities for a run."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head": head,
        "checkout_clean": not bool(status.strip()),
        "module_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in GATE01_SCIENTIFIC_MODULES
        },
    }


def _background_leakage(
    prediction: torch.Tensor, support_mask: torch.Tensor
) -> dict[str, float | int]:
    background = prediction.detach().cpu().to(torch.float64)[
        ~support_mask.detach().cpu()
    ].abs()
    if background.numel() == 0:
        return {
            "voxel_count": 0,
            "nonzero_voxel_count": 0,
            "nonzero_fraction": 0.0,
            "mean_abs": 0.0,
            "rms": 0.0,
            "max_abs": 0.0,
        }
    return {
        "voxel_count": int(background.numel()),
        "nonzero_voxel_count": int(torch.count_nonzero(background).item()),
        "nonzero_fraction": float(torch.count_nonzero(background).item() / background.numel()),
        "mean_abs": float(background.mean().item()),
        "rms": float(torch.sqrt(torch.mean(background.square())).item()),
        "max_abs": float(background.max().item()),
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_evidence_scope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Gate 0.1 evidence_scope must be a mapping.")
    required = {
        "role",
        "evidence_kind",
        "traveller_identity_sha256",
        "private_data_run",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"Gate 0.1 evidence_scope is incomplete; missing {missing}."
        )
    if not isinstance(value["private_data_run"], bool):
        raise ValueError("Gate 0.1 evidence_scope.private_data_run must be boolean.")
    if value["evidence_kind"] not in {"private", "synthetic", "development"}:
        raise ValueError(
            "Gate 0.1 evidence_scope.evidence_kind must be private, synthetic, or development."
        )
    if not str(value["role"]).strip():
        raise ValueError("Gate 0.1 evidence_scope role must be non-empty.")
    if not _is_sha256(str(value["traveller_identity_sha256"])):
        raise ValueError(
            "Gate 0.1 evidence_scope traveller identity must be a SHA-256 digest."
        )
    return dict(value)


def _resolve_manifest_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Gate 0.1 volume path must be a non-empty string.")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _assert_exact_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted((allowed - optional) - set(payload))
    unexpected = sorted(set(payload) - allowed)
    if missing or unexpected:
        raise ValueError(
            f"{name} schema mismatch: missing={missing}, unexpected={unexpected}."
        )


def _official_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    array = tensor.detach().cpu().to(torch.float32).numpy().squeeze()
    if array.ndim != 3:
        raise ValueError(
            f"Gate 0.1 {name} must resolve to one 3-D full volume; got {array.shape}."
        )
    return array.astype(np.float64)


def _validated_metric_result(
    values: Mapping[str, float], metrics: Sequence[str], method: str
) -> dict[str, float]:
    missing = sorted(set(metrics) - set(values))
    unexpected = sorted(set(values) - set(metrics))
    if missing or unexpected:
        raise ValueError(
            f"Gate 0.1 metric result for {method} is incompatible: "
            f"missing={missing}, unexpected={unexpected}."
        )
    result = {name: float(values[name]) for name in metrics}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"Gate 0.1 metric result for {method} is non-finite.")
    return result


def _paired_comparison(
    calibrated_identity: Mapping[str, float],
    calibrated_sb: Mapping[str, float],
    metrics: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metrics:
        raw_delta = float(calibrated_sb[metric] - calibrated_identity[metric])
        improvement = -raw_delta if LOWER_IS_BETTER[metric] else raw_delta
        tolerance = 1e-12
        winner = "sb_v2" if improvement > tolerance else (
            "identity" if improvement < -tolerance else "tie"
        )
        result[metric] = {
            "calibrated_sb_minus_calibrated_identity": raw_delta,
            "improvement_favoring_sb": improvement,
            "winner": winner,
        }
    return result


def _reduce_background_leakage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = (
        "raw_identity",
        "raw_sb_v2",
        "stage1_reconstruction_ceiling",
    )
    result: dict[str, Any] = {"num_pairs": len(rows), "methods": {}}
    for method in methods:
        blocks = [row["raw_pre_mask_background_leakage"][method] for row in rows]
        voxel_count = sum(int(block["voxel_count"]) for block in blocks)
        nonzero_count = sum(int(block["nonzero_voxel_count"]) for block in blocks)
        result["methods"][method] = {
            "background_voxel_count": voxel_count,
            "nonzero_voxel_count": nonzero_count,
            "nonzero_fraction": (
                float(nonzero_count / voxel_count) if voxel_count else 0.0
            ),
            "mean_case_mean_abs": _mean_or_none(
                [float(block["mean_abs"]) for block in blocks]
            ),
            "mean_case_rms": _mean_or_none(
                [float(block["rms"]) for block in blocks]
            ),
            "max_abs": max(
                (float(block["max_abs"]) for block in blocks), default=0.0
            ),
        }
    return result


def _reduce_rows(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, Any]:
    return {
        "num_pairs": len(rows),
        "methods": {
            method: {
                metric: _mean_or_none(
                    [row["methods"][method][metric] for row in rows]
                )
                for metric in metrics
            }
            for method in methods
        },
        "central_comparison": _reduce_central(rows, metrics)["overall"],
    }


def _reduce_central(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, Any]:
    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"num_pairs": len(subset), "metrics": {}}
        for metric in metrics:
            items = [row["central_comparison"][metric] for row in subset]
            result["metrics"][metric] = {
                "mean_calibrated_sb_minus_calibrated_identity": _mean_or_none(
                    [item["calibrated_sb_minus_calibrated_identity"] for item in items]
                ),
                "mean_improvement_favoring_sb": _mean_or_none(
                    [item["improvement_favoring_sb"] for item in items]
                ),
                "wins": {
                    name: sum(item["winner"] == name for item in items)
                    for name in ("sb_v2", "identity", "tie")
                },
            }
        return result

    return {
        "definition": "calibrated SB-v2 versus calibrated identity on identical pairs",
        "overall": block(rows),
        "by_contrast": {
            contrast.value: block(
                [row for row in rows if row["contrast"] == contrast.value]
            )
            for contrast in CONTRASTS
        },
    }


def _directed_pair_reductions(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {}
    matrices: dict[str, Any] = {}
    field_labels = [f"{field:g}T" for field in FIELD_STRENGTHS_T]
    for contrast in CONTRASTS:
        contrast_rows: list[dict[str, Any]] = []
        for source in FIELD_STRENGTHS_T:
            for target in FIELD_STRENGTHS_T:
                if source == target:
                    continue
                subset = [
                    row
                    for row in rows
                    if row["contrast"] == contrast.value
                    and row["source_field_t"] == source
                    and row["target_field_t"] == target
                ]
                reduced = _reduce_rows(subset, methods, metrics)
                contrast_rows.append(
                    {
                        "source_field_t": source,
                        "target_field_t": target,
                        **reduced,
                    }
                )
        results[contrast.value] = contrast_rows
        matrices[contrast.value] = {
            method: {
                metric: {
                    f"{source:g}T": {
                        f"{target:g}T": (
                            None
                            if source == target
                            else next(
                                row["methods"][method][metric]
                                for row in contrast_rows
                                if row["source_field_t"] == source
                                and row["target_field_t"] == target
                            )
                        )
                        for target in FIELD_STRENGTHS_T
                    }
                    for source in FIELD_STRENGTHS_T
                }
                for metric in metrics
            }
            for method in methods
        }
        matrices[contrast.value]["axes"] = {
            "rows": "source_field_t",
            "columns": "target_field_t",
            "labels": field_labels,
            "diagonal": None,
        }
    return {"results": results, "matrices": matrices}


def _wrong_target_diagnostic(
    case: Gate01Case,
    *,
    calibrator: PosthocTargetCalibrator,
    requested_raw_metrics: Mapping[str, float],
    requested_calibrated_metrics: Mapping[str, float],
    metrics: Sequence[str],
    device: str,
    metric_fn: MetricFn,
) -> dict[str, Any] | None:
    if not case.wrong_target_sb_v2:
        return None
    wrong_rows: list[dict[str, Any]] = []
    contrast = Contrast.parse(case.target_domain.contrast)
    for label, prediction in sorted(case.wrong_target_sb_v2.items()):
        wrong_domain = _wrong_target_domain(label, contrast)
        raw = _validated_metric_result(
            metric_fn(prediction, case.target, metrics, device), metrics, "wrong_target_raw"
        )
        common_requested_prediction = calibrator.apply(
            prediction,
            case.target_domain,
            support_mask=case.support_mask,
            mode="histogram",
        )
        common_requested = _validated_metric_result(
            metric_fn(common_requested_prediction, case.target, metrics, device),
            metrics,
            "wrong_target_common_requested_domain_calibrated",
        )
        condition_native_prediction = calibrator.apply(
            prediction,
            wrong_domain,
            support_mask=case.support_mask,
            mode="histogram",
        )
        condition_native = _validated_metric_result(
            metric_fn(condition_native_prediction, case.target, metrics, device),
            metrics,
            "wrong_target_condition_native_calibrated",
        )
        wrong_rows.append(
            {
                "conditioned_target_field_t": wrong_domain.field_strength_t,
                "raw_metrics_against_requested_target": raw,
                "common_requested_domain_calibrated_metrics_against_requested_target": (
                    common_requested
                ),
                "condition_native_calibrated_metrics_against_requested_target": (
                    condition_native
                ),
            }
        )

    comparisons: dict[str, Any] = {}
    for mode, requested, wrong_key in (
        ("raw", requested_raw_metrics, "raw_metrics_against_requested_target"),
        (
            "common_requested_domain_calibration",
            requested_calibrated_metrics,
            "common_requested_domain_calibrated_metrics_against_requested_target",
        ),
        (
            "condition_native_calibration",
            requested_calibrated_metrics,
            "condition_native_calibrated_metrics_against_requested_target",
        ),
    ):
        comparisons[mode] = {}
        for metric in metrics:
            wrong_values = [
                row[wrong_key][metric]
                for row in wrong_rows
            ]
            wrong_mean = float(sum(wrong_values) / len(wrong_values))
            improvement = (
                wrong_mean - requested[metric]
                if LOWER_IS_BETTER[metric]
                else requested[metric] - wrong_mean
            )
            comparisons[mode][metric] = {
                "requested_value": float(requested[metric]),
                "mean_wrong_value": wrong_mean,
                "requested_better_than_mean_wrong": improvement > 0,
                "margin_favoring_requested": improvement,
            }
    return {
        "wrong_conditions": wrong_rows,
        "comparisons": comparisons,
        "interpretation": {
            "common_requested_domain_calibration": (
                "equal-photometry mechanistic comparison; conditioning varies while both "
                "outputs use the same requested-domain photometric template"
            ),
            "condition_native_calibration": (
                "endpoint diagnostic; conditioning and photometric template both vary and "
                "therefore this endpoint does not isolate target control"
            ),
        },
    }


def _reduce_wrong_target(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, Any]:
    available = [row for row in rows if row["requested_vs_wrong_target"] is not None]
    result: dict[str, Any] = {
        "role": "diagnostic-only requested-versus-wrong conditioning evidence",
        "available_pairs": len(available),
        "not_rerun": True,
        "modes": {},
        "mode_roles": {
            "raw": "uncalibrated endpoint diagnostic",
            "common_requested_domain_calibration": (
                "equal-photometry mechanistic comparison using one requested-domain template"
            ),
            "condition_native_calibration": (
                "condition-native endpoint diagnostic; does not isolate target control"
            ),
        },
    }
    for mode in (
        "raw",
        "common_requested_domain_calibration",
        "condition_native_calibration",
    ):
        result["modes"][mode] = {}
        for metric in metrics:
            entries = [
                row["requested_vs_wrong_target"]["comparisons"][mode][metric]
                for row in available
            ]
            result["modes"][mode][metric] = {
                "mean_margin_favoring_requested": _mean_or_none(
                    [entry["margin_favoring_requested"] for entry in entries]
                ),
                "requested_better_fraction": _mean_or_none(
                    [
                        float(entry["requested_better_than_mean_wrong"])
                        for entry in entries
                    ]
                ),
            }
    return result


def _scientific_promotion_decision(*, private_data_run: bool, eligible: bool) -> str:
    if not private_data_run:
        return "unset_pending_private_data_run"
    if eligible:
        return "unset_pending_scientific_review"
    return "unset_ineligible_scientific_contract"


def _wrong_target_domain(label: str, contrast: Contrast) -> Domain:
    normalized = str(label).strip()
    if normalized.endswith("T"):
        normalized = normalized[:-1]
    return Domain(float(normalized), contrast)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(sum(float(value) for value in values) / len(values)) if values else None


def _markdown_scientific_status(status: Mapping[str, Any]) -> str:
    if status["eligible_for_scientific_conclusions"]:
        return (
            "The strict 60-case protocol lock and private-data execution contract are "
            "validated. Scientific promotion remains **unset pending review**; no "
            "population or challenge claim is made."
        )
    reasons = ", ".join(str(value) for value in status["ineligibility_reasons"])
    prefix = (
        "This is development evidence only and is "
        if status["execution_mode"] != "scientific"
        else "This result is "
    )
    pending = (
        "**unset pending a private-data run** validated against this contract"
        if status.get("private_data_run") is not True
        else "**unset because the scientific contract is ineligible pending correction/review**"
    )
    return (
        prefix + "explicitly ineligible for scientific conclusions "
        f"(`{reasons}`). Scientific promotion remains {pending}; no population or "
        "challenge claim is made."
    )


def _markdown_metrics_table(methods: Mapping[str, Mapping[str, Any]]) -> str:
    metric_names = list(next(iter(methods.values()), {}).keys())
    header = "| Method | " + " | ".join(metric_names) + " |"
    divider = "|---|" + "---:|" * len(metric_names)
    rows = [header, divider]
    for method, values in methods.items():
        cells = [
            "—" if values[name] is None else f"{float(values[name]):.6f}"
            for name in metric_names
        ]
        rows.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _markdown_central_table(block: Mapping[str, Any]) -> str:
    rows = [
        "| Metric | Mean raw delta | Mean improvement | SB wins | Identity wins | Ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in block["metrics"].items():
        delta = values["mean_calibrated_sb_minus_calibrated_identity"]
        improvement = values["mean_improvement_favoring_sb"]
        wins = values["wins"]
        rows.append(
            f"| {metric} | {_fmt_optional(delta)} | {_fmt_optional(improvement)} | "
            f"{wins['sb_v2']} | {wins['identity']} | {wins['tie']} |"
        )
    return "\n".join(rows)


def _markdown_background_leakage(block: Mapping[str, Any]) -> str:
    rows = [
        "| Method | Nonzero / background voxels | Fraction | Mean case | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in block["methods"].items():
        rows.append(
            f"| {method} | {values['nonzero_voxel_count']} / "
            f"{values['background_voxel_count']} | {values['nonzero_fraction']:.6f} | "
            f"{_fmt_optional(values['mean_case_mean_abs'])} | {values['max_abs']:.6f} |"
        )
    return "\n".join(rows)


def _markdown_wrong_target(block: Mapping[str, Any]) -> str:
    rows = [
        f"Available directed cases: {block['available_pairs']}",
        "",
        "| Mode | Metric | Mean margin favoring requested | Requested-better fraction |",
        "|---|---|---:|---:|",
    ]
    for mode, metrics in block["modes"].items():
        for metric, values in metrics.items():
            rows.append(
                f"| {mode} | {metric} | "
                f"{_fmt_optional(values['mean_margin_favoring_requested'])} | "
                f"{_fmt_optional(values['requested_better_fraction'])} |"
            )
    return "\n".join(rows)


def _fmt_optional(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def _write_text_atomic(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


__all__ = [
    "GATE01_CONTRACT_VERSION",
    "GATE01_INPUT_CONTRACT_VERSION",
    "GATE01_VERIFIED_PRODUCER_RECEIPT_VERSION",
    "Gate01Case",
    "Gate01InputManifest",
    "canonical_loaded_array_sha256",
    "evaluate_gate01",
    "fixed_montage_specifications",
    "frozen_artifact_provenance",
    "gate01_code_provenance",
    "gate01_selection_fingerprint",
    "load_gate01_input_manifest",
    "validate_gate01_verified_producer_receipt",
    "official_gate01_metric_fn",
    "render_gate01_markdown",
    "validate_gate01_scientific_hash_graph",
    "validate_frozen_artifact_provenance",
    "write_gate01_outputs",
]
