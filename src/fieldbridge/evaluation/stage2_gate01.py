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
from collections.abc import Callable, Mapping, Sequence
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
    official_task3_ssim,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    GATE0_DIAGNOSTIC_COMMIT,
    GATE01_CALIBRATION_SEMANTICS,
    PosthocTargetCalibrator,
    RESPLIT_FINGERPRINT,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
    reject_target_derived_calibration_fields,
)

GATE01_CONTRACT_VERSION = "stage2-gate01-equal-photometry-v1"
GATE01_INPUT_CONTRACT_VERSION = "stage2-gate01-input-v1"

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
    wrong_target_sb_v2: Mapping[str, torch.Tensor] = field(default_factory=dict)

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

        for label in self.wrong_target_sb_v2:
            wrong = _wrong_target_domain(label, target_contrast)
            if wrong.field_strength_t == self.target_domain.field_strength_t:
                raise ValueError(
                    "wrong_target_sb_v2 must not repeat the requested target field."
                )


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
    cases: Sequence[Gate01Case],
    *,
    calibrator: PosthocTargetCalibrator,
    artifact_provenance: Mapping[str, Any],
    code_commit: str,
    evidence_scope: Mapping[str, Any],
    input_manifest_sha256: str,
    metrics: Sequence[str] = OFFICIAL_METRICS,
    device: str = "cuda",
    include_robust_affine: bool = False,
    metric_fn: MetricFn = official_gate01_metric_fn,
) -> dict[str, Any]:
    """Evaluate raw/equal-calibrated references and reduce the frozen Gate 0.1 tables."""

    if not cases:
        raise ValueError("Gate 0.1 requires at least one full-volume case.")
    requested_metrics = tuple(metrics)
    unsupported = sorted(set(requested_metrics) - set(OFFICIAL_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported Gate 0.1 official metrics: {unsupported}.")
    if "nrmse" not in requested_metrics:
        raise ValueError("Gate 0.1 requires official nRMSE to freeze difficulty strata.")
    if not code_commit:
        raise ValueError("Gate 0.1 requires the evaluation code commit.")
    if not input_manifest_sha256:
        raise ValueError("Gate 0.1 requires the input manifest SHA-256.")
    evidence_scope = _validated_evidence_scope(evidence_scope)
    validate_frozen_artifact_provenance(artifact_provenance)
    calibrator.assert_split_fingerprint(RESPLIT_FINGERPRINT)

    method_names = list(CORE_METHODS)
    if include_robust_affine:
        method_names.extend(DIAGNOSTIC_METHODS)

    rows: list[dict[str, Any]] = []
    for case in cases:
        calibrated_identity = calibrator.apply(
            case.raw_identity, case.target_domain, mode="histogram"
        )
        calibrated_sb = calibrator.apply(
            case.raw_sb_v2, case.target_domain, mode="histogram"
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
                        mode="robust_affine",
                    ),
                    "diagnostic_robust_affine_sb_v2": calibrator.apply(
                        case.raw_sb_v2,
                        case.target_domain,
                        mode="robust_affine",
                    ),
                }
            )

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
            "case_id": case.case_id,
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
            row["case_id"],
        )
    )
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

    contract = {
        "contract_version": GATE01_CONTRACT_VERSION,
        "code_commit": code_commit,
        "input_manifest_sha256": input_manifest_sha256,
        "evidence_scope": dict(evidence_scope),
        "scientific_promotion_decision": "unset_pending_private_data_run",
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "training_cohort_identity": calibrator.provenance[
            "training_cohort_identity"
        ],
        "calibrator": {
            "contract_version": calibrator.to_dict()["contract_version"],
            "semantics": GATE01_CALIBRATION_SEMANTICS,
            "template_sha256": calibrator.template_sha256,
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
            "fit_code_commit": calibrator.provenance["code_commit"],
        },
        "artifact_provenance": dict(artifact_provenance),
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
            "calibrator_signature": "prediction + requested target domain only",
            "template_fit": "retrospective training records only",
            "paired_target_use": "official metrics and qualitative comparison only",
            "paired_target_for_calibration": False,
            "paired_target_mask_for_calibration": False,
            "paired_target_statistics_for_calibration": False,
            "background": "prediction exact-zero voxels remain exact zero",
        },
    }

    return {
        "contract_version": GATE01_CONTRACT_VERSION,
        "evidence_scope": dict(evidence_scope),
        "scientific_status": {
            "evidence": "development diagnostic",
            "promotion_decision": "unset_pending_private_data_run",
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
                "requested_vs_wrong_target": (
                    "official metric formulas used for a mechanistic comparison, not "
                    "an official challenge score"
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
        "This is development evidence only. Scientific promotion remains "
        "**unset pending a private-data run**; no population or challenge claim is made.",
        "",
        "Evidence scope: " + json.dumps(result["evidence_scope"], sort_keys=True),
        "",
        "## Calibration contract",
        "",
        "Both identity and SB-v2 are projected from their own prediction CDF onto the same "
        "frozen requested-target-domain foreground CDF fitted from retrospective training "
        "records. The paired evaluation target is used only by the official metrics and is "
        "never passed to calibration. Exact-zero prediction background remains zero.",
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
    lines.extend(["## Per contrast", ""])
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
            "## Directed-pair matrices and qualitative panels",
            "",
            "All 20 directed field pairs for every contrast, their paired deltas/wins, and "
            "the frozen montage specification are included in the machine-readable JSON.",
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
    "evidence_scope",
    "split_fingerprint",
    "artifact_provenance",
    "cases",
}
_CASE_INPUT_KEYS = {
    "case_id",
    "source_domain",
    "target_domain",
    "target",
    "raw_identity",
    "raw_sb_v2",
    "stage1_reconstruction_ceiling",
    "wrong_target_sb_v2",
}


def load_gate01_input_manifest(
    path: str | Path,
) -> tuple[list[Gate01Case], dict[str, Any]]:
    """Load a strict external prediction manifest without copying paths into outputs."""

    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 input manifest root must be a mapping.")
    reject_target_derived_calibration_fields(payload)
    _assert_exact_keys(payload, _ROOT_INPUT_KEYS, "Gate 0.1 input manifest")
    if payload["contract_version"] != GATE01_INPUT_CONTRACT_VERSION:
        raise ValueError("Gate 0.1 input manifest contract is incompatible.")
    if payload["split_fingerprint"] != RESPLIT_FINGERPRINT:
        raise ValueError("Gate 0.1 input manifest has a stale split fingerprint.")
    validate_frozen_artifact_provenance(payload["artifact_provenance"])
    evidence_scope = _validated_evidence_scope(payload["evidence_scope"])

    cases: list[Gate01Case] = []
    manifest_root = source.resolve().parent
    for index, item in enumerate(payload["cases"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"Gate 0.1 case {index} must be a mapping.")
        _assert_exact_keys(
            item,
            _CASE_INPUT_KEYS,
            f"Gate 0.1 case {index}",
            optional={"wrong_target_sb_v2"},
        )
        source_domain = Domain.from_dict(dict(item["source_domain"]))
        target_domain = Domain.from_dict(dict(item["target_domain"]))
        wrong = {
            str(label): _load_external_volume(_resolve_manifest_path(manifest_root, value))
            for label, value in dict(item.get("wrong_target_sb_v2", {})).items()
        }
        cases.append(
            Gate01Case(
                case_id=str(item["case_id"]),
                source_domain=source_domain,
                target_domain=target_domain,
                target=_load_external_volume(
                    _resolve_manifest_path(manifest_root, item["target"])
                ),
                raw_identity=_load_external_volume(
                    _resolve_manifest_path(manifest_root, item["raw_identity"])
                ),
                raw_sb_v2=_load_external_volume(
                    _resolve_manifest_path(manifest_root, item["raw_sb_v2"])
                ),
                stage1_reconstruction_ceiling=_load_external_volume(
                    _resolve_manifest_path(
                        manifest_root, item["stage1_reconstruction_ceiling"]
                    )
                ),
                wrong_target_sb_v2=wrong,
            )
        )
    metadata = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_scope": dict(evidence_scope),
        "artifact_provenance": dict(payload["artifact_provenance"]),
        "split_fingerprint": str(payload["split_fingerprint"]),
    }
    return cases, metadata


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


def _validated_evidence_scope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Gate 0.1 evidence_scope must be a mapping.")
    required = {"role", "traveller", "private_data_run"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"Gate 0.1 evidence_scope is incomplete; missing {missing}."
        )
    if not isinstance(value["private_data_run"], bool):
        raise ValueError("Gate 0.1 evidence_scope.private_data_run must be boolean.")
    if not str(value["role"]).strip() or not str(value["traveller"]).strip():
        raise ValueError(
            "Gate 0.1 evidence_scope role and traveller must be non-empty."
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
        calibrated_prediction = calibrator.apply(
            prediction, wrong_domain, mode="histogram"
        )
        calibrated = _validated_metric_result(
            metric_fn(calibrated_prediction, case.target, metrics, device),
            metrics,
            "wrong_target_calibrated",
        )
        wrong_rows.append(
            {
                "conditioned_target_field_t": wrong_domain.field_strength_t,
                "raw_metrics_against_requested_target": raw,
                "calibrated_metrics_against_requested_target": calibrated,
            }
        )

    comparisons: dict[str, Any] = {}
    for mode, requested in (
        ("raw", requested_raw_metrics),
        ("calibrated", requested_calibrated_metrics),
    ):
        comparisons[mode] = {}
        for metric in metrics:
            wrong_values = [
                row[f"{mode}_metrics_against_requested_target"][metric]
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
    return {"wrong_conditions": wrong_rows, "comparisons": comparisons}


def _reduce_wrong_target(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, Any]:
    available = [row for row in rows if row["requested_vs_wrong_target"] is not None]
    result: dict[str, Any] = {
        "role": "diagnostic-only mechanistic target-control evidence",
        "available_pairs": len(available),
        "not_rerun": True,
        "modes": {},
    }
    for mode in ("raw", "calibrated"):
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


def _wrong_target_domain(label: str, contrast: Contrast) -> Domain:
    normalized = str(label).strip()
    if normalized.endswith("T"):
        normalized = normalized[:-1]
    return Domain(float(normalized), contrast)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(sum(float(value) for value in values) / len(values)) if values else None


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
    "Gate01Case",
    "evaluate_gate01",
    "fixed_montage_specifications",
    "frozen_artifact_provenance",
    "load_gate01_input_manifest",
    "official_gate01_metric_fn",
    "render_gate01_markdown",
    "validate_frozen_artifact_provenance",
    "write_gate01_outputs",
]
