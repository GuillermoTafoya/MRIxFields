"""Protocol-level paired evaluation for the non-learned Stage-2 Variant-A baseline.

This module never constructs cross-field pairs.  It consumes only an externally sealed
manifest whose source and target endpoints share one canonical subject group.  Gate 0.1
metrics remain an external continuity track and never enter same-case reductions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import Domain
from fieldbridge.data.photometry_factorization import (
    FrozenPhotometryArtifact,
    assert_variant_a_external_path,
    canonical_tensor_sha256,
    classify_variant_a_cohort,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
)
from fieldbridge.evaluation.metrics import gradient_mae
from fieldbridge.evaluation.stage2_photometry_baseline import (
    FIXED_MAP_METHOD,
    OFFICIAL_METRICS,
    RAW_IDENTITY_METHOD,
    STAGE1_CEILING_METHOD,
    VARIANT_A_CASE_SHARD_VERSION,
    VARIANT_A_CONTINUITY_REFERENCE_VERSION,
    VARIANT_A_DUAL_BASELINE_RESULT_VERSION,
    VARIANT_A_PAIRED_MANIFEST_VERSION,
    ContinuityReference,
    MetricFunction,
    PairedEvaluationCase,
)

PAIRED_EVALUATION_ROLE = "externally-authorized-genuinely-paired-evaluation"
RAW_IDENTITY_CATASTROPHIC_BOUNDARY = 1.0
_LOWER_BETTER = frozenset({"nrmse", "lpips"})
_CONTROL_NAMES = (
    "same_support_intensity_mae_to_source",
    "same_support_gradient_mae_to_source",
    "same_support_edge_mae_to_source",
    "same_support_gradient_mae_to_target",
    "same_support_edge_mae_to_target",
)

ProgressFunction = Callable[[int, int, str], None]


def seal_paired_evaluation_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic self-hashed paired-manifest payload."""

    body = _json_mapping(payload)
    body.pop("membership_fingerprint", None)
    body.pop("manifest_sha256", None)
    cases = body.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("Paired evaluation manifest requires at least one case.")
    membership = _paired_membership_fingerprint(cases)
    split = body.get("split_provenance")
    if not isinstance(split, Mapping):
        raise ValueError("Paired evaluation manifest requires split provenance.")
    split_payload = dict(split)
    existing = split_payload.get("evaluation_membership_fingerprint")
    if existing not in (None, membership):
        raise ValueError("Paired evaluation split membership fingerprint conflicts with cases.")
    split_payload["evaluation_membership_fingerprint"] = membership
    body["split_provenance"] = split_payload
    body["membership_fingerprint"] = membership
    body["manifest_sha256"] = sha256_json(body)
    return body


def load_paired_evaluation_manifest(
    path: str | Path,
    *,
    artifact: FrozenPhotometryArtifact,
) -> tuple[dict[str, Any], tuple[PairedEvaluationCase, ...]]:
    """Load, hash-verify, and materialize externally sealed genuinely paired cases."""

    manifest_path = assert_variant_a_external_path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load paired evaluation manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Paired evaluation manifest root must be an object.")
    normalized = _json_mapping(payload)
    if normalized.get("contract_version") != VARIANT_A_PAIRED_MANIFEST_VERSION:
        raise ValueError("Unsupported paired evaluation manifest contract.")
    stored_hash = str(normalized.get("manifest_sha256", ""))
    body = {key: value for key, value in normalized.items() if key != "manifest_sha256"}
    if stored_hash != sha256_json(body):
        raise ValueError("Paired evaluation manifest content hash mismatch.")
    if normalized.get("data_role") != PAIRED_EVALUATION_ROLE:
        raise ValueError("Paired evaluation manifest lacks the authorized paired data role.")
    if not str(normalized.get("evaluation_identity", "")):
        raise ValueError("Paired evaluation manifest requires an evaluation identity.")
    _validate_metric_names(normalized.get("metrics"))
    if float(normalized.get("raw_identity_catastrophic_boundary", math.nan)) != (
        RAW_IDENTITY_CATASTROPHIC_BOUNDARY
    ):
        raise ValueError("Paired evaluation catastrophic boundary is incompatible.")
    if normalized.get("photometry_artifact_sha256") != artifact.artifact_sha256:
        raise ValueError("Paired evaluation manifest photometry artifact mismatch.")
    if normalized.get("photometry_config_sha256") != artifact.provenance[
        "resolved_config_sha256"
    ]:
        raise ValueError("Paired evaluation manifest photometry config mismatch.")
    split = normalized.get("split_provenance")
    if not isinstance(split, Mapping):
        raise ValueError("Paired evaluation split provenance is missing.")
    _require_sha256(str(split.get("file_sha256", "")), "paired split file")
    if not str(split.get("role", "")):
        raise ValueError("Paired evaluation split provenance requires a role.")
    _require_sha256(
        str(split.get("source_membership_fingerprint", "")),
        "paired source split membership",
    )
    _require_sha256(
        str(split.get("source_recovery_fingerprint", "")),
        "paired source split recovery",
    )
    provenance = normalized.get("provenance")
    if not isinstance(provenance, Mapping) or not str(
        provenance.get("authorization_reference", "")
    ):
        raise ValueError("Paired evaluation manifest requires authorization provenance.")

    cases_raw = normalized.get("cases")
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, (str, bytes)):
        raise ValueError("Paired evaluation manifest cases must be a sequence.")
    membership = _paired_membership_fingerprint(cases_raw)
    if normalized.get("membership_fingerprint") != membership:
        raise ValueError("Paired evaluation membership fingerprint mismatch.")
    if split.get("evaluation_membership_fingerprint") != membership:
        raise ValueError("Paired split membership does not match manifest cases.")

    cases: list[PairedEvaluationCase] = []
    identities: set[str] = set()
    for item in cases_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Paired evaluation case must be an object.")
        case_identity = str(item.get("case_identity", ""))
        if not case_identity or case_identity in identities:
            raise ValueError("Paired evaluation case identity is missing or duplicated.")
        identities.add(case_identity)
        if item.get("genuinely_paired") is not True:
            raise ValueError("Paired evaluation refuses an unsealed or synthetic endpoint pair.")
        source, source_domain, source_identity, source_provenance = _load_endpoint(
            item.get("source"), "source"
        )
        target, target_domain, target_identity, target_provenance = _load_endpoint(
            item.get("target"), "target"
        )
        if source_identity.subject_group_identity != target_identity.subject_group_identity:
            raise ValueError("Paired evaluation endpoints must belong to the same subject group.")
        if source_domain.contrast != target_domain.contrast:
            raise ValueError("Paired evaluation supports same-contrast edges only.")
        if source_domain.field_strength_t == target_domain.field_strength_t:
            raise ValueError("Paired evaluation requires a cross-field edge.")
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError("Paired evaluation source and target shapes do not match.")
        stage1, stage1_provenance = _load_optional_stage1(
            item.get("stage1_reconstruction"), source
        )
        cases.append(
            PairedEvaluationCase(
                case_identity=case_identity,
                source=source,
                target=target,
                source_domain=source_domain,
                target_domain=target_domain,
                subject_group_identity=source_identity.subject_group_identity,
                source_provenance=source_provenance,
                target_provenance=target_provenance,
                stage1_reconstruction=stage1,
                stage1_provenance=stage1_provenance,
            )
        )
    return normalized, tuple(cases)


def evaluate_paired_variant_a(
    artifact: FrozenPhotometryArtifact,
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    continuity: ContinuityReference,
    metric_function: MetricFunction,
    metric_runtime_provenance: Mapping[str, Any],
    evaluation_code_provenance: Mapping[str, Any],
    resume: bool = False,
    progress: ProgressFunction | None = None,
) -> dict[str, Any]:
    """Evaluate one sealed paired manifest with deterministic atomic per-case resume."""

    manifest, cases = load_paired_evaluation_manifest(manifest_path, artifact=artifact)
    if continuity.evaluation_identity != manifest["evaluation_identity"]:
        raise ValueError("Continuity and paired manifest evaluation identities do not match.")
    runtime = _json_mapping(metric_runtime_provenance)
    code = _json_mapping(evaluation_code_provenance)
    if not runtime or not code:
        raise ValueError("Paired evaluation requires metric-runtime and code provenance.")
    root = assert_variant_a_external_path(output_dir)
    existing = list(root.iterdir()) if root.exists() else []
    if existing and not resume:
        raise FileExistsError("Paired evaluation output is nonempty; pass resume=True to verify it.")
    if existing and resume and not (root / "run_contract.json").is_file():
        raise ValueError("Paired evaluation cannot resume a nonempty uncontracted directory.")
    root.mkdir(parents=True, exist_ok=True)
    shard_dir = root / "case_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_contract: dict[str, Any] = {
        "contract_version": "stage2-photometry-paired-evaluation-run-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_membership_fingerprint": manifest["membership_fingerprint"],
        "evaluation_identity": manifest["evaluation_identity"],
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "photometry_config_sha256": artifact.provenance["resolved_config_sha256"],
        "fit_code_commit": artifact.provenance["code_commit"],
        "fit_code_provenance": artifact.provenance["code_provenance"],
        "evaluation_code_provenance": code,
        "official_metrics": list(OFFICIAL_METRICS),
        "metric_runtime_provenance": runtime,
        "continuity_reference_sha256": continuity.artifact_sha256,
        "split_provenance": dict(manifest["split_provenance"]),
        "case_count": len(cases),
    }
    run_contract["run_contract_sha256"] = sha256_json(run_contract)
    contract_path = root / "run_contract.json"
    if contract_path.exists():
        if _load_hashed_json(contract_path, "run_contract_sha256") != run_contract:
            raise ValueError("Paired evaluation resume contract mismatch.")
    else:
        write_json_atomic(contract_path, run_contract, refuse_existing=True)

    final_path = root / "result.json"
    if final_path.exists():
        if not resume:
            raise FileExistsError("Paired evaluation final result already exists.")
        result = _load_hashed_json(final_path, "result_sha256")
        if result.get("run_contract_sha256") != run_contract["run_contract_sha256"]:
            raise ValueError("Paired evaluation final result belongs to another run.")
        return result

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        input_fingerprint = _case_input_fingerprint(case, run_contract)
        shard_path = shard_dir / f"{sha256_text(case.case_identity)}.json"
        if shard_path.exists():
            if not resume:
                raise FileExistsError("Paired evaluation shard exists without resume authorization.")
            shard = _load_hashed_json(shard_path, "result_sha256")
            if (
                shard.get("contract_version") != VARIANT_A_CASE_SHARD_VERSION
                or shard.get("run_contract_sha256") != run_contract["run_contract_sha256"]
                or shard.get("case_input_sha256") != input_fingerprint
            ):
                raise ValueError("Paired evaluation shard resume verification failed.")
            row = dict(shard["case"])
        else:
            row = _evaluate_case(artifact, case, metric_function)
            shard = {
                "contract_version": VARIANT_A_CASE_SHARD_VERSION,
                "run_contract_sha256": run_contract["run_contract_sha256"],
                "case_input_sha256": input_fingerprint,
                "case": row,
            }
            shard["result_sha256"] = sha256_json(shard)
            write_json_atomic(shard_path, shard, refuse_existing=True)
        rows.append(row)
        if progress is not None:
            progress(index, len(cases), case.case_identity)

    rows.sort(key=lambda item: str(item["case_identity"]))
    result: dict[str, Any] = {
        "contract_version": VARIANT_A_DUAL_BASELINE_RESULT_VERSION,
        "run_contract_sha256": run_contract["run_contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "membership_fingerprint": manifest["membership_fingerprint"],
        "evaluation_identity": manifest["evaluation_identity"],
        "same_case_methods": [
            method
            for method in (RAW_IDENTITY_METHOD, FIXED_MAP_METHOD, STAGE1_CEILING_METHOD)
            if any(method in row["methods"] for row in rows)
        ],
        "cases": rows,
        "reductions": _aggregate_rows(rows),
        "strata": _stratified_reductions(rows),
        "external_continuity_track": {
            "contract_version": VARIANT_A_CONTINUITY_REFERENCE_VERSION,
            "reference_sha256": continuity.artifact_sha256,
            "source_result_sha256": continuity.source_result_sha256,
            "same_cases_recomputed": False,
            "included_in_same_case_reductions": False,
            "methods": {key: dict(value) for key, value in continuity.methods.items()},
            "provenance": dict(continuity.provenance),
            "semantics": (
                "Gate 0.1 posthoc calibration is an external continuity reference only"
            ),
        },
        "anatomy_reporting_semantics": (
            "unthresholded same-support intensity, spatial-gradient, and edge differences; "
            "not an anatomy score or promotion decision"
        ),
        "qualification_vs_promotion": (
            "this result is engineering/scientific baseline evidence and declares no learned "
            "candidate promotion"
        ),
        "provenance": {
            "photometry_artifact_sha256": artifact.artifact_sha256,
            "photometry_config_sha256": artifact.provenance["resolved_config_sha256"],
            "split_provenance": dict(manifest["split_provenance"]),
            "metric_runtime_provenance": runtime,
            "evaluation_code_provenance": code,
        },
    }
    result["result_sha256"] = sha256_json(result)
    write_json_atomic(final_path, result, refuse_existing=True)
    return result


def _load_endpoint(
    raw: Any, role: str
) -> tuple[torch.Tensor, Domain, Any, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Paired evaluation {role} endpoint must be an object.")
    identity = classify_variant_a_cohort(
        case_identity=str(raw.get("case_id", "")),
        metadata_prefix=raw.get("metadata_prefix"),
        supplied_cohort=raw.get("cohort"),
        subject_identity=raw.get("subject_id"),
        allowed_cohorts=("R", "P"),
    )
    domain_raw = raw.get("domain")
    if not isinstance(domain_raw, Mapping):
        raise ValueError(f"Paired evaluation {role} domain is missing.")
    domain = Domain.from_dict(dict(domain_raw))
    array, provenance = _load_array_spec(raw, role)
    provenance.update(
        {
            "case_id": identity.case_identity,
            "subject_identity": identity.subject_identity,
            "subject_group_identity": identity.subject_group_identity,
            "cohort": identity.cohort,
            "metadata_prefix": str(raw.get("metadata_prefix")),
            "domain": domain.to_dict(),
        }
    )
    return array, domain, identity, provenance


def _load_optional_stage1(
    raw: Any, reference: torch.Tensor
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, Mapping):
        raise ValueError("Stage-1 reconstruction spec must be an object when supplied.")
    array, provenance = _load_array_spec(raw, "stage1 reconstruction")
    if tuple(array.shape) != tuple(reference.shape):
        raise ValueError("Stage-1 reconstruction shape does not match paired source/target.")
    return array, provenance


def _load_array_spec(raw: Mapping[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
    content_identity = str(raw.get("content_identity", "")).strip()
    if not content_identity:
        raise ValueError(f"Paired evaluation {role} requires a content identity.")
    path = assert_variant_a_external_path(str(raw.get("array_path", "")))
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Paired evaluation {role} accepts external .npy only.")
    expected_file = str(raw.get("file_sha256", ""))
    expected_loaded = str(raw.get("loaded_array_sha256", ""))
    _require_sha256(expected_file, f"{role} file")
    _require_sha256(expected_loaded, f"{role} loaded array")
    if sha256_file(path) != expected_file:
        raise ValueError(f"Paired evaluation {role} file SHA-256 mismatch.")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load paired evaluation {role} array: {exc}") from exc
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Paired evaluation {role} array must be floating point.")
    tensor = torch.from_numpy(np.array(array, copy=True)).float()
    _validate_volume(tensor, role)
    actual_loaded = canonical_tensor_sha256(tensor)
    if actual_loaded != expected_loaded:
        raise ValueError(f"Paired evaluation {role} loaded-array SHA-256 mismatch.")
    declared_shape = raw.get("shape")
    if not isinstance(declared_shape, Sequence) or isinstance(declared_shape, (str, bytes)):
        raise ValueError(f"Paired evaluation {role} requires a declared loaded shape.")
    try:
        normalized_shape = tuple(int(value) for value in declared_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Paired evaluation {role} loaded shape is invalid.") from exc
    if normalized_shape != tuple(tensor.shape):
        raise ValueError(f"Paired evaluation {role} loaded shape mismatch.")
    if raw.get("dtype") != str(tensor.dtype):
        raise ValueError(f"Paired evaluation {role} loaded dtype mismatch.")
    return tensor, {
        "content_identity": content_identity,
        "content_identity_sha256": sha256_text(content_identity),
        "file_sha256": expected_file,
        "loaded_array_sha256": actual_loaded,
        "path_identity_sha256": sha256_text(str(path)),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _evaluate_case(
    artifact: FrozenPhotometryArtifact,
    case: PairedEvaluationCase,
    metric_function: MetricFunction,
) -> dict[str, Any]:
    source = _validate_volume(case.source, "paired source")
    target = _validate_volume(case.target, "paired target")
    support = source != 0
    if not bool(support.any()):
        raise ValueError("Paired evaluation source support is empty.")
    source_masked = _apply_support(source, support)
    target_masked = _apply_support(target, support)
    fixed = _apply_support(
        artifact.factorized_identity(source, case.source_domain, case.target_domain),
        support,
    )
    predictions: dict[str, torch.Tensor] = {
        RAW_IDENTITY_METHOD: source_masked,
        FIXED_MAP_METHOD: fixed,
    }
    if case.stage1_reconstruction is not None:
        predictions[STAGE1_CEILING_METHOD] = _apply_support(
            _validate_volume(case.stage1_reconstruction, "Stage-1 reconstruction"),
            support,
        )
    methods: dict[str, Any] = {}
    for name, prediction in predictions.items():
        metrics = _exact_metrics(metric_function(prediction, target_masked), name)
        methods[name] = {
            "metrics": metrics,
            "spatial_ssim": metrics["ssim"],
            "controls": _continuity_controls(
                prediction, source_masked, target_masked, support
            ),
            "prediction_sha256": canonical_tensor_sha256(prediction),
        }
    comparison = _fixed_vs_raw(methods)
    raw_nrmse = float(methods[RAW_IDENTITY_METHOD]["metrics"]["nrmse"])
    return {
        "case_identity": case.case_identity,
        "case_identity_sha256": sha256_text(case.case_identity),
        "subject_group_identity": case.subject_group_identity,
        "source_domain": case.source_domain.to_dict(),
        "target_domain": case.target_domain.to_dict(),
        "direction": f"{case.source_domain.label}->{case.target_domain.label}",
        "contrast": case.source_domain.contrast.value,
        "source": dict(case.source_provenance),
        "target": dict(case.target_provenance),
        "stage1_reconstruction": (
            dict(case.stage1_provenance) if case.stage1_provenance is not None else None
        ),
        "support": _support_provenance(support),
        "methods": methods,
        "fixed_vs_raw": comparison,
        "raw_identity_stratum": (
            "catastrophic"
            if raw_nrmse > RAW_IDENTITY_CATASTROPHIC_BOUNDARY
            else "ordinary"
        ),
    }


def _continuity_controls(
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, float]:
    pred5, source5, target5, mask5 = (
        _as_5d(prediction),
        _as_5d(source),
        _as_5d(target),
        _as_5d(support.to(torch.float32)),
    )
    return {
        "same_support_intensity_mae_to_source": _masked_mae(prediction, source, support),
        "same_support_gradient_mae_to_source": float(
            gradient_mae(pred5, source5, mask5).cpu()
        ),
        "same_support_edge_mae_to_source": _edge_mae(pred5, source5, mask5),
        "same_support_gradient_mae_to_target": float(
            gradient_mae(pred5, target5, mask5).cpu()
        ),
        "same_support_edge_mae_to_target": _edge_mae(pred5, target5, mask5),
    }


def _fixed_vs_raw(methods: Mapping[str, Any]) -> dict[str, Any]:
    raw = methods[RAW_IDENTITY_METHOD]["metrics"]
    fixed = methods[FIXED_MAP_METHOD]["metrics"]
    deltas = {metric: float(fixed[metric]) - float(raw[metric]) for metric in OFFICIAL_METRICS}
    wins = {
        metric: (deltas[metric] < 0 if metric in _LOWER_BETTER else deltas[metric] > 0)
        for metric in OFFICIAL_METRICS
    }
    return {
        "delta_fixed_minus_raw": deltas,
        "fixed_map_better": wins,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["direction"])].append(row)
    per_domain = {
        label: _reduce_cases(items) for label, items in sorted(by_domain.items())
    }
    by_contrast: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for label, payload in per_domain.items():
        contrast = str(payload["contrast"])
        by_contrast[contrast].append(payload)
    per_contrast = {
        contrast: _reduce_domain_payloads(items)
        for contrast, items in sorted(by_contrast.items())
    }
    return {
        "per_domain_equal_case": per_domain,
        "per_contrast_equal_domain": per_contrast,
        "macro_equal_domain": _reduce_domain_payloads(list(per_domain.values())),
    }


def _reduce_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    method_names = sorted(
        {str(name) for row in rows for name in row["methods"]}
    )
    for method in method_names:
        available = [row["methods"][method] for row in rows if method in row["methods"]]
        methods[method] = {
            "case_count": len(available),
            "metrics": {
                metric: _mean(item["metrics"][metric] for item in available)
                for metric in OFFICIAL_METRICS
            },
            "controls": {
                control: _mean(item["controls"][control] for item in available)
                for control in _CONTROL_NAMES
            },
        }
    return {
        "contrast": str(rows[0]["contrast"]),
        "case_count": len(rows),
        "methods": methods,
        "fixed_vs_raw": {
            "mean_delta_fixed_minus_raw": {
                metric: _mean(
                    row["fixed_vs_raw"]["delta_fixed_minus_raw"][metric] for row in rows
                )
                for metric in OFFICIAL_METRICS
            },
            "fixed_map_win_count": {
                metric: sum(
                    bool(row["fixed_vs_raw"]["fixed_map_better"][metric]) for row in rows
                )
                for metric in OFFICIAL_METRICS
            },
            "fixed_map_win_fraction": {
                metric: _mean(
                    float(bool(row["fixed_vs_raw"]["fixed_map_better"][metric]))
                    for row in rows
                )
                for metric in OFFICIAL_METRICS
            },
        },
    }


def _reduce_domain_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not payloads:
        return None
    methods: dict[str, Any] = {}
    method_names = sorted(
        {str(name) for payload in payloads for name in payload["methods"]}
    )
    for method in method_names:
        available = [payload["methods"][method] for payload in payloads if method in payload["methods"]]
        methods[method] = {
            "domain_count": len(available),
            "metrics": {
                metric: _mean(item["metrics"][metric] for item in available)
                for metric in OFFICIAL_METRICS
            },
            "controls": {
                control: _mean(item["controls"][control] for item in available)
                for control in _CONTROL_NAMES
            },
        }
    return {
        "domain_count": len(payloads),
        "case_count": sum(int(payload["case_count"]) for payload in payloads),
        "methods": methods,
        "fixed_vs_raw": {
            "mean_delta_fixed_minus_raw": {
                metric: _mean(
                    payload["fixed_vs_raw"]["mean_delta_fixed_minus_raw"][metric]
                    for payload in payloads
                )
                for metric in OFFICIAL_METRICS
            },
            "fixed_map_win_fraction": {
                metric: _mean(
                    payload["fixed_vs_raw"]["fixed_map_win_fraction"][metric]
                    for payload in payloads
                )
                for metric in OFFICIAL_METRICS
            },
        },
    }


def _stratified_reductions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordinary = [row for row in rows if row["raw_identity_stratum"] == "ordinary"]
    catastrophic = [row for row in rows if row["raw_identity_stratum"] == "catastrophic"]
    return {
        "frozen_definition": "raw identity official nRMSE > 1.0",
        "assignment_source": "same-case raw_identity only",
        "ordinary_case_count": len(ordinary),
        "catastrophic_case_count": len(catastrophic),
        "ordinary": _aggregate_rows(ordinary) if ordinary else None,
        "catastrophic": _aggregate_rows(catastrophic) if catastrophic else None,
    }


def _paired_membership_fingerprint(cases: Sequence[Any]) -> str:
    identities: list[dict[str, Any]] = []
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise ValueError("Paired manifest contains a malformed case.")
        source, target = raw.get("source"), raw.get("target")
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError("Paired manifest case endpoints are malformed.")
        identities.append(
            {
                "case_identity": raw.get("case_identity"),
                "genuinely_paired": raw.get("genuinely_paired"),
                "source_case_id": source.get("case_id"),
                "target_case_id": target.get("case_id"),
                "source_subject_id": source.get("subject_id"),
                "target_subject_id": target.get("subject_id"),
                "source_metadata_prefix": source.get("metadata_prefix"),
                "target_metadata_prefix": target.get("metadata_prefix"),
                "source_cohort": source.get("cohort"),
                "target_cohort": target.get("cohort"),
                "source_content_identity": source.get("content_identity"),
                "target_content_identity": target.get("content_identity"),
                "source_domain": source.get("domain"),
                "target_domain": target.get("domain"),
                "source_file_sha256": source.get("file_sha256"),
                "target_file_sha256": target.get("file_sha256"),
                "source_loaded_array_sha256": source.get("loaded_array_sha256"),
                "target_loaded_array_sha256": target.get("loaded_array_sha256"),
                "source_shape": source.get("shape"),
                "target_shape": target.get("shape"),
                "source_dtype": source.get("dtype"),
                "target_dtype": target.get("dtype"),
                "stage1_loaded_array_sha256": (
                    raw.get("stage1_reconstruction", {}).get("loaded_array_sha256")
                    if isinstance(raw.get("stage1_reconstruction"), Mapping)
                    else None
                ),
                "stage1_file_sha256": (
                    raw.get("stage1_reconstruction", {}).get("file_sha256")
                    if isinstance(raw.get("stage1_reconstruction"), Mapping)
                    else None
                ),
                "stage1_content_identity": (
                    raw.get("stage1_reconstruction", {}).get("content_identity")
                    if isinstance(raw.get("stage1_reconstruction"), Mapping)
                    else None
                ),
                "stage1_shape": (
                    raw.get("stage1_reconstruction", {}).get("shape")
                    if isinstance(raw.get("stage1_reconstruction"), Mapping)
                    else None
                ),
                "stage1_dtype": (
                    raw.get("stage1_reconstruction", {}).get("dtype")
                    if isinstance(raw.get("stage1_reconstruction"), Mapping)
                    else None
                ),
            }
        )
    identities.sort(key=lambda item: str(item["case_identity"]))
    return sha256_json(identities)


def _case_input_fingerprint(
    case: PairedEvaluationCase, run_contract: Mapping[str, Any]
) -> str:
    return sha256_json(
        {
            "case_identity": case.case_identity,
            "source": dict(case.source_provenance),
            "target": dict(case.target_provenance),
            "stage1": dict(case.stage1_provenance or {}),
            "source_domain": case.source_domain.to_dict(),
            "target_domain": case.target_domain.to_dict(),
            "run_contract_sha256": run_contract["run_contract_sha256"],
        }
    )


def _load_hashed_json(path: Path, hash_key: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load resumable Variant-A JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Resumable Variant-A JSON must be an object.")
    normalized = _json_mapping(payload)
    stored = normalized.get(hash_key)
    body = {key: value for key, value in normalized.items() if key != hash_key}
    if stored != sha256_json(body):
        raise ValueError(f"Resumable Variant-A JSON {path.name} hash mismatch.")
    return normalized


def _support_provenance(support: torch.Tensor) -> dict[str, Any]:
    mask = support.detach().cpu().to(torch.bool).contiguous()
    canonical = np.ascontiguousarray(mask.numpy(), dtype=np.uint8)
    header = json.dumps(
        {"dtype": "bool-uint8", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    import hashlib

    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return {
        "shape": list(mask.shape),
        "dtype": "bool",
        "voxel_count": int(mask.sum()),
        "canonical_byte_sha256": digest.hexdigest(),
        "source_only": True,
    }


def _apply_support(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if support.dtype != torch.bool or tuple(values.shape) != tuple(support.shape):
        raise ValueError("Paired evaluation support must be boolean and shape-matched.")
    output = torch.zeros_like(values)
    output[support] = values[support]
    return output


def _validate_volume(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or not values.dtype.is_floating_point:
        raise ValueError(f"{name} must be a floating tensor.")
    if values.ndim < 3 or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be a finite full volume.")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError(f"{name} violates the official [0,1] range.")
    return values


def _exact_metrics(values: Mapping[str, float], name: str) -> dict[str, float]:
    result = {str(key): float(value) for key, value in values.items()}
    if set(result) != set(OFFICIAL_METRICS) or len(result) != len(OFFICIAL_METRICS):
        raise ValueError(f"{name} metrics must be exactly nrmse, ssim, and lpips.")
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name} metrics contain nonfinite values.")
    return {metric: result[metric] for metric in OFFICIAL_METRICS}


def _validate_metric_names(values: Any) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Paired evaluation metrics must be a sequence.")
    names = tuple(str(value) for value in values)
    if len(names) != len(set(names)) or set(names) != set(OFFICIAL_METRICS):
        raise ValueError("Paired scientific evaluation requires exactly nrmse, ssim, and lpips.")


def _masked_mae(first: torch.Tensor, second: torch.Tensor, support: torch.Tensor) -> float:
    return float((first[support] - second[support]).abs().mean().cpu())


def _as_5d(values: torch.Tensor) -> torch.Tensor:
    tensor = values
    while tensor.ndim < 5:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError("Paired spatial controls require a single 3-D volume.")
    return tensor


def _edge_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    first_edges = _edge_magnitude(first)
    second_edges = _edge_magnitude(second)
    valid = mask > 0
    return float((first_edges[valid] - second_edges[valid]).abs().mean().cpu())


def _edge_magnitude(values: torch.Tensor) -> torch.Tensor:
    squared = torch.zeros_like(values)
    for dim in range(2, 5):
        difference = values.diff(dim=dim)
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (4 - dim) + 1] = 1
        squared = squared + torch.nn.functional.pad(difference, padding).square()
    return squared.sqrt()


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if not isinstance(decoded, dict):
        raise TypeError("Variant-A protocol value must be an object.")
    return decoded


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} SHA-256 is invalid.")


def _mean(values: Sequence[float] | Any) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot average an empty Variant-A reduction.")
    return float(sum(items) / len(items))


__all__ = [
    "PAIRED_EVALUATION_ROLE",
    "RAW_IDENTITY_CATASTROPHIC_BOUNDARY",
    "evaluate_paired_variant_a",
    "load_paired_evaluation_manifest",
    "seal_paired_evaluation_manifest",
]
