"""Selected-slice and complete-volume evaluation for real-paired LOSO Track A."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from fieldbridge.data.domains import Domain
from fieldbridge.data.paired_loso import (
    AffineCalibration,
    full_volume_preprocessing_spec,
    reconstruct_native_grid_volume,
    verify_full_slice_coverage,
)
from fieldbridge.data.preprocessing import (
    SlicePreprocessingSpec,
    from_model_range,
    preprocess_volume_slice,
    to_model_range,
)
from fieldbridge.evaluation.metrics import (
    gradient_mae,
    masked_mae,
    normalized_cross_correlation,
    nrmse,
    ssim3d,
)
from fieldbridge.evaluation.prospective_paired import (
    CONTRAST,
    SOURCE_FIELD,
    TARGET_FIELDS,
    compute_paired_metrics,
    conditioning_margins,
    foreground_and_outside_masks,
    validate_preprocessed_geometry,
)
from fieldbridge.models.translators.base import BaseTranslator

EVIDENCE_SCOPE = "prospective_paired_loso_development"
NEURAL_ARMS = ("identity_initialization", "synthetic_initialization")
ALL_ARMS = ("source", "affine", *NEURAL_ARMS)


def evaluate_selected_case(
    *,
    fold_slot: str,
    case_slot: str,
    source_volume: torch.Tensor,
    target_volumes: Mapping[float, torch.Tensor],
    calibrations: Mapping[float, AffineCalibration],
    models: Mapping[str, BaseTranslator],
    preprocessing: SlicePreprocessingSpec,
    slice_indices: Sequence[int],
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(TARGET_FIELDS) * len(slice_indices)
    completed = 0
    for field in TARGET_FIELDS:
        target_volume = target_volumes[field]
        for slice_index in slice_indices:
            source_01, source_geometry = preprocess_volume_slice(
                source_volume,
                int(slice_index),
                preprocessing,
                apply_model_range=False,
            )
            target_01, target_geometry = preprocess_volume_slice(
                target_volume,
                int(slice_index),
                preprocessing,
                apply_model_range=False,
            )
            validate_preprocessed_geometry(source_geometry, target_geometry)
            source_01 = source_01.unsqueeze(0).to(device)
            target_01 = target_01.unsqueeze(0).to(device)
            foreground, outside = foreground_and_outside_masks(target_01, target_geometry)
            arm_metrics: dict[str, dict[str, float]] = {
                "source": compute_paired_metrics(
                    source_01,
                    target_01,
                    source_01,
                    foreground,
                    outside,
                )
            }
            affine = calibrations[field].apply(source_01)
            arm_metrics["affine"] = compute_paired_metrics(
                affine,
                target_01,
                source_01,
                foreground,
                outside,
            )
            conditioning: dict[str, Any] = {}
            source_model = to_model_range(source_01, preprocessing.model_range)
            with torch.inference_mode():
                for arm in NEURAL_ARMS:
                    model = models[arm].to(device).eval()
                    requested_metrics: dict[str, dict[str, float]] = {}
                    for requested_field in TARGET_FIELDS:
                        output = model(
                            source_model,
                            Domain(SOURCE_FIELD, CONTRAST),
                            Domain(requested_field, CONTRAST),
                        )
                        prediction = from_model_range(
                            output,
                            preprocessing.model_range,
                        ).clamp(0.0, 1.0)
                        requested_metrics[_field_label(requested_field)] = compute_paired_metrics(
                            prediction,
                            target_01,
                            source_01,
                            foreground,
                            outside,
                        )
                    true_label = _field_label(field)
                    correct = requested_metrics[true_label]
                    wrong = {
                        label: metrics
                        for label, metrics in requested_metrics.items()
                        if label != true_label
                    }
                    arm_metrics[arm] = correct
                    conditioning[arm] = {
                        "requested": requested_metrics,
                        "margins": {
                            label: conditioning_margins(correct, metrics)
                            for label, metrics in wrong.items()
                        },
                        "correct_best_nrmse": correct["nrmse"]
                        <= min(metrics["nrmse"] for metrics in wrong.values()),
                        "margin_vs_best_wrong_nrmse": min(
                            metrics["nrmse"] for metrics in wrong.values()
                        )
                        - correct["nrmse"],
                    }
            rows.append(
                {
                    "fold_slot": fold_slot,
                    "case_slot": case_slot,
                    "target_field": _field_label(field),
                    "slice_index": int(slice_index),
                    "arms": arm_metrics,
                    "conditioning": conditioning,
                }
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
    return rows


def aggregate_selected_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty paired LOSO rows.")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["fold_slot"]), str(row["case_slot"]), str(row["target_field"]))
        grouped[key].append(row)

    case_field_units: list[dict[str, Any]] = []
    for (fold_slot, case_slot, field), unit_rows in sorted(grouped.items()):
        arms = {
            arm: _mean_metrics([row["arms"][arm] for row in unit_rows])
            for arm in ALL_ARMS
        }
        conditioning: dict[str, Any] = {}
        for arm in NEURAL_ARMS:
            requested = _aggregate_requested(unit_rows, arm)
            correct_nrmse = requested[field]["nrmse"]
            wrong_nrmse = [
                metrics["nrmse"]
                for requested_field, metrics in requested.items()
                if requested_field != field
            ]
            best_wrong_nrmse = min(wrong_nrmse)
            conditioning[arm] = {
                "correct_best_nrmse": correct_nrmse <= best_wrong_nrmse,
                "mean_margin_vs_best_wrong_nrmse": best_wrong_nrmse - correct_nrmse,
                "requested": requested,
            }
        case_field_units.append(
            {
                "fold_slot": fold_slot,
                "case_slot": case_slot,
                "target_field": field,
                "selected_slices": len(unit_rows),
                "arms": arms,
                "conditioning": conditioning,
            }
        )

    per_field: dict[str, Any] = {}
    for field in sorted({_field_label(value) for value in TARGET_FIELDS}):
        units = [unit for unit in case_field_units if unit["target_field"] == field]
        per_field[field] = {
            "case_fold_units": len(units),
            "arms": {
                arm: _mean_metrics([unit["arms"][arm] for unit in units])
                for arm in ALL_ARMS
            },
            "conditioning": {
                arm: {
                    "mean_margin_vs_best_wrong_nrmse": _mean(
                        [
                            unit["conditioning"][arm]["mean_margin_vs_best_wrong_nrmse"]
                            for unit in units
                        ]
                    )
                }
                for arm in NEURAL_ARMS
            },
        }
    macro = {
        arm: _mean_metrics([per_field[field]["arms"][arm] for field in per_field])
        for arm in ALL_ARMS
    }
    return {
        "weighting": {
            "selected_slices_within_case_field": "equal",
            "case_folds_within_field": "equal",
            "target_fields_for_macro": "equal",
        },
        "case_field_units": case_field_units,
        "per_target_field": per_field,
        "macro": macro,
    }


def evaluate_viability(
    aggregate: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    source_macro = aggregate["macro"]["source"]
    affine_macro = aggregate["macro"]["affine"]
    units = aggregate["case_field_units"]
    per_field = aggregate["per_target_field"]
    payload: dict[str, Any] = {"preregistered_rules": dict(rules)}
    for arm in NEURAL_ARMS:
        arm_macro = aggregate["macro"][arm]
        improved_fields = sum(
            per_field[field]["arms"][arm]["nrmse"]
            < per_field[field]["arms"]["source"]["nrmse"]
            for field in per_field
        )
        case_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for unit in units:
            case_groups[str(unit["case_slot"])].append(unit)
        improved_cases = sum(
            _mean([unit["arms"][arm]["nrmse"] for unit in case_units])
            < _mean([unit["arms"]["source"]["nrmse"] for unit in case_units])
            for case_units in case_groups.values()
        )
        correct_best = sum(
            bool(unit["conditioning"][arm]["correct_best_nrmse"])
            for unit in units
        )
        positive_margins = all(
            per_field[field]["conditioning"][arm]["mean_margin_vs_best_wrong_nrmse"] > 0.0
            for field in per_field
        )
        max_regression = max(
            per_field[field]["arms"][arm]["nrmse"]
            - per_field[field]["arms"]["source"]["nrmse"]
            for field in per_field
        )
        checks = {
            "macro_nrmse_below_source": arm_macro["nrmse"] < source_macro["nrmse"],
            "macro_ssim_not_below_source": arm_macro["ssim"] >= source_macro["ssim"],
            "fields_improved_nrmse": improved_fields
            >= int(rules["min_fields_improved_nrmse"]),
            "held_out_cases_improved_nrmse": improved_cases
            >= int(rules["min_held_out_cases_improved_nrmse"]),
            "correct_best_case_field_units": correct_best
            >= int(rules["min_case_field_units_correct_best_nrmse"]),
            "positive_mean_conditioning_margin_every_field": positive_margins,
            "no_material_field_regression": max_regression
            <= float(rules["max_absolute_nrmse_regression_per_field"]),
            "macro_nrmse_below_affine": arm_macro["nrmse"] < affine_macro["nrmse"],
            "macro_ssim_not_below_affine": arm_macro["ssim"] >= affine_macro["ssim"],
        }
        payload[arm] = {
            "viable": all(checks.values()),
            "checks": checks,
            "counts": {
                "fields_improved_nrmse": improved_fields,
                "held_out_cases_improved_nrmse": improved_cases,
                "correct_best_case_field_units": correct_best,
            },
            "max_absolute_field_nrmse_regression": max_regression,
        }
    identity = payload["identity_initialization"]
    synthetic = payload["synthetic_initialization"]
    identity_macro = aggregate["macro"]["identity_initialization"]
    synthetic_macro = aggregate["macro"]["synthetic_initialization"]
    identity_margins = [
        per_field[field]["conditioning"]["identity_initialization"][
            "mean_margin_vs_best_wrong_nrmse"
        ]
        for field in per_field
    ]
    synthetic_margins = [
        per_field[field]["conditioning"]["synthetic_initialization"][
            "mean_margin_vs_best_wrong_nrmse"
        ]
        for field in per_field
    ]
    payload["synthetic_initialization_retention"] = {
        "retain": bool(synthetic["viable"])
        and synthetic_macro["nrmse"] < identity_macro["nrmse"]
        and synthetic_macro["ssim"] >= identity_macro["ssim"]
        and _mean(synthetic_margins) >= _mean(identity_margins),
        "identity_viable": bool(identity["viable"]),
        "synthetic_viable": bool(synthetic["viable"]),
    }
    return payload


def reconstruct_complete_candidate(
    *,
    source_volume: torch.Tensor,
    target_volume: torch.Tensor,
    preprocessing: SlicePreprocessingSpec,
    candidate: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    depth = int(source_volume.shape[3])
    if int(target_volume.shape[3]) != depth:
        raise ValueError("Complete paired volumes must have equal z depth.")
    spec = full_volume_preprocessing_spec(preprocessing, depth=depth)
    indices = tuple(range(depth))
    verify_full_slice_coverage(indices, depth)
    source_slices: list[torch.Tensor] = []
    target_slices: list[torch.Tensor] = []
    candidate_slices: list[torch.Tensor] = []
    geometries = []
    with torch.inference_mode():
        for index in indices:
            source_01, source_geometry = preprocess_volume_slice(
                source_volume,
                index,
                spec,
                apply_model_range=False,
            )
            target_01, target_geometry = preprocess_volume_slice(
                target_volume,
                index,
                spec,
                apply_model_range=False,
            )
            validate_preprocessed_geometry(source_geometry, target_geometry)
            predicted_01 = candidate(source_01.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)
            if predicted_01.shape != source_01.shape:
                raise ValueError("Complete-volume candidate changed model-grid slice shape.")
            source_slices.append(source_01)
            target_slices.append(target_01)
            candidate_slices.append(predicted_01)
            geometries.append(target_geometry)
    source_model_grid = torch.stack(source_slices, dim=-1)
    target_model_grid = torch.stack(target_slices, dim=-1)
    candidate_model_grid = torch.stack(candidate_slices, dim=-1)
    source_native = reconstruct_native_grid_volume(source_slices, geometries, depth=depth)
    target_native = reconstruct_native_grid_volume(target_slices, geometries, depth=depth)
    candidate_native = reconstruct_native_grid_volume(candidate_slices, geometries, depth=depth)
    if candidate_native.shape != source_volume.shape or target_native.shape != target_volume.shape:
        raise ValueError("Inverse geometry did not restore the original native volume shape.")
    return {
        "complete_volume": True,
        "processed_slices": depth,
        "model_grid": _complete_metrics(candidate_model_grid, target_model_grid),
        "reconstructed_native_grid": _complete_metrics(candidate_native, target_volume),
        "source_model_grid": _complete_metrics(source_model_grid, target_model_grid),
        "raw_native_source_baseline": _complete_metrics(source_volume, target_volume),
        "roundtrip_native_source_baseline": _complete_metrics(source_native, target_volume),
    }


def sanitized_loso_handoff(
    *,
    audit_commit: str,
    training_checkpoint_commit: str,
    experiment_commit: str,
    aggregate: Mapping[str, Any],
    full_volume: Mapping[str, Any],
    viability: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "evidence_scope": EVIDENCE_SCOPE,
        "evidence_role": "observed_development_not_confirmatory",
        "audit_commit": audit_commit,
        "training_checkpoint_commit": training_checkpoint_commit,
        "experiment_commit": experiment_commit,
        "selected_slice_evidence": dict(aggregate),
        "complete_volume_evidence": dict(full_volume),
        "viability": dict(viability),
        "provenance": dict(provenance),
    }
    _assert_sanitized(payload)
    return payload


def _complete_metrics(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError("Complete-volume metrics require matching (C,H,W,Z) tensors.")
    candidate_5d = candidate.permute(0, 3, 1, 2).unsqueeze(0)
    target_5d = target.permute(0, 3, 1, 2).unsqueeze(0)
    minimum_spatial = min(int(value) for value in candidate_5d.shape[2:])
    window_size = min(7, minimum_spatial)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 1:
        raise ValueError("Complete volume has an empty spatial dimension.")
    mask = (target_5d > 0.0).to(target_5d.dtype)
    if not bool(mask.any()):
        raise ValueError("Complete target foreground is empty.")
    values = {
        "masked_mae": masked_mae(candidate_5d, target_5d, mask),
        "nrmse": nrmse(candidate_5d, target_5d, data_range=1.0),
        "ssim3d": ssim3d(
            candidate_5d,
            target_5d,
            data_range=1.0,
            window_size=window_size,
        ),
        "correlation": normalized_cross_correlation(candidate_5d, target_5d, mask),
        "gradient_mae": gradient_mae(candidate_5d, target_5d, mask),
    }
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _aggregate_requested(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    labels = sorted(rows[0]["conditioning"][arm]["requested"])
    return {
        label: _mean_metrics(
            [row["conditioning"][arm]["requested"][label] for row in rows]
        )
        for label in labels
    }


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            values[str(name)].append(float(value))
    return {name: _mean(items) for name, items in sorted(values.items())}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(float(value) for value in values) / len(values)


def _field_label(field: float) -> str:
    return f"{float(field):g}T"


def _assert_sanitized(payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True).lower()
    forbidden = (
        "subject_id",
        "case_id",
        "sample_id",
        "raw_uri",
        "relative_path",
        "checkpoint_path",
        "image_path",
        ".nii",
        ".png",
        ".pt",
        "/content/drive",
        "\\",
    )
    matched = [term for term in forbidden if term in text]
    if matched:
        raise ValueError(f"Sanitized LOSO handoff contains forbidden material: {matched}.")
