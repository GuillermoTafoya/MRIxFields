"""Contracts for the prospective paired, selected-slice Track-A audit.

The public helpers in this module are deliberately data-location agnostic.  Real
NIfTI files, checkpoints, diagnostic images, and private reports remain external
to the repository; tests use synthetic tensors and metadata only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import SliceGeometry
from fieldbridge.evaluation.metrics import (
    gradient_mae,
    masked_mae,
    normalized_cross_correlation,
    nrmse,
    ssim,
)
from fieldbridge.official.data_manifest import MRIxFieldsDataRecord

EVIDENCE_SCOPE = "prospective_paired_selected_slice_development"
SELECTED_SLICE_INDICES = (72, 103, 135, 166, 197, 228, 260, 291)
CASE_IDS = ("0006", "0007", "0009")
SOURCE_FIELD = 0.1
TARGET_FIELDS = (1.5, 3.0, 5.0, 7.0)
ALL_FIELDS = (SOURCE_FIELD, *TARGET_FIELDS)
CONTRAST = "T2-FLAIR"
OFFICIAL_MODALITY = "T2FLAIR"

LOWER_IS_BETTER = {
    "masked_mae",
    "nrmse",
    "gradient_mae",
    "outside_mask_error",
    "signed_foreground_bias",
}
HIGHER_IS_BETTER = {"ssim", "correlation"}


@dataclass(frozen=True, slots=True)
class AcquisitionGeometry:
    """Physical geometry needed to reject misregistered paired acquisitions."""

    shape: tuple[int, int, int]
    affine: tuple[tuple[float, ...], ...]
    orientation: tuple[str, str, str]
    voxel_sizes: tuple[float, float, float]

    @classmethod
    def from_arrays(
        cls,
        *,
        shape: Sequence[int],
        affine: np.ndarray,
        orientation: Sequence[str],
        voxel_sizes: Sequence[float],
    ) -> "AcquisitionGeometry":
        affine_array = np.asarray(affine, dtype=np.float64)
        if affine_array.shape != (4, 4):
            raise ValueError(f"NIfTI affine must be 4x4, got {affine_array.shape}.")
        return cls(
            shape=tuple(int(value) for value in shape),  # type: ignore[arg-type]
            affine=tuple(tuple(float(value) for value in row) for row in affine_array),
            orientation=tuple(str(value) for value in orientation),  # type: ignore[arg-type]
            voxel_sizes=tuple(float(value) for value in voxel_sizes),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LoadedAcquisition:
    volume: torch.Tensor
    geometry: AcquisitionGeometry


def select_required_acquisitions(
    records: Sequence[MRIxFieldsDataRecord],
    *,
    split_name: str = "Training_prospective",
) -> dict[str, dict[float, MRIxFieldsDataRecord]]:
    """Select exactly one required acquisition per case and field, failing closed."""

    selected: dict[tuple[str, float], list[MRIxFieldsDataRecord]] = defaultdict(list)
    for record in records:
        if (
            record.split_name == split_name
            and record.cohort == "prospective"
            and record.is_paired
            and record.modality == OFFICIAL_MODALITY
            and record.subject_id in CASE_IDS
            and float(record.field_value) in ALL_FIELDS
        ):
            selected[(record.subject_id, float(record.field_value))].append(record)

    failures: list[str] = []
    result: dict[str, dict[float, MRIxFieldsDataRecord]] = {}
    for case_id in CASE_IDS:
        result[case_id] = {}
        for field in ALL_FIELDS:
            matches = selected[(case_id, field)]
            if len(matches) != 1:
                failures.append(
                    f"case {case_id} field {field:g}T requires exactly one acquisition; "
                    f"found {len(matches)}"
                )
            else:
                result[case_id][field] = matches[0]
    if failures:
        raise ValueError(
            "Prospective paired acquisition contract failed: " + "; ".join(failures) + "."
        )

    paths = [record.raw_uri for case in result.values() for record in case.values()]
    duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicates:
        raise ValueError("Prospective paired acquisition contract has duplicate raw_uri values.")
    return result


def load_nifti_acquisition(path: str | Path) -> LoadedAcquisition:
    """Load one NIfTI without intensity normalization and retain physical geometry."""

    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional real-data path
        raise ImportError(
            'Prospective paired audit requires pip install -e ".[dev,nifti]".'
        ) from exc

    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Prospective acquisition must be 3D, got shape {image.shape}.")
    data = image.get_fdata(dtype="float32")
    volume = torch.from_numpy(np.asarray(data)).unsqueeze(0)
    if not torch.isfinite(volume).all():
        raise ValueError("Prospective acquisition contains non-finite values.")
    geometry = AcquisitionGeometry.from_arrays(
        shape=image.shape,
        affine=image.affine,
        orientation=nib.aff2axcodes(image.affine),
        voxel_sizes=image.header.get_zooms()[:3],
    )
    return LoadedAcquisition(volume=volume, geometry=geometry)


def validate_paired_geometry(
    acquisitions: Mapping[float, LoadedAcquisition],
    *,
    source_field: float = SOURCE_FIELD,
) -> None:
    """Require exact array and physical geometry equality to the source acquisition."""

    if source_field not in acquisitions:
        raise ValueError(f"Missing source geometry for {source_field:g}T.")
    source = acquisitions[source_field].geometry
    for field in ALL_FIELDS:
        if field not in acquisitions:
            raise ValueError(f"Missing paired geometry for {field:g}T.")
        candidate = acquisitions[field].geometry
        mismatches: list[str] = []
        if candidate.shape != source.shape:
            mismatches.append("shape")
        if candidate.affine != source.affine:
            mismatches.append("affine")
        if candidate.orientation != source.orientation:
            mismatches.append("orientation")
        if candidate.voxel_sizes != source.voxel_sizes:
            mismatches.append("voxel_sizes")
        if mismatches:
            raise ValueError(
                f"Paired geometry mismatch for {field:g}T versus {source_field:g}T: "
                + ", ".join(mismatches)
                + "."
            )


def validate_preprocessed_geometry(
    source: SliceGeometry,
    target: SliceGeometry,
) -> None:
    """Reject any source/target disagreement introduced during fit-pad preprocessing."""

    if source != target:
        raise ValueError("Source and target SliceGeometry differ after preprocess_volume_slice.")
    if source.resize_mode != "fit_pad":
        raise ValueError("Prospective paired audit requires resize_mode='fit_pad'.")


def validate_checkpoint_contract(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    historical_training_config: Mapping[str, Any],
) -> None:
    """Validate the frozen residual checkpoint and its recorded training contract."""

    expected = {
        "trainer": str(contract["trainer"]),
        "model_class": str(contract["model_class"]),
        "pseudo_pair_pipeline_version": int(contract["pseudo_pair_pipeline_version"]),
        "epoch": int(contract["epoch"]),
        "global_step": int(contract["global_step"]),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Checkpoint {key} mismatch: expected {value!r}, got {state.get(key)!r}."
            )
    metadata = state.get("_meta")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint is missing _meta mapping.")
    if metadata.get("git_commit") != contract["git_commit"]:
        raise ValueError("Checkpoint Git commit does not match the frozen residual commit.")
    recorded = state.get("pseudo_pair_config")
    saved = metadata.get("config")
    expected_training = historical_training_config.get("training")
    if not isinstance(recorded, Mapping) or not isinstance(saved, Mapping):
        raise ValueError("Checkpoint is missing its recorded pseudo-pair training config.")
    if dict(recorded) != dict(saved):
        raise ValueError("Checkpoint training config copies disagree.")
    if not isinstance(expected_training, Mapping):
        raise ValueError("Historical training YAML is missing training configuration.")
    # The epoch trainer serializes its normalized config, so compare the fields that
    # originated in the historical YAML and fail on any changed value.
    for key, value in expected_training.items():
        if key in {"checkpoint_dir", "resume_from"}:
            continue
        if key not in recorded or recorded[key] != value:
            raise ValueError(f"Checkpoint recorded training config differs at {key!r}.")
    if "model" not in state:
        raise ValueError("Checkpoint is missing model state.")


def foreground_and_outside_masks(
    target_01: torch.Tensor,
    geometry: SliceGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic nonzero-target foreground and in-frame outside masks."""

    valid = torch.zeros_like(target_01)
    valid[
        :,
        :,
        geometry.pad_top : geometry.output_height - geometry.pad_bottom,
        geometry.pad_left : geometry.output_width - geometry.pad_right,
    ] = 1.0
    foreground = (target_01 > 0.0).to(target_01.dtype) * valid
    if not bool(foreground.any().detach().cpu().item()):
        raise ValueError("Actual paired target has an empty foreground mask.")
    outside = (valid - foreground).clamp(0.0, 1.0)
    return foreground, outside


def compute_paired_metrics(
    candidate_01: torch.Tensor,
    target_01: torch.Tensor,
    source_01: torch.Tensor,
    foreground: torch.Tensor,
    outside: torch.Tensor,
) -> dict[str, float]:
    """Compute the predeclared descriptive paired metrics in official [0, 1] units."""

    for name, tensor in {
        "candidate": candidate_01,
        "target": target_01,
        "source": source_01,
        "foreground": foreground,
        "outside": outside,
    }.items():
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape (B,C,H,W).")
    outside_count = outside.sum()
    outside_error = (
        torch.zeros((), device=candidate_01.device, dtype=candidate_01.dtype)
        if not bool((outside_count > 0).detach().cpu().item())
        else (torch.abs(candidate_01 - target_01) * outside).sum() / outside_count
    )
    foreground_count = foreground.sum()
    signed_bias = ((candidate_01 - target_01) * foreground).sum() / foreground_count
    residual = (torch.abs(candidate_01 - source_01) * foreground).sum() / foreground_count
    with torch.inference_mode():
        metrics = {
            "masked_mae": masked_mae(candidate_01, target_01, foreground),
            "nrmse": nrmse(candidate_01, target_01, data_range=1.0),
            "ssim": ssim(candidate_01, target_01, data_range=1.0),
            "correlation": normalized_cross_correlation(candidate_01, target_01, foreground),
            "gradient_mae": gradient_mae(candidate_01, target_01, foreground),
            "outside_mask_error": outside_error,
            "signed_foreground_bias": signed_bias,
            "prediction_minus_source_residual_magnitude": residual,
        }
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


def error_improvement_map(
    source_01: torch.Tensor,
    prediction_01: torch.Tensor,
    target_01: torch.Tensor,
) -> torch.Tensor:
    """Return abs(source-target) - abs(prediction-target); positive is improvement."""

    return torch.abs(source_01 - target_01) - torch.abs(prediction_01 - target_01)


def fixed_edge_map(image_01: torch.Tensor) -> torch.Tensor:
    """Return a fixed Sobel edge magnitude for alignment visualization."""

    if image_01.ndim != 4 or image_01.shape[1] != 1:
        raise ValueError("fixed_edge_map expects (B,1,H,W).")
    kernels = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
         [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=image_01.dtype,
        device=image_01.device,
    ).unsqueeze(1)
    gradients = F.conv2d(image_01, kernels, padding=1)
    return torch.sqrt(gradients.square().sum(dim=1, keepdim=True))


def conditioning_margins(
    correct: Mapping[str, float],
    wrong: Mapping[str, float],
) -> dict[str, float]:
    """Return positive-is-better correct-versus-wrong descriptive margins."""

    margins: dict[str, float] = {}
    for metric, correct_value in correct.items():
        if metric == "prediction_minus_source_residual_magnitude" or metric not in wrong:
            continue
        if metric == "signed_foreground_bias":
            margins[metric] = abs(float(wrong[metric])) - abs(float(correct_value))
        elif metric in LOWER_IS_BETTER:
            margins[metric] = float(wrong[metric]) - float(correct_value)
        elif metric in HIGHER_IS_BETTER:
            margins[metric] = float(correct_value) - float(wrong[metric])
    return margins


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-weight slices within case, cases within field, then fields for macro."""

    if not rows:
        raise ValueError("Cannot aggregate an empty prospective paired audit.")
    by_case_field: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_field[(str(row["case_slot"]), str(row["target_field"]))].append(row)
    per_case: dict[str, dict[str, Any]] = defaultdict(dict)
    for (case_slot, field), group in sorted(by_case_field.items()):
        per_case[case_slot][field] = _mean_named_metric_sets(group)

    by_field: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fields in per_case.values():
        for field, payload in fields.items():
            by_field[field].append(payload)
    per_field = {
        field: _mean_named_metric_sets(payloads)
        for field, payloads in sorted(by_field.items())
    }
    macro = _mean_named_metric_sets(list(per_field.values()))
    return {
        "weighting": {
            "slices_within_case": "equal",
            "cases_within_target_field": "equal",
            "target_fields_for_macro": "equal",
        },
        "per_case": dict(per_case),
        "per_target_field": per_field,
        "macro": macro,
    }


def _mean_named_metric_sets(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("source", "correct", "wrong_mean")
    result: dict[str, Any] = {}
    for name in names:
        metric_rows = [row[name] for row in rows if isinstance(row.get(name), Mapping)]
        if metric_rows:
            result[name] = _mean_metrics(metric_rows)
    margin_rows = [
        row["margins_mean"]
        for row in rows
        if isinstance(row.get("margins_mean"), Mapping)
    ]
    if margin_rows:
        result["margins_mean"] = _mean_metrics(margin_rows)
    return result


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            values[str(name)].append(float(value))
    return {name: sum(items) / len(items) for name, items in sorted(values.items())}


def sanitized_handoff(
    *,
    checkpoint_contract: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    counts: Mapping[str, int],
    target_conditioning_sweep: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a path-, identity-, and image-free descriptive handoff."""

    payload = {
        "evidence_scope": EVIDENCE_SCOPE,
        "complete_volume": False,
        "evidence_role": "observed_development_not_confirmatory",
        "scientific_thresholds": None,
        "training_experiment": "not_implemented",
        "checkpoint_contract": {
            "git_commit": str(checkpoint_contract["git_commit"]),
            "model_class": str(checkpoint_contract["model_class"]),
            "pipeline_version": int(checkpoint_contract["pseudo_pair_pipeline_version"]),
            "epoch": int(checkpoint_contract["epoch"]),
            "global_step": int(checkpoint_contract["global_step"]),
        },
        "counts": {str(key): int(value) for key, value in counts.items()},
        "aggregation": dict(aggregate),
        "error_improvement_definition": "abs(source-target) - abs(prediction-target)",
        "positive_error_improvement_means": "improvement",
    }
    if target_conditioning_sweep is not None:
        payload["target_conditioning_sweep"] = dict(target_conditioning_sweep)
    return payload


def assert_sanitized_handoff(payload: Mapping[str, Any]) -> None:
    """Fail before writing if a handoff contains private locations or identities."""

    import json

    text = json.dumps(payload, sort_keys=True).lower()
    forbidden = (
        "subject_id",
        "case_id",
        "sample_id",
        "record_id",
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
        raise ValueError(f"Sanitized handoff contains forbidden material: {matched}.")
