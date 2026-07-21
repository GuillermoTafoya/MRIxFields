"""Real-paired T2-FLAIR LOSO data contracts with subject-first splitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import (
    SliceGeometry,
    SlicePreprocessingSpec,
    preprocess_volume_slice,
    reverse_preprocessed_slice,
    to_model_range,
)
from fieldbridge.data.pseudo_pairs import PseudoPairSliceSample
from fieldbridge.evaluation.prospective_paired import (
    ALL_FIELDS,
    CONTRAST,
    SOURCE_FIELD,
    TARGET_FIELDS,
    validate_preprocessed_geometry,
)


@dataclass(frozen=True, slots=True)
class LosoFold:
    fold: int
    train_case_ids: tuple[str, str]
    held_out_case_id: str

    @property
    def fold_slot(self) -> str:
        return f"fold_{self.fold:02d}"


@dataclass(frozen=True, slots=True)
class AffineCalibration:
    scale: float
    bias: float
    fitted_cases: int
    fitted_pixels: int

    def apply(self, source_01: torch.Tensor) -> torch.Tensor:
        return (source_01 * self.scale + self.bias).clamp(0.0, 1.0)


def build_loso_folds(case_ids: Sequence[str]) -> tuple[LosoFold, ...]:
    cases = tuple(str(value) for value in case_ids)
    if cases != ("0006", "0007", "0009"):
        raise ValueError("LOSO v1 requires cases 0006, 0007, and 0009 in that order.")
    folds = tuple(
        LosoFold(
            fold=index + 1,
            train_case_ids=tuple(  # type: ignore[arg-type]
                case for case in cases if case != held_out
            ),
            held_out_case_id=held_out,
        )
        for index, held_out in enumerate(cases)
    )
    validate_loso_folds(folds, cases)
    return folds


def validate_loso_folds(folds: Sequence[LosoFold], case_ids: Sequence[str]) -> None:
    expected = set(str(value) for value in case_ids)
    if len(folds) != len(expected):
        raise ValueError("LOSO requires one fold per case.")
    held_out: list[str] = []
    for fold in folds:
        train = set(fold.train_case_ids)
        test = {fold.held_out_case_id}
        if train & test:
            raise ValueError(f"Subject leakage in {fold.fold_slot}.")
        if train | test != expected:
            raise ValueError(f"{fold.fold_slot} does not cover the declared cases exactly.")
        held_out.append(fold.held_out_case_id)
    if set(held_out) != expected or len(held_out) != len(set(held_out)):
        raise ValueError("Each case must be held out exactly once.")


class RealPairedSliceDataset(Dataset[PseudoPairSliceSample]):
    """All declared real source/target slices for training cases only."""

    def __init__(
        self,
        volumes: Mapping[str, Mapping[float, torch.Tensor]],
        *,
        case_ids: Sequence[str],
        preprocessing: SlicePreprocessingSpec,
        slice_indices: Sequence[int],
    ) -> None:
        self.volumes = volumes
        self.case_ids = tuple(str(case) for case in case_ids)
        self.preprocessing = preprocessing
        self.slice_indices = tuple(int(index) for index in slice_indices)
        if not self.case_ids or not self.slice_indices:
            raise ValueError("Paired dataset requires cases and slice indices.")
        self.samples = tuple(
            (case_id, field, slice_index)
            for case_id in self.case_ids
            for field in TARGET_FIELDS
            for slice_index in self.slice_indices
        )
        _validate_volume_bank(volumes, self.case_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PseudoPairSliceSample:
        case_id, target_field, slice_index = self.samples[int(index)]
        source_01, source_geometry = preprocess_volume_slice(
            self.volumes[case_id][SOURCE_FIELD],
            slice_index,
            self.preprocessing,
            apply_model_range=False,
        )
        target_01, target_geometry = preprocess_volume_slice(
            self.volumes[case_id][target_field],
            slice_index,
            self.preprocessing,
            apply_model_range=False,
        )
        validate_preprocessed_geometry(source_geometry, target_geometry)
        mask = valid_fit_pad_mask(target_01, target_geometry)
        return PseudoPairSliceSample(
            x_low=to_model_range(source_01, self.preprocessing.model_range),
            x_high=to_model_range(target_01, self.preprocessing.model_range),
            mask=mask,
            source_domain=Domain(SOURCE_FIELD, CONTRAST),
            target_domain=Domain(target_field, CONTRAST),
            record_id=f"paired:{case_id}:{target_field:g}T",
            subject_id=case_id,
            volume_path="external",
            slice_index=slice_index,
            degradation_seed=0,
            degradation_strength=0.0,
            geometry=target_geometry,
        )


def valid_fit_pad_mask(image: torch.Tensor, geometry: SliceGeometry) -> torch.Tensor:
    mask = torch.zeros_like(image, dtype=torch.float32)
    mask[
        :,
        geometry.pad_top : geometry.output_height - geometry.pad_bottom,
        geometry.pad_left : geometry.output_width - geometry.pad_right,
    ] = 1.0
    return mask


def fit_train_only_affine_calibrations(
    volumes: Mapping[str, Mapping[float, torch.Tensor]],
    *,
    train_case_ids: Sequence[str],
    preprocessing: SlicePreprocessingSpec,
    slice_indices: Sequence[int],
) -> dict[float, AffineCalibration]:
    """Fit target-specific least squares using only declared training cases."""

    train_cases = tuple(str(case) for case in train_case_ids)
    _validate_volume_bank(volumes, train_cases)
    calibrations: dict[float, AffineCalibration] = {}
    for field in TARGET_FIELDS:
        count = 0
        sum_x = sum_y = sum_xx = sum_xy = 0.0
        for case_id in train_cases:
            for slice_index in slice_indices:
                source, source_geometry = preprocess_volume_slice(
                    volumes[case_id][SOURCE_FIELD],
                    int(slice_index),
                    preprocessing,
                    apply_model_range=False,
                )
                target, target_geometry = preprocess_volume_slice(
                    volumes[case_id][field],
                    int(slice_index),
                    preprocessing,
                    apply_model_range=False,
                )
                validate_preprocessed_geometry(source_geometry, target_geometry)
                valid = valid_fit_pad_mask(target, target_geometry)
                mask = (target > 0.0).to(target.dtype) * valid
                selected_x = source[mask.bool()].to(torch.float64)
                selected_y = target[mask.bool()].to(torch.float64)
                count += int(selected_x.numel())
                sum_x += float(selected_x.sum())
                sum_y += float(selected_y.sum())
                sum_xx += float((selected_x * selected_x).sum())
                sum_xy += float((selected_x * selected_y).sum())
        if count <= 1:
            raise ValueError(f"Affine fit for {field:g}T has insufficient training pixels.")
        denominator = count * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-12:
            raise ValueError(f"Affine fit for {field:g}T is singular.")
        scale = (count * sum_xy - sum_x * sum_y) / denominator
        bias = (sum_y - scale * sum_x) / count
        calibrations[field] = AffineCalibration(
            scale=scale,
            bias=bias,
            fitted_cases=len(train_cases),
            fitted_pixels=count,
        )
    return calibrations


def verify_full_slice_coverage(indices: Sequence[int], depth: int) -> None:
    expected = tuple(range(int(depth)))
    actual = tuple(int(index) for index in indices)
    if actual != expected:
        raise ValueError(
            f"Complete-volume coverage requires every z slice exactly once; "
            f"received {len(actual)} of {depth}."
        )


def reconstruct_native_grid_volume(
    model_grid_slices: Sequence[torch.Tensor],
    geometries: Sequence[SliceGeometry],
    *,
    depth: int,
) -> torch.Tensor:
    if len(model_grid_slices) != len(geometries):
        raise ValueError("Model-grid slices and inverse geometries must have equal length.")
    verify_full_slice_coverage([geometry.slice_index for geometry in geometries], depth)
    restored: list[torch.Tensor] = []
    expected_shape: tuple[int, int, int] | None = None
    for image, geometry in zip(model_grid_slices, geometries, strict=True):
        native = reverse_preprocessed_slice(image, geometry)
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in native.shape)
        if tuple(native.shape) != expected_shape:
            raise ValueError("Inverse fit-pad geometry produced inconsistent native slices.")
        restored.append(native)
    volume = torch.stack(restored, dim=-1)
    if int(volume.shape[-1]) != int(depth):
        raise ValueError("Inverse geometry did not reconstruct the complete z depth.")
    return volume


def full_volume_preprocessing_spec(
    template: SlicePreprocessingSpec,
    *,
    depth: int,
) -> SlicePreprocessingSpec:
    return SlicePreprocessingSpec(
        slice_start=0,
        slice_end=int(depth),
        slices_per_volume=None,
        normalization=template.normalization,
        model_range=template.model_range,
        resize_mode=template.resize_mode,
        output_height=template.output_height,
        output_width=template.output_width,
        slice_axis=template.slice_axis,
    )


def _validate_volume_bank(
    volumes: Mapping[str, Mapping[float, torch.Tensor]],
    case_ids: Sequence[str],
) -> None:
    for case_id in case_ids:
        if case_id not in volumes:
            raise ValueError(f"Missing paired case {case_id}.")
        missing = [field for field in ALL_FIELDS if field not in volumes[case_id]]
        if missing:
            raise ValueError(f"Paired case {case_id} is missing fields {missing}.")
        for field in ALL_FIELDS:
            volume = volumes[case_id][field]
            if volume.ndim != 4 or not torch.isfinite(volume).all():
                raise ValueError(f"Invalid volume tensor for case {case_id}, field {field:g}T.")
