"""Pure metadata validation helpers for MRIxFields2026."""

from __future__ import annotations

from clbfield.official.mrixfields2026 import (
    FULL_SHAPE,
    INTENSITY_RANGE,
    SUBMISSION_SHAPE,
    SUBMISSION_Z_CLIP,
)


def validate_shape(shape: tuple[int, ...], expected: tuple[int, ...]) -> list[str]:
    observed = tuple(int(dim) for dim in shape)
    wanted = tuple(int(dim) for dim in expected)
    if observed == wanted:
        return []
    return [f"Expected shape {wanted}, got {observed}."]


def validate_dtype(dtype: str, expected: str = "float32") -> list[str]:
    observed = str(dtype).strip().lower()
    wanted = str(expected).strip().lower()
    if observed == wanted:
        return []
    return [f"Expected dtype {wanted!r}, got {dtype!r}."]


def validate_intensity_range(min_value: float, max_value: float) -> list[str]:
    lower, upper = INTENSITY_RANGE
    observed_min = float(min_value)
    observed_max = float(max_value)
    errors: list[str] = []
    if observed_min > observed_max:
        errors.append(f"Intensity min {observed_min} is greater than max {observed_max}.")
    if observed_min < lower or observed_max > upper:
        errors.append(
            f"Expected intensity range inside [{lower}, {upper}], "
            f"got [{observed_min}, {observed_max}]."
        )
    return errors


def validate_prediction_metadata(
    shape: tuple[int, ...],
    dtype: str,
    min_value: float,
    max_value: float,
    submission: bool = True,
) -> list[str]:
    expected_shape = SUBMISSION_SHAPE if submission else FULL_SHAPE
    return [
        *validate_shape(shape, expected_shape),
        *validate_dtype(dtype),
        *validate_intensity_range(min_value, max_value),
    ]


def submission_shape_from_full_shape(full_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(full_shape) != 3:
        raise ValueError(f"Full shape must be 3D, got {full_shape}.")
    z_start, z_stop = SUBMISSION_Z_CLIP
    if full_shape[2] < z_stop:
        raise ValueError(
            f"Full shape z dimension must be at least {z_stop} for submission clipping, "
            f"got {full_shape[2]}."
        )
    return (int(full_shape[0]), int(full_shape[1]), z_stop - z_start)
