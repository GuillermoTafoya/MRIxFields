import pytest

from clbfield.official.mrixfields2026 import FULL_SHAPE, SUBMISSION_SHAPE
from clbfield.official.validation import (
    submission_shape_from_full_shape,
    validate_dtype,
    validate_intensity_range,
    validate_prediction_metadata,
    validate_shape,
)


def test_validate_shape_uses_metadata_tuples_only() -> None:
    assert validate_shape(SUBMISSION_SHAPE, SUBMISSION_SHAPE) == []
    errors = validate_shape((364, 436, 31), SUBMISSION_SHAPE)
    assert errors
    assert "Expected shape" in errors[0]


def test_validate_dtype_requires_float32() -> None:
    assert validate_dtype("float32") == []
    assert validate_dtype("FLOAT32") == []
    assert validate_dtype("float64") == ["Expected dtype 'float32', got 'float64'."]


def test_validate_intensity_range_requires_zero_to_one_bounds() -> None:
    assert validate_intensity_range(0.0, 1.0) == []
    assert validate_intensity_range(0.2, 0.8) == []
    assert validate_intensity_range(-0.1, 1.0)
    assert validate_intensity_range(0.0, 1.1)
    assert len(validate_intensity_range(0.9, 0.1)) == 1


def test_validate_prediction_metadata_switches_between_submission_and_full_shapes() -> None:
    assert validate_prediction_metadata(SUBMISSION_SHAPE, "float32", 0.0, 1.0) == []
    assert validate_prediction_metadata(FULL_SHAPE, "float32", 0.0, 1.0, submission=False) == []

    errors = validate_prediction_metadata(FULL_SHAPE, "float64", -0.2, 1.2)
    assert len(errors) == 3
    assert any("Expected shape" in error for error in errors)
    assert any("Expected dtype" in error for error in errors)
    assert any("Expected intensity range" in error for error in errors)


def test_submission_shape_from_full_shape_uses_official_z_clip() -> None:
    assert submission_shape_from_full_shape(FULL_SHAPE) == SUBMISSION_SHAPE
    assert submission_shape_from_full_shape((10, 20, 200)) == (10, 20, 30)
    with pytest.raises(ValueError):
        submission_shape_from_full_shape((364, 436))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        submission_shape_from_full_shape((364, 436, 179))
