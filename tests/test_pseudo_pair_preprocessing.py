import torch

from fieldbridge.data.preprocessing import (
    SlicePreprocessingSpec,
    from_model_range,
    preprocess_volume_slices,
    reverse_preprocessed_slice,
    selected_slice_indices,
    to_model_range,
)


def test_official_01_mode_preserves_intensities_without_zscore() -> None:
    volume = torch.linspace(0.0, 1.0, 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    spec = SlicePreprocessingSpec(
        slice_start=0,
        slice_end=4,
        slices_per_volume=None,
        model_range="zero_one",
        resize_mode="native",
        slice_axis="x",
    )

    batch = preprocess_volume_slices(volume, spec)

    assert torch.equal(batch.image[0], volume[:, 0])
    assert torch.equal(batch.image[-1], volume[:, 3])
    assert not torch.isclose(batch.image.mean(), torch.tensor(0.0))


def test_zero_one_minus_one_one_conversion_is_reversible() -> None:
    x = torch.rand(1, 8, 9)

    converted = to_model_range(x, "minus_one_one")
    restored = from_model_range(converted, "minus_one_one")

    assert torch.allclose(restored, x)


def test_uniform_slice_indices_are_in_range() -> None:
    spec = SlicePreprocessingSpec(slice_start=2, slice_end=12, slices_per_volume=4)

    indices = selected_slice_indices(spec, depth=20)

    assert len(indices) == 4
    assert indices == tuple(sorted(indices))
    assert all(2 <= index < 12 for index in indices)
    assert indices[0] == 2
    assert indices[-1] == 11


def test_fit_pad_preserves_aspect_ratio_and_output_shape() -> None:
    volume = torch.ones(1, 1, 10, 20)
    spec = SlicePreprocessingSpec(
        slice_start=0,
        slice_end=1,
        slices_per_volume=1,
        model_range="zero_one",
        resize_mode="fit_pad",
        output_height=10,
        output_width=10,
        slice_axis="x",
    )

    batch = preprocess_volume_slices(volume, spec)
    geometry = batch.geometry[0]

    assert batch.image.shape == (1, 1, 10, 10)
    assert geometry.resized_height == 5
    assert geometry.resized_width == 10
    assert geometry.pad_top + geometry.pad_bottom == 5
    assert geometry.pad_left == 0
    assert geometry.pad_right == 0


def test_reverse_preprocessed_slice_restores_original_shape() -> None:
    volume = torch.rand(1, 1, 10, 20)
    spec = SlicePreprocessingSpec(
        slice_start=0,
        slice_end=1,
        slices_per_volume=1,
        model_range="zero_one",
        resize_mode="fit_pad",
        output_height=10,
        output_width=10,
        slice_axis="x",
    )

    batch = preprocess_volume_slices(volume, spec)
    restored = reverse_preprocessed_slice(batch.image[0], batch.geometry[0])

    assert restored.shape == volume[:, 0].shape


def test_configured_axial_z_slice_uses_raw_nifti_z_axis() -> None:
    x = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
    y = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
    z = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6)
    volume = (x * 0.1 + y * 0.01 + z * 0.001).clamp(0.0, 1.0)
    spec = SlicePreprocessingSpec(
        slice_start=3,
        slice_end=4,
        slices_per_volume=1,
        model_range="zero_one",
        resize_mode="native",
        slice_axis="z",
    )

    batch = preprocess_volume_slices(volume, spec)

    assert batch.image.shape == (1, 1, 4, 5)
    assert torch.equal(batch.image[0], volume[:, :, :, 3])
    assert batch.geometry[0].slice_axis == "z"
    assert batch.geometry[0].original_height == 4
    assert batch.geometry[0].original_width == 5


def test_official_axial_geometry_has_small_horizontal_padding_for_micro_shape() -> None:
    volume = torch.zeros(1, 364, 436, 1)
    spec = SlicePreprocessingSpec(
        slice_start=0,
        slice_end=1,
        slices_per_volume=1,
        model_range="zero_one",
        resize_mode="fit_pad",
        output_height=128,
        output_width=160,
        slice_axis="z",
    )

    batch = preprocess_volume_slices(volume, spec)
    geometry = batch.geometry[0]

    assert batch.image.shape == (1, 1, 128, 160)
    assert geometry.original_height == 364
    assert geometry.original_width == 436
    assert geometry.resized_height == 128
    assert geometry.resized_width == 153
    assert geometry.pad_left == 3
    assert geometry.pad_right == 4
