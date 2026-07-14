import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import SlicePreprocessingSpec, to_model_range
from fieldbridge.data.pseudo_pairs import PseudoPairSliceDataset, make_field_balanced_sampler


def _record(field: float = 7.0) -> VolumeRecord:
    return VolumeRecord(
        case_id=f"case-{field:g}",
        image_path=f"case-{field:g}.nii.gz",
        domain=Domain(field, "T2-FLAIR"),
        subject_id=f"subject-{field:g}",
    )


def _volume() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 1 * 4 * 6 * 8, dtype=torch.float32).reshape(1, 4, 6, 8)


def _loader(path, record) -> torch.Tensor:  # type: ignore[no-untyped-def]
    del path, record
    return _volume()


def _spec(model_range: str = "zero_one") -> SlicePreprocessingSpec:
    return SlicePreprocessingSpec(
        slice_start=0,
        slice_end=4,
        slices_per_volume=2,
        model_range=model_range,  # type: ignore[arg-type]
        resize_mode="native",
        slice_axis="x",
    )


def test_pseudo_pair_slice_dataset_length_and_metadata() -> None:
    dataset = PseudoPairSliceDataset(
        [_record(7.0), _record(3.0)],
        image_loader=_loader,
        source_field=0.1,
        sequence="T2-FLAIR",
        preprocessing=_spec(),
        mode="validation",
    )

    sample = dataset[0]

    assert len(dataset) == 4
    assert sample.record_id == "case-7"
    assert sample.slice_index in (0, 3)
    assert sample.x_low.shape == sample.x_high.shape == sample.mask.shape
    assert sample.source_domain == Domain(0.1, "T2-FLAIR")
    assert sample.target_domain == Domain(7.0, "T2-FLAIR")


def test_train_degradation_changes_between_accesses() -> None:
    dataset = PseudoPairSliceDataset(
        [_record(7.0)],
        image_loader=_loader,
        source_field=0.1,
        sequence="T2-FLAIR",
        preprocessing=_spec(),
        mode="train",
        seed=4,
    )

    first = dataset[0]
    second = dataset[0]

    assert first.degradation_seed != second.degradation_seed
    assert not torch.allclose(first.x_low, second.x_low)


def test_validation_degradation_is_repeatable() -> None:
    dataset = PseudoPairSliceDataset(
        [_record(7.0)],
        image_loader=_loader,
        source_field=0.1,
        sequence="T2-FLAIR",
        preprocessing=_spec(),
        mode="validation",
        seed=4,
    )

    first = dataset[0]
    second = dataset[0]

    assert first.degradation_seed == second.degradation_seed
    assert torch.allclose(first.x_low, second.x_low)


def test_target_equals_original_high_field_tensor_after_model_boundary_mapping() -> None:
    spec = _spec(model_range="minus_one_one")
    dataset = PseudoPairSliceDataset(
        [_record(7.0)],
        image_loader=_loader,
        source_field=0.1,
        sequence="T2-FLAIR",
        preprocessing=spec,
        mode="validation",
        seed=5,
    )

    sample = dataset[0]
    expected = to_model_range(_volume()[:, sample.slice_index], "minus_one_one")

    assert torch.allclose(sample.x_high, expected)


def test_default_mask_keeps_dark_anatomy_inside_geometry() -> None:
    def dark_loader(path, record):  # type: ignore[no-untyped-def]
        del path, record
        volume = torch.ones(1, 1, 4, 8)
        volume[:, :, 1:3, 2:6] = 0.0
        return volume

    dataset = PseudoPairSliceDataset(
        [_record(7.0)],
        image_loader=dark_loader,
        source_field=0.1,
        sequence="T2-FLAIR",
        preprocessing=SlicePreprocessingSpec(
            slice_start=0,
            slice_end=1,
            slices_per_volume=1,
            model_range="zero_one",
            resize_mode="fit_pad",
            output_height=8,
            output_width=8,
            slice_axis="x",
        ),
        mode="validation",
    )

    sample = dataset[0]

    assert sample.mask[:, 3:5, 2:6].sum().item() == 8.0
    assert sample.mask[:, 0, :].sum().item() == 0.0


def test_field_balanced_sampler_equalizes_total_field_weight() -> None:
    records = [_record(1.5)] + [_record(3.0) for _ in range(3)]
    # Give the duplicate field unique ids so dataset indexing is unambiguous.
    records = [
        VolumeRecord(
            case_id=f"{record.case_id}-{index}",
            image_path=f"{record.case_id}-{index}.nii.gz",
            domain=record.domain,
            subject_id=f"subject-{index}",
        )
        for index, record in enumerate(records)
    ]
    dataset = PseudoPairSliceDataset(
        records,
        image_loader=_loader,
        source_field=0.1,
        sequence="T2-FLAIR",
        preprocessing=SlicePreprocessingSpec(
            slice_start=0,
            slice_end=1,
            slices_per_volume=1,
            model_range="zero_one",
            resize_mode="native",
        ),
        mode="validation",
    )

    sampler = make_field_balanced_sampler(dataset, seed=1)
    field_weight = {1.5: 0.0, 3.0: 0.0}
    for index, weight in enumerate(sampler.weights.tolist()):
        field_weight[dataset.field_for_index(index)] += weight

    assert field_weight[1.5] == field_weight[3.0]
