import pytest

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.volume_splits import (
    VolumeSplitError,
    VolumeSplits,
    audit_volume_splits,
    build_volume_splits,
    load_volume_splits,
    save_volume_splits,
    summarize_volume_splits,
    validate_pseudo_pair_manifest_records,
)


def _records(fields: tuple[float, ...] = (1.5, 3.0), count: int = 5) -> list[VolumeRecord]:
    records = []
    for field in fields:
        for index in range(count):
            records.append(
                VolumeRecord(
                    case_id=f"{field:g}T-case-{index}",
                    image_path=f"{field:g}T-case-{index}.nii.gz",
                    domain=Domain(field, "T2-FLAIR"),
                    subject_id=f"{field:g}T-subject-{index}",
                )
            )
    return records


def test_build_volume_splits_is_deterministic_and_balanced() -> None:
    kwargs = {
        "sequence": "T2FLAIR",
        "target_fields": (1.5, 3.0),
        "train_volumes_per_field": 2,
        "val_volumes_per_field": 1,
        "test_volumes_per_field": 1,
        "seed": 7,
    }

    first = build_volume_splits(_records(), **kwargs)
    second = build_volume_splits(_records(), **kwargs)
    summary = summarize_volume_splits(first, slices_per_volume=3)

    assert [record.case_id for record in first.train] == [record.case_id for record in second.train]
    assert summary["splits"]["train"]["by_field"]["1.5T"]["volumes"] == 2
    assert summary["splits"]["train"]["by_field"]["3T"]["volumes"] == 2
    assert summary["splits"]["train"]["slices"] == 12


def test_volume_splits_json_round_trip(tmp_path) -> None:
    splits = build_volume_splits(
        _records(),
        sequence="T2-FLAIR",
        target_fields=(1.5, 3.0),
        train_volumes_per_field=2,
        val_volumes_per_field=1,
        test_volumes_per_field=1,
        seed=3,
    )

    path = save_volume_splits(splits, tmp_path / "splits.json")
    loaded = load_volume_splits(path)

    assert loaded.to_dict() == splits.to_dict()


def test_insufficient_records_raise_useful_error() -> None:
    with pytest.raises(VolumeSplitError, match="Insufficient"):
        build_volume_splits(
            _records(count=2),
            sequence="T2-FLAIR",
            target_fields=(1.5,),
            train_volumes_per_field=2,
            val_volumes_per_field=1,
            test_volumes_per_field=1,
            seed=1,
        )


def test_pseudo_pair_manifest_validation_requires_subject_ids() -> None:
    records = [
        VolumeRecord(
            case_id="case-1",
            image_path="case-1.nii.gz",
            domain=Domain(1.5, "T2-FLAIR"),
            subject_id=None,
        )
    ]

    report = validate_pseudo_pair_manifest_records(
        records,
        sequence="T2-FLAIR",
        target_fields=(1.5,),
    )

    assert not report.ok
    assert any("subject_id" in error for error in report.errors)


def test_volume_leakage_is_rejected() -> None:
    shared_subject_train = VolumeRecord(
        case_id="train-case",
        image_path="train.nii.gz",
        domain=Domain(1.5, "T2-FLAIR"),
        subject_id="subject-a",
    )
    shared_subject_val = VolumeRecord(
        case_id="val-case",
        image_path="val.nii.gz",
        domain=Domain(3.0, "T2-FLAIR"),
        subject_id="subject-a",
    )
    splits = VolumeSplits(
        train=(shared_subject_train,),
        validation=(shared_subject_val,),
        test=(),
        sequence="T2-FLAIR",
        target_fields=(1.5, 3.0),
        seed=0,
    )

    audit = audit_volume_splits(splits)

    assert not audit.ok
    with pytest.raises(VolumeSplitError, match="leakage"):
        audit.raise_for_leakage()
