import pytest

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.official.mrixfields2026 import (
    FIELDS,
    FIELD_VALUES,
    FULL_SHAPE,
    INTENSITY_RANGE,
    OFFICIAL_MODALITIES,
    SUBMISSION_SHAPE,
    SUBMISSION_Z_CLIP,
    TASK1_PAIRS,
    TASK2_PAIRS,
    TASK3_PAIRS,
    TRAIN_SLICE_RANGE,
    VALIDATION_RELEASED_IDS,
    MRIxFieldsFilename,
    build_prediction_filename,
    expected_prediction_file_count,
    expected_segmentation_file_count,
    expected_subject_ids_for_pair,
    expected_subtask_count,
    get_task_pairs,
    internal_modality_from_official,
    is_allowed_pair,
    normalize_modality,
    pair_name,
    parse_mrixfields_filename,
    parse_pair_name,
    requires_segmentation,
)


def test_official_constants_match_mrixfields2026() -> None:
    assert FIELDS == ("0.1T", "1.5T", "3T", "5T", "7T")
    assert FIELD_VALUES == {"0.1T": 0.1, "1.5T": 1.5, "3T": 3.0, "5T": 5.0, "7T": 7.0}
    assert OFFICIAL_MODALITIES == ("T1W", "T2W", "T2FLAIR")
    assert FULL_SHAPE == (364, 436, 364)
    assert SUBMISSION_Z_CLIP == (150, 180)
    assert SUBMISSION_SHAPE == (364, 436, 30)
    assert INTENSITY_RANGE == (0.0, 1.0)
    assert TRAIN_SLICE_RANGE == (72, 292)


@pytest.mark.parametrize(
    ("alias", "official"),
    [
        ("T1W", "T1W"),
        ("T1w", "T1W"),
        ("t1w", "T1W"),
        ("T2W", "T2W"),
        ("T2w", "T2W"),
        ("t2w", "T2W"),
        ("T2FLAIR", "T2FLAIR"),
        ("T2-FLAIR", "T2FLAIR"),
        ("T2Flair", "T2FLAIR"),
        ("t2flair", "T2FLAIR"),
        ("t2-flair", "T2FLAIR"),
    ],
)
def test_modality_aliases_normalize_to_official_labels(alias: str, official: str) -> None:
    assert normalize_modality(alias) == official


def test_official_modalities_map_to_existing_internal_domain_labels() -> None:
    assert internal_modality_from_official("T1W") == "T1w"
    assert internal_modality_from_official("T2W") == "T2w"
    assert internal_modality_from_official("T2FLAIR") == "T2-FLAIR"
    assert Domain(3.0, "T1w").contrast == Contrast.T1W
    assert Domain(3.0, "T2w").contrast == Contrast.T2W
    assert Domain(3.0, "T2-FLAIR").contrast == Contrast.T2_FLAIR
    assert Domain(3.0, "T2FLAIR").contrast == Contrast.T2_FLAIR
    assert Domain(3.0, "t2flair").contrast == Contrast.T2_FLAIR


def test_task_pairs_and_expected_counts() -> None:
    assert get_task_pairs("task1") == TASK1_PAIRS
    assert get_task_pairs("task2") == TASK2_PAIRS
    assert get_task_pairs("task3") == TASK3_PAIRS
    assert len(TASK1_PAIRS) == 4
    assert len(TASK2_PAIRS) == 4
    assert len(TASK3_PAIRS) == 20
    assert all(source != target for source, target in TASK3_PAIRS)
    assert ("7T", "0.1T") in TASK3_PAIRS
    assert requires_segmentation("task1")
    assert requires_segmentation("task2")
    assert not requires_segmentation("task3")
    assert expected_subtask_count("task1") == 12
    assert expected_subtask_count("task2") == 12
    assert expected_subtask_count("task3") == 60
    assert expected_prediction_file_count("task1") == 36
    assert expected_segmentation_file_count("task1") == 36
    assert expected_prediction_file_count("task2") == 36
    assert expected_segmentation_file_count("task2") == 36
    assert expected_prediction_file_count("task3") == 180
    assert expected_segmentation_file_count("task3") == 0


def test_validation_ids_are_keyed_by_source_field() -> None:
    assert VALIDATION_RELEASED_IDS == {
        "0.1T": ("0001", "0002", "0003"),
        "1.5T": ("0004", "0005", "0008"),
        "3T": ("0010", "0011", "0012"),
        "5T": ("0013", "0014", "0015"),
        "7T": ("0016", "0017", "0018"),
    }
    assert expected_subject_ids_for_pair("task1", "0.1T", "7T") == ("0001", "0002", "0003")
    assert expected_subject_ids_for_pair("task3", "7T", "0.1T") == ("0016", "0017", "0018")


def test_pair_name_and_parse_pair_name_round_trip() -> None:
    for source, target in TASK3_PAIRS:
        name = pair_name(source, target)
        assert parse_pair_name(name) == (source, target)
    assert pair_name("3.0T", "7.0T") == "3T_to_7T"
    assert is_allowed_pair("task1", "3T", "7T")
    assert not is_allowed_pair("task1", "7T", "3T")
    with pytest.raises(ValueError):
        parse_pair_name("3T-7T")


def test_filename_parser_handles_prediction_reference_and_segmentation_names() -> None:
    parsed = parse_mrixfields_filename("R_T2FLAIR_0.1T_0001.nii.gz")
    assert parsed == MRIxFieldsFilename(
        prefix="R",
        modality="T2FLAIR",
        field="0.1T",
        subject_id="0001",
        is_segmentation=False,
    )

    parsed_seg = parse_mrixfields_filename("P_T1W_7T_0001_seg.nii.gz")
    assert parsed_seg.prefix == "P"
    assert parsed_seg.modality == "T1W"
    assert parsed_seg.field == "7T"
    assert parsed_seg.subject_id == "0001"
    assert parsed_seg.is_segmentation


def test_filename_builder_uses_target_field_and_preserves_subject_id() -> None:
    assert build_prediction_filename("T1W", "7T", "0001") == "P_T1W_7T_0001.nii.gz"
    assert (
        build_prediction_filename("T1w", "7T", "0001", segmentation=True)
        == "P_T1W_7T_0001_seg.nii.gz"
    )
    assert build_prediction_filename("T2-FLAIR", "3.0T", "0016") == "P_T2FLAIR_3T_0016.nii.gz"


@pytest.mark.parametrize(
    "filename",
    [
        "X_T1W_7T_0001.nii.gz",
        "P_PD_7T_0001.nii.gz",
        "P_T1W_9T_0001.nii.gz",
        "P_T1W_7T_001.nii.gz",
        "P_T1W_7T_0001.nii",
    ],
)
def test_malformed_filenames_raise_value_error(filename: str) -> None:
    with pytest.raises(ValueError):
        parse_mrixfields_filename(filename)


@pytest.mark.parametrize(
    ("modality", "field", "subject_id"),
    [
        ("PD", "7T", "0001"),
        ("T1W", "9T", "0001"),
        ("T1W", "7T", "abc1"),
    ],
)
def test_build_prediction_filename_rejects_invalid_parts(
    modality: str,
    field: str,
    subject_id: str,
) -> None:
    with pytest.raises(ValueError):
        build_prediction_filename(modality, field, subject_id)
