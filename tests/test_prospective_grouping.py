from fieldbridge.data.prospective import (
    find_multifield_groups,
    group_prospective_paths,
    leave_one_subject_out_folds,
    parse_prospective_path,
)


def _five_field_case(case_id: str) -> list[str]:
    return [
        f"P_T2FLAIR_0.1T_{case_id}.nii.gz",
        f"P_T2FLAIR_1.5T_{case_id}.nii.gz",
        f"P_T2FLAIR_3T_{case_id}.nii.gz",
        f"P_T2FLAIR_5T_{case_id}.nii.gz",
        f"P_T2FLAIR_7T_{case_id}.nii.gz",
    ]


def test_parse_prospective_path() -> None:
    record = parse_prospective_path("P_T2FLAIR_7T_0006.nii.gz")

    assert record.sequence == "T2FLAIR"
    assert record.case_id == "0006"
    assert record.field_strength_t == 7.0


def test_group_prospective_paths_groups_by_case_id_and_sequence() -> None:
    paths = [*_five_field_case("0006"), "P_T1W_3T_0006.nii.gz", "P_T2FLAIR_7T_0007.nii.gz"]

    groups = group_prospective_paths(paths)

    assert ("0006", "T2FLAIR") in groups
    assert ("0006", "T1W") in groups
    assert len(groups[("0006", "T2FLAIR")]) == 5


def test_find_multifield_groups_finds_five_field_groups() -> None:
    paths = [*_five_field_case("0006"), "P_T2FLAIR_7T_0007.nii.gz"]

    groups = find_multifield_groups(paths, min_fields=5)

    assert list(groups) == [("0006", "T2FLAIR")]
    assert {record.field_strength_t for record in groups[("0006", "T2FLAIR")]} == {
        0.1,
        1.5,
        3.0,
        5.0,
        7.0,
    }


def test_leave_one_subject_out_folds_for_case_ids() -> None:
    folds = leave_one_subject_out_folds(["0006", "0007", "0009"])

    assert [fold.held_out_case_id for fold in folds] == ["0006", "0007", "0009"]
    assert folds[0].test_case_ids == ("0006",)
    assert folds[0].train_case_ids == ("0007", "0009")
    assert folds[1].train_case_ids == ("0006", "0009")
    assert folds[2].train_case_ids == ("0006", "0007")
