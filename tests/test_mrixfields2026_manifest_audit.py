from fieldbridge.official.submissions import (
    audit_prediction_manifest_rows,
    expected_submission_entries,
)


def test_manifest_rows_can_audit_complete_task3_without_files() -> None:
    rows = [{"relative_path": entry.relative_path} for entry in expected_submission_entries("task3")]

    report = audit_prediction_manifest_rows(rows, "task3")

    assert report.ok
    assert report.found_pred_count == 180
    assert report.found_seg_count == 0


def test_manifest_rows_validate_ids_target_field_and_task_seg_rules() -> None:
    entries = expected_submission_entries("task3")
    bad_id = "task3/T1W/0.1T_to_1.5T/pred/P_T1W_1.5T_0016.nii.gz"
    rows = [{"relative_path": bad_id}, *({"relative_path": entry.relative_path} for entry in entries[1:])]

    report = audit_prediction_manifest_rows(rows, "task3")

    assert not report.ok
    assert any("not expected for source field" in error for error in report.errors)

    seg_row = {
        "modality": "T1W",
        "source_field": "0.1T",
        "target_field": "1.5T",
        "subject_id": "0001",
        "kind": "seg",
    }
    seg_report = audit_prediction_manifest_rows([seg_row], "task3")
    assert not seg_report.ok
    assert any("prediction files only" in error for error in seg_report.errors)


def test_manifest_rows_validate_metadata_against_relative_path() -> None:
    rows = [
        {
            "relative_path": "task1/T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz",
            "modality": "T2W",
            "pair": "0.1T_to_7T",
            "source_field": "0.1T",
            "target_field": "7T",
            "subject_id": "0001",
            "kind": "pred",
        }
    ]

    report = audit_prediction_manifest_rows(rows, "task1")

    assert not report.ok
    assert any("modality does not match path" in error for error in report.errors)
