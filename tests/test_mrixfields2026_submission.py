from __future__ import annotations

from zipfile import ZipFile

from fieldbridge.official.submissions import (
    build_submission_zip,
    expected_submission_entries,
    validate_submission_dir,
    validate_submission_zip,
)


def test_expected_submission_entries_counts_and_target_field_filenames() -> None:
    task1 = expected_submission_entries("task1")
    task2 = expected_submission_entries("task2")
    task3 = expected_submission_entries("task3")

    assert sum(entry.kind == "pred" for entry in task1) == 36
    assert sum(entry.kind == "seg" for entry in task1) == 36
    assert sum(entry.kind == "pred" for entry in task2) == 36
    assert sum(entry.kind == "seg" for entry in task2) == 36
    assert sum(entry.kind == "pred" for entry in task3) == 180
    assert sum(entry.kind == "seg" for entry in task3) == 0
    assert any(entry.pair == "7T_to_0.1T" for entry in task3)
    assert (
        "task1/T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz"
        in {entry.relative_path for entry in task1}
    )


def test_valid_synthetic_submission_trees_pass(tmp_path) -> None:
    task1_root = tmp_path / "task1-root"
    task2_root = tmp_path / "task2-root"
    task3_root = tmp_path / "task3-root"
    _touch_expected_tree(task1_root, "task1")
    _touch_expected_tree(task2_root, "task2")
    _touch_expected_tree(task3_root, "task3")

    assert validate_submission_dir(task1_root, "task1").ok
    assert validate_submission_dir(task2_root, "task2").ok
    assert validate_submission_dir(task3_root, "task3").ok


def test_root_may_be_task_directory_itself(tmp_path) -> None:
    _touch_expected_tree(tmp_path, "task3")
    report = validate_submission_dir(tmp_path / "task3", "task3")
    assert report.ok
    assert report.found_pred_count == 180


def test_missing_prediction_and_segmentation_files_are_reported(tmp_path) -> None:
    entries = _touch_expected_tree(tmp_path, "task3")
    missing_pred = next(entry for entry in entries if entry.kind == "pred")
    (tmp_path / missing_pred.relative_path).unlink()

    report = validate_submission_dir(tmp_path, "task3")
    assert not report.ok
    assert missing_pred.relative_path in report.missing

    task1_entries = _touch_expected_tree(tmp_path / "strict", "task1")
    missing_seg = next(entry for entry in task1_entries if entry.kind == "seg")
    (tmp_path / "strict" / missing_seg.relative_path).unlink()

    strict_report = validate_submission_dir(tmp_path / "strict", "task1")
    assert not strict_report.ok
    assert missing_seg.relative_path in strict_report.missing


def test_segmentation_rules_for_task3_and_optional_task1(tmp_path) -> None:
    task3_entries = _touch_expected_tree(tmp_path / "task3-seg", "task3")
    first = task3_entries[0]
    bad_seg = (
        tmp_path
        / "task3-seg"
        / "task3"
        / first.modality
        / first.pair
        / "seg"
        / f"P_{first.modality}_{first.target_field}_{first.subject_id}_seg.nii.gz"
    )
    bad_seg.parent.mkdir(parents=True, exist_ok=True)
    bad_seg.touch()

    task3_report = validate_submission_dir(tmp_path / "task3-seg", "task3")
    assert not task3_report.ok
    assert any("prediction files only" in error for error in task3_report.errors)

    _touch_expected_tree(tmp_path / "task1-pred-only", "task1", include_segmentation=False)
    strict_report = validate_submission_dir(tmp_path / "task1-pred-only", "task1")
    assert not strict_report.ok

    relaxed_report = validate_submission_dir(
        tmp_path / "task1-pred-only",
        "task1",
        strict_segmentation=False,
    )
    assert relaxed_report.ok
    assert relaxed_report.warnings

    partial_entries = _touch_expected_tree(
        tmp_path / "task1-partial-seg",
        "task1",
        include_segmentation=False,
    )
    first_pred = partial_entries[0]
    one_seg = (
        tmp_path
        / "task1-partial-seg"
        / "task1"
        / first_pred.modality
        / first_pred.pair
        / "seg"
        / f"P_{first_pred.modality}_{first_pred.target_field}_{first_pred.subject_id}_seg.nii.gz"
    )
    one_seg.parent.mkdir(parents=True, exist_ok=True)
    one_seg.touch()

    partial_report = validate_submission_dir(
        tmp_path / "task1-partial-seg",
        "task1",
        strict_segmentation=False,
    )
    assert not partial_report.ok
    assert partial_report.missing


def test_filename_and_path_mismatches_fail(tmp_path) -> None:
    _touch_expected_tree(tmp_path, "task1")

    expected = (
        tmp_path
        / "task1"
        / "T1W"
        / "0.1T_to_7T"
        / "pred"
        / "P_T1W_7T_0001.nii.gz"
    )
    expected.unlink()
    (expected.parent / "P_T1W_0.1T_0001.nii.gz").touch()

    report = validate_submission_dir(tmp_path, "task1")
    assert not report.ok
    assert any("Filename target field" in error for error in report.errors)

    mismatch_root = tmp_path / "mismatch"
    bad_files = [
        "task1/T2W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz",
        "task1/T1W/0.1T_to_7T/pred/P_T1W_7T_0016.nii.gz",
        "task1/T1W/0.1T_to_7T/pred/not_official.nii.gz",
    ]
    for rel in bad_files:
        path = mismatch_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    mismatch_report = validate_submission_dir(mismatch_root, "task1")
    assert not mismatch_report.ok
    assert any("does not match filename modality" in error for error in mismatch_report.errors)
    assert any("is not expected for source field" in error for error in mismatch_report.errors)
    assert any("malformed filename" in error for error in mismatch_report.errors)


def test_extra_files_are_rejected_unless_allowed(tmp_path) -> None:
    _touch_expected_tree(tmp_path, "task3")
    extra = tmp_path / "task3" / "README.txt"
    extra.write_text("extra", encoding="utf-8")

    report = validate_submission_dir(tmp_path, "task3")
    assert not report.ok
    assert "task3/README.txt" in report.extra

    allowed = validate_submission_dir(tmp_path, "task3", allow_extra_files=True)
    assert allowed.ok
    assert allowed.extra == []


def test_build_and_validate_submission_zip(tmp_path) -> None:
    _touch_expected_tree(tmp_path, "task3")
    out_zip = build_submission_zip(tmp_path, "task3", tmp_path / "submission.zip")

    with ZipFile(out_zip, "r") as archive:
        names = archive.namelist()
    assert names
    assert all(name.startswith("task3/") for name in names)

    report = validate_submission_zip(out_zip, "task3")
    assert report.ok
    assert report.found_pred_count == 180


def test_invalid_zip_roots_fail(tmp_path) -> None:
    modality_root_zip = tmp_path / "modality-root.zip"
    with ZipFile(modality_root_zip, "w") as archive:
        archive.writestr("T1W/0.1T_to_1.5T/pred/P_T1W_1.5T_0001.nii.gz", "")
    modality_report = validate_submission_zip(modality_root_zip, "task3")
    assert not modality_report.ok
    assert any("modality folders at root" in error for error in modality_report.errors)

    wrong_task_zip = tmp_path / "wrong-task.zip"
    with ZipFile(wrong_task_zip, "w") as archive:
        archive.writestr("task1/T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz", "")
    wrong_task_report = validate_submission_zip(wrong_task_zip, "task3")
    assert not wrong_task_report.ok
    assert any("task3/" in error for error in wrong_task_report.errors)


def _touch_expected_tree(
    root,
    task: str,
    include_segmentation: bool | None = None,
):
    entries = expected_submission_entries(task, include_segmentation=include_segmentation)
    for entry in entries:
        path = root / entry.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return entries
