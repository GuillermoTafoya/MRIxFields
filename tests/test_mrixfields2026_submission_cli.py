import json

from clbfield.cli import main
from clbfield.official.submissions import (
    expected_submission_entries,
    validate_submission_zip,
)


def test_audit_submission_cli_outputs_success_json_for_valid_task3(tmp_path, capsys) -> None:
    _touch_expected_tree(tmp_path, "task3")

    exit_code = main(
        [
            "mrixfields2026-audit-submission",
            "--root",
            str(tmp_path),
            "--task",
            "task3",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["found_pred_count"] == 180


def test_audit_submission_cli_returns_nonzero_for_invalid_tree(tmp_path, capsys) -> None:
    _touch_expected_tree(tmp_path, "task3")
    first = expected_submission_entries("task3")[0]
    (tmp_path / first.relative_path).unlink()

    exit_code = main(
        [
            "mrixfields2026-audit-submission",
            "--root",
            str(tmp_path),
            "--task",
            "task3",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert first.relative_path in payload["missing"]


def test_zip_submission_cli_creates_valid_task_root_zip(tmp_path, capsys) -> None:
    _touch_expected_tree(tmp_path, "task3")
    out_zip = tmp_path / "submission.zip"

    exit_code = main(
        [
            "mrixfields2026-zip-submission",
            "--submission-root",
            str(tmp_path),
            "--task",
            "task3",
            "--out",
            str(out_zip),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["out"] == str(out_zip)
    assert out_zip.exists()
    assert validate_submission_zip(out_zip, "task3").ok


def _touch_expected_tree(root, task: str) -> None:
    for entry in expected_submission_entries(task):
        path = root / entry.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
