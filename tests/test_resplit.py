from __future__ import annotations

import json

import pytest

from fieldbridge.data.resplit import promote_subjects_to_split, resplit_file


def _split_fixture():
    def rec(subject, arr):
        return {"case_id": f"P_T1W_0.1T_{subject}", "subject_id": subject, "split": "Training_prospective"}

    return {
        "seed": 13,
        "splits": {
            "train": [rec("0006", "train"), rec("0007", "train"), {"case_id": "R_x", "subject_id": "1000"}],
            "validation": [{"case_id": "R_y", "subject_id": "1001"}],
            "test": [rec("0009", "test")],
        },
    }


def test_promote_moves_all_subject_records_to_target() -> None:
    updated = promote_subjects_to_split(_split_fixture(), ["0006", "0007"], "validation")
    val_subjects = {r["subject_id"] for r in updated["splits"]["validation"]}
    train_subjects = {r["subject_id"] for r in updated["splits"]["train"]}
    assert {"0006", "0007"}.issubset(val_subjects)  # both travellers now in validation
    assert "0006" not in train_subjects and "0007" not in train_subjects
    assert updated["resplit"]["moved_records_from"]["train"] == 2


def test_promote_raises_on_missing_subject() -> None:
    with pytest.raises(ValueError, match="not found"):
        promote_subjects_to_split(_split_fixture(), ["9999"], "validation")


def test_resplit_file_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(_split_fixture()), encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        resplit_file(path, path, ["0006"], "validation")


def test_resplit_file_writes_new_split(tmp_path) -> None:
    src = tmp_path / "split.json"
    src.write_text(json.dumps(_split_fixture()), encoding="utf-8")
    out = tmp_path / "split_gate.json"
    summary = resplit_file(src, out, ["0006"], "validation")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert summary["counts"]["validation"] == 2  # R_y + promoted 0006
    assert {r["subject_id"] for r in written["splits"]["validation"]} == {"1001", "0006"}
