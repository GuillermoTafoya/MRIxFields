from __future__ import annotations

import json

import pytest

from fieldbridge.data.resplit import promote_subjects_to_split, resplit_file
from fieldbridge.data.vae_splits import load_vae_splits


def _record(case_id, subject_id, prefix, field=0.1, contrast="T1w"):
    """A record shaped like a real split entry, so fingerprints can be recomputed from it."""

    return {
        "case_id": case_id,
        "image_path": f"data/{case_id}.nii.gz",
        "domain": {"field_strength_t": field, "contrast": contrast},
        "subject_id": subject_id,
        "split": None,
        "metadata": {"prefix": prefix},
    }


def _split_fixture():
    def rec(subject):
        return _record(f"P_T1W_0.1T_{subject}", subject, "P")

    return {
        "seed": 13,
        "fractions": [0.8, 0.1, 0.1],
        "metadata": {},
        "splits": {
            "train": [rec("0006"), rec("0007"), _record("R_x", "1000", "R")],
            "validation": [_record("R_y", "1001", "R")],
            "test": [rec("0009")],
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


def test_resplit_output_is_loadable_by_the_split_reader(tmp_path) -> None:
    """The whole point of a resplit is that the downstream gate can read it.

    Membership fingerprints are membership-sensitive, so carrying the input's forward made
    every resplit file fail `load_vae_splits` as "stale or altered" — silently blocking the
    Stage-2 eval gate from ever using the held-out anchor the resplit exists to create.
    """

    src = tmp_path / "split.json"
    src.write_text(json.dumps(_split_fixture()), encoding="utf-8")
    out = tmp_path / "split_v4.json"
    resplit_file(src, out, ["P:0006"], "validation")

    reloaded = load_vae_splits(out)
    assert {r.subject_id for r in reloaded.validation} == {"1001", "0006"}
    assert "0006" not in {r.subject_id for r in reloaded.train}


def test_promote_drops_the_now_stale_fingerprints(tmp_path) -> None:
    stale = _split_fixture()
    stale["fingerprint"] = "stale-membership-fingerprint"
    stale["recovery_fingerprint_v3"] = "stale-recovery-fingerprint"

    updated = promote_subjects_to_split(stale, ["P:0006"], "validation")
    assert "fingerprint" not in updated
    assert "recovery_fingerprint_v3" not in updated

    src = tmp_path / "split.json"
    src.write_text(json.dumps(stale), encoding="utf-8")
    out = tmp_path / "split_v4.json"
    resplit_file(src, out, ["P:0006"], "validation")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["fingerprint"] != "stale-membership-fingerprint"
    assert written["recovery_fingerprint_v3"] != "stale-recovery-fingerprint"
