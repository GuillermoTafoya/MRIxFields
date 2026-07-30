"""A bare subject number names two different people; the resplit must not conflate them.

The official data description gives the cohorts overlapping numeric ranges (retrospective
0001-1056, field-scoped; prospective 0001-0040), and split_v3 really does contain both a
traveller P_..._0006 and a 0.1T retrospective volunteer R_..._0006, both in train.
"""

from __future__ import annotations

import pytest

from fieldbridge.data.resplit import promote_subjects_to_split


def _split() -> dict:
    def record(prefix: str, subject: str, field: float, split_hint: str) -> dict:
        return {"case_id": f"{prefix}_T1W_{field}T_{subject}", "subject_id": subject,
                "image_path": f"/data/{prefix}_{subject}_{field}_{split_hint}.nii.gz",
                "domain": {"field_strength_t": field, "contrast": "T1w"}}

    return {"splits": {
        "train": [record("P", "0006", 0.1, "t"), record("P", "0006", 7.0, "t"),
                  record("R", "0006", 0.1, "t"), record("P", "0007", 0.1, "t")],
        "validation": [record("R", "0007", 1.5, "v")],
        "test": [record("P", "0009", 0.1, "s"), record("R", "0100", 0.1, "s")],
    }}


def test_qualified_id_moves_only_the_traveller() -> None:
    out = promote_subjects_to_split(_split(), ["P:0006"], "validation")

    moved = out["splits"]["validation"]
    assert sorted(r["case_id"] for r in moved) == [
        "P_T1W_0.1T_0006", "P_T1W_7.0T_0006", "R_T1W_1.5T_0007",
    ]
    # The retrospective 0006 is a different person and must stay put.
    assert any(r["case_id"] == "R_T1W_0.1T_0006" for r in out["splits"]["train"])


def test_underscore_form_is_accepted_too() -> None:
    out = promote_subjects_to_split(_split(), ["P_0006"], "validation")

    assert any(r["case_id"] == "R_T1W_0.1T_0006" for r in out["splits"]["train"])


def test_bare_ambiguous_id_raises_instead_of_guessing() -> None:
    with pytest.raises(ValueError, match="Ambiguous subject id"):
        promote_subjects_to_split(_split(), ["0006"], "validation")


def test_bare_unambiguous_id_still_works() -> None:
    out = promote_subjects_to_split(_split(), ["0009"], "validation")

    assert any(r["case_id"] == "P_T1W_0.1T_0009" for r in out["splits"]["validation"])


def test_unknown_subject_raises() -> None:
    with pytest.raises(ValueError, match="not found in the split"):
        promote_subjects_to_split(_split(), ["P:9999"], "validation")


def test_no_record_is_lost_or_duplicated() -> None:
    before = _split()
    total = sum(len(v) for v in before["splits"].values())

    out = promote_subjects_to_split(before, ["P:0006"], "validation")

    after = [r["case_id"] for v in out["splits"].values() for r in v]
    assert len(after) == total
    assert len(set(after)) == len(after)
