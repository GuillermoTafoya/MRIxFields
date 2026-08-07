from __future__ import annotations

import json
from copy import deepcopy

import pytest

import fieldbridge.data.resplit as resplit_module
from fieldbridge.data.resplit import promote_subjects_to_split, resplit_file
from fieldbridge.data.vae_splits import load_vae_splits
from fieldbridge.data.volume_splits import VolumeSplitError


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
    def rec(subject, field=0.1, contrast="T1w"):
        field_token = str(field).replace(".", "p")
        return _record(
            f"P_{contrast.upper()}_{field_token}T_{subject}",
            subject,
            "P",
            field=field,
            contrast=contrast,
        )

    return {
        "seed": 13,
        "fractions": [0.8, 0.1, 0.1],
        "metadata": {},
        "splits": {
            "train": [
                rec("0006"),
                rec("0006", field=1.5, contrast="T2w"),
                rec("0007"),
                _record("R_T1W_0.1T_0006", "0006", "R"),
                _record("R_x", "1000", "R"),
            ],
            "validation": [_record("R_y", "1001", "R")],
            "test": [rec("0009")],
        },
    }


def test_promote_moves_all_subject_records_to_target() -> None:
    updated = promote_subjects_to_split(_split_fixture(), ["0006", "0007"], "validation")
    validation = updated["splits"]["validation"]
    train = updated["splits"]["train"]
    prospective_0006 = [
        r for r in validation if r["subject_id"] == "0006" and r["metadata"]["prefix"] == "P"
    ]
    assert len(prospective_0006) == 2
    assert any(r["subject_id"] == "0007" for r in validation)
    assert not any(
        r["subject_id"] in {"0006", "0007"} and r["metadata"]["prefix"] == "P"
        for r in train
    )
    assert any(r["subject_id"] == "0006" and r["metadata"]["prefix"] == "R" for r in train)
    assert updated["resplit"]["moved_records_from"]["train"] == 3


def test_promote_raises_on_missing_subject() -> None:
    with pytest.raises(ValueError, match="not found"):
        promote_subjects_to_split(_split_fixture(), ["9999"], "validation")


def test_promote_rejects_id_that_exists_only_retrospectively() -> None:
    with pytest.raises(ValueError, match="1000"):
        promote_subjects_to_split(_split_fixture(), ["1000"], "validation")


def test_resplit_file_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(_split_fixture()), encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        resplit_file(path, path, ["0006"], "validation")


def test_resplit_file_writes_new_split(tmp_path) -> None:
    src = tmp_path / "split.json"
    src.write_text(json.dumps(_fixture_with_valid_fingerprints()), encoding="utf-8")
    out = tmp_path / "split_gate.json"
    summary = resplit_file(src, out, ["0006"], "validation")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert summary["counts"]["validation"] == 3  # R_y + both prospective 0006 records
    assert {r["subject_id"] for r in written["splits"]["validation"]} == {"1001", "0006"}


def _fixture_with_valid_fingerprints():
    """The fixture carrying fingerprints that are correct *for its own contents*.

    This is the state a real split file is in: `save_vae_splits` always writes both. Starting
    from a valid file is what makes the pre-fix failure the observed one -- "stale or altered"
    -- rather than the weaker "no persisted fingerprint".
    """

    from fieldbridge.data.manifests import record_from_mapping
    from fieldbridge.data.vae_splits import (
        VaeSplits,
        vae_splits_fingerprint,
        vae_splits_recovery_fingerprint_v3,
    )

    data = _split_fixture()
    built = VaeSplits(
        train=tuple(record_from_mapping(r) for r in data["splits"]["train"]),
        validation=tuple(record_from_mapping(r) for r in data["splits"]["validation"]),
        test=tuple(record_from_mapping(r) for r in data["splits"]["test"]),
        seed=data["seed"],
        fractions=tuple(data["fractions"]),
        metadata=data["metadata"],
    )
    data["fingerprint"] = vae_splits_fingerprint(built)
    data["recovery_fingerprint_v3"] = vae_splits_recovery_fingerprint_v3(built)
    return data


def test_resplit_file_output_survives_load_vae_splits(tmp_path) -> None:
    """The regression that made the Stage-2 traveller gate unrunnable.

    `promote_subjects_to_split` copied the input's membership fingerprints forward while
    changing membership. Both fingerprints are membership-sensitive, so `load_vae_splits`
    rejected every resplit file as "stale or altered" -- including the split whose only
    purpose is to give the gate a held-out anchor. Reproduced from split_v3.json.
    """

    src = tmp_path / "split.json"
    src.write_text(json.dumps(_fixture_with_valid_fingerprints()), encoding="utf-8")
    assert load_vae_splits(src) is not None  # the input is loadable to begin with

    out = tmp_path / "split_v4.json"
    resplit_file(src, out, ["0006"], "validation")

    reloaded = load_vae_splits(out)  # raised VolumeSplitError("stale or altered") before the fix
    assert {r.subject_id for r in reloaded.validation} == {"1001", "0006"}
    assert len([r for r in reloaded.validation if r.subject_id == "0006"]) == 2
    assert any(r.subject_id == "0006" and r.metadata["prefix"] == "R" for r in reloaded.train)
    assert not any(r.subject_id == "0006" and r.metadata["prefix"] == "P" for r in reloaded.train)
    assert {r.subject_id for r in reloaded.test} == {"0009"}


def test_promote_drops_the_now_stale_fingerprints() -> None:
    valid = _fixture_with_valid_fingerprints()
    updated = promote_subjects_to_split(valid, ["0006"], "validation")
    assert "fingerprint" not in updated
    assert "recovery_fingerprint_v3" not in updated


def _changed_membership_with_stale_hashes(data):
    data["splits"]["validation"].append(data["splits"]["train"].pop(0))


def _changed_record_with_stale_recovery_hash(data):
    data["splits"]["train"][0]["image_path"] = "altered/source.nii.gz"


def _missing_membership_fingerprint(data):
    del data["fingerprint"]


def _missing_recovery_fingerprint(data):
    del data["recovery_fingerprint_v3"]


@pytest.mark.parametrize(
    "mutate",
    [
        _changed_membership_with_stale_hashes,
        _changed_record_with_stale_recovery_hash,
        _missing_membership_fingerprint,
        _missing_recovery_fingerprint,
    ],
)
def test_resplit_file_rejects_untrusted_source_without_publishing(tmp_path, mutate) -> None:
    stale = deepcopy(_fixture_with_valid_fingerprints())
    mutate(stale)

    src = tmp_path / "split.json"
    src.write_text(json.dumps(stale), encoding="utf-8")
    out = tmp_path / "split_v4.json"
    temporary = out.with_name(f".{out.name}.tmp")

    with pytest.raises(VolumeSplitError):
        resplit_file(src, out, ["0006"], "validation")

    assert not out.exists()
    assert not temporary.exists()


def test_resplit_file_cleans_temporary_output_when_revalidation_fails(
    tmp_path, monkeypatch
) -> None:
    src = tmp_path / "split.json"
    src.write_text(json.dumps(_fixture_with_valid_fingerprints()), encoding="utf-8")
    out = tmp_path / "split_v4.json"
    temporary = out.with_name(f".{out.name}.tmp")
    canonical_load = resplit_module.load_vae_splits
    load_count = 0

    def fail_output_validation(path):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            raise VolumeSplitError("synthetic temporary-output validation failure")
        return canonical_load(path)

    monkeypatch.setattr(resplit_module, "load_vae_splits", fail_output_validation)
    with pytest.raises(VolumeSplitError, match="temporary-output"):
        resplit_file(src, out, ["0006"], "validation")

    assert load_count == 2
    assert not out.exists()
    assert not temporary.exists()
