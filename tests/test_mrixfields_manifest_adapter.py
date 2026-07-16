from dataclasses import replace

import pytest

from fieldbridge.data.mrixfields_adapter import (
    adapt_mrixfields_manifest,
    load_adapted_mrixfields_manifest,
)
from fieldbridge.official.data_manifest import (
    parse_mrixfields_data_path,
    write_manifest_jsonl,
)


def _same_subject_different_fields():
    return [
        parse_mrixfields_data_path(
            "Training_retrospective/T1W/1.5T/R_T1W_1.5T_0001.nii.gz"
        ),
        parse_mrixfields_data_path(
            "Training_retrospective/T1W/3T/R_T1W_3T_0001.nii.gz"
        ),
    ]


def test_adapter_uses_unique_sample_id_and_preserves_subject_and_split() -> None:
    records = _same_subject_different_fields()

    adapted = adapt_mrixfields_manifest(records)

    volumes = adapted.manifest.records
    assert [record.case_id for record in volumes] == [record.sample_id for record in records]
    assert len({record.case_id for record in volumes}) == 2
    assert {record.subject_id for record in volumes} == {"0001"}
    assert {record.split for record in volumes} == {"Training_retrospective"}
    assert all(record.metadata["split_name"] == record.split for record in volumes)
    assert all(record.metadata["sample_id"] == record.case_id for record in volumes)
    assert adapted.official_audit.ok
    assert adapted.volume_audit["ok"]
    assert adapted.volume_audit["duplicate_case_ids"] == []


def test_adapter_stops_on_official_manifest_audit_failure() -> None:
    first, second = _same_subject_different_fields()
    duplicate = replace(second, sample_id=first.sample_id)

    with pytest.raises(ValueError, match="Official.*audit failed"):
        adapt_mrixfields_manifest([first, duplicate])


def test_adapter_stops_on_missing_paths_when_strict(tmp_path) -> None:
    records = _same_subject_different_fields()

    with pytest.raises(ValueError, match="volume-manifest audit failed"):
        adapt_mrixfields_manifest(records, strict_paths=True)


def test_adapter_jsonl_round_trip_uses_same_contract(tmp_path) -> None:
    path = write_manifest_jsonl(_same_subject_different_fields(), tmp_path / "official.jsonl")

    adapted = load_adapted_mrixfields_manifest(path)

    assert len(adapted.manifest.records) == 2
    assert adapted.manifest.metadata["identity_contract"] == "case_id_is_official_sample_id"
