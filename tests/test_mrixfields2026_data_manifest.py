from __future__ import annotations

from dataclasses import replace

import pytest

from fieldbridge.official.data_manifest import (
    OFFICIAL_SPLITS,
    MRIxFieldsDataRecord,
    all_domain_specs,
    audit_mrixfields_manifest,
    build_mrixfields_manifest_from_paths,
    domain_id_for,
    domain_label_for,
    parse_mrixfields_data_path,
    read_manifest_jsonl,
    scan_mrixfields_data_root,
    write_manifest_jsonl,
)
from fieldbridge.official.mrixfields2026 import FIELDS, OFFICIAL_MODALITIES


def test_domain_ids_are_stable_and_round_trip() -> None:
    specs = all_domain_specs()
    ids = {spec.domain_id for spec in specs}

    assert len(specs) == 15
    assert ids == set(range(15))

    for modality in OFFICIAL_MODALITIES:
        for field in FIELDS:
            domain_id = domain_id_for(modality, field)
            assert domain_label_for(domain_id) == (modality, field)

    assert domain_id_for("T1w", "3.0T") == domain_id_for("T1W", "3T")
    assert domain_label_for(0) == ("T1W", "0.1T")
    assert domain_label_for(14) == ("T2FLAIR", "7T")
    with pytest.raises(ValueError):
        domain_label_for(15)


@pytest.mark.parametrize(
    ("path", "prefix", "cohort", "is_paired"),
    [
        ("Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz", "R", "retrospective", False),
        ("Training_prospective/T2W/7T/P_T2W_7T_0006.nii.gz", "P", "prospective", True),
        ("Validating_prospective/T2FLAIR/3T/P_T2FLAIR_3T_0010.nii.gz", "P", "prospective", True),
        ("Testing_prospective/T1W/5T/P_T1W_5T_0021.nii.gz", "P", "prospective", True),
    ],
)
def test_parse_mrixfields_data_path_for_official_splits(
    path: str,
    prefix: str,
    cohort: str,
    is_paired: bool,
) -> None:
    record = parse_mrixfields_data_path(path)

    assert record.split_name in OFFICIAL_SPLITS
    assert record.prefix == prefix
    assert record.cohort == cohort
    assert record.is_paired is is_paired
    assert record.relative_path == path
    assert record.filename.endswith(".nii.gz")
    assert record.internal_modality in {"T1w", "T2w", "T2-FLAIR"}


@pytest.mark.parametrize(
    "path",
    [
        "Training_retrospective/T2W/0.1T/R_T1W_0.1T_0001.nii.gz",
        "Training_retrospective/T1W/1.5T/R_T1W_0.1T_0001.nii.gz",
        "Training_prospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz",
        "Training_retrospective/T1W/0.1T/P_T1W_0.1T_0001.nii.gz",
        "Training_retrospective/T1W/0.1T/not_official.nii.gz",
        "Unknown_split/T1W/0.1T/R_T1W_0.1T_0001.nii.gz",
        "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001_seg.nii.gz",
    ],
)
def test_parse_mrixfields_data_path_rejects_invalid_paths(path: str) -> None:
    with pytest.raises(ValueError):
        parse_mrixfields_data_path(path)


def test_build_manifest_jsonl_round_trip_and_sorting(tmp_path) -> None:
    paths = [
        "Training_prospective/T2W/7T/P_T2W_7T_0006.nii.gz",
        "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz",
        "Validating_prospective/T2FLAIR/3T/P_T2FLAIR_3T_0010.nii.gz",
    ]

    records = build_mrixfields_manifest_from_paths(paths)
    out_path = write_manifest_jsonl(records, tmp_path / "manifest.jsonl")
    loaded = read_manifest_jsonl(out_path)

    assert [record.relative_path for record in records] == [
        "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz",
        "Training_prospective/T2W/7T/P_T2W_7T_0006.nii.gz",
        "Validating_prospective/T2FLAIR/3T/P_T2FLAIR_3T_0010.nii.gz",
    ]
    assert loaded == records
    assert len({record.sample_id for record in records}) == len(records)


def test_scan_mrixfields_data_root_ignores_hidden_and_non_nifti(tmp_path) -> None:
    kept = [
        "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz",
        "Training_prospective/T2W/7T/P_T2W_7T_0006.nii.gz",
    ]
    ignored = [
        ".hidden/Training_retrospective/T1W/0.1T/R_T1W_0.1T_9999.nii.gz",
        "Training_retrospective/T1W/0.1T/.R_T1W_0.1T_9999.nii.gz",
        "Training_retrospective/T1W/0.1T/readme.txt",
    ]
    for relative_path in [*kept, *ignored]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    records = scan_mrixfields_data_root(tmp_path)

    assert [record.relative_path for record in records] == kept
    assert all(record.raw_uri.startswith(str(tmp_path)) for record in records)


def test_audit_manifest_detects_duplicates_and_bad_metadata() -> None:
    base = parse_mrixfields_data_path("Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz")
    valid = audit_mrixfields_manifest([base])
    assert valid.ok

    duplicate_uri = audit_mrixfields_manifest([base, replace(base, sample_id="other")])
    assert not duplicate_uri.ok
    assert duplicate_uri.duplicate_raw_uris == [base.raw_uri]

    duplicate_sample = audit_mrixfields_manifest([base, replace(base, raw_uri="other")])
    assert not duplicate_sample.ok
    assert duplicate_sample.duplicate_sample_ids == [base.sample_id]

    bad = replace(base, shape=(1, 2, 3), dtype="float64", intensity_min=-0.1, intensity_max=1.1)
    bad_report = audit_mrixfields_manifest([bad])
    assert not bad_report.ok
    assert any("Expected shape" in error for error in bad_report.errors)
    assert any("Expected dtype" in error for error in bad_report.errors)
    assert any("Expected intensity range" in error for error in bad_report.errors)


def test_record_from_mapping_preserves_optional_metadata() -> None:
    record = parse_mrixfields_data_path("Training_retrospective/T2FLAIR/7T/R_T2FLAIR_7T_0001.nii.gz")
    payload = record.to_dict()
    payload["shape"] = [364, 436, 364]
    payload["dtype"] = "float32"
    payload["intensity_min"] = 0.0
    payload["intensity_max"] = 1.0

    restored = MRIxFieldsDataRecord.from_mapping(payload)

    assert restored.shape == (364, 436, 364)
    assert restored.dtype == "float32"
    assert restored.intensity_min == 0.0
    assert restored.intensity_max == 1.0
