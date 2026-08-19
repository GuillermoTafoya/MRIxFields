from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentRecord,
    FactoredLatentStats,
)
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json
from fieldbridge.data.vae_splits import VaeSplits, save_vae_splits
from fieldbridge.evaluation.stage2_unified_preflight import (
    BASELINE_SOURCE_CONTRACT,
    BASELINE_SOURCE_PRODUCER_CONTRACT,
    MATERIALIZED_VALIDATION_ARRAYS_CONTRACT,
    MATERIALIZED_VALIDATION_PRODUCER_CONTRACT,
    audit_retrospective_paired_feasibility,
    build_baseline_prediction_manifest,
    build_retrospective_paired_manifest,
    quantify_factored_domain_separability,
    seal_long_run_evaluation_readiness,
)


class SeparabilityIndex:
    artifact_sha256 = "a" * 64

    def __init__(self, split: str) -> None:
        self.split = split
        self.records = []
        self.values = []
        offset = 0 if split == "train" else 100
        for label, (contrast, field) in enumerate(
            (contrast, field) for contrast in Contrast for field in FIELD_STRENGTHS_T
        ):
            for replicate in range(2):
                identity = f"R_{split}_{label}_{replicate}"
                self.records.append(
                    FactoredLatentRecord(
                        case_id=identity,
                        subject_group_id=f"R:{offset + label * 2 + replicate}",
                        domain=Domain(field, contrast),
                        split=split,
                        path=Path("unused.pt"),
                        resume_key=f"{offset + label * 2 + replicate:064x}",
                        sidecar={"cohort": "R", "split": split},
                    )
                )
                tensor = torch.zeros(4, 2, 2, 2)
                tensor[label % 4] = float(label + 1)
                tensor[(label + 1) % 4] = float(label) / 10.0
                self.values.append(tensor)

    def load(self, index: int):
        return self.values[index], torch.ones(2, 2, 2, dtype=torch.bool)


def _stats() -> FactoredLatentStats:
    return FactoredLatentStats(
        mean=torch.zeros(4),
        std=torch.ones(4),
        supported_count=torch.ones(4, dtype=torch.int64),
        artifact_sha256="b" * 64,
    )


def test_domain_separability_is_subject_grouped_and_reports_all_15_domains(tmp_path: Path) -> None:
    output = tmp_path / "separability.json"
    result = quantify_factored_domain_separability(
        SeparabilityIndex("train"),  # type: ignore[arg-type]
        SeparabilityIndex("validation"),  # type: ignore[arg-type]
        _stats(),
        output_path=output,
    )
    assert len(result["domain_order"]) == 15
    assert len(result["validation_confusion"]) == 15
    assert set(result["per_domain"]) == set(result["domain_order"])
    assert result["subject_grouped_train_validation"] is True
    assert json.loads(output.read_text())["result_sha256"] == result["result_sha256"]


def _record(tmp_path: Path, case_id: str, subject: str, field: float, *, cohort: str = "R") -> VolumeRecord:
    path = tmp_path / f"{case_id}.bin"
    if cohort == "R":
        path.write_bytes(case_id.encode("utf-8"))
    return VolumeRecord(
        case_id=case_id,
        image_path=path,
        domain=Domain(field, Contrast.T1W),
        subject_id=subject,
        split="validation",
        metadata={"prefix": cohort, "cohort": cohort},
    )


def test_paired_feasibility_builds_complete_directed_inventory_and_excludes_p_preload(
    tmp_path: Path,
) -> None:
    records = (
        _record(tmp_path, "R_same_01", "same", 0.1),
        _record(tmp_path, "R_same_3", "same", 3.0),
        _record(tmp_path, "R_other_7", "other", 7.0),
        _record(tmp_path, "P_forbidden", "traveller", 7.0, cohort="P"),
    )
    splits = VaeSplits((), records, (), 13, (0.0, 1.0, 0.0))
    split_path = save_vae_splits(splits, tmp_path / "split.json")
    result = audit_retrospective_paired_feasibility(split_path)
    assert result["paired_evaluation_possible"] is True
    assert result["directed_pair_count"] == 2
    assert {item["source_record_identity"] for item in result["pairs"]} == {
        "R_same_01",
        "R_same_3",
    }
    assert result["excluded_prospective"][0]["case_id"] == "P_forbidden"
    assert "P_forbidden" not in result["source_files"]
    assert result["classification_before_source_file_access"] is True
    assert result["array_payloads_opened"] == 0


def test_paired_feasibility_fails_closed_when_r_validation_has_no_cross_field_subject(
    tmp_path: Path,
) -> None:
    records = (
        _record(tmp_path, "R_one", "one", 0.1),
        _record(tmp_path, "R_two", "two", 3.0),
    )
    split_path = save_vae_splits(
        VaeSplits((), records, (), 13, (0.0, 1.0, 0.0)), tmp_path / "split.json"
    )
    result = audit_retrospective_paired_feasibility(split_path)
    assert result["paired_evaluation_possible"] is False
    assert result["directed_pair_count"] == 0
    assert "do not fabricate" in result["failure_instruction"]


def test_corrupted_r_identity_raises_before_source_access_or_complete_inventory_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    valid = _record(tmp_path, "R_valid", "same", 0.1)
    corrupt = VolumeRecord(
        case_id="R_corrupt",
        image_path=tmp_path / "must-not-open.bin",
        domain=Domain(3.0, Contrast.T1W),
        subject_id="same",
        split="validation",
        metadata={"prefix": "R", "cohort": "P"},
    )
    split_path = save_vae_splits(
        VaeSplits((), (valid, corrupt), (), 13, (0.0, 1.0, 0.0)),
        tmp_path / "corrupt-split.json",
    )
    accessed = False

    def forbidden(*args, **kwargs):
        nonlocal accessed
        accessed = True
        raise AssertionError("source identity accessed")

    monkeypatch.setattr(
        "fieldbridge.evaluation.stage2_unified_preflight._source_file_identity",
        forbidden,
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="complete_inventory_no_selection"):
        audit_retrospective_paired_feasibility(split_path, output_path=output)
    assert accessed is False
    assert not output.exists()


class ArtifactIdentity:
    artifact_sha256 = "e" * 64
    provenance = {"resolved_config_sha256": "f" * 64}


def test_paired_manifest_builder_uses_complete_feasible_inventory_and_stage1_ceilings(
    tmp_path: Path,
) -> None:
    records = (
        _record(tmp_path, "R_same_01", "same", 0.1),
        _record(tmp_path, "R_same_3", "same", 3.0),
    )
    split_path = save_vae_splits(
        VaeSplits((), records, (), 13, (0.0, 1.0, 0.0)), tmp_path / "split.json"
    )
    feasibility_path = tmp_path / "feasibility.json"
    feasibility = audit_retrospective_paired_feasibility(
        split_path, output_path=feasibility_path
    )
    materialized_body = {
        "contract_version": MATERIALIZED_VALIDATION_ARRAYS_CONTRACT,
        "producer": {
            "contract_version": MATERIALIZED_VALIDATION_PRODUCER_CONTRACT,
            "deterministic": True,
            "complete_inventory_no_selection": True,
            "full_volume_arithmetic": True,
            "source_code_provenance_sha256": "a" * 64,
            "resolved_config_sha256": "b" * 64,
            "feasibility_result_sha256": feasibility["result_sha256"],
        },
        "records": [],
    }
    for record in records:
        array_spec = {
            "record_identity": record.case_id,
            "subject_identity": record.subject_id,
            "array_path": str(tmp_path / f"{record.case_id}.npy"),
            "file_sha256": "1" * 64,
            "loaded_array_sha256": "2" * 64,
            "content_identity": f"content:{record.case_id}",
            "shape": [1, 4, 4, 4],
            "dtype": "torch.float32",
            "stage1_reconstruction": {
                "array_path": str(tmp_path / f"{record.case_id}_stage1.npy"),
                "file_sha256": "3" * 64,
                "loaded_array_sha256": "4" * 64,
                "content_identity": f"stage1:{record.case_id}",
                "shape": [1, 4, 4, 4],
                "dtype": "torch.float32",
            },
        }
        materialized_body["records"].append(array_spec)
    materialized = dict(materialized_body)
    materialized["manifest_sha256"] = sha256_json(materialized_body)
    materialized_path = tmp_path / "materialized.json"
    materialized_path.write_text(json.dumps(materialized), encoding="utf-8")
    output = tmp_path / "paired.json"
    result = build_retrospective_paired_manifest(
        feasibility_path,
        materialized_path,
        ArtifactIdentity(),  # type: ignore[arg-type]
        output_path=output,
        authorization_reference="synthetic-complete-inventory",
    )
    assert len(result["cases"]) == feasibility["directed_pair_count"] == 2
    assert all(case["stage1_reconstruction"] for case in result["cases"])
    assert result["provenance"]["complete_inventory_no_selection"] is True
    assert json.loads(output.read_text())["manifest_sha256"] == result["manifest_sha256"]

    source_cases = []
    for edge in feasibility["pairs"]:
        methods = {}
        for method in ("gate01_calibrated_identity", "original_sb_v2"):
            prediction = tmp_path / f"{edge['case_identity']}-{method}.npy"
            prediction.write_bytes(f"{edge['case_identity']}:{method}".encode())
            methods[method] = {
                "path": str(prediction),
                "file_sha256": sha256_file(prediction),
            }
        source_cases.append(
            {
                "case_identity": edge["case_identity"],
                "record_identity": edge["source_record_identity"],
                "subject_identity": edge["subject_group_identity"].split(":", 1)[-1],
                "metadata_prefix": "R",
                "cohort": "R",
                "split": "validation",
                **methods,
            }
        )
    baseline_source_body = {
        "contract_version": BASELINE_SOURCE_CONTRACT,
        "producer": {
            "contract_version": BASELINE_SOURCE_PRODUCER_CONTRACT,
            "deterministic": True,
            "complete_inventory_no_selection": True,
            "full_volume_arithmetic": True,
            "source_code_provenance_sha256": "c" * 64,
            "resolved_config_sha256": "d" * 64,
            "paired_manifest_sha256": result["manifest_sha256"],
        },
        "cases": source_cases,
    }
    baseline_source = dict(baseline_source_body)
    baseline_source["manifest_sha256"] = sha256_json(baseline_source_body)
    baseline_source_path = tmp_path / "baseline-source.json"
    baseline_source_path.write_text(json.dumps(baseline_source), encoding="utf-8")
    baseline_predictions_path = tmp_path / "baseline-predictions.json"
    baseline_predictions = build_baseline_prediction_manifest(
        output,
        baseline_source_path,
        output_path=baseline_predictions_path,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness = seal_long_run_evaluation_readiness(
        feasibility_path,
        materialized_path,
        output,
        baseline_source_path,
        baseline_predictions_path,
        output_path=readiness_path,
    )
    assert readiness["long_run_authorized_by_evaluation_path"] is True
    assert readiness["prospective_protocol_used"] is False
    assert readiness["directed_pair_count"] == len(baseline_predictions["cases"])


def test_long_run_readiness_hard_stops_when_genuine_pairs_do_not_exist(
    tmp_path: Path,
) -> None:
    split_path = save_vae_splits(
        VaeSplits(
            (),
            (
                _record(tmp_path, "R_one", "one", 0.1),
                _record(tmp_path, "R_two", "two", 3.0),
            ),
            (),
            13,
            (0.0, 1.0, 0.0),
        ),
        tmp_path / "no-pairs-split.json",
    )
    feasibility_path = tmp_path / "no-pairs.json"
    audit_retrospective_paired_feasibility(split_path, output_path=feasibility_path)
    with pytest.raises(ValueError, match="Long training is blocked"):
        seal_long_run_evaluation_readiness(
            feasibility_path,
            tmp_path / "not-opened-arrays.json",
            tmp_path / "not-opened-pairs.json",
            tmp_path / "not-opened-source.json",
            tmp_path / "not-opened-baselines.json",
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()
