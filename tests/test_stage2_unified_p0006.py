from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentRecord
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json, sha256_text
from fieldbridge.evaluation.stage2_gate01 import GATE01_CONTRACT_VERSION, Gate01Case
from fieldbridge.training.stage2_unified import build_unified_validation_plan


class _Index:
    def __init__(self, split: str) -> None:
        self.split = split
        self.artifact_sha256 = sha256_text(f"bank:{split}")
        self.records = []
        subjects = ("train-a", "train-b") if split == "train" else ("val-a", "val-b")
        for contrast in Contrast:
            for field in FIELD_STRENGTHS_T:
                for subject in subjects:
                    domain = Domain(field, contrast)
                    identity = f"R_{split}_{contrast.value}_{field:g}_{subject}"
                    self.records.append(
                        FactoredLatentRecord(
                            case_id=identity,
                            subject_group_id=f"R:{subject}",
                            domain=domain,
                            split=split,
                            path=Path("unused.pt"),
                            resume_key=sha256_text(identity),
                            sidecar={"cohort": "R", "split": split},
                        )
                    )


class _Lock:
    traveller_identity_sha256 = p0006.P0006_IDENTITY_SHA256
    split_fingerprint = "split"
    calibrator_template_sha256 = "t" * 64
    calibrator_artifact_sha256 = "c" * 64
    artifact_sha256 = "l" * 64


class _Calibrator:
    artifact_sha256 = "c" * 64

    def apply(self, prediction, requested_target, *, support_mask):
        del requested_target
        return prediction + support_mask.to(prediction.dtype) * 0.125


class _GateManifest:
    def __init__(self, root: Path, cases: list[Gate01Case]) -> None:
        self.root = root
        self._cases = cases
        reference = {"path": "array.npy", "sha256": "a" * 64}
        self.case_specs = tuple(
            {
                "source_image": reference,
                "source_support_mask": reference,
                "target": reference,
                "raw_identity": reference,
                "raw_sb_v2": reference,
                "stage1_reconstruction_ceiling": reference,
                "wrong_target_sb_v2": {
                    label: reference for label in case.wrong_target_sb_v2
                },
            }
            for case in cases
        )

    def __iter__(self):
        return iter(self._cases)


def _gate_cases() -> list[Gate01Case]:
    result = []
    volume = torch.ones(1, 3, 3, 3)
    support = torch.ones_like(volume, dtype=torch.bool)
    for contrast in Contrast:
        for source_field in FIELD_STRENGTHS_T:
            source_domain = Domain(source_field, contrast)
            for target_field in FIELD_STRENGTHS_T:
                if target_field == source_field:
                    continue
                target_domain = Domain(target_field, contrast)
                case_identity = f"P_0006_{contrast.value}_{source_field:g}_{target_field:g}"
                wrong_labels = [
                    f"{field:g}T"
                    for field in FIELD_STRENGTHS_T
                    if field not in {source_field, target_field}
                ]
                hashes = {
                    "source_image": sha256_text(f"acquisition:{source_domain.label}"),
                    "source_support_mask": sha256_text(f"support:{source_domain.label}"),
                    "target": sha256_text(f"acquisition:{target_domain.label}"),
                    "raw_identity": sha256_text(f"identity:{source_domain.label}"),
                    "raw_sb_v2": sha256_text(f"sb:{case_identity}"),
                    "stage1_reconstruction_ceiling": sha256_text(
                        f"stage1:{target_domain.label}"
                    ),
                    **{
                        f"wrong_target_sb_v2[{label}]": sha256_text(
                            f"wrong:{case_identity}:{label}"
                        )
                        for label in wrong_labels
                    },
                }
                result.append(
                    Gate01Case(
                        case_id=case_identity,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        target=volume * target_field,
                        raw_identity=volume * source_field,
                        raw_sb_v2=volume * (source_field + target_field),
                        stage1_reconstruction_ceiling=volume * target_field,
                        support_mask=support,
                        traveller_identity_sha256=p0006.P0006_IDENTITY_SHA256,
                        array_sha256=hashes,
                        wrong_target_sb_v2={label: volume for label in wrong_labels},
                        source_image=volume * source_field,
                    )
                )
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_gate01_p0006_import_and_reload_seal_complete_evaluation_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "Gate01Private_8012a3f"
    archive.mkdir()
    result_body = {"contract_version": GATE01_CONTRACT_VERSION, "value": 1}
    result = {**result_body, "result_sha256": sha256_json(result_body)}
    result_path = archive / "result.json"
    manifest_path = archive / "manifest.json"
    lock_path = archive / "lock.json"
    calibrator_path = archive / "calibrator.json"
    _write_json(result_path, result)
    _write_json(manifest_path, {"contract_version": p0006.GATE01_INPUT_CONTRACT_VERSION})
    _write_json(lock_path, {"contract_version": p0006.GATE01_PROTOCOL_LOCK_CONTRACT_VERSION})
    _write_json(calibrator_path, {"contract_version": p0006.GATE01_CALIBRATOR_CONTRACT_VERSION})
    with (archive / "sha256-inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Algorithm", "Hash", "Path"))
        writer.writeheader()
        for path in (result_path, manifest_path, lock_path, calibrator_path):
            writer.writerow({"Algorithm": "SHA256", "Hash": sha256_file(path), "Path": str(path)})

    indexes = {"train": _Index("train"), "validation": _Index("validation")}
    plan = build_unified_validation_plan(indexes["validation"], validation_seed=20260818)  # type: ignore[arg-type]
    plan_path = tmp_path / "validation-plan.json"
    _write_json(plan_path, plan)
    cases = _gate_cases()
    (archive / "array.npy").write_bytes(b"synthetic-array-placeholder")
    gate_manifest = _GateManifest(archive, cases)
    calibrator = _Calibrator()
    metadata = {
        "execution_mode": "scientific",
        "evidence_scope": {
            "private_data_run": True,
            "evidence_kind": "private",
            "traveller_identity_sha256": p0006.P0006_IDENTITY_SHA256,
        },
        "selection_fingerprint_sha256": "s" * 64,
        "scientific_hash_graph": {
            "validated": True,
            "acquisition_count": 15,
            "direction_count": 60,
        },
        "producer_receipt": {
            "producer_receipt": {
                "decode_strategy": "full",
                "path_used": ["full"],
                "direction_count": 60,
                "wrong_target_reference_count": 180,
            }
        },
    }
    monkeypatch.setattr(
        p0006,
        "PhotometryFactoredLatentBankIndex",
        lambda root, split: indexes[split],
    )
    monkeypatch.setattr(p0006.Gate01ProtocolLock, "load", lambda path: _Lock())
    monkeypatch.setattr(
        p0006.PosthocTargetCalibrator,
        "load",
        lambda path, **kwargs: calibrator,
    )
    monkeypatch.setattr(
        p0006,
        "load_gate01_input_manifest",
        lambda *args, **kwargs: (gate_manifest, metadata),
    )
    output = tmp_path / "p0006-protocol.json"
    protocol = p0006.import_gate01_p0006_evaluation_protocol(
        archive,
        expected_gate01_result_sha256=sha256_file(result_path),
        bank_dir=tmp_path / "bank",
        validation_plan_path=plan_path,
        output_path=output,
    )
    assert protocol["acquisition_count"] == 15
    assert protocol["directed_pair_count"] == 60
    assert protocol["wrong_target_reference_count"] == 180
    assert protocol["factored_bank"]["P_record_count"] == 0
    assert protocol["frozen_unpaired_validation"]["P_endpoint_count"] == 0
    assert protocol["full_volume_decode_proof"]["path_used"] == ["full"]
    assert protocol["data_role"] == p0006.P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
    assert protocol["evidence_interpretation"] == p0006.P0006_EVIDENCE_LIMITATION
    assert protocol["population_or_generalization_claims_authorized"] is False
    assert protocol["P0009_confirmation_status"] == p0006.P0009_CONFIRMATION_STATUS
    assert protocol["P0009_executed"] is False
    assert len(protocol["case_receipts"]) == 60

    reloaded, paired_cases, baselines = p0006.load_gate01_p0006_evaluation_protocol(output)
    assert reloaded["protocol_sha256"] == protocol["protocol_sha256"]
    assert len(paired_cases) == len(baselines) == 60
    assert {case.subject_group_identity for case in paired_cases} == {"P:0006"}
    assert all(case.source_provenance["data_role"] == protocol["data_role"] for case in paired_cases)

    obsolete_body = dict(protocol)
    obsolete_body.pop("protocol_sha256")
    obsolete_body["contract_version"] = (
        "stage2-unified-gate01-p0006-evaluation-only-protocol-v1"
    )
    obsolete_path = tmp_path / "obsolete-p0006-protocol.json"
    _write_json(
        obsolete_path,
        {**obsolete_body, "protocol_sha256": sha256_json(obsolete_body)},
    )
    with pytest.raises(ValueError, match="Unsupported"):
        p0006.load_gate01_p0006_evaluation_protocol(obsolete_path)

    cases[0].array_sha256["target"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="changed"):
        p0006.load_gate01_p0006_evaluation_protocol(output)
