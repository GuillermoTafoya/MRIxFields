from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import pytest
import torch

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentRecord
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json, sha256_text
from fieldbridge.data.vae_splits import VaeSplits, save_vae_splits, vae_splits_fingerprint
from fieldbridge.evaluation.stage2_gate01 import GATE01_CONTRACT_VERSION, Gate01Case
from fieldbridge.evaluation.stage2_gate01_protocol import GATE01_SCIENTIFIC_MODULES
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
    bank_source_split_fingerprint = vae_splits_fingerprint(
        VaeSplits((), (), (), 8012, (1.0, 0.0, 0.0), {"synthetic": True})
    )
    evaluation_git_commit = "c" * 40
    evaluation_module_sha256 = {
        module: sha256_text(module) for module in GATE01_SCIENTIFIC_MODULES
    }


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


class _MetadataOnlyGateManifest(_GateManifest):
    def __iter__(self):
        raise AssertionError("metadata preflight must not open private arrays")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _pad_json(path: Path, size_bytes: int) -> None:
    payload = path.read_bytes()
    assert len(payload) <= size_bytes
    path.write_bytes(payload + b" " * (size_bytes - len(payload)))


def _authentic_gate01_result() -> dict[str, object]:
    result: dict[str, object] = {
        "contract_version": GATE01_CONTRACT_VERSION,
        "evidence_scope": {"synthetic": True},
        "scientific_status": {"synthetic": True},
        "central_question": "synthetic-only",
        "metric_roles": {},
        "method_roles": {},
        "num_pairs": 0,
        "methods": [],
        "overall": {},
        "strata": {},
        "raw_pre_mask_background_leakage": {},
        "by_contrast": {},
        "directed_pair_results": {},
        "directed_pair_matrices": {},
        "central_paired_deltas_and_wins": {},
        "requested_vs_wrong_target_diagnostic": {},
        "montage_specifications": {},
        "pairs": [],
        "contract": {},
        "montage_rendering": {},
    }
    assert len(result) == 20
    return result


@pytest.mark.parametrize("inventory_format", ["csv", "reviewed_legacy_json"])
def test_gate01_p0006_import_and_reload_seal_complete_evaluation_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inventory_format: str
) -> None:
    archive = tmp_path / "Gate01Private_8012a3f"
    archive.mkdir()
    result = _authentic_gate01_result()
    result_path = archive / "gate01-results.json"
    manifest_path = archive / "gate01-private-manifest.json"
    lock_path = archive / "gate01-protocol-lock.json"
    calibrator_path = archive / "gate01-target-calibrator.json"
    _write_json(result_path, result)
    result_file_sha256 = sha256_file(result_path)
    monkeypatch.setattr(
        p0006,
        "REVIEWED_GATE01_RESULT_FILE_SHA256",
        result_file_sha256,
    )
    _write_json(manifest_path, {"contract_version": p0006.GATE01_INPUT_CONTRACT_VERSION})
    _write_json(lock_path, {"contract_version": p0006.GATE01_PROTOCOL_LOCK_CONTRACT_VERSION})
    _write_json(calibrator_path, {"contract_version": p0006.GATE01_CALIBRATOR_CONTRACT_VERSION})
    additional = {
        "gate01-private-build-state.json",
        "gate01-result-contract.json",
        "gate01-protocol-spec.json",
        "gate01-producer-spec.json",
        "gate01-prospective-selection.json",
        "frozen-scientific-resplit.json",
    }
    if inventory_format == "csv":
        additional |= {
            "original-split-v3.json",
            "producer-state.json",
            "private-build-plan.json",
        }
    else:
        additional |= {
            "gate01-report.md",
        }
    for name in additional:
        _write_json(archive / name, {"synthetic_dependency": name})
    if inventory_format == "csv":
        (archive / "gate01-report.md").write_text("synthetic", encoding="utf-8")
    if inventory_format == "csv":
        (archive / "gate01-report.md").write_text("synthetic", encoding="utf-8")
        inventoried = sorted(path for path in archive.iterdir() if path.is_file())
        with (archive / "sha256-inventory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("Algorithm", "Hash", "Path"))
            writer.writeheader()
            for path in inventoried:
                writer.writerow(
                    {"Algorithm": "SHA256", "Hash": sha256_file(path), "Path": str(path)}
                )
    else:
        nested = archive / "archive"
        nested.mkdir()
        split_path = nested / "split_v3.json"
        _write_json(split_path, {"reviewed_split": True})
        _pad_json(split_path, p0006._GATE01_LEGACY_SPLIT_SIZE_BYTES)
        _write_json(
            archive / "producer-state/producer-state.json",
            {"synthetic_dependency": "producer-state/producer-state.json"},
        )
        _write_json(
            archive / "producer-output/private-build-plan.json",
            {"synthetic_dependency": "producer-output/private-build-plan.json"},
        )
        operational_split = archive / "colab-operational-source-split.json"
        save_vae_splits(
            VaeSplits(
                (),
                (),
                (),
                8012,
                (1.0, 0.0, 0.0),
                {"synthetic": True},
            ),
            operational_split,
        )
        _pad_json(operational_split, 799986)
        reviewed_modules = archive / "gate01-reviewed-module-sha256-8012a3f.json"
        previous_modules = dict(_Lock.evaluation_module_sha256)
        previous_modules[
            "src/fieldbridge/models/translators/flow_transport.py"
        ] = sha256_text("previous:flow_transport")
        reviewed_modules.write_text(
            json.dumps(
                {
                    "changed_modules": [
                        "src/fieldbridge/models/translators/flow_transport.py"
                    ],
                    "evaluation_git_commit": _Lock.evaluation_git_commit,
                    "evaluation_module_sha256": _Lock.evaluation_module_sha256,
                    "previous_evaluation_git_commit": "b" * 40,
                    "previous_evaluation_module_sha256": previous_modules,
                }
            ),
            encoding="utf-8",
        )
        _pad_json(reviewed_modules, 7634)
        real_sha256_file = sha256_file
        pinned = {
            split_path.resolve(): p0006._GATE01_LEGACY_SPLIT_SHA256,
            operational_split.resolve(): (
                "972e497e2d29755e928414a4aa51f906951674ec0a950b0e9ac73881fffd0c54"
            ),
            reviewed_modules.resolve(): (
                "ea5f40b580cbba26766ee60ce243d466ab93d32b1856125c067eace9a7d1ed36"
            ),
        }

        def pinned_sha256(path: str | Path) -> str:
            candidate = Path(path).resolve()
            return pinned.get(candidate, real_sha256_file(candidate))

        monkeypatch.setattr(p0006, "sha256_file", pinned_sha256)
        legacy_paths = (
            "frozen-scientific-resplit.json",
            "archive/split_v3.json",
            "gate01-target-calibrator.json",
            "gate01-prospective-selection.json",
            "gate01-protocol-spec.json",
            "gate01-protocol-lock.json",
            "gate01-producer-spec.json",
            "producer-state/producer-state.json",
            "producer-output/private-build-plan.json",
            "gate01-private-manifest.json",
            "gate01-private-build-state.json",
            "gate01-results.json",
            "gate01-report.md",
            "gate01-result-contract.json",
        )
        rows = [
            {
                "path": (
                    "/content/drive/MyDrive/MRIxFields2026/split_v3.json"
                    if relative_path == "archive/split_v3.json"
                    else "/content/historical/Gate01Private_8012a3f/"
                    + relative_path
                ),
                "sha256": p0006.sha256_file(
                    archive.joinpath(*relative_path.split("/"))
                ),
                "size_bytes": archive.joinpath(
                    *relative_path.split("/")
                ).stat().st_size,
            }
            for relative_path in legacy_paths
        ]
        (nested / "sha256-inventory.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )

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
    manifest_for_load = [_MetadataOnlyGateManifest(archive, cases)]
    monkeypatch.setattr(
        p0006,
        "load_gate01_input_manifest",
        lambda *args, **kwargs: (manifest_for_load[0], metadata),
    )
    preflight = p0006.preflight_gate01_p0006_archive(
        archive,
        expected_gate01_result_sha256=sha256_file(result_path),
    )
    assert preflight["private_array_payloads_opened"] == 0
    assert preflight["verified_gate01_result_file_sha256"] == result_file_sha256
    assert "result_sha256" not in result
    manifest_for_load[0] = gate_manifest
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
    assert protocol["gate01_result"] == {
        "path": str(result_path.resolve()),
        "file_sha256": result_file_sha256,
        "internal_self_hash_defined": False,
    }
    assert "result_sha256" not in json.dumps(protocol, sort_keys=True)
    assert len(protocol["case_receipts"]) == 60
    assert protocol["contract_version"].endswith("protocol-v4")
    expected_layout = (
        p0006.GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1
        if inventory_format == "csv"
        else p0006.GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2
    )
    assert protocol["archive_identity"]["layout_contract"] == expected_layout
    assert protocol["archive_identity"]["inventory_format"] == inventory_format
    assert protocol["archive_inventory"][
        "stored_absolute_paths_trusted_for_file_access"
    ] is False
    assert protocol["archive_inventory"]["verified_entries"]
    assert protocol["supplemental_dependencies"]["included_in_archive_inventory"] is False
    if inventory_format == "reviewed_legacy_json":
        assert len(protocol["supplemental_dependencies"]["verified_dependencies"]) == 2

    progress_events: list[dict[str, object]] = []
    progress_output = tmp_path / "p0006-protocol-with-progress.json"
    protocol_with_progress = p0006.import_gate01_p0006_evaluation_protocol(
        archive,
        expected_gate01_result_sha256=sha256_file(result_path),
        bank_dir=tmp_path / "bank",
        validation_plan_path=plan_path,
        output_path=progress_output,
        progress_callback=progress_events.append,
    )
    assert protocol_with_progress == protocol
    assert sha256_file(progress_output) == sha256_file(output)
    assert progress_events[0]["status"] == "start"
    assert any(event["status"] == "periodic" for event in progress_events)
    assert progress_events[-1]["status"] == "end"
    allowed_progress_fields = {
        "stage",
        "status",
        "verified_inventory_entry_count",
        "train_record_count",
        "validation_record_count",
        "case_count",
        "expected_case_count",
        "acquisition_node_count",
    }
    assert all(set(event) <= allowed_progress_fields for event in progress_events)
    assert all(
        not any(
            token in key
            for key in event
            for token in ("path", "sha", "identity", "subject")
        )
        for event in progress_events
    )

    load_progress: list[dict[str, object]] = []
    reloaded, paired_cases, baselines = p0006.load_gate01_p0006_evaluation_protocol(
        output, progress_callback=load_progress.append
    )
    assert reloaded["protocol_sha256"] == protocol["protocol_sha256"]
    assert len(paired_cases) == len(baselines) == 60
    assert {case.subject_group_identity for case in paired_cases} == {"P:0006"}
    assert all(case.source_provenance["data_role"] == protocol["data_role"] for case in paired_cases)
    assert load_progress[0] == {
        "stage": "p0006_load",
        "status": "start",
        "case_count": 0,
    }
    assert load_progress[-1]["status"] == "end"
    assert load_progress[-1]["case_count"] == 60
    protocol_file_sha256 = sha256_file(output)
    reloaded_without_progress, _, _ = p0006.load_gate01_p0006_evaluation_protocol(
        output
    )
    assert reloaded_without_progress == reloaded
    assert sha256_file(output) == protocol_file_sha256

    if inventory_format == "csv":
        v3_body = copy.deepcopy(protocol)
        v3_body.pop("protocol_sha256")
        v3_body["contract_version"] = p0006.GATE01_P0006_EVALUATION_PROTOCOL_V3
        v3_body["gate01_result"].pop("internal_self_hash_defined")
        v3_body["gate01_result"]["result_sha256"] = "0" * 64
        old_entries = sorted(
            [
                {
                    "basename": entry["basename"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in v3_body["archive_inventory"]["verified_entries"]
            ],
            key=lambda entry: entry["basename"],
        )
        old_normalized_sha = sha256_json(old_entries)
        v3_body["archive_inventory"]["verified_entries"] = old_entries
        v3_body["archive_inventory"]["normalized_inventory_sha256"] = old_normalized_sha
        v3_body["archive_inventory"]["file_access_derivation"] = (
            "verified_basename_as_direct_logical_root_child"
        )
        v3_body["archive_identity"]["normalized_inventory_sha256"] = old_normalized_sha
        v3_body["archive_identity"].pop("supplemental_dependency_count")
        v3_body["archive_identity"].pop("supplemental_dependencies_sha256")
        v3_body.pop("supplemental_dependencies")
        v3_path = tmp_path / "compatible-v3-p0006-protocol.json"
        _write_json(v3_path, {**v3_body, "protocol_sha256": sha256_json(v3_body)})
        compatible_v3, compatible_v3_cases, _ = (
            p0006.load_gate01_p0006_evaluation_protocol(v3_path)
        )
        assert compatible_v3["contract_version"] == (
            p0006.GATE01_P0006_EVALUATION_PROTOCOL_V3
        )
        assert len(compatible_v3_cases) == 60

    v2_body = dict(protocol)
    v2_body.pop("protocol_sha256")
    v2_body["contract_version"] = p0006.GATE01_P0006_EVALUATION_PROTOCOL_V2
    v2_body["gate01_result"] = dict(v2_body["gate01_result"])
    v2_body["gate01_result"].pop("internal_self_hash_defined")
    v2_body["gate01_result"]["result_sha256"] = "0" * 64
    v2_path = tmp_path / "compatible-v2-p0006-protocol.json"
    _write_json(v2_path, {**v2_body, "protocol_sha256": sha256_json(v2_body)})
    compatible, compatible_cases, _ = p0006.load_gate01_p0006_evaluation_protocol(
        v2_path
    )
    assert compatible["contract_version"] == p0006.GATE01_P0006_EVALUATION_PROTOCOL_V2
    assert len(compatible_cases) == 60

    malformed_v4 = copy.deepcopy(protocol)
    malformed_v4.pop("protocol_sha256")
    malformed_v4["gate01_result"]["result_sha256"] = "0" * 64
    malformed_v4_path = tmp_path / "malformed-v4-result-provenance.json"
    _write_json(
        malformed_v4_path,
        {**malformed_v4, "protocol_sha256": sha256_json(malformed_v4)},
    )
    with pytest.raises(ValueError, match="result provenance is malformed"):
        p0006.load_gate01_p0006_evaluation_protocol(malformed_v4_path)

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
