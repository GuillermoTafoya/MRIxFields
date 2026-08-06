from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
import torch

from fieldbridge.cli import main
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.evaluation.stage2_gate01 import (
    GATE01_INPUT_CONTRACT_VERSION,
    canonical_loaded_array_sha256,
    frozen_artifact_provenance,
    gate01_selection_fingerprint,
    load_gate01_input_manifest,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    RESPLIT_FINGERPRINT,
    TrainingTemplateVolume,
    fit_posthoc_target_calibrator,
)


def _write_calibrator(tmp_path):
    records = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            values = torch.linspace(0.0, 0.9, 64).reshape(4, 4, 4)
            values *= 0.5 + 0.02 * contrast_index + 0.03 * field_index
            records.append(
                TrainingTemplateVolume(
                    volume=values,
                    domain=Domain(field, contrast),
                    record_identity=f"template-{contrast.value}-{field:g}",
                )
            )
    calibrator = fit_posthoc_target_calibrator(
        records,
        split_fingerprint=RESPLIT_FINGERPRINT,
        training_cohort_identity="synthetic-retrospective-training",
        code_commit="fit-commit",
        num_quantiles=9,
    )
    return calibrator.save(tmp_path / "calibrator.json")


def _write_manifest(tmp_path, *, forbidden_field: str | None = None):
    arrays = {
        "source": np.full((4, 4, 4), 0.3, dtype=np.float32),
        "target": np.full((4, 4, 4), 0.4, dtype=np.float32),
        "identity": np.full((4, 4, 4), 0.8, dtype=np.float32),
        "sb": np.full((4, 4, 4), 0.5, dtype=np.float32),
        "ceiling": np.full((4, 4, 4), 0.41, dtype=np.float32),
        "wrong": np.full((4, 4, 4), 0.77, dtype=np.float32),
    }
    for array in arrays.values():
        array[0, 0, 0] = 0.0
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array, allow_pickle=False)
    support = arrays["source"] != 0
    np.save(tmp_path / "support.npy", support, allow_pickle=False)

    def reference(name, array):
        return {
            "path": f"{name}.npy",
            "sha256": canonical_loaded_array_sha256(array),
        }

    traveller_hash = hashlib.sha256(b"synthetic-traveller").hexdigest()

    case = {
        "case_id": "synthetic-case",
        "traveller_identity_sha256": traveller_hash,
        "source_domain": Domain(0.1, "T1w").to_dict(),
        "target_domain": Domain(7.0, "T1w").to_dict(),
        "source_image": reference("source", arrays["source"]),
        "source_support_mask": reference("support", support),
        "target": reference("target", arrays["target"]),
        "raw_identity": reference("identity", arrays["identity"]),
        "raw_sb_v2": reference("sb", arrays["sb"]),
        "stage1_reconstruction_ceiling": reference("ceiling", arrays["ceiling"]),
        "wrong_target_sb_v2": {"1.5T": reference("wrong", arrays["wrong"])},
    }
    if forbidden_field is not None:
        case[forbidden_field] = [1, 2, 3]
    payload = {
        "contract_version": GATE01_INPUT_CONTRACT_VERSION,
        "execution_mode": "development-incomplete",
        "selection_fingerprint_sha256": gate01_selection_fingerprint(
            [
                {
                    "case_identity_sha256": hashlib.sha256(
                        b"synthetic-case"
                    ).hexdigest(),
                    "traveller_identity_sha256": traveller_hash,
                    "contrast": "T1w",
                    "source_field_t": 0.1,
                    "target_field_t": 7.0,
                }
            ]
        ),
        "evidence_scope": {
            "role": "synthetic development evidence",
            "traveller_identity_sha256": traveller_hash,
            "private_data_run": False,
        },
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "artifact_provenance": frozen_artifact_provenance(),
        "source_support_contract": {
            "derivation": "abs(source_image)>threshold",
            "threshold": 0.0,
        },
        "cases": [case],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_gate01_cli_prints_json_and_supports_optional_output_files(
    tmp_path, capsys
) -> None:
    calibrator = _write_calibrator(tmp_path)
    manifest = _write_manifest(tmp_path)

    exit_code = main(
        [
            "gate01-equal-photometry",
            "--manifest",
            str(manifest),
            "--calibrator",
            str(calibrator),
            "--metrics",
            "nrmse",
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["contract_version"] == "stage2-gate01-equal-photometry-v1"
    assert printed["num_pairs"] == 1
    assert printed["written"] == {}
    verified = printed["contract"]["verified_loaded_array_sha256"][0]["arrays"]
    assert {
        "target",
        "raw_identity",
        "raw_sb_v2",
        "stage1_reconstruction_ceiling",
        "wrong_target_sb_v2[1.5T]",
    } <= set(verified)

    json_out = tmp_path / "result.json"
    markdown_out = tmp_path / "report.md"
    contract_out = tmp_path / "contract.json"
    montage_dir = tmp_path / "montages"
    exit_code = main(
        [
            "gate01-equal-photometry",
            "--manifest",
            str(manifest),
            "--calibrator",
            str(calibrator),
            "--metrics",
            "nrmse",
            "--device",
            "cpu",
            "--out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--contract-out",
            str(contract_out),
            "--montage-dir",
            str(montage_dir),
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["written"]) == {"json", "markdown", "contract"}
    assert json_out.is_file()
    assert markdown_out.is_file()
    assert contract_out.is_file()
    assert (montage_dir / "montage_manifest.json").is_file()
    assert len(list(montage_dir.glob("*.png"))) == 1
    assert printed["montage_rendering"]["entries"][0]["png_sha256"]
    assert json.loads(json_out.read_text(encoding="utf-8"))["num_pairs"] == 1
    assert "promotion remains **unset" in markdown_out.read_text(encoding="utf-8")


def test_gate01_cli_rejects_target_derived_calibration_fields(tmp_path) -> None:
    calibrator = _write_calibrator(tmp_path)
    manifest = _write_manifest(tmp_path, forbidden_field="target_mask")
    with pytest.raises(ValueError, match="forbids target-derived"):
        main(
            [
                "gate01-equal-photometry",
                "--manifest",
                str(manifest),
                "--calibrator",
                str(calibrator),
                "--metrics",
                "nrmse",
                "--device",
                "cpu",
            ]
        )


@pytest.mark.parametrize("name", ["target", "identity", "sb", "ceiling", "wrong"])
def test_loaded_array_hash_rejects_mutation_after_manifest(tmp_path, name: str) -> None:
    manifest = _write_manifest(tmp_path)
    cases, _ = load_gate01_input_manifest(manifest)
    mutated = np.load(tmp_path / f"{name}.npy", allow_pickle=False)
    mutated[1, 1, 1] += 0.125
    np.save(tmp_path / f"{name}.npy", mutated, allow_pickle=False)

    with pytest.raises(ValueError, match="loaded-array SHA-256 mismatch"):
        list(cases)


def test_manifest_selection_fingerprint_and_source_support_fail_closed(tmp_path) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["selection_fingerprint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection fingerprint"):
        load_gate01_input_manifest(manifest)

    manifest = _write_manifest(tmp_path)
    support = np.load(tmp_path / "support.npy", allow_pickle=False)
    support[1, 1, 1] = False
    np.save(tmp_path / "support.npy", support, allow_pickle=False)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][0]["source_support_mask"]["sha256"] = (
        canonical_loaded_array_sha256(support)
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    cases, _ = load_gate01_input_manifest(manifest)
    with pytest.raises(ValueError, match="does not match its source-derived contract"):
        list(cases)


def test_gate01_modules_are_importable_from_the_package() -> None:
    from fieldbridge.evaluation import stage2_gate01
    from fieldbridge.evaluation import stage2_gate01_calibration
    from fieldbridge.evaluation import stage2_gate01_montage

    assert stage2_gate01.GATE01_CONTRACT_VERSION.endswith("v1")
    assert stage2_gate01_calibration.GATE01_CALIBRATOR_CONTRACT_VERSION.endswith("v2")
    assert stage2_gate01_montage.Gate01MontageCollector
