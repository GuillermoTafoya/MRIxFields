from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fieldbridge.cli import main
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.evaluation.stage2_gate01 import (
    GATE01_INPUT_CONTRACT_VERSION,
    frozen_artifact_provenance,
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
        "target": np.full((4, 4, 4), 0.4, dtype=np.float32),
        "identity": np.full((4, 4, 4), 0.8, dtype=np.float32),
        "sb": np.full((4, 4, 4), 0.5, dtype=np.float32),
        "ceiling": np.full((4, 4, 4), 0.41, dtype=np.float32),
    }
    for array in arrays.values():
        array[0, 0, 0] = 0.0
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array, allow_pickle=False)

    case = {
        "case_id": "synthetic-case",
        "source_domain": Domain(0.1, "T1w").to_dict(),
        "target_domain": Domain(7.0, "T1w").to_dict(),
        "target": "target.npy",
        "raw_identity": "identity.npy",
        "raw_sb_v2": "sb.npy",
        "stage1_reconstruction_ceiling": "ceiling.npy",
    }
    if forbidden_field is not None:
        case[forbidden_field] = [1, 2, 3]
    payload = {
        "contract_version": GATE01_INPUT_CONTRACT_VERSION,
        "evidence_scope": {
            "role": "synthetic development evidence",
            "traveller": "synthetic-traveller",
            "private_data_run": False,
        },
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "artifact_provenance": frozen_artifact_provenance(),
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

    json_out = tmp_path / "result.json"
    markdown_out = tmp_path / "report.md"
    contract_out = tmp_path / "contract.json"
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
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["written"]) == {"json", "markdown", "contract"}
    assert json_out.is_file()
    assert markdown_out.is_file()
    assert contract_out.is_file()
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


def test_gate01_modules_are_importable_from_the_package() -> None:
    from fieldbridge.evaluation import stage2_gate01
    from fieldbridge.evaluation import stage2_gate01_calibration

    assert stage2_gate01.GATE01_CONTRACT_VERSION.endswith("v1")
    assert stage2_gate01_calibration.GATE01_CALIBRATOR_CONTRACT_VERSION.endswith("v1")
