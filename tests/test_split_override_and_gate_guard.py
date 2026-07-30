"""A resplit must actually take effect, and the gate must refuse contaminated subjects.

Both defend the same failure: the bank bakes each record's split into its manifest at build
time, so a resplit is a silent no-op for every Stage-2 consumer unless the split is passed
explicitly — and with the `nn` coupling a training traveller retrieves its own target-field
volume, so scoring it measures memorization while looking identical to a clean read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fieldbridge.data.latent_bank_dataset import LatentBankIndex, split_assignment_from_json


def _bank(root: Path) -> Path:
    bank = root / "bank"
    records = []
    for split, case_id in (("train", "P_T1W_0.1T_0006"), ("train", "P_T1W_7.0T_0006"),
                           ("train", "R_T1W_0.1T_0100"), ("test", "P_T1W_0.1T_0009")):
        (bank / split).mkdir(parents=True, exist_ok=True)
        path = bank / split / f"{case_id}.pt"
        torch.save({"case_id": case_id, "subject_id": case_id.rsplit("_", 1)[1], "split": split,
                    "domain": {"field_strength_t": 0.1, "contrast": "T1w"},
                    "latent": torch.zeros(2, 2, 2, 2)}, path)
        records.append({"case_id": case_id, "subject_id": case_id.rsplit("_", 1)[1],
                        "split": split, "path": f"{split}/{case_id}.pt",
                        "domain": {"field_strength_t": 0.1, "contrast": "T1w"}})
    (bank / "latent_bank_manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    return bank


def _split_json(root: Path, moved_to_validation: list[str]) -> Path:
    def rec(case_id: str) -> dict:
        return {"case_id": case_id, "subject_id": case_id.rsplit("_", 1)[1],
                "image_path": f"/data/{case_id}.nii.gz",
                "domain": {"field_strength_t": 0.1, "contrast": "T1w"}}

    train = [c for c in ("P_T1W_0.1T_0006", "P_T1W_7.0T_0006", "R_T1W_0.1T_0100")
             if c not in moved_to_validation]
    path = root / "split.json"
    path.write_text(json.dumps({"splits": {
        "train": [rec(c) for c in train],
        "validation": [rec(c) for c in moved_to_validation],
        "test": [rec("P_T1W_0.1T_0009")],
    }}), encoding="utf-8")
    return path


def test_without_the_override_a_resplit_is_silently_ignored(tmp_path: Path) -> None:
    bank = _bank(tmp_path)
    _split_json(tmp_path, ["P_T1W_0.1T_0006", "P_T1W_7.0T_0006"])

    index = LatentBankIndex(bank, "train")  # no split_json

    assert "P_T1W_0.1T_0006" in {r.case_id for r in index.records}


def test_the_override_relocates_records_without_touching_the_bank(tmp_path: Path) -> None:
    bank = _bank(tmp_path)
    split = _split_json(tmp_path, ["P_T1W_0.1T_0006", "P_T1W_7.0T_0006"])

    train = LatentBankIndex(bank, "train", split_json=split)
    validation = LatentBankIndex(bank, "validation", split_json=split)

    assert {r.case_id for r in train.records} == {"R_T1W_0.1T_0100"}
    assert {r.case_id for r in validation.records} == {"P_T1W_0.1T_0006", "P_T1W_7.0T_0006"}
    # The latent files never moved; only the membership did.
    assert all(r.path.is_file() for r in validation.records)
    assert all("train" in str(r.path) for r in validation.records)


def test_override_reports_an_empty_split_instead_of_training_on_nothing(tmp_path: Path) -> None:
    bank = _bank(tmp_path)
    split = _split_json(tmp_path, [])

    with pytest.raises(ValueError, match="no records for split 'validation'"):
        LatentBankIndex(bank, "validation", split_json=split)


def test_split_assignment_reads_all_three_arrays(tmp_path: Path) -> None:
    split = _split_json(tmp_path, ["P_T1W_0.1T_0006"])

    assignment = split_assignment_from_json(split)

    assert assignment["P_T1W_0.1T_0006"] == "validation"
    assert assignment["R_T1W_0.1T_0100"] == "train"
    assert assignment["P_T1W_0.1T_0009"] == "test"


def _traveller_records(subject: str, fields: tuple[float, ...]) -> list:
    from fieldbridge.data.contracts import VolumeRecord
    from fieldbridge.data.domains import Domain

    return [
        VolumeRecord(
            case_id=f"P_T1W_{f}T_{subject}", subject_id=subject,
            image_path=Path(f"/data/P_T1W_{f}T_{subject}.nii.gz"),
            domain=Domain(field_strength_t=f, contrast="T1w"),
        )
        for f in fields
    ]


def _gate_bank(root: Path, records) -> tuple[Path, dict]:
    bank = root / "gate_bank"
    (bank / "x").mkdir(parents=True, exist_ok=True)
    manifest = {"records": []}
    for record in records:
        path = bank / "x" / f"{record.case_id}.pt"
        torch.save({"latent": torch.zeros(2, 2, 2, 2)}, path)
        manifest["records"].append({"case_id": record.case_id, "path": f"x/{record.case_id}.pt"})
    return bank, manifest


def _run_gate(tmp_path: Path, split_of_case: dict, **kwargs):
    from fieldbridge.data.latent_bank_dataset import LatentStats
    from fieldbridge.evaluation.stage2_transport_eval import (
        DecodeSpec, TransportSamplerConfig, evaluate_transport_travellers,
    )

    records = _traveller_records("0006", (0.1, 7.0))
    bank, manifest = _gate_bank(tmp_path, records)

    class _Id(torch.nn.Module):
        downsample_factor = 1

        def forward(self, z, *a, **k):
            return torch.zeros_like(z)

        def decode(self, z, domain):
            return z[:, :1]

    return evaluate_transport_travellers(
        translator=_Id(), decoder=_Id(), records=records, bank_manifest=manifest,
        bank_dir=bank, stats=LatentStats(mean=torch.zeros(2), std=torch.ones(2)),
        sampler=TransportSamplerConfig(solver="euler", n_steps=1),
        decode=DecodeSpec(precision="float32", strategy="full"),
        device=torch.device("cpu"), metrics=("nrmse",),
        volume_loader=lambda record: torch.zeros(1, 1, 2, 2, 2),
        split_of_case=split_of_case, log=False, **kwargs,
    )


def test_gate_refuses_a_subject_from_the_training_split(tmp_path: Path) -> None:
    split_of_case = {f"P_T1W_{f}T_0006": "train" for f in (0.1, 7.0)}

    with pytest.raises(ValueError, match="Refusing to score subject"):
        _run_gate(tmp_path, split_of_case)


def test_gate_allows_it_only_behind_the_explicit_flag(tmp_path: Path) -> None:
    split_of_case = {f"P_T1W_{f}T_0006": "train" for f in (0.1, 7.0)}

    result = _run_gate(tmp_path, split_of_case, allow_training_subjects=True)

    assert result["scored_training_subjects"] == ["0006"]
    assert result["subject_splits"] == {"0006": "train"}


def test_gate_runs_clean_on_a_held_out_subject_and_records_provenance(tmp_path: Path) -> None:
    split_of_case = {f"P_T1W_{f}T_0006": "validation" for f in (0.1, 7.0)}

    result = _run_gate(tmp_path, split_of_case)

    assert result["scored_training_subjects"] == []
    assert result["subject_splits"] == {"0006": "validation"}
    assert result["num_pairs"] == 2
