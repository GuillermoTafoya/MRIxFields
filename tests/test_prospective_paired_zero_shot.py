import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data.preprocessing import (
    SliceGeometry,
    SlicePreprocessingSpec,
    selected_slice_indices,
)
from fieldbridge.evaluation.prospective_paired import (
    ALL_FIELDS,
    CASE_IDS,
    EVIDENCE_SCOPE,
    SELECTED_SLICE_INDICES,
    AcquisitionGeometry,
    LoadedAcquisition,
    aggregate_rows,
    assert_sanitized_handoff,
    compute_paired_metrics,
    conditioning_margins,
    error_improvement_map,
    fixed_edge_map,
    sanitized_handoff,
    select_required_acquisitions,
    validate_checkpoint_contract,
    validate_paired_geometry,
    validate_preprocessed_geometry,
)
from fieldbridge.official.data_manifest import MRIxFieldsDataRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "prospective_paired_zero_shot_v1.yaml"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "prospective_paired_zero_shot_colab.ipynb"
RUNNER_PATH = PROJECT_ROOT / "notebooks" / "prospective_paired_zero_shot_runner.py"
FROZEN_COMMIT = "e1e526ea5fa0a58f5682823f85a3957d5cc8647c"


def _record(case_id: str, field: float, *, suffix: str = "") -> MRIxFieldsDataRecord:
    label = f"{field:g}T"
    return MRIxFieldsDataRecord(
        sample_id=f"sample-{case_id}-{label}{suffix}",
        split_name="Training_prospective",
        cohort="prospective",
        is_paired=True,
        prefix="P",
        modality="T2FLAIR",
        internal_modality="T2-FLAIR",
        field=label,
        field_value=field,
        subject_id=case_id,
        domain_id=0,
        relative_path=f"private-{case_id}-{label}{suffix}",
        raw_uri=f"private-{case_id}-{label}{suffix}",
        filename=f"P_T2FLAIR_{label}_{case_id}.nii.gz",
    )


def _records() -> list[MRIxFieldsDataRecord]:
    return [_record(case, field) for case in CASE_IDS for field in ALL_FIELDS]


def _geometry(*, affine_x: float = 1.0) -> AcquisitionGeometry:
    affine = np.eye(4)
    affine[0, 0] = affine_x
    return AcquisitionGeometry.from_arrays(
        shape=(8, 9, 300),
        affine=affine,
        orientation=("R", "A", "S"),
        voxel_sizes=(affine_x, 1.0, 1.0),
    )


def test_config_freezes_scope_indices_fit_pad_and_has_no_thresholds() -> None:
    config = load_yaml_config(CONFIG_PATH)
    audit = config["audit"]
    preprocessing = SlicePreprocessingSpec.from_mapping(config["preprocessing"])

    assert audit["evidence_scope"] == EVIDENCE_SCOPE
    assert audit["complete_volume"] is False
    assert audit["evidence_role"] == "observed_development_not_confirmatory"
    assert tuple(audit["case_ids"]) == CASE_IDS
    assert tuple(audit["selected_slice_indices"]) == SELECTED_SLICE_INDICES
    assert selected_slice_indices(preprocessing) == SELECTED_SLICE_INDICES
    assert preprocessing.resize_mode == "fit_pad"
    assert (preprocessing.output_height, preprocessing.output_width) == (128, 160)
    assert preprocessing.normalization == "official_01"
    assert config["checkpoint_contract"] == {
        "git_commit": FROZEN_COMMIT,
        "trainer": "pseudo_pair_epochs",
        "model_class": "ConditionalResidualUNetFieldTranslator",
        "pseudo_pair_pipeline_version": 2,
        "epoch": 10,
        "global_step": 160,
    }
    assert config["reporting"]["scientific_thresholds"] is None
    assert config["reporting"]["training_experiment"] == "not_implemented"


def test_selection_requires_every_unique_paired_acquisition() -> None:
    selected = select_required_acquisitions(_records())
    assert tuple(selected) == CASE_IDS
    assert all(tuple(case) == ALL_FIELDS for case in selected.values())

    with pytest.raises(ValueError, match="found 0"):
        select_required_acquisitions(_records()[:-1])
    duplicate = _records() + [_record("0006", 1.5, suffix="-duplicate")]
    with pytest.raises(ValueError, match="found 2"):
        select_required_acquisitions(duplicate)
    reused_uri = replace(_record("0006", 1.5), raw_uri=_record("0006", 0.1).raw_uri)
    records = [
        record
        for record in _records()
        if not (record.subject_id == "0006" and record.field_value == 1.5)
    ]
    with pytest.raises(ValueError, match="duplicate raw_uri"):
        select_required_acquisitions([*records, reused_uri])


def test_physical_and_fit_pad_geometry_fail_closed() -> None:
    tensor = torch.zeros(1, 8, 9, 300)
    loaded = {field: LoadedAcquisition(tensor, _geometry()) for field in ALL_FIELDS}
    validate_paired_geometry(loaded)

    changed = dict(loaded)
    changed[7.0] = LoadedAcquisition(tensor, _geometry(affine_x=2.0))
    with pytest.raises(ValueError, match="affine.*voxel_sizes"):
        validate_paired_geometry(changed)

    geometry = SliceGeometry(
        slice_index=72,
        original_height=8,
        original_width=9,
        resized_height=8,
        resized_width=9,
        output_height=128,
        output_width=160,
        pad_top=60,
        pad_bottom=60,
        pad_left=75,
        pad_right=76,
        resize_mode="fit_pad",
    )
    validate_preprocessed_geometry(geometry, geometry)
    with pytest.raises(ValueError, match="SliceGeometry"):
        validate_preprocessed_geometry(geometry, replace(geometry, pad_left=74))


def test_checkpoint_contract_verifies_all_frozen_identity_fields() -> None:
    training = {
        "epochs": 10,
        "batch_size": 4,
        "checkpoint_dir": "historical",
        "resume_from": None,
    }
    recorded = {
        "epochs": 10,
        "batch_size": 4,
        "checkpoint_dir": "private",
        "resume_from": None,
    }
    contract = {
        "git_commit": FROZEN_COMMIT,
        "trainer": "pseudo_pair_epochs",
        "model_class": "ConditionalResidualUNetFieldTranslator",
        "pseudo_pair_pipeline_version": 2,
        "epoch": 10,
        "global_step": 160,
    }
    state = {
        **{key: value for key, value in contract.items() if key != "git_commit"},
        "pseudo_pair_config": recorded,
        "model": {},
        "_meta": {"git_commit": FROZEN_COMMIT, "config": dict(recorded)},
    }
    validate_checkpoint_contract(state, contract, historical_training_config={"training": training})
    for key, bad_value in {
        "model_class": "OtherModel",
        "pseudo_pair_pipeline_version": 1,
        "epoch": 9,
        "global_step": 159,
    }.items():
        changed = dict(state)
        changed[key] = bad_value
        with pytest.raises(ValueError, match=key):
            validate_checkpoint_contract(
                changed,
                contract,
                historical_training_config={"training": training},
            )
    changed = dict(state)
    changed["_meta"] = {**state["_meta"], "git_commit": "0" * 40}
    with pytest.raises(ValueError, match="Git commit"):
        validate_checkpoint_contract(
            changed,
            contract,
            historical_training_config={"training": training},
        )


def test_metrics_maps_edges_and_conditioning_margins_are_deterministic() -> None:
    source = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.2, 0.4], [0.0, 0.2, 0.4]]]])
    target = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.4, 0.8], [0.0, 0.4, 0.8]]]])
    prediction = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.3, 0.7], [0.0, 0.3, 0.7]]]])
    foreground = (target > 0).float()
    outside = 1.0 - foreground
    source_metrics = compute_paired_metrics(source, target, source, foreground, outside)
    prediction_metrics = compute_paired_metrics(prediction, target, source, foreground, outside)

    assert source_metrics["prediction_minus_source_residual_magnitude"] == 0.0
    assert prediction_metrics["masked_mae"] == pytest.approx(0.1)
    assert prediction_metrics["signed_foreground_bias"] == pytest.approx(-0.1)
    expected = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.1, 0.3], [0.0, 0.1, 0.3]]]])
    assert torch.allclose(error_improvement_map(source, prediction, target), expected)
    assert fixed_edge_map(torch.nn.functional.pad(source, (2, 2, 2, 2))).shape == (1, 1, 7, 7)
    margins = conditioning_margins(prediction_metrics, source_metrics)
    assert margins["masked_mae"] > 0
    assert margins["signed_foreground_bias"] > 0


def test_hierarchical_aggregation_equal_weights_each_level() -> None:
    rows = []
    for case_slot, base in (("case_01", 1.0), ("case_02", 3.0)):
        for field in ("1.5T", "3T"):
            for offset in (0.0, 2.0):
                metric = {"masked_mae": base + offset}
                rows.append(
                    {
                        "case_slot": case_slot,
                        "target_field": field,
                        "source": metric,
                        "correct": metric,
                        "wrong_mean": metric,
                        "margins_mean": metric,
                    }
                )
    aggregate = aggregate_rows(rows)
    assert aggregate["per_case"]["case_01"]["1.5T"]["correct"]["masked_mae"] == 2.0
    assert aggregate["per_target_field"]["1.5T"]["correct"]["masked_mae"] == 3.0
    assert aggregate["macro"]["correct"]["masked_mae"] == 3.0


def test_sanitized_handoff_excludes_identifiers_paths_images_and_checkpoint_paths() -> None:
    contract = {
        "git_commit": FROZEN_COMMIT,
        "model_class": "ConditionalResidualUNetFieldTranslator",
        "pseudo_pair_pipeline_version": 2,
        "epoch": 10,
        "global_step": 160,
    }
    payload = sanitized_handoff(
        checkpoint_contract=contract,
        aggregate={"macro": {"correct": {"nrmse": 0.1}}},
        counts={"cases": 3},
        target_conditioning_sweep={"1.5T": {"3T": {"nrmse": 0.2}}},
    )
    assert_sanitized_handoff(payload)
    text = json.dumps(payload).lower()
    for forbidden in ("case_id", "subject_id", "raw_uri", ".nii", ".png", ".pt"):
        assert forbidden not in text
    with pytest.raises(ValueError, match="forbidden"):
        assert_sanitized_handoff({**payload, "image_path": "/private/a.png"})


def test_notebook_is_unexecuted_and_preflights_gpu_before_clone_or_install() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert code_cells
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"prospective_cell_{index}", "exec")
    assert (
        source.index("nvidia-smi")
        < source.index("git', 'clone")
        < source.index("'pip', 'install'")
    )
    assert "fetch', 'origin'" in source
    assert "checkout', '--detach', EXPECTED_AUDIT_COMMIT" in source
    assert "actual != EXPECTED_AUDIT_COMMIT" in source
    assert "REPO_DIR not in PACKAGE_FILE.parents" in source
    assert "Training" not in source


def test_runner_contains_no_training_or_percentile_normalization_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "torch.inference_mode()" in source
    assert "preprocess_volume_slice" in source
    assert "apply_model_range=False" in source
    assert "fit_pad" in source
    assert "percentile" not in source.lower()
    assert "optimizer" not in source.lower()
    assert "backward(" not in source
    # The canonical definition lives in the contract module, not duplicated here.
    assert "abs(source-target) - abs(prediction-target)" not in source
    assert "fixed 50/50 overlay" in source
    assert "correct_vs_requested_margin" in source
