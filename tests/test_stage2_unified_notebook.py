from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.photometry_factorization import sha256_json
from fieldbridge.evaluation.stage2_photometry_baseline import PairedEvaluationCase
from fieldbridge.evaluation.stage2_unified import _load_baseline_predictions
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
    P0006_EVIDENCE_LIMITATION,
    P0009_CONFIRMATION_STATUS,
)
from fieldbridge.training.stage2_unified import UnifiedStage2Config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "stage2_unified_retrospective_full_model_colab.ipynb"
OPERATOR = ROOT / "notebooks" / "stage2_gate01_legacy_recovery_operator.py"


def _notebook_source() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    return payload, source


def test_recovery_notebook_is_unexecuted_output_free_and_ast_valid() -> None:
    payload, source = _notebook_source()
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells
    )
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
    assert "TRAINING_EVIDENCE_COMMIT = '82633d66e5ea47f96b149ea22cc192fcf4526f06'" in source
    pin = re.search(r"OPERATOR_IMPLEMENTATION_COMMIT = '([^']+)'", source)
    assert pin
    assert pin.group(1) == "__OPERATOR_IMPLEMENTATION_COMMIT__" or re.fullmatch(
        r"[0-9a-f]{40}", pin.group(1)
    )
    assert "input(" not in source
    assert "drive.mount" not in source
    assert "stage2_gate01_legacy_recovery_operator.py" in source


def test_operator_scope_guard_separates_commits_and_rejects_training_diff() -> None:
    _, source = _notebook_source()
    for required in (
        "merge-base",
        "--is-ancestor",
        "TRAINING_EVIDENCE_COMMIT",
        "OPERATOR_IMPLEMENTATION_COMMIT",
        "src/fieldbridge/evaluation/stage2_unified_gate01_p0006.py",
        "src/fieldbridge/evaluation/stage2_unified_preflight.py",
        "src/fieldbridge/models/",
        "src/fieldbridge/training/",
        "configs/",
        "operator_diff_touches_training_critical_code",
        "training_critical_code_byte_identical",
    ):
        assert required in source


def test_gate01_metadata_preflight_precedes_every_expensive_recovery_operation() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    ast.parse(source)
    topology_preflight = source.index("stage2_drive_layout = drive_retry")
    preflight = source.index("gate01_preflight = drive_retry")
    pair_preflight = source.index("pair_feasibility = drive_retry")
    assert source.index("preflight_gate01_p0006_archive(", preflight) > preflight
    for operation in (
        "bank_restore = drive_retry",
        "completed_evidence = verify_completed_stage2_pilot_evidence",
        "subprocess.Popen",
        "import-stage2-gate01-p0006-evaluation",
        "seal-stage2-long-run-evaluation-readiness",
    ):
        assert preflight < source.index(operation)
    assert source.index("drive.mount") < topology_preflight < preflight < pair_preflight
    assert pair_preflight < source.index("bank_restore = drive_retry")
    assert "resolve-exact-stage2-drive-layout" in source
    assert "Early step-200 selection-receipt file SHA-256 mismatch" in source
    assert "private_array_payloads_opened" in source
    assert "early_gate01_metadata_preflight" in source


def test_fresh_colab_dependency_import_preflight_precedes_drive_and_artifacts() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    dependency_preflight = source.index("dependency_versions = {}")
    drive_mount = source.index('drive.mount("/content/drive")')
    gate_preflight = source.index("gate01_preflight = drive_retry")
    assert dependency_preflight < drive_mount < gate_preflight
    for dependency in ('"numpy"', '"scipy"', '"torch"', '"yaml"'):
        assert dependency in source[dependency_preflight:drive_mount]
    assert "import fieldbridge.cli" in source[dependency_preflight:drive_mount]
    assert "fresh_colab_dependency_import_preflight" in source
    assert '"packages_installed_or_downloaded": False' in source


def test_recovery_operator_uses_actual_drive_topology_and_completed_namespace() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    assert 'GATE01_PRIVATE_ARCHIVE_ROOT = DRIVE_ROOT / "Gate01Private_8012a3f"' in source
    assert (
        'GATE01_PRIVATE_ARCHIVE_ROOT = DRIVE_ROOT / "Gate01Private_8012a3f" / "archive"'
        not in source
    )
    assert 'OUTPUT_ROOT = DRIVE_ROOT / "UnifiedStage2_1ca2b4a_01"' in source
    assert 'STAGE2_V7_ROOT = OUTPUT_ROOT / "stage2_unified_v7"' in source
    assert 'BANK_NAMESPACE = STAGE2_V7_ROOT / "bank_8081ce89a0ea"' in source
    assert 'TRAINING_NAMESPACE = BANK_NAMESPACE / "implementation_82633d66e5ea"' in source
    assert 'BANK_ARCHIVE = OUTPUT_ROOT / "photometry_factored_latent_bank_v2.tar"' in source
    assert (
        'PAIR_FEASIBILITY = OUTPUT_ROOT / "stage2_retrospective_pair_feasibility_v2.json"'
        in source
    )
    assert (
        'UNRECEIPTED_BANK_DIRECTORY = OUTPUT_ROOT / "photometry_factored_latent_bank_v2"'
        in source
    )
    assert 'STAGE2_V7_ROOT = DRIVE_ROOT / "stage2_unified_v7"' not in source
    assert "unique_existing_directory" not in source
    assert "unique_existing(" not in source
    assert "attempt-0001" in source
    assert "stage2_unified_full_selection_step000000200.json" in source
    assert "c8d73fec48815224fcb87333dfd093c15738cc41dce89c4fb8ccf2cd874ef828" in source
    assert "3afca2bab6a440529f88e7c8d9a9294fed9ecbf07eea1e308ed0910e2ba16421" in source
    assert "fd15be634185a29d5ddedec3f2d7a24527bf5e59a49731f101f62cafcf1b06d6" in source


def test_recovery_operator_pins_and_verifies_reviewed_bank_tar() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    for required in (
        "78d323c02ceccdfcb054307da3c9e14575210869d22cade6c5ecd4afa4baf8d5",
        "f9cb09bfa177a3e389f87f087b0d756a2709e2054559a39c85e8272d5e1cfaa3",
        "8081ce89a0eac1522b4fb28cd7919de4a4ecf1d5af72552d141a0ee9b9944194",
        "EXPECTED_BANK_FILE_COUNT = 3312",
        "EXPECTED_BANK_TOTAL_BYTES = 12873486620",
        "restore_verified_stage2_bank_tar",
        "ignored_empty_unreceipted_bank_directory",
        "resolve_stage2_recovery_drive_layout",
    ):
        assert required in source


def test_recovery_operator_reuses_without_training_and_writes_new_namespace_only() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    assert "verify_completed_stage2_pilot_evidence" in source
    assert '"training_reused": True' in source
    assert '"training_invoked": False' in source
    assert "recovery_training_" in source
    assert '"training_namespace_read_only": True' in source
    assert "TRAINING_NAMESPACE.mkdir" not in source
    assert "train-stage2-unified" not in source
    assert "--device" not in source
    assert "cuda" not in source.casefold()
    assert '"pip"' not in source.casefold()
    assert "pip install" not in source.casefold()
    assert "restore_verified_stage2_bank_tar" in source
    assert "restore_bank_archive_to_scratch" not in source
    assert "drive_retry" in source
    assert "archive_no_clobber" in source
    assert "refuse_existing=True" in source
    assert "Existing recovery receipt changed" in source


def test_recovery_operator_preserves_scientific_role_and_stops_before_long_run() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    for required in (
        "P0006_DEVELOPMENT_VALIDATION_DATA_ROLE",
        "P0006_EVIDENCE_LIMITATION",
        "population_or_generalization_claims_authorized",
        '"P0009_executed": False',
        '"descriptor_coupling": False',
        '"learned_disentanglement_claim": False',
        '"StarGAN_control_claim": False',
        "AUTHORIZE_100K_TRAINING = False",
        "AUTHORIZE_LONG_FULL_MODEL = False",
        "AUTHORIZE_BACKWARD_ABLATIONS_AFTER_FULL_REVIEW = False",
        "STOP_FOR_RESOURCE_BOUNDED_TRAINING_DESIGN_REVIEW",
    ):
        assert required in source
    assert (
        P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
        == "development_validation_P0006_evaluation_only"
    )
    assert P0006_EVIDENCE_LIMITATION == (
        "development/model assessment only; cannot support population or "
        "generalization claims"
    )
    assert (
        P0009_CONFIRMATION_STATUS
        == "frozen_and_unused_for_possible_later_confirmation"
    )


def test_full_config_uses_reviewed_initial_weights() -> None:
    import yaml

    payload = yaml.safe_load(
        (ROOT / "configs/experiment/stage2_unified_full_retrospective_v7.yaml").read_text()
    )
    config = UnifiedStage2Config.from_mapping(payload)
    assert config.loss_weights == {
        "sb": 1.0,
        "identity": 0.1,
        "anatomy": 0.02,
        "graph": 0.01,
        "adversarial": 0.05,
        "domain": 0.1,
    }
    assert config.batch_size == 1
    assert config.precision == "bf16"
    assert config.integration_steps == 4
    assert config.integration_solver == "heun"
    assert config.decoder_activation_checkpoint_mode == "fine_grained_full_volume_v1"
    assert payload["training"]["decoder_activation_checkpoint"][
        "outer_full_decoder_checkpoint"
    ] == "forbidden"
    assert payload["training"]["generator_gradient_accumulation"][
        "optimizer_updates_per_step"
    ] == 1
    accumulation = payload["training"]["generator_gradient_accumulation"]
    assert accumulation["contract"] == "stage2-unified-term-wise-recomputation-v6"
    assert accumulation["graph_construction"] == "one_term_at_a_time"
    assert accumulation["backward"] == "immediate_without_retain_graph"
    assert accumulation["saved_tensor_policy"] == "save_on_cpu"
    assert accumulation["gradient_measurement_scope"] == "pilot_steps_only"
    assert accumulation["long_run_hook_measurement"] == "disabled_after_pilot"
    assert payload["training"]["pilot"]["a100_peak_allocated_limit_bytes"] == (
        72 * 1024**3
    )
    assert config.critic_space == "latent"
    assert config.pilot_steps == 200
    assert config.validation_complete_inventory is True
    assert config.validation_plan_seed == 20260818
    assert config.critic_spectral_normalization is False
    assert config.critic_lazy_r1 is False
    assert payload["training"]["pilot"]["projection_scope"] == (
        "training_plus_complete_validation"
    )
    assert payload["development_validation_evidence"] == {
        "P0006_data_role": "development_validation_P0006_evaluation_only",
        "interpretation": "development/model assessment only",
        "population_or_generalization_claims_authorized": False,
        "P0009_status": "frozen_and_unused_for_possible_later_confirmation",
        "P0009_execution": "forbidden",
    }
    obsolete = dict(payload)
    obsolete["contract"] = "stage2-unified-retrospective-full-model-config-v4"
    with pytest.raises(ValueError, match="obsolete or unsupported"):
        UnifiedStage2Config.from_mapping(obsolete)
    with pytest.raises(ValueError, match="obsolete"):
        UnifiedStage2Config.from_mapping(
            {"training": {"sanity": {"steps": 20}, "loss_weights": config.loss_weights}}
        )


def test_baseline_manifest_rejects_p_before_array_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = PairedEvaluationCase(
        case_identity="R_case",
        source=torch.zeros(1, 2, 2, 2),
        target=torch.zeros(1, 2, 2, 2),
        source_domain=Domain(0.1, Contrast.T1W),
        target_domain=Domain(3.0, Contrast.T1W),
        subject_group_identity="R_subject",
        source_provenance={"case_id": "R_case"},
        target_provenance={"case_id": "R_target"},
        stage1_reconstruction=torch.zeros(1, 2, 2, 2),
    )
    body = {
        "contract_version": "stage2-unified-retrospective-baseline-predictions-v1",
        "cases": [
            {
                "case_identity": "R_case",
                "record_identity": "P_any_identity",
                "metadata_prefix": "P",
                "cohort": "P",
                "split": "validation",
            }
        ],
    }
    payload = dict(body)
    payload["manifest_sha256"] = sha256_json(body)
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    opened = False

    def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("array opened")

    monkeypatch.setattr("numpy.load", forbidden)
    with pytest.raises(ValueError):
        _load_baseline_predictions(path, [case])
    assert opened is False
