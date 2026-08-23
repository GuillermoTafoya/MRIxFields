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
from fieldbridge.training.stage2_unified import UnifiedStage2Config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "stage2_unified_retrospective_full_model_colab.ipynb"


def test_complete_operator_notebook_is_unexecuted_and_ordered() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    for identity in (
        "f6a19d7a31c4c3bb73edd92088ea078192e88ee4b276309bad81c548ab7f94d5",
        "cbe885f73a307065418ea80296d6cfd6d634edeb3281f503cf90e149800409e7",
        "569a17b1316a47a0c95c42c649f4aab61f8fe8c9cf7d0582c411f544c9b23173",
        "454747cd3e4b1376855915244a7c40fe281b758150e86f584fbea96f94d531f5",
    ):
        assert identity in source
    required = [
        "fit-stage2-photometry",
        "audit-stage2-photometry",
        "preflight-photometry-factored-latent-bank",
        "build-photometry-factored-latent-bank",
        "audit-photometry-factored-latent-bank",
        "preflight-stage2-factored-domain-separability",
        "audit-stage2-retrospective-pair-feasibility",
        "train-stage2-unified",
        "eval-stage2-unified",
    ]
    positions = [source.index(value) for value in required]
    assert positions == sorted(positions)
    assert source.count("AUTHORIZE_LONG_FULL_MODEL = False") == 1
    assert source.count("AUTHORIZE_BACKWARD_ABLATIONS_AFTER_FULL_REVIEW = False") == 1
    assert "RUN_LONG_FULL_AND_BACKWARD_ABLATIONS" not in source
    assert "--steps', '200', '--pilot-steps', '200'" in source
    assert "No ablation is launched by this flag" in source
    assert "Complete paired R/validation manifest with Stage-1 ceilings" not in source
    assert "AUTHORIZE_FIT_AFTER" not in source
    assert "AUTHORIZE_QUALIFICATION_AFTER" not in source
    assert "descriptor_coupling': False" in source
    assert "Path outside reviewed Windows root" in source
    assert "classification_before_array_load" in source
    assert "StarGAN_control_claim': False" in source
    assert "learned_disentanglement_claim': False" in source
    assert source.index("seal-stage2-long-run-evaluation-readiness") < source.index(
        "AUTHORIZE_LONG_FULL_MODEL = False"
    )
    assert "find_latest_stage2_selection_receipt" in source
    assert "--selection-receipt" in source
    assert "FROZEN_VALIDATION_PLAN_SHA256" in source
    assert "selected_best_checkpoint" in source
    assert "final_checkpoint_diagnostic_only" in source
    assert "import-stage2-gate01-p0006-evaluation" in source
    assert "import-stage2-retrospective-paired-evaluation" in source
    assert "Gate01Private_8012a3f" in source
    assert "stage2_unified_validation_plan_v2.json" in source
    assert "required_directed_domain_cell_count" in source
    assert "validation_directed_domain_cell_count': 60" in source
    assert "prospective_training_or_model_selection': False" in source
    assert "development_validation_P0006_evaluation_only" in source
    assert "development/model assessment only; cannot support population or generalization claims" in source
    assert "frozen_and_unused_for_possible_later_confirmation" in source
    assert "'P0009_executed': False" in source
    assert "projected_training_plus_complete_validation" in source
    assert "planned_validation_run_count" in source
    assert "projected_validation_seconds" in source
    assert "measured_complete_validation_directed_domain_cell_count" in source
    assert "projected_total_hours" in source
    assert "projected_total_gpu_cost_usd" in source
    assert "peak_cuda_bytes_across_training_and_validation" in source
    assert source.index("LONG_RUN_AUTHORIZATION_RESOURCE_ESTIMATE") < source.index(
        "AUTHORIZE_LONG_FULL_MODEL = False"
    )
    prohibited_p0006_claims = re.compile(
        r"(?:P:0006|P0006).{0,80}(?:held.?out|final|blind|independent.?test|definitive)"
        r"|(?:held.?out|final|blind|independent.?test|definitive).{0,80}(?:P:0006|P0006)",
        re.IGNORECASE,
    )
    assert prohibited_p0006_claims.search(source) is None
    assert "MATERIALIZED_VALIDATION_ARRAYS_RAW" not in source
    assert "BASELINE_SOURCE_ARTIFACT_RAW" not in source
    assert "stage2_unified_full_retrospective_v2.yaml" not in source
    assert 'stage2_unified_full_retrospective_v7.yaml' in source
    assert "output_root / 'stage2_unified_v7'" in source
    assert 'fine_grained_full_volume_v1' in source
    assert 'outer_full_decoder_checkpoint' in source
    assert 'one_step_anatomy_memory_qualification' in source
    assert 'anatomy_cuda_peak_memory' in source
    assert 'generator_optimizer_updates_per_step' in source
    assert '_memory_checkpointed_decode' not in source


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
    assert config.precision == 'bf16'
    assert config.integration_steps == 4
    assert config.integration_solver == 'heun'
    assert config.decoder_activation_checkpoint_mode == 'fine_grained_full_volume_v1'
    assert payload['training']['decoder_activation_checkpoint'][
        'outer_full_decoder_checkpoint'
    ] == 'forbidden'
    assert payload['training']['generator_gradient_accumulation'][
        'optimizer_updates_per_step'
    ] == 1
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


def test_baseline_manifest_rejects_p_before_array_open(monkeypatch, tmp_path: Path) -> None:
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
