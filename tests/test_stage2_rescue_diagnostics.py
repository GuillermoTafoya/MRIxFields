from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.evaluation.stage2_rescue_diagnostics import (
    LatentDiagnosticRecord,
    build_rescue_scorecard,
    diagnose_conditioning_plumbing,
    diagnose_off_manifold_latent_drift,
    diagnose_per_term_gradients,
    diagnose_real_domain_identifiability,
    run_synthetic_micro_overfit,
    same_source_five_target_sweep,
    self_hashed,
    validate_data_boundary,
    verify_self_hash,
    write_diagnostic_json,
)
from fieldbridge.models.conditioning import DomainEmbedding
from fieldbridge.models.discriminators import DomainProjectionDiscriminator
from fieldbridge.models.translators.flow_transport import FlowMatchingLatentTranslator
from fieldbridge.training.stage2_unified import UnifiedStage2Config, _TrainingBatch


class TinyConditionedTranslator(nn.Module):
    def __init__(self, *, ignore_target: bool = False) -> None:
        super().__init__()
        self.domain_embedding = DomainEmbedding(
            cond_dim=8, contrast_embedding_dim=2, field_embedding_dim=2
        )
        self.projection = nn.Linear(8, 1)
        self.ignore_target = ignore_target

    def forward(self, x, source, target, t=None):
        used_target = source if self.ignore_target else target
        conditioning = self.domain_embedding(
            source,
            used_target,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        value = self.projection(conditioning)
        return value.reshape(x.shape[0], 1, *([1] * (x.ndim - 2))).expand_as(x)


class ZeroVelocityTranslator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, source, target, t=None):
        return torch.zeros_like(x) + self.anchor * 0


class LearnableAffineVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offsets = nn.Parameter(torch.zeros(len(FIELD_STRENGTHS_T)))

    def forward(self, x, source, target, t=None):
        source_ids = torch.tensor(
            [FIELD_STRENGTHS_T.index(item.field_strength_t) for item in source],
            device=x.device,
        )
        target_ids = torch.tensor(
            [FIELD_STRENGTHS_T.index(item.field_strength_t) for item in target],
            device=x.device,
        )
        velocity = self.offsets[target_ids] - self.offsets[source_ids]
        return velocity.reshape(x.shape[0], 1, 1, 1).expand_as(x)


class IgnoredVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, source, target, t=None):
        return torch.zeros_like(x) + self.anchor * 0


def _stats() -> FactoredLatentStats:
    return FactoredLatentStats(
        mean=torch.zeros(1),
        std=torch.ones(1),
        supported_count=torch.tensor([100]),
        artifact_sha256="a" * 64,
    )


def test_conditioning_swap_and_nonzero_conditioning_gradient() -> None:
    torch.manual_seed(4)
    source = Domain(1.5, Contrast.T1W)
    requested = Domain(3.0, Contrast.T1W)
    alternate = Domain(7.0, Contrast.T1W)
    result = diagnose_conditioning_plumbing(
        TinyConditionedTranslator(),
        torch.randn(1, 1, 4, 4),
        [source],
        [requested],
        [alternate],
        time_values=torch.tensor([0.5]),
    )
    assert result["status"] == "PASS"
    assert result["target_embedding_l2_delta"] > 0
    assert result["conditioning_output_gradient_norm"] > 0
    assert result["translator_observed_target_labels"] == [requested.label]
    verify_self_hash(result)


def test_completely_ignored_target_fails() -> None:
    result = diagnose_conditioning_plumbing(
        TinyConditionedTranslator(ignore_target=True),
        torch.randn(1, 1, 4, 4),
        [Domain(1.5, Contrast.T2W)],
        [Domain(3.0, Contrast.T2W)],
        [Domain(7.0, Contrast.T2W)],
        time_values=torch.tensor([0.5]),
    )
    assert result["status"] == "FAIL"
    assert "target_conditioning_ignored" in result["failures"]


def test_renderer_only_change_is_separated_from_learned_transport() -> None:
    source = Domain(3.0, Contrast.T1W)

    def renderer(decoded: torch.Tensor, target: Domain) -> torch.Tensor:
        return decoded + target.field_strength_t / 10.0

    result = same_source_five_target_sweep(
        ZeroVelocityTranslator(),
        nn.Identity(),
        torch.ones(1, 1, 3, 3),
        torch.ones(1, 1, 3, 3, dtype=torch.bool),
        source,
        _stats(),
        renderer=renderer,
        integration_steps=1,
        solver="euler",
    )
    assert result["learned_target_sensitivity"]["status"] == "FAIL"
    assert result["renderer_only_sensitivity"]["status"] == "PASS"
    assert result["final_target_sensitivity"]["status"] == "PASS"


def test_off_manifold_synthetic_outlier_is_detected() -> None:
    support = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    domain = Domain(3.0, Contrast.T2_FLAIR)
    real = [
        LatentDiagnosticRecord(
            latent=torch.full((1, 1, 4, 4), value),
            support=support,
            source_domain=domain,
            target_domain=domain,
            contrast=domain.contrast,
        )
        for value in (-0.1, -0.05, 0.05, 0.1)
    ]
    generated = [
        LatentDiagnosticRecord(
            latent=torch.full((1, 1, 4, 4), 20.0),
            support=support,
            source_domain=domain,
            target_domain=Domain(7.0, domain.contrast),
            contrast=domain.contrast,
        )
    ]
    result = diagnose_off_manifold_latent_drift(real, generated, _stats())
    assert result["status"] == "FAIL"
    assert result["outlier_fraction"] == 1.0
    assert result["rows"][0]["outlier"] is True


def _identifiability_data(*, separable: bool) -> tuple[np.ndarray, list[Domain]]:
    features = []
    domains = []
    for contrast_index, contrast in enumerate(Contrast):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            for repeat in range(3):
                domains.append(Domain(field, contrast))
                if separable:
                    features.append(
                        [contrast_index * 20.0 + repeat * 0.001, field_index * 5.0]
                    )
                else:
                    features.append([0.0, 0.0])
    return np.asarray(features), domains


def test_domain_classifier_reports_chance_vs_separable_data() -> None:
    separable, domains = _identifiability_data(separable=True)
    good = diagnose_real_domain_identifiability(separable, domains, separable, domains)
    assert good["status"] == "PASS"
    assert good["contrast_probe"]["validation"]["balanced_accuracy"] == 1.0
    chance, chance_domains = _identifiability_data(separable=False)
    bad = diagnose_real_domain_identifiability(chance, chance_domains, chance, chance_domains)
    assert bad["status"] == "FAIL"
    assert bad["contrast_probe"]["validation"]["balanced_accuracy"] == pytest.approx(1 / 3)
    for probe in bad["field_within_contrast_probes"].values():
        assert probe["validation"]["balanced_accuracy"] == pytest.approx(1 / 5)


def test_micro_overfit_success_and_failure() -> None:
    passed = run_synthetic_micro_overfit(
        LearnableAffineVelocity(), steps=160, learning_rate=0.05
    )
    assert passed["status"] == "PASS", passed
    failed = run_synthetic_micro_overfit(IgnoredVelocity(), steps=10)
    assert failed["status"] == "FAIL"
    assert "known_target_transform_not_learned" in failed["failures"]


def test_data_boundary_refuses_p_for_fitting_and_always_refuses_p0009() -> None:
    with pytest.raises(ValueError, match="Prospective records"):
        validate_data_boundary(
            [{"cohort": "P", "split": "test", "subject_id": "P:0006"}],
            purpose="fit",
        )
    with pytest.raises(ValueError, match="P:0009"):
        validate_data_boundary(
            [{"cohort": "P", "split": "test", "subject_id": "P:0009"}],
            purpose="final_p0006",
        )
    accepted = validate_data_boundary(
        [{"cohort": "P", "split": "test", "subject_id": "P:0006"}],
        purpose="final_p0006",
    )
    assert accepted["accepted_count"] == 1


def test_per_term_gradients_use_real_recomputation_without_optimizer_step() -> None:
    torch.manual_seed(12)
    source_domain = Domain(1.5, Contrast.T1W)
    target_domain = Domain(3.0, Contrast.T1W)
    source = torch.randn(1, 1, 4, 4, 4)
    target = source + 0.2
    support = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
    batch = _TrainingBatch(
        source=source,
        source_support=support,
        source_domains=[source_domain],
        target=target,
        target_support=support,
        target_domains=[target_domain],
        source_records=[SimpleNamespace(case_id="R/train/source")],
        target_records=[SimpleNamespace(case_id="R/train/target")],
    )
    translator = FlowMatchingLatentTranslator(
        latent_channels=1,
        hidden_channels=(4,),
        bottleneck_channels=8,
        cond_dim=8,
        time_embed_dim=8,
        spatial_dims=3,
        use_norm=False,
    )
    critic = DomainProjectionDiscriminator(2, (4,))
    cfg = UnifiedStage2Config(
        batch_size=1,
        integration_steps=1,
        integration_solver="euler",
        precision="fp32",
        critic_channels=(4,),
        anatomy_pool_scales=(1,),
        anatomy_support_erosion=0,
        decoder_activation_checkpoint_mode="disabled",
    )
    result = diagnose_per_term_gradients(
        cfg,
        translator,
        critic,
        nn.Identity(),
        batch,
        _stats(),
        maximum_exact_cosine_parameters=1_000_000,
        gradient_epsilon=0.0,
    )
    assert result["optimizer_step_called"] is False
    assert result["graphs_retained_across_terms"] is False
    assert set(result["terms"]) == {
        "sb",
        "identity",
        "anatomy",
        "graph",
        "adversarial",
        "domain",
    }
    assert all(term["finite"] for term in result["terms"].values())
    assert result["critic"]["status"] == "PASS"


def test_diagnostic_json_exact_resume_no_clobber_and_self_hash(tmp_path) -> None:
    path = tmp_path / "diagnostic.json"
    payload = {"contract_version": "test-v1", "status": "PASS", "value": 1}
    write_diagnostic_json(path, payload)
    stored = json.loads(path.read_text(encoding="utf-8"))
    verify_self_hash(stored)
    write_diagnostic_json(path, payload, resume=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_diagnostic_json(path, payload)
    with pytest.raises(FileExistsError, match="not an exact resume"):
        write_diagnostic_json(path, {**payload, "value": 2}, resume=True)


def test_scorecard_keeps_human_verdict_and_separates_renderer() -> None:
    base = self_hashed({"contract_version": "test", "status": "PASS"})
    sweep = self_hashed(
        {
            "contract_version": "sweep",
            "status": "PASS",
            "learned_target_sensitivity": {"status": "FAIL"},
            "renderer_only_sensitivity": {"status": "PASS"},
        }
    )
    scorecard = build_rescue_scorecard(
        {
            "conditioning_plumbing": base,
            "real_domain_identifiability": base,
            "target_sweep": sweep,
            "off_manifold_drift": base,
            "gradient_health": base,
            "micro_overfit": base,
        }
    )
    assert scorecard["architecture_verdict"] == "human_decision_required"
    assert scorecard["rows"]["learned_target_sensitivity"]["status"] == "FAIL"
    assert scorecard["rows"]["renderer_only_sensitivity"]["status"] == "PASS"
    assert scorecard["current_step200_promotion_authorized"] is False


def test_rescue_notebook_is_unexecuted_and_delegates_to_operator() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "stage2_rescue_diagnostics_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    assert "stage2_rescue_diagnostics_operator.py" in code
    assert "exec(compile(operator.read_text" in code
    assert "drive.mount('/content/drive')" in code
    assert "git', 'checkout', '--detach'" in code
    assert "nvidia-smi" in code
    assert "optimizer.step" not in code
    assert "integrate_transport" not in code


def test_rescue_operator_uses_reviewed_paths_and_new_no_clobber_root() -> None:
    root = Path(__file__).resolve().parents[1]
    operator = (
        root / "notebooks" / "stage2_rescue_diagnostics_operator.py"
    ).read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/MRIxFields2026" in operator
    assert "Gate01Private_8012a3f" in operator
    assert "UnifiedStage2_1ca2b4a_01" in operator
    assert '"stage2_unified_v7"' in operator
    assert '"bank_8081ce89a0ea"' in operator
    assert '"implementation_82633d66e5ea"' in operator
    assert '"photometry_factored_latent_bank_v2.tar"' in operator
    assert '"stage2_photometry_factorization_v1.json"' in operator
    assert '"vae_kl_vae_best.pt"' in operator
    assert '"stage1-run-c.yaml"' in operator
    assert 'Path("/content/stage2_gate01_recovery_v8_scratch")' in operator
    assert '"stage2_rescue_2026_09_01" / "diagnostics_v1"' in operator
    assert "DIAGNOSTIC_ROOT.mkdir" in operator
    assert "TRAINING_NAMESPACE / \"stage2_rescue" not in operator
