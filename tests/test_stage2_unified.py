from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import fieldbridge.data.photometry_factored_bank_dataset as factored_reader
import fieldbridge.training.stage2_unified as unified
from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentRecord,
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import sha256_json
from fieldbridge.models.discriminators import (
    DOMAIN_COUNT,
    DomainProjectionDiscriminator,
    domain_labels,
    supported_critic_input,
)
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.stage2_unified import (
    DEFAULT_UNIFIED_WEIGHTS,
    UNIFIED_SELECTION_RULE,
    UNIFIED_SELECTION_RULE_SHA256,
    UnifiedStage2Config,
    anatomy_preservation_components,
    build_unified_validation_plan,
    directed_domain_macro_means,
    find_latest_stage2_selection_receipt,
    graph_consistency_loss,
    integrate_transport,
    load_stage2_selection_receipt,
    run_stage2_unified_train,
    unified_validation_selection_score,
)


class TinyTranslator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 4, 1)

    def forward(self, z, source_domains, target_domains, t):
        del source_domains, target_domains
        time = torch.as_tensor(t, device=z.device, dtype=z.dtype).reshape(-1, 1, 1, 1, 1)
        return self.conv(z) + time * 0.01


class FieldVelocity(nn.Module):
    def forward(self, z, source_domains, target_domains, t):
        del t
        values = [target.field_strength_t - source.field_strength_t for source, target in zip(source_domains, target_domains)]
        return torch.tensor(values, device=z.device, dtype=z.dtype).reshape(-1, 1, 1, 1, 1).expand_as(z)


class OOMTranslator(TinyTranslator):
    def forward(self, z, source_domains, target_domains, t):
        raise torch.OutOfMemoryError("synthetic oom")


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv3d(4, 1, 1)

    def decode(self, z, domains):
        del domains
        return torch.sigmoid(self.projection(z))


class DecoderSpy(TinyDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def decode(self, z, domains):
        self.calls.append(z.detach().clone())
        return super().decode(z, domains)


class SyntheticFactoredIndex:
    artifact_sha256 = "a" * 64

    def __init__(self, split: str = "train") -> None:
        self.split = split
        self.records = []
        self._latents = []
        self._supports = []
        for contrast in Contrast:
            for field in FIELD_STRENGTHS_T:
                for subject in (("s1", "s2") if split == "train" else ("v1", "v2")):
                    domain = Domain(field, contrast)
                    identity = f"R_{contrast.value}_{field:g}_{subject}"
                    self.records.append(
                        FactoredLatentRecord(
                            case_id=identity,
                            subject_group_id=subject,
                            domain=domain,
                            split=split,
                            path=Path(f"{identity}.pt"),
                            resume_key="r" * 64,
                            sidecar={"cohort": "R", "split": split},
                        )
                    )
                    generator = torch.Generator().manual_seed(len(self.records))
                    self._latents.append(torch.randn(4, 4, 4, 4, generator=generator))
                    self._supports.append(torch.ones(4, 4, 4, dtype=torch.bool))

    def load_batch(self, indices):
        records = [self.records[index] for index in indices]
        return (
            torch.stack([self._latents[index] for index in indices]),
            torch.stack([self._supports[index] for index in indices])[:, None],
            [record.domain for record in records],
            records,
        )


def _stats() -> FactoredLatentStats:
    return FactoredLatentStats(
        mean=torch.zeros(4),
        std=torch.ones(4),
        supported_count=torch.full((4,), 100, dtype=torch.int64),
        artifact_sha256="b" * 64,
    )


def _nontrivial_stats() -> FactoredLatentStats:
    return FactoredLatentStats(
        mean=torch.tensor([2.0, -1.0, 0.5, 3.0]),
        std=torch.tensor([0.5, 2.0, 1.5, 4.0]),
        supported_count=torch.full((4,), 100, dtype=torch.int64),
        artifact_sha256="c" * 64,
    )


def _config(tmp_path: Path, **overrides) -> UnifiedStage2Config:
    value = UnifiedStage2Config(
        steps=1,
        batch_size=1,
        device="cpu",
        precision="fp32",
        integration_steps=1,
        anatomy_pool_scales=(1,),
        anatomy_support_erosion=0,
        critic_channels=(4,),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every_steps=1,
        checkpoint_max_bytes=20_000_000,
        history_jsonl=tmp_path / "history.jsonl",
        pilot_steps=0,
        validation_every_steps=1,
        loss_weights=dict(DEFAULT_UNIFIED_WEIGHTS),
    )
    return replace(value, **overrides)


def test_shared_critic_covers_all_15_domains_and_support() -> None:
    domains = [Domain(field, contrast) for contrast in Contrast for field in FIELD_STRENGTHS_T]
    labels = domain_labels(domains, len(domains), torch.device("cpu"))
    assert sorted(labels.tolist()) == list(range(DOMAIN_COUNT))
    tensor = torch.randn(15, 4, 4, 4, 4)
    support = torch.ones(15, 1, 4, 4, 4, dtype=torch.bool)
    critic_input = supported_critic_input(tensor, support)
    critic = DomainProjectionDiscriminator(5, (4,))
    score, logits = critic(critic_input, domains)
    assert score.shape == (15,)
    assert logits.shape == (15, 15)


def test_path_composition_telescopes_for_field_velocity() -> None:
    translator = FieldVelocity()
    z = torch.zeros(1, 4, 3, 3, 3)
    support = torch.ones(1, 1, 3, 3, 3, dtype=torch.bool)
    source = [Domain(0.1, Contrast.T1W)]
    middle = [Domain(3.0, Contrast.T1W)]
    target = [Domain(7.0, Contrast.T1W)]
    loss, direct, composed = graph_consistency_loss(
        translator, z, source, middle, target, support, steps=2, solver="heun"
    )
    assert torch.equal(direct, composed)
    assert loss.item() == 0.0
    same = integrate_transport(translator, z, source, source, steps=2, solver="heun")
    assert torch.equal(same, z)


def test_anatomy_loss_ignores_unsupported_arbitrary_values() -> None:
    source = torch.zeros(1, 1, 9, 9, 9)
    source[:, :, 3:6, 3:6, 3:6] = 0.5
    support = torch.zeros(1, 1, 9, 9, 9, dtype=torch.bool)
    support[:, :, 2:7, 2:7, 2:7] = True
    prediction = source.clone()
    outside = ~support
    prediction[outside] = 1.0e8
    components = anatomy_preservation_components(
        source, prediction, support, pool_scales=(1,), support_erosion=1
    )
    assert components["total"].item() == 0.0


def test_stats_force_unsupported_cells_to_zero() -> None:
    stats = _stats()
    latent = torch.zeros(4, 2, 2, 2)
    latent[:, 0] = 1.0e12
    support = torch.ones(2, 2, 2, dtype=torch.bool)
    support[0] = False
    normalized = stats.normalize(latent, support)
    assert torch.equal(normalized[:, 0], torch.zeros_like(normalized[:, 0]))


def test_image_critic_decoder_receives_denormalized_latent_with_nontrivial_stats() -> None:
    stats = _nontrivial_stats()
    decoder = DecoderSpy().requires_grad_(False)
    latent = torch.randn(2, 4, 3, 3, 3)
    support = torch.ones(2, 1, 3, 3, 3, dtype=torch.bool)
    domains = [Domain(0.1, Contrast.T1W), Domain(3.0, Contrast.T1W)]
    view = unified._critic_view(latent, support, domains, decoder, stats, "image")
    assert torch.equal(decoder.calls[0], stats.denormalize(latent))
    assert view.shape[1] == 2


def test_real_fake_critic_views_have_identical_support_and_mask_only_score() -> None:
    stats = _nontrivial_stats()
    support = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    support[:, :, 1:3, 1:3, 1:3] = True
    real = torch.randn(1, 4, 4, 4, 4)
    fake = torch.randn(1, 4, 4, 4, 4)
    domains = [Domain(3.0, Contrast.T2W)]
    decoder = DecoderSpy().requires_grad_(False)
    real_view = unified._critic_view(real, support, domains, decoder, stats, "latent")
    fake_view = unified._critic_view(fake, support, domains, decoder, stats, "latent")
    assert torch.equal(real_view[:, -1:], fake_view[:, -1:])
    real_outside = real_view[:, :-1].masked_select(~support.expand_as(real))
    fake_outside = fake_view[:, :-1].masked_select(~support.expand_as(fake))
    assert torch.equal(real_outside, torch.zeros_like(real_outside))
    assert torch.equal(fake_outside, torch.zeros_like(fake_outside))
    mask_only_real = real_view[:, -1:].mean(dim=(1, 2, 3, 4))
    mask_only_fake = fake_view[:, -1:].mean(dim=(1, 2, 3, 4))
    assert torch.equal(mask_only_real, mask_only_fake)


def test_every_generator_term_has_finite_nonzero_gradient_and_decoder_stays_frozen(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    translator = TinyTranslator()
    decoder = DecoderSpy().requires_grad_(False)
    decoder_before = {name: value.clone() for name, value in decoder.state_dict().items()}
    critic = DomainProjectionDiscriminator(5, (4,))
    stats = _nontrivial_stats()
    source = torch.randn(1, 4, 5, 5, 5)
    target = torch.randn(1, 4, 5, 5, 5) + 0.75
    support = torch.ones(1, 1, 5, 5, 5, dtype=torch.bool)
    source_domain = Domain(0.1, Contrast.T1W)
    target_domain = Domain(3.0, Contrast.T1W)
    record = FactoredLatentRecord(
        case_id="R_fixture",
        subject_group_id="R:fixture",
        domain=source_domain,
        split="train",
        path=tmp_path / "unused.pt",
        resume_key="d" * 64,
        sidecar={"cohort": "R", "split": "train"},
    )
    batch = unified._TrainingBatch(
        source,
        support,
        [source_domain],
        target,
        support.clone(),
        [target_domain],
        [record],
        [replace(record, case_id="R_target", domain=target_domain)],
    )
    cfg = _config(tmp_path, checkpoint_dir=None, history_jsonl=None)
    expected_generated = integrate_transport(
        translator,
        source,
        [source_domain],
        [target_domain],
        steps=cfg.integration_steps,
        solver=cfg.integration_solver,
    ).detach()
    generator_optimizer = torch.optim.AdamW(translator.parameters(), lr=1.0e-4)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1.0e-4)
    row = unified._train_step(
        cfg,
        translator,
        critic,
        decoder,
        generator_optimizer,
        critic_optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        batch,
        torch.Generator().manual_seed(17),
        stats,
        qualify_term_gradients=True,
    )
    for term in DEFAULT_UNIFIED_WEIGHTS:
        assert torch.isfinite(torch.tensor(row[f"raw/{term}"]))
        assert row[f"gradient/term_{term}"] > 0.0
        disabled = dict(DEFAULT_UNIFIED_WEIGHTS)
        disabled[term] = 0.0
        norms = unified.generator_term_gradient_norms(
            {name: (translator.conv.weight * (position + 1)).sum() for position, name in enumerate(DEFAULT_UNIFIED_WEIGHTS)},
            translator,
            disabled,
        )
        assert norms[term] == 0.0
    assert torch.equal(decoder.calls[0], stats.denormalize(source))
    assert torch.allclose(decoder.calls[1], stats.denormalize(expected_generated))
    assert all(parameter.grad is None for parameter in decoder.parameters())
    for name, value in decoder_before.items():
        assert torch.equal(value, decoder.state_dict()[name])


def test_graph_loss_backpropagates_through_direct_and_composed_paths() -> None:
    torch.manual_seed(43)
    translator = TinyTranslator()
    latent = torch.randn(1, 4, 4, 4, 4)
    support = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
    loss, direct, composed = graph_consistency_loss(
        translator,
        latent,
        [Domain(0.1, Contrast.T1W)],
        [Domain(3.0, Contrast.T1W)],
        [Domain(7.0, Contrast.T1W)],
        support,
        steps=1,
        solver="euler",
    )
    direct.retain_grad()
    composed.retain_grad()
    loss.backward()
    assert direct.grad is not None and float(direct.grad.abs().sum()) > 0
    assert composed.grad is not None and float(composed.grad.abs().sum()) > 0
    assert translator.conv.weight.grad is not None
    assert float(translator.conv.weight.grad.abs().sum()) > 0


def test_full_objective_pilot_reports_gradients_gan_runtime_and_cost(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        pilot_steps=2,
        pilot_smoothing_window=2,
        gpu_hourly_cost_usd=2.5,
    )
    rows = []
    for step in range(2):
        row = {
            "weighted/generator_total": 2.0 - step * 0.1,
            "weighted/aux_to_flow_ratio": 0.25,
            "gradient/generator_norm": 1.0,
            "gradient/critic_norm": 1.0,
            "critic/score_saturation_fraction": 0.0,
            "critic/real_score_mean": 0.5,
            "critic/fake_score_mean": -0.5,
            "critic/score_separation": 1.0,
            "critic/real_domain_accuracy": 0.5,
            "critic/generated_domain_accuracy": 0.25,
            "step_seconds": 2.0,
            "peak_cuda_bytes": 1234,
        }
        for branch in ("real", "fake"):
            for quantile in ("p05", "p50", "p95"):
                row[f"critic/{branch}_score_{quantile}"] = 0.1
        for term, weight in cfg.loss_weights.items():
            row[f"raw/{term}"] = 1.0
            row[f"weighted/{term}"] = weight
            row[f"gradient/term_{term}"] = 0.5
        rows.append(row)
    report = unified._pilot_report(rows, cfg)
    assert report["status"] == "pass"
    assert set(report["term_gradient_norms"]) == set(DEFAULT_UNIFIED_WEIGHTS)
    assert report["critic"]["real_score_distribution"]["p50"] == 0.1
    assert report["runtime"]["projected_seconds"] == 200_000.0
    assert report["runtime"]["projected_cost_usd"] == pytest.approx(138.8888889)


def test_validation_plan_is_step_and_variant_independent_and_freezes_all_draws() -> None:
    index = SyntheticFactoredIndex("validation")
    first = build_unified_validation_plan(index, validation_seed=20260818)
    second = build_unified_validation_plan(index, validation_seed=20260818)
    assert first == second
    assert first["derivation"]["training_step_dependency"] is False
    assert first["derivation"]["model_or_variant_dependency"] is False
    assert len(first["entries"]) == len(index.records) * 4
    assert first["required_directed_domain_cell_count"] == 60
    assert len(first["directed_domain_cell_counts"]) == 60
    assert set(first["directed_domain_cell_counts"].values()) == {2}
    assert all(value == 4 for value in first["source_usage_counts"].values())
    cells = set(first["directed_domain_cell_counts"])
    assert "T1w:0.1T->7T" in cells
    assert "T1w:7T->0.1T" in cells
    assert all(
        entry["source_subject_group_identity"]
        != entry["target_subject_group_identity"]
        and 0.0 < entry["bridge_t"] < 1.0
        and isinstance(entry["noise_seed"], int)
        for entry in first["entries"]
    )
    changed = build_unified_validation_plan(index, validation_seed=20260819)
    assert changed["validation_plan_sha256"] != first["validation_plan_sha256"]


def test_directed_domain_macro_means_prevent_record_count_dominance() -> None:
    cells = ["T1w:0.1T->3T", "T1w:3T->0.1T"]
    balanced = [
        {"directed_domain_cell": cells[0], "sb": 0.0, "identity": 0.0},
        {"directed_domain_cell": cells[1], "sb": 10.0, "identity": 2.0},
    ]
    imbalanced = [balanced[0], *([balanced[1]] * 99)]
    balanced_macro, _, balanced_weighted = directed_domain_macro_means(
        balanced, required_cells=cells
    )
    imbalanced_macro, _, imbalanced_weighted = directed_domain_macro_means(
        imbalanced, required_cells=cells
    )
    assert balanced_macro == imbalanced_macro == {"identity": 1.0, "sb": 5.0}
    assert balanced_weighted["sb"] == 5.0
    assert imbalanced_weighted["sb"] == pytest.approx(9.9)


def test_validation_draws_are_identical_at_every_checkpoint_step(tmp_path: Path) -> None:
    index = SyntheticFactoredIndex("validation")
    cfg = _config(tmp_path, checkpoint_dir=None, history_jsonl=None)
    plan = build_unified_validation_plan(index, validation_seed=cfg.validation_plan_seed)
    translator = TinyTranslator()
    critic = DomainProjectionDiscriminator(5, (4,))
    decoder = TinyDecoder().requires_grad_(False)
    first = unified._evaluate_unpaired_validation(
        cfg, translator, critic, decoder, index, plan, _stats(), torch.device("cpu"), step=1
    )
    last = unified._evaluate_unpaired_validation(
        cfg,
        translator,
        critic,
        decoder,
        index,
        plan,
        _stats(),
        torch.device("cpu"),
        step=100_000,
    )
    assert first["validation_plan_sha256"] == last["validation_plan_sha256"]
    assert first["means"] == last["means"]
    assert first["selection_score"] == last["selection_score"]


def test_selection_rule_is_critic_independent_and_meaningful_for_sb_only() -> None:
    means = {"sb": 1.0, "identity": 2.0, "anatomy": 3.0, "graph": 4.0}
    score = unified_validation_selection_score(
        {**means, "generated_domain_correct": 0.0, "real_score": -1.0e6}
    )
    changed_critic = unified_validation_selection_score(
        {**means, "generated_domain_correct": 1.0, "real_score": 1.0e6}
    )
    assert score == changed_critic == pytest.approx(1.3)
    assert UNIFIED_SELECTION_RULE["training_critic_inputs"] == []
    assert UNIFIED_SELECTION_RULE["applies_to_all_variants_including_sb_only"] is True


def test_prospective_manifest_is_rejected_before_payload_load(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "artifact_sha256": "a" * 64,
        "records": [
            {
                "path": "P_forbidden.pt",
                "sidecar": {
                    "record_identity": "P_arbitrary",
                    "subject_group_identity": "P_arbitrary",
                    "cohort": "P",
                    "split": "train",
                    "domain": Domain(3.0, Contrast.T1W).to_dict(),
                    "resume_key": "r" * 64,
                },
            }
        ],
    }
    monkeypatch.setattr(factored_reader, "load_photometry_factored_latent_bank_manifest", lambda *a, **k: manifest)
    opened = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("payload opened")

    monkeypatch.setattr(factored_reader, "_load_latent_record", forbidden_open)
    with pytest.raises(ValueError, match="Prospective"):
        PhotometryFactoredLatentBankIndex(tmp_path, "train")
    assert opened is False


def test_full_step_updates_critic_keeps_vae_frozen_and_exactly_resumes(tmp_path: Path) -> None:
    index = SyntheticFactoredIndex()
    decoder = TinyDecoder().requires_grad_(False)
    before = {key: value.clone() for key, value in decoder.state_dict().items()}
    translator = TinyTranslator()
    cfg = _config(tmp_path)
    result = run_stage2_unified_train(
        cfg, translator=translator, decoder=decoder, train_index=index,
        validation_index=SyntheticFactoredIndex("validation"), stats=_stats()
    )
    assert result.completed_steps == 1
    checkpoint = Path(result.checkpoint)
    state = load_checkpoint(checkpoint)
    assert state["training_cursor"] == 1
    assert state["critic"]
    assert state["generator_scheduler"] and state["critic_scheduler"]
    assert state["sampler_rng"].numel() > 0
    assert state["validation_plan_sha256"]
    assert state["selection_rule_sha256"] == UNIFIED_SELECTION_RULE_SHA256
    assert state["validation_selection"]["paired_targets_used"] is False
    assert state["validation_selection"]["best_step"] == 1
    receipt_path, receipt = find_latest_stage2_selection_receipt(
        cfg.checkpoint_dir, variant="full", require_complete=True
    )
    assert receipt_path.name == "stage2_unified_full_selection_step000000001.json"
    assert receipt["best_checkpoint"] == str(checkpoint.resolve())
    assert receipt["validation_plan_sha256"] == state["validation_plan_sha256"]
    assert receipt["checkpoint_hashes"]["best"]["file_sha256"]
    assert receipt["checkpoint_hashes"]["final"] == receipt["checkpoint_hashes"]["latest"]
    assert load_stage2_selection_receipt(receipt_path) == receipt
    for field, message in (
        ("validation_plan_sha256", "another validation plan"),
        ("selection_rule_sha256", "rule provenance"),
    ):
        mutated = dict(receipt)
        mutated.pop("selection_sha256")
        mutated[field] = "f" * 64
        mutated["selection_sha256"] = sha256_json(mutated)
        mutation_path = tmp_path / f"mutated-{field}.json"
        mutation_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_stage2_selection_receipt(mutation_path)
    assert before.keys() == decoder.state_dict().keys()
    for key, value in before.items():
        assert torch.equal(value, decoder.state_dict()[key])
    with pytest.raises(ValueError, match="config/bank/code identity"):
        run_stage2_unified_train(
            replace(cfg, resume_from=checkpoint),
            translator=TinyTranslator(),
            decoder=TinyDecoder().requires_grad_(False),
            train_index=index,
            validation_index=SyntheticFactoredIndex("validation"),
            stats=_stats(),
        )
    restored = TinyTranslator()
    resume_decoder = TinyDecoder()
    resume_decoder.load_state_dict(before)
    resume_decoder.requires_grad_(False)
    resumed = run_stage2_unified_train(
        replace(cfg, resume_from=checkpoint),
        translator=restored,
        decoder=resume_decoder,
        train_index=index,
        validation_index=SyntheticFactoredIndex("validation"),
        stats=_stats(),
    )
    assert resumed.completed_steps == 1
    for key, value in state["translator"].items():
        assert torch.equal(value, restored.state_dict()[key])
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    step_rows = [row for row in rows if "raw/sb" in row]
    assert len(step_rows) == 1
    assert all(name in step_rows[0] for name in ("raw/sb", "raw/identity", "raw/anatomy", "raw/graph", "raw/adversarial", "raw/domain"))
    assert any(row.get("event") == "unpaired_validation" for row in rows)


def test_oom_is_hard_stop_and_diagnostic_is_preserved(tmp_path: Path) -> None:
    with pytest.raises(torch.OutOfMemoryError):
        run_stage2_unified_train(
            _config(tmp_path),
            translator=OOMTranslator(),
            decoder=TinyDecoder().requires_grad_(False),
            train_index=SyntheticFactoredIndex(),
            validation_index=SyntheticFactoredIndex("validation"),
            stats=_stats(),
        )
    row = json.loads((tmp_path / "history.jsonl").read_text().strip())
    assert row["event"] == "oom_hard_stop"
    assert row["fallback"] == "forbidden"


def test_sb_only_backward_ablation_disables_every_auxiliary_path(tmp_path: Path) -> None:
    weights = {name: 0.0 for name in DEFAULT_UNIFIED_WEIGHTS}
    weights["sb"] = 1.0
    validation_index = SyntheticFactoredIndex("validation")
    cfg = _config(tmp_path, loss_weights=weights, variant="sb_only")
    result = run_stage2_unified_train(
        cfg,
        translator=TinyTranslator(),
        decoder=TinyDecoder().requires_grad_(False),
        train_index=SyntheticFactoredIndex(),
        validation_index=validation_index,
        stats=_stats(),
    )
    state = load_checkpoint(Path(result.checkpoint))
    expected_plan = build_unified_validation_plan(
        validation_index, validation_seed=cfg.validation_plan_seed
    )
    assert state["validation_plan_sha256"] == expected_plan["validation_plan_sha256"]
    _, receipt = find_latest_stage2_selection_receipt(
        cfg.checkpoint_dir, variant="sb_only", require_complete=True
    )
    assert receipt["validation_plan_sha256"] == expected_plan["validation_plan_sha256"]
    row = next(
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text().splitlines()
        if "raw/sb" in line
    )
    assert row["graph_path"] == "disabled"
    assert row["gradient/critic_norm"] == 0.0
    assert row["critic/total"] == 0.0
    for term in ("identity", "anatomy", "graph", "adversarial", "domain"):
        assert row[f"raw/{term}"] == 0.0
        assert row[f"weighted/{term}"] == 0.0


def test_interrupted_resume_reproduces_uninterrupted_next_step(tmp_path: Path) -> None:
    index = SyntheticFactoredIndex()
    decoder = TinyDecoder().requires_grad_(False)
    cfg = _config(tmp_path / "uninterrupted", steps=2)
    uninterrupted = TinyTranslator()
    run_stage2_unified_train(
        cfg, translator=uninterrupted, decoder=decoder, train_index=index,
        validation_index=SyntheticFactoredIndex("validation"), stats=_stats()
    )
    step1 = cfg.checkpoint_dir / "stage2_unified_full_step000000001.pt"
    step2 = cfg.checkpoint_dir / "stage2_unified_full_step000000002.pt"
    expected = load_checkpoint(step2)

    resumed_model = TinyTranslator()
    resumed_cfg = replace(
        cfg,
        checkpoint_dir=tmp_path / "resumed" / "checkpoints",
        history_jsonl=cfg.history_jsonl,
        resume_from=step1,
    )
    resumed_result = run_stage2_unified_train(
        resumed_cfg,
        translator=resumed_model,
        decoder=decoder,
        train_index=index,
        validation_index=SyntheticFactoredIndex("validation"),
        stats=_stats(),
    )
    actual = load_checkpoint(Path(resumed_result.checkpoint))
    for component in ("translator", "critic"):
        for name, value in expected[component].items():
            assert torch.equal(value, actual[component][name])
    assert expected["sampler_rng"].equal(actual["sampler_rng"])
    assert expected["training_cursor"] == actual["training_cursor"] == 2
