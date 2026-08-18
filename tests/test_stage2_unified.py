from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import fieldbridge.data.photometry_factored_bank_dataset as factored_reader
from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentRecord,
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.models.discriminators import (
    DOMAIN_COUNT,
    DomainProjectionDiscriminator,
    domain_labels,
    supported_critic_input,
)
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.stage2_unified import (
    DEFAULT_UNIFIED_WEIGHTS,
    UnifiedStage2Config,
    anatomy_preservation_components,
    graph_consistency_loss,
    integrate_transport,
    run_stage2_unified_train,
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


class SyntheticFactoredIndex:
    split = "train"
    artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.records = []
        self._latents = []
        self._supports = []
        for contrast in Contrast:
            for field in FIELD_STRENGTHS_T:
                for subject in ("s1", "s2"):
                    domain = Domain(field, contrast)
                    identity = f"R_{contrast.value}_{field:g}_{subject}"
                    self.records.append(
                        FactoredLatentRecord(
                            case_id=identity,
                            subject_group_id=subject,
                            domain=domain,
                            split="train",
                            path=Path(f"{identity}.pt"),
                            resume_key="r" * 64,
                            sidecar={"cohort": "R", "split": "train"},
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
        sanity_steps=0,
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
        cfg, translator=translator, decoder=decoder, train_index=index, stats=_stats()
    )
    assert result.completed_steps == 1
    checkpoint = Path(result.checkpoint)
    state = load_checkpoint(checkpoint)
    assert state["training_cursor"] == 1
    assert state["critic"]
    assert state["generator_scheduler"] and state["critic_scheduler"]
    assert state["sampler_rng"].numel() > 0
    assert before.keys() == decoder.state_dict().keys()
    for key, value in before.items():
        assert torch.equal(value, decoder.state_dict()[key])
    with pytest.raises(ValueError, match="config/bank/code identity"):
        run_stage2_unified_train(
            replace(cfg, resume_from=checkpoint),
            translator=TinyTranslator(),
            decoder=TinyDecoder().requires_grad_(False),
            train_index=index,
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
        stats=_stats(),
    )
    assert resumed.completed_steps == 1
    for key, value in state["translator"].items():
        assert torch.equal(value, restored.state_dict()[key])
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert all(name in rows[0] for name in ("raw/sb", "raw/identity", "raw/anatomy", "raw/graph", "raw/adversarial", "raw/domain"))


def test_oom_is_hard_stop_and_diagnostic_is_preserved(tmp_path: Path) -> None:
    with pytest.raises(torch.OutOfMemoryError):
        run_stage2_unified_train(
            _config(tmp_path),
            translator=OOMTranslator(),
            decoder=TinyDecoder().requires_grad_(False),
            train_index=SyntheticFactoredIndex(),
            stats=_stats(),
        )
    row = json.loads((tmp_path / "history.jsonl").read_text().strip())
    assert row["event"] == "oom_hard_stop"
    assert row["fallback"] == "forbidden"


def test_sb_only_backward_ablation_disables_every_auxiliary_path(tmp_path: Path) -> None:
    weights = {name: 0.0 for name in DEFAULT_UNIFIED_WEIGHTS}
    weights["sb"] = 1.0
    run_stage2_unified_train(
        _config(tmp_path, loss_weights=weights),
        translator=TinyTranslator(),
        decoder=TinyDecoder().requires_grad_(False),
        train_index=SyntheticFactoredIndex(),
        stats=_stats(),
    )
    row = json.loads((tmp_path / "history.jsonl").read_text().strip())
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
        cfg, translator=uninterrupted, decoder=decoder, train_index=index, stats=_stats()
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
        stats=_stats(),
    )
    actual = load_checkpoint(Path(resumed_result.checkpoint))
    for component in ("translator", "critic"):
        for name, value in expected[component].items():
            assert torch.equal(value, actual[component][name])
    assert expected["sampler_rng"].equal(actual["sampler_rng"])
    assert expected["training_cursor"] == actual["training_cursor"] == 2
