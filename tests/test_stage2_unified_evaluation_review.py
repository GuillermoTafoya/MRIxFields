from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fieldbridge.evaluation.stage2_unified as evaluation
from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.evaluation.stage2_photometry_baseline import PairedEvaluationCase
from fieldbridge.evaluation.stage2_unified import (
    _all_intermediates,
    _reductions,
    _wrong_target_controls,
    decode_unified_endpoint,
)


class FieldTranslator(nn.Module):
    def forward(self, z, source_domains, target_domains, t):
        del source_domains, t
        values = torch.tensor(
            [domain.field_strength_t for domain in target_domains],
            dtype=z.dtype,
            device=z.device,
        ).reshape(-1, 1, 1, 1, 1)
        return values.expand_as(z) * 0.01


class CaptureDecoder(nn.Module):
    def decode(self, z, domains):
        self.domains = list(domains)
        self.latent = z.clone()
        return z.mean(dim=1, keepdim=True)


class Decoder(nn.Module):
    def decode(self, z, domains):
        del domains
        return z[:, :1]


@dataclass
class Context:
    values: torch.Tensor
    support_mask: torch.Tensor

    def with_values(self, values: torch.Tensor):
        return Context(values, self.support_mask)


class ArtifactSpy:
    def __init__(self) -> None:
        self.rendered_domains: list[str] = []

    def render_target(self, context, domain):
        self.rendered_domains.append(domain.label)
        return context.values


def _stats() -> FactoredLatentStats:
    return FactoredLatentStats(
        mean=torch.tensor([2.0, -1.0, 0.5, 3.0]),
        std=torch.tensor([0.5, 2.0, 1.5, 4.0]),
        supported_count=torch.ones(4, dtype=torch.int64),
        artifact_sha256="a" * 64,
    )


@pytest.mark.parametrize("policy", [None, "historical", "native-exterior"])
def test_wrong_target_mechanistic_control_holds_requested_render_map_fixed(policy) -> None:
    requested = Domain(3.0, Contrast.T1W)
    case = PairedEvaluationCase(
        case_identity="R_pair",
        source=torch.zeros(1, 3, 3, 3),
        target=torch.zeros(1, 3, 3, 3),
        source_domain=Domain(0.1, Contrast.T1W),
        target_domain=requested,
        subject_group_identity="R:s",
        source_provenance={"case_id": "R_source"},
        target_provenance={"case_id": "R_target"},
    )
    artifact = ArtifactSpy()
    decoder_domains = []

    class ConditionedDecoder(nn.Module):
        def decode(self, z, domains):
            decoder_domains.append(domains[0].label)
            return z[:, :1] * 0.01 + domains[0].field_strength_t / 10

    rows = _wrong_target_controls(
        FieldTranslator(),
        ConditionedDecoder(),
        artifact,  # type: ignore[arg-type]
        _stats(),
        torch.zeros(1, 4, 3, 3, 3),
        Context(torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3, dtype=torch.bool)),
        case,
        1,
        "euler",
        torch.device("cpu"),
        decoder_context=policy,
        native_source=torch.zeros(1, 4, 3, 3, 3),
        valid_core=torch.ones(1, 1, 3, 3, 3, dtype=torch.bool),
    )
    if policy is None:
        assert decoder_domains == [row["conditioned_domain"] for row in rows]
    else:
        assert decoder_domains == [requested.label] * 4
        assert [row["decoder_domain"] for row in rows] == decoder_domains
    assert len(rows) == 4
    assert all(row["requested_render_domain"] == requested.label for row in rows)
    if policy is None:
        assert artifact.rendered_domains[::2] == [requested.label] * 4
        assert artifact.rendered_domains[1::2] == [row["conditioned_domain"] for row in rows]
        assert all("condition_native_render_nrmse" in row for row in rows)
    else:
        assert artifact.rendered_domains == [requested.label] * 4
        assert all("condition_native_render_nrmse" not in row for row in rows)


def test_graph_evaluation_enumerates_every_valid_intermediate_field() -> None:
    values = _all_intermediates(
        Domain(0.1, Contrast.T2W), Domain(7.0, Contrast.T2W)
    )
    assert [item.field_strength_t for item in values] == [1.5, 3.0, 5.0]


def test_reductions_include_source_target_domains_and_raw_decoder_leakage() -> None:
    methods = {
        "raw_identity": {"nrmse": 1.0, "ssim": 0.1, "lpips": 0.9},
        "full_unified_model": {"nrmse": 0.5, "ssim": 0.5, "lpips": 0.4},
    }
    row = {
        "source_domain": "0.1T/T1w",
        "target_domain": "3T/T1w",
        "contrast": "T1w",
        "directed_field_pair": "0.1T->3T",
        "raw_identity_stratum": "ordinary",
        "methods": methods,
        "identical_support_methods": methods,
        "requested_vs_wrong_target": {"requested_better_than_every_wrong_target": True},
        "graph_consistency": {"mean_direct_vs_composed_l1": 0.25},
        "anatomy_preservation": {
            "low_mid": 0.1,
            "edge": 0.2,
            "gradient": 0.3,
            "total": 0.2,
        },
        "raw_pre_mask_decoder_background_leakage_mae": 0.75,
    }
    result = _reductions([row])
    assert "per_source_domain" in result and "per_target_domain" in result
    assert (
        result["overall"]["all"]["raw_pre_mask_decoder_background_leakage_mae"]
        == 0.75
    )


@pytest.mark.parametrize("learned", [False, True])
def test_decode_uses_one_endpoint_for_both_exterior_policies(learned):
    from fieldbridge.training.stage2_unified import integrate_transport

    stats = _stats()
    native = torch.arange(2 * 4 * 2**3, dtype=torch.float32).reshape(2, 4, 2, 2, 2) / 8
    core = torch.zeros_like(native[:, :1], dtype=torch.bool)
    core[0, :, 0] = True
    core[1, :, 1] = True
    start = stats.normalize(native, core)
    domains = [Domain(3, Contrast.T1W), Domain(0.1, Contrast.T2W)]
    source_domains = [Domain(0.1, Contrast.T1W), domains[1]]
    endpoint = (
        integrate_transport(FieldTranslator(), start, source_domains, domains, steps=4)
        if learned else start
    )
    source_before, core_before, endpoint_before = native.clone(), core.clone(), endpoint.clone()
    decoder = CaptureDecoder()
    historical = decode_unified_endpoint(decoder, endpoint, domains, stats=stats)
    denormalized = stats.denormalize(endpoint)
    assert torch.equal(decoder.latent, denormalized)
    assert torch.equal(historical, denormalized.mean(1, keepdim=True))
    preserved = decode_unified_endpoint(
        decoder, endpoint, domains, stats=stats, decoder_context="native-exterior",
        native_source=native, valid_core=core,
    )
    expected = torch.where(core, denormalized, native)
    assert torch.equal(decoder.latent, expected)
    assert torch.equal(preserved, expected.mean(1, keepdim=True))
    assert decoder.domains == domains
    assert torch.equal(endpoint, endpoint_before) and torch.equal(native, source_before)
    assert torch.equal(core, core_before)
    if not learned:
        torch.testing.assert_close(decoder.latent, native)


def test_preserving_context_rejects_a_missing_source_or_broadcastable_wrong_mask():
    stats = _stats()
    endpoint = torch.zeros(2, 4, 3, 3, 3)
    kwargs = dict(stats=stats, decoder_context="native-exterior")
    args = Decoder(), endpoint, [Domain(3, Contrast.T1W)] * 2
    with pytest.raises(ValueError, match="original source"):
        decode_unified_endpoint(*args, **kwargs)
    for core in (torch.ones_like(endpoint[:, :1]), torch.ones(1, 1, 3, 3, 3, dtype=torch.bool)):
        with pytest.raises(ValueError, match="boolean"):
            decode_unified_endpoint(*args, native_source=endpoint, valid_core=core, **kwargs)


def test_retrospective_decoder_context_comparison_and_resume(tmp_path, monkeypatch):
    source = torch.full((1, 8, 8, 8), 0.25)
    case = PairedEvaluationCase(
        "pair", source, source, Domain(0.1, Contrast.T1W), Domain(3, Contrast.T1W),
        "R:s", {"case_id": "source"}, {"case_id": "target"}, stage1_reconstruction=source,
    )
    native = source.expand(4, -1, -1, -1).clone()
    core = torch.ones_like(source[0], dtype=torch.bool)
    core[0] = False
    bank = SimpleNamespace(
        split="validation", artifact_sha256="bank", records=[SimpleNamespace(case_id="source")],
        load=lambda _: (native, core),
    )

    class Artifact(ArtifactSpy):
        artifact_sha256 = "photometry"

        def normalize_source(self, volume, domain):
            return Context(volume, volume != 0)

        def render_target(self, context, domain):
            assert bool(((context.values >= 0) & (context.values <= 1)).all())
            return context.values

    def metrics(prediction, target):
        return {"nrmse": float((prediction-target).square().mean().sqrt()), "ssim": 0, "lpips": 0}

    monkeypatch.setattr(evaluation, "_preflight_paired_retrospective_manifest", lambda path: None)
    monkeypatch.setattr(evaluation, "load_paired_evaluation_manifest",
                        lambda *a, **kw: ({"manifest_sha256": "manifest"}, [case]))
    baselines = {"pair": {"gate01_calibrated_identity": source, "original_sb_v2": source}}
    monkeypatch.setattr(evaluation, "_load_baseline_predictions", lambda *a: (baselines, {}))
    monkeypatch.setattr(evaluation, "OfficialQualificationMetrics", lambda **kw: metrics)
    monkeypatch.setattr(evaluation, "_render_montage", lambda path, *a: path.write_text("synthetic"))
    kwargs = dict(
        translator=FieldTranslator(), encoder=None, decoder=Decoder(), artifact=Artifact(),
        bank=bank, stats=_stats(), paired_manifest_path="pairs", baseline_predictions_path="baselines",
        sb_only_translator=FieldTranslator(), ablation_translators={"fixture": FieldTranslator()},
        device="cpu", integration_steps=2,
    )
    def run(policy, **extra):
        return evaluation.evaluate_stage2_unified(
            **kwargs, decoder_context=policy, output_dir=tmp_path / policy, **extra
        )

    historical = run("historical")
    preserved = run("native-exterior")
    assert historical["run_contract_sha256"] != preserved["run_contract_sha256"]
    assert preserved["result_sha256"] == run("native-exterior", resume=True)["result_sha256"]
    for name in ("full_unified_model", "zero_transport", "photometry_factored_sb_only",
                 "ablation_fixture"):
        assert (historical["cases"][0]["methods"][name]["nrmse"]
                > preserved["cases"][0]["methods"][name]["nrmse"])
    assert preserved["cases"][0]["methods"]["zero_transport"]["nrmse"] == 0
    with pytest.raises(ValueError, match="exact-resume contract mismatch"):
        evaluation.evaluate_stage2_unified(
            **kwargs, decoder_context="historical", output_dir=tmp_path / "native-exterior",
            resume=True,
        )
    with pytest.raises(ValueError, match="Choose either genuine R pairs or P:0006"):
        evaluation.evaluate_stage2_unified(
            **kwargs, decoder_context="native-exterior", output_dir=tmp_path / "forbidden",
            p0006_evaluation_protocol_path="sealed",
        )
