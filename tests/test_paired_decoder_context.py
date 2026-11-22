"""Exercise both paired evaluators on synthetic data with observable domain labels."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge import cli
from fieldbridge.data.domains import FIELD_STRENGTHS_T, Contrast, Domain
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.data.photometry_factored_latent_bank import (
    derive_encoder_local_support_rule,
    propagate_encoder_local_valid_core_support,
)
from fieldbridge.data.photometry_factorization import SourceCanonicalizedVolume, sha256_file
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256
from fieldbridge.evaluation import paired_decoder_context as paired
from fieldbridge.evaluation import stage2_unified as unified
from fieldbridge.evaluation.decoder_context import HISTORICAL_DENORMALIZE, SOURCE_NATIVE_EXTERIOR
from fieldbridge.evaluation.low_memory_vae import enable_low_memory
from fieldbridge.evaluation.stage2_photometry_baseline import PairedEvaluationCase
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder


SOURCE = Domain(3.0, Contrast.T1W)
REQUEST = Domain(7.0, Contrast.T1W)
WRONG = Domain(0.1, Contrast.T1W)


class ObservedVelocity(nn.Module):
    def __init__(self, rate=0.03125):
        super().__init__()
        self.register_buffer("rate", torch.tensor([rate]))
        self.calls = []

    def forward(self, z, sources, requests, t):
        self.calls.append((z.clone(), sources[0], requests[0], float(t.item())))
        return torch.ones_like(z) * self.rate * requests[0].field_strength_t


class ObservedDecoder(nn.Module):
    output_activation = "none"

    def __init__(self):
        super().__init__()
        self.calls = []

    def decode(self, z, domains):
        self.calls.append((z.clone(), domains[0]))
        return F.interpolate(z[:, :1].sigmoid(), scale_factor=4, mode="nearest")


class FixedPhotometry:
    artifact_sha256 = "a" * 64

    def __init__(self):
        self.requests = []

    def normalize_source(self, source, domain):
        return SourceCanonicalizedVolume(source * 0.8, source != 0, domain, self.artifact_sha256)

    def render_target(self, context, domain):
        self.requests.append(domain)
        rendered = context.values + domain.field_strength_t / 100
        return rendered.masked_fill(~context.support_mask, 0)


@pytest.fixture
def synthetic_pair():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(709)
        encoder = KLVAEEncoder(base_channels=2, latent_channels=2, num_res_blocks=1, spatial_dims=3)
    encoder.eval().requires_grad_(False)
    artifact = FixedPhotometry()
    source = torch.linspace(0, 1, 24**3).reshape(1, 24, 24, 24)
    target = 0.2 + 0.5 * source
    target[..., 1, 1, 1] = 0
    case = PairedEvaluationCase(
        case_identity="synthetic-t1w", source=source, target=target, source_domain=SOURCE,
        target_domain=REQUEST, subject_group_identity="P:0006",
        source_provenance={"case_id": "synthetic-source"},
        target_provenance={"case_id": "synthetic-target"}, stage1_reconstruction=target * 0.98,
    )
    stats = FactoredLatentStats(
        torch.tensor([1.25, -0.75]), torch.tensor([0.4, 0.7]), torch.tensor([10, 10]), "b" * 64
    )
    rule = derive_encoder_local_support_rule(encoder)["rule_sha256"]
    core = propagate_encoder_local_valid_core_support(source[None] != 0, encoder)[None, None]
    assert core.any() and (~core).any()
    with torch.inference_mode():
        canonical = artifact.normalize_source(source[None], SOURCE)
        native, _ = encoder.encode_dist(canonical.values, SOURCE)
    return SimpleNamespace(
        encoder=encoder, decoder=ObservedDecoder().eval(), artifact=artifact, case=case,
        native=native, stats=stats, rule=rule, core=core, start=stats.normalize(native, core),
    )


def test_matched_pilot_keeps_decoder_and_renderer_fixed_for_zero_correct_wrong(synthetic_pair):
    fixture = synthetic_pair
    translator = ObservedVelocity().eval()
    prepared = paired.prepare_source(
        encoder=fixture.encoder, artifact=fixture.artifact, stats=fixture.stats,
        support_rule_sha256=fixture.rule, source=fixture.case.source, source_domain=SOURCE,
        device="cpu", mark=lambda *_: None,
    )
    calibration_calls = []

    def calibrate(image):
        calibration_calls.append(image.clone())
        return image * 0.9

    result = paired.score_direction(
        prepared=prepared, decoder=fixture.decoder, translator=translator,
        artifact=fixture.artifact,
        stats=fixture.stats, target=fixture.case.target, target_domain=REQUEST, wrong_domain=WRONG,
        raw_stage1_source=fixture.case.source, calibrated_stage1_source=fixture.case.source * 0.9,
        calibrate=calibrate, case_seed=42, device="cpu", mark=lambda *_: None,
    )
    assert len(translator.calls) == 16
    for i, request in enumerate([REQUEST, WRONG]):
        calls = translator.calls[i * 8:(i + 1) * 8]
        assert [(s, r) for _, s, r, _ in calls] == [(SOURCE, request)] * 8
        assert [t for _, _, _, t in calls] == [0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1]
        assert torch.equal(calls[0][0], fixture.start)
    assert fixture.artifact.requests == [REQUEST] * 4  # photometry-only plus the three arms
    assert [label for _, label in fixture.decoder.calls] == [REQUEST] * 3
    for (given, _), request in zip(fixture.decoder.calls, [None, REQUEST, WRONG], strict=True):
        endpoint = fixture.start
        if request:
            velocity = translator.rate * request.field_strength_t
            for _ in range(4):
                endpoint = endpoint + (velocity + velocity) * 0.125
        expected = torch.where(fixture.core, fixture.stats.denormalize(endpoint), fixture.native)
        torch.testing.assert_close(given, expected, atol=0, rtol=0)
    assert len(calibration_calls) == 5  # Stage 1 uses the supplied archived calibrated source.
    assert result["trace"]["requests"]["zero"]["velocity_evaluations"] == 0
    assert result["raw_scores"].keys() == result["calibrated_scores"].keys()
    assert result["direct_correct_vs_wrong"]["before"]["source_foreground"]["rmse"] > 0


def _synthetic_metrics(prediction, target):
    error = (prediction - target).float()
    return {
        "nrmse": float(error.norm() / target.norm()), "ssim": 1.0 - float(error.abs().mean()),
        "lpips": float(error.abs().mean()),
    }


def test_eval_stage2_unified_cli_restores_p0006_context_and_isolates_replay(
    tmp_path, monkeypatch, synthetic_pair
):
    f = synthetic_pair
    # Replace external artifacts and perceptual weights. Real CLI dispatch, source encoding,
    # valid-core propagation, transport, decoding, wrong-target loop and exact resume all run.
    vae_config = tmp_path / "synthetic.yaml"
    vae_config.write_text('model: {name: kl_vae, spatial_dims: 3}', encoding="utf-8")
    checkpoint = tmp_path / "synthetic-vae"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    photometry_path = tmp_path / "synthetic-artifact"
    photometry_path.write_text("{}", encoding="utf-8")
    manifest = {
        "vae": {
            "config_sha256": sha256_file(vae_config), "checkpoint_sha256": sha256_file(checkpoint),
        },
        "operational_support_rule": {"rule_sha256": f.rule},
    }
    bank = SimpleNamespace(
        split="validation", manifest=manifest, records=[], artifact_sha256="c" * 64,
    )
    monkeypatch.setattr(cli, "PhotometryFactoredLatentBankIndex", lambda *_: bank)
    monkeypatch.setattr(cli.FactoredLatentStats, "from_bank", lambda *_: f.stats)
    monkeypatch.setattr(cli.FrozenPhotometryArtifact, "load", lambda *_: f.artifact)
    monkeypatch.setattr(cli, "build_encoder", lambda *args, **kwargs: f.encoder)
    monkeypatch.setattr(cli, "build_decoder", lambda *args, **kwargs: f.decoder)
    translators = []
    rates = {"full": 0.03125, "sb": 0.015625, "ablation": 0.0625}

    def build_translator(*args, **kwargs):
        model = ObservedVelocity()
        translators.append(model)
        return model

    def weights(path):
        if path == checkpoint:
            return {"encoder": f.encoder.state_dict(), "decoder": f.decoder.state_dict()}
        return {"contract_version": cli.UNIFIED_RESUME_CONTRACT,
                "translator": ObservedVelocity(rates[path.name]).state_dict()}

    monkeypatch.setattr(cli, "build_translator", build_translator)
    monkeypatch.setattr(cli, "load_checkpoint", weights)
    monkeypatch.setattr(unified, "load_gate01_p0006_evaluation_protocol", lambda _: (
        {"protocol_sha256": "d" * 64, "contract_version": "synthetic-protocol"},
        [f.case], {f.case.case_identity: {
            "gate01_calibrated_identity": f.case.source, "original_sb_v2": f.case.source,
        }},
    ))
    monkeypatch.setattr(unified, "OfficialQualificationMetrics", lambda **kw: _synthetic_metrics)
    monkeypatch.setattr(
        unified, "_render_montage", lambda path, *_: path.write_bytes(b"stub montage")
    )

    def run(out, policy=None, *, resume=False):
        args = [
            "eval-stage2-unified", "--bank-dir", "synthetic-bank", "--checkpoint", "full",
            "--sb-only-checkpoint", "sb", "--ablation-checkpoint", "test=ablation",
            "--vae-config", str(vae_config), "--vae-checkpoint", str(checkpoint),
            "--photometry-artifact", str(photometry_path),
            "--p0006-evaluation-protocol", "synthetic",
            "--out", str(out), "--integration-steps", "1", "--solver", "euler", "--device", "cpu",
        ]
        if policy is not None:
            args += ["--decoder-context", policy]
        if resume:
            args += ["--resume"]
        assert cli.main(args) == 0
        return json.loads((out / "result.json").read_text(encoding="utf-8"))

    wrong_fields = [Domain(field, SOURCE.contrast) for field in FIELD_STRENGTHS_T if field != 7]
    modes = [
        (None, "v4"), (HISTORICAL_DENORMALIZE, "historical"), (SOURCE_NATIVE_EXTERIOR, "native")
    ]
    reports = {}
    for policy, dirname in modes:
        out = tmp_path / dirname
        f.decoder.calls.clear()
        f.artifact.requests.clear()
        result = run(out, policy)
        reports[dirname] = result
        assert result["case_count"] == 1
        full, sb_only, ablation = translators[-3:]
        assert [(s, r) for _, s, r, _ in full.calls[-4:]] == [(SOURCE, d) for d in wrong_fields]
        for model in (full, sb_only, ablation):
            assert torch.equal(model.calls[0][0], f.start)
        # Decode order is full, optional zero, SB-only, ablation, then wrong conditions.
        models = [full] + ([None] if policy else []) + [sb_only, ablation]
        conditions = [REQUEST] * len(models) + wrong_fields
        models += [full] * len(wrong_fields)
        assert len(f.decoder.calls) == len(conditions)
        observed = zip(f.decoder.calls, conditions, models, strict=True)
        for (actual, label), request, model in observed:
            endpoint = f.start if model is None else f.start + model.rate * request.field_strength_t
            expected = f.stats.denormalize(endpoint)
            if policy == SOURCE_NATIVE_EXTERIOR:
                expected = torch.where(f.core, expected, f.native)
            assert torch.equal(actual, expected)
            assert label == (REQUEST if policy else request)
        wrong_rows = result["cases"][0]["requested_vs_wrong_target"]["wrong_targets"]
        if policy is not None:
            assert f.artifact.requests == [REQUEST] * len(conditions)
            assert all("condition_native_render_nrmse" not in row for row in wrong_rows)
            assert result["decoder_context"] == policy
            assert "zero_transport" in result["methods"]
        else:
            assert f.artifact.requests[3::2] == [REQUEST] * 4
            assert f.artifact.requests[4::2] == wrong_fields
            assert all("condition_native_render_nrmse" in row for row in wrong_rows)
            assert "zero_transport" not in result["methods"]
            assert result["contract_version"].endswith("v4")

    assert len({report["run_contract_sha256"] for report in reports.values()}) == 3
    previous_calls = len(f.decoder.calls)
    for policy, dirname in modes:
        assert run(tmp_path / dirname, policy, resume=True) == reports[dirname]
        for other_policy, _ in modes:
            if other_policy != policy:
                with pytest.raises(ValueError, match="contract"):
                    run(tmp_path / dirname, other_policy, resume=True)
    # A stale shard in an interrupted native run must fail even with a matching outer contract.
    mixed = tmp_path / "interrupted"
    shards = mixed / "case_shards"
    shards.mkdir(parents=True)
    contract = (tmp_path / "native" / "run_contract.json").read_bytes()
    (mixed / "run_contract.json").write_bytes(contract)
    legacy_shard, = (tmp_path / "v4" / "case_shards").glob("*.json")
    (shards / legacy_shard.name).write_bytes(legacy_shard.read_bytes())
    with pytest.raises(ValueError, match="shard"):
        run(mixed, SOURCE_NATIVE_EXTERIOR, resume=True)
    assert len(f.decoder.calls) == previous_calls


@torch.inference_mode()
def test_pilot_operator_calibrates_and_checks_independent_reference(
    tmp_path, monkeypatch, synthetic_pair
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks"))
    operator = import_module("stage2_decoder_context_parity_operator")
    f = synthetic_pair
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(340)
        decoder = KLVAEDecoder(base_channels=2, latent_channels=2, num_res_blocks=1, spatial_dims=3)
    decoder.eval().requires_grad_(False)
    translator = ObservedVelocity().eval()
    enable_low_memory(f.encoder, decoder, slab_depth=3)
    source, target = f.case.source[None], f.case.target[None]
    foreground, shared = source != 0, (source != 0) & (target != 0)
    raw_identity = source * 0.8
    seed = 1091568062

    def calibrate(image, domain, *, support_mask, mode):
        assert domain == REQUEST and mode == "histogram"
        assert torch.equal(support_mask, foreground)
        return (image * 0.9).masked_fill(~support_mask, 0)

    array_hashes = {k: storage_tensor_sha256(v) for k, v in {
        "source_image": source, "source_support_mask": foreground, "target": target,
        "raw_identity": raw_identity,
    }.items()}
    calibrated_identity = raw_identity * 0.9
    spec = SimpleNamespace(
        case_id=operator.SENTINEL_CASE_ID, source_domain=SOURCE, target_domain=REQUEST, wrong_domain=WRONG,
        ordinal=12, seed=seed, array_sha256=array_hashes,
        calibrated_identity_sha256=storage_tensor_sha256(calibrated_identity),
    )
    cohort = SimpleNamespace(provenance={"synthetic": True}, calibrator=SimpleNamespace(apply=calibrate))
    source_data = SimpleNamespace(source_image=source, support_mask=foreground, raw_identity=raw_identity)
    monkeypatch.setattr(operator, "load_pilot_inputs",
                        lambda _: (cohort, spec, source_data, target, calibrated_identity))
    model_provenance = {"slab_depth": 3}
    monkeypatch.setattr(operator, "stage_and_load_models", lambda *_: (
        f.encoder, decoder, translator, f.artifact, f.stats, f.rule, model_provenance,
    ))

    # Independent oracle for the known constant velocity, fixed photometry and calibrator.
    # Direct KL-VAE calls and tensor reductions bypass both packaged scoring functions.
    canonical = source * 0.8
    native, _ = f.encoder.encode_dist(canonical, SOURCE)
    start = f.stats.normalize(native, f.core)
    images = {"acquired_source": source, "stage1_source": raw_identity,
              "photometry_only": (canonical + 0.07).masked_fill(~foreground, 0)}
    requests = {}
    for name, request in (("zero", None), ("correct", REQUEST), ("wrong", WRONG)):
        endpoint = start
        if request is not None:
            increment = translator.rate * request.field_strength_t / 4
            for _ in range(4):
                endpoint = endpoint + increment
        z = torch.where(f.core, f.stats.denormalize(endpoint), native)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            image = decoder.decode(z, [REQUEST]).float().clamp(0, 1)
        images[name] = (image + 0.07).masked_fill(~foreground, 0)
        times = [0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1] if request else []
        label = request.label if request else None
        requests[name] = dict(
            neural_request=label, velocity_evaluations=len(times), velocity_times=times,
            velocity_labels=[label] * len(times), decoder_domain=REQUEST.label,
            renderer_domain=REQUEST.label, normalized_endpoint_sha256=storage_tensor_sha256(endpoint),
            decoder_input_sha256=storage_tensor_sha256(z),
        )

    def scores(image):
        delta = image.double() - target.double()
        return dict(full_nrmse=float(delta.norm() / target.double().norm()),
                    shared_foreground_nrmse=float(delta[shared].norm() / target[shared].double().norm()),
                    shared_foreground_rmse=float(delta[shared].square().mean().sqrt()),
                    rendered_prediction_sha256=storage_tensor_sha256(image))

    trace = dict(
        requests=requests, case_seed=seed, encoder_calls_scientific_source=1, integration_count=2,
        masked_start_sha256=storage_tensor_sha256(start), native_posterior_sha256=storage_tensor_sha256(native),
        core_mask_sha256=storage_tensor_sha256(f.core[0, 0]), latent_core_count=int(f.core.sum()),
        latent_grid_count=f.core.numel(), source_foreground_count=int(foreground.sum()),
        target_foreground_count=int((target != 0).sum()), shared_foreground_count=int(shared.sum()),
        source_shape=list(source.shape), spatial_shape=list(source.shape[2:]),
    )
    expected = dict(trace=trace, raw_scores={k: scores(v) for k, v in images.items()},
                    calibrated_scores={k: scores(v * 0.9) for k, v in images.items()})
    pilot = dict(source_domain=SOURCE.label, target_domain=REQUEST.label, wrong_neural_domain=WRONG.label,
                 inference_seed=seed, paired_input_provenance=dict(array_sha256=array_hashes,
                 calibrated_identity_sha256=spec.calibrated_identity_sha256), analysis=expected)
    result = operator.evaluate_pilot(tmp_path, pilot, "cpu")
    assert result["input_provenance"] == cohort.provenance
    assert result["model_provenance"] == model_provenance
    assert result["parity"]["status"] == "passed", result["parity"]["failures"]
    expected["calibrated_scores"]["wrong"]["shared_foreground_rmse"] += 1e-6
    mismatch = operator.evaluate_pilot(tmp_path, pilot, "cpu")["parity"]
    assert mismatch["status"] == "failed"
    assert mismatch["failures"] == ["calibrated_scores/wrong/shared_foreground_rmse"]
