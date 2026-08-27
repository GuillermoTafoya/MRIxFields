from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import SourceCanonicalizedVolume
from fieldbridge.evaluation import stage2_step200_inference_audit as audit
from fieldbridge.evaluation.stage2_gate01 import Gate01Case
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import P0006_IDENTITY_SHA256


def _case(shape: tuple[int, ...]) -> Gate01Case:
    spatial = shape[-3:]
    spatial_values = torch.linspace(0.1, 0.8, steps=int(torch.tensor(spatial).prod())).reshape(
        spatial
    )
    values = spatial_values.reshape((1,) * (len(shape) - 3) + spatial).expand(shape).clone()
    target = values + 0.05
    return Gate01Case(
        case_id="synthetic-layout-case",
        source_domain=Domain(3.0, CONTRASTS[0]),
        target_domain=Domain(7.0, CONTRASTS[0]),
        source_image=values,
        target=target,
        raw_identity=values.clone(),
        raw_sb_v2=values + 0.01,
        stage1_reconstruction_ceiling=target - 0.005,
        support_mask=torch.ones(shape, dtype=torch.bool),
        traveller_identity_sha256=P0006_IDENTITY_SHA256,
        array_sha256={"source_image": "1" * 64},
    )


def _plan(case: Gate01Case) -> audit.FrozenStep200InferencePlan:
    return audit.FrozenStep200InferencePlan(
        {
            "inference_plan_sha256": "9" * 64,
            "case_seeds": [
                {
                    "source_image_sha256": case.array_sha256["source_image"],
                    "source_domain": case.source_domain.to_dict(),
                    "target_domain": case.target_domain.to_dict(),
                    "seed": 17,
                }
            ],
            "memory_gate_case": {
                "contrast": case.source_domain.contrast.value,
                "source_field_t": 3.0,
                "target_field_t": 7.0,
            },
            "graph_paths": [
                {
                    "contrast": case.source_domain.contrast.value,
                    "source_field_t": 3.0,
                    "intermediate_field_t": 1.5,
                    "target_field_t": 7.0,
                }
            ],
            "field_sweep": {
                "source_field_t": 3.0,
                "target_fields_t": [float(value) for value in FIELD_STRENGTHS_T],
            },
        }
    )


class _EncoderSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.input_references: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.input_references.append(value)
        self.inputs.append(value.detach().clone())
        return value


class _DecoderSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        self.outputs.append(result.detach().clone())
        return result


class _ArtifactSpy:
    artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.rendered_context_shapes: list[tuple[int, ...]] = []

    def normalize_source(self, source: torch.Tensor, domain: Domain):
        return SourceCanonicalizedVolume(
            values=source,
            support_mask=torch.ones_like(source, dtype=torch.bool),
            source_domain=domain,
            artifact_sha256=self.artifact_sha256,
        )

    def render_target(self, context, target_domain: Domain) -> torch.Tensor:
        del target_domain
        self.rendered_context_shapes.append(tuple(context.values.shape))
        return context.values.clone()


class _StatsSpy:
    def normalize(self, value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        assert value.shape == support.shape
        return value

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _CalibratorSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def apply(self, prediction, domain, *, support_mask, mode):
        del domain, mode
        self.calls.append((tuple(prediction.shape), tuple(support_mask.shape)))
        assert prediction.shape == support_mask.shape
        return prediction.clone()


def _runtime(monkeypatch):
    encoder = _EncoderSpy()
    decoder = _DecoderSpy()
    artifact = _ArtifactSpy()
    anatomy_shapes: list[tuple[tuple[int, ...], ...]] = []
    propagated_supports: list[torch.Tensor] = []

    def encode(encoder_module, value, *args, **kwargs):
        del args, kwargs
        return encoder_module(value), "full"

    def propagate(support, encoder_module, **kwargs):
        del encoder_module, kwargs
        propagated_supports.append(support.detach().clone())
        assert support.ndim == 3
        return support

    def transport(translator, value, *args, **kwargs):
        del translator, args, kwargs
        return value

    def graph(translator, value, *args, **kwargs):
        del translator, args, kwargs
        return torch.tensor(0.0), value.clone(), value.clone()

    def anatomy(canonical, decoded, support):
        anatomy_shapes.append(
            (tuple(canonical.shape), tuple(decoded.shape), tuple(support.shape))
        )
        return {"gradient": torch.tensor(0.0)}

    monkeypatch.setattr(audit, "encode_latent", encode)
    monkeypatch.setattr(
        audit, "propagate_encoder_local_valid_core_support", propagate
    )
    monkeypatch.setattr(audit, "integrate_transport", transport)
    monkeypatch.setattr(audit, "graph_consistency_loss", graph)
    monkeypatch.setattr(audit, "anatomy_preservation_components", anatomy)

    runtime = object.__new__(audit.UnifiedStep200InferenceRuntime)
    runtime.translator = torch.nn.Identity()
    runtime.encoder = encoder
    runtime.decoder = decoder
    runtime.artifact = artifact
    runtime.stats = _StatsSpy()
    runtime.support_rule_sha256 = "b" * 64
    runtime.device = torch.device("cpu")
    runtime.gpu_identity = {"name": "synthetic CPU", "total_memory_bytes": 0}
    runtime.frozen_stage1_vae_provenance = {"synthetic_test_runtime": True}
    runtime.reviewed_photometry_provenance = {"synthetic_test_runtime": True}
    return runtime, encoder, decoder, artifact, anatomy_shapes, propagated_supports


def _metric(prediction, target, metrics, device):
    del device
    assert prediction.ndim == target.ndim == 3
    error = float(torch.mean(torch.abs(prediction - target)))
    values = {"nrmse": error, "ssim": 1.0 - error, "lpips": error * 2.0}
    return {name: values[name] for name in metrics}


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 2, 3, 4), (1, 1, 2, 3, 4)])
def test_supported_representations_produce_identical_model_batches_and_outputs(
    monkeypatch, shape
):
    case = _case(shape)
    runtime, encoder, decoder, artifact, anatomy_shapes, supports = _runtime(monkeypatch)
    calibrator = _CalibratorSpy()
    outputs = runtime.infer_case(case, calibrator, plan=_plan(case))
    spatial = case.source_image.reshape(case.source_image.shape[-3:])
    expected_batch = spatial.reshape((1, 1, *spatial.shape))
    assert len(encoder.inputs) == 1
    assert torch.equal(encoder.inputs[0], expected_batch)
    if len(shape) == 5:
        assert encoder.input_references[0] is case.source_image
    assert all(tuple(value.shape) == tuple(expected_batch.shape) for value in decoder.outputs)
    assert all(shape == tuple(case.source_image.shape) for shape in artifact.rendered_context_shapes)
    assert len(artifact.rendered_context_shapes) == 8  # primary + graph pair + five sweeps
    assert anatomy_shapes == [
        (tuple(expected_batch.shape), tuple(expected_batch.shape), tuple(expected_batch.shape))
    ]
    assert len(supports) == 1 and tuple(supports[0].shape) == tuple(spatial.shape)
    assert calibrator.calls == [(shape, shape), (shape, shape), (shape, shape)]
    assert all(tuple(value.shape) == shape for value in outputs.methods.values())
    assert outputs.full_volume_layout_provenance[
        "authenticated_canonical_rank"
    ] == len(shape)
    scored = audit._score_case(outputs, case, metric_fn=_metric, device="cpu")
    assert set(scored) == set(outputs.methods)


def test_representation_wrappers_are_numerically_equivalent(monkeypatch):
    results = []
    for shape in ((2, 3, 4), (1, 2, 3, 4), (1, 1, 2, 3, 4)):
        case = _case(shape)
        runtime, _, _, _, _, _ = _runtime(monkeypatch)
        outputs = runtime.infer_case(case, _CalibratorSpy(), plan=_plan(case))
        results.append(
            (
                outputs.methods["raw_unified_step200"].reshape(2, 3, 4),
                audit._score_case(outputs, case, metric_fn=_metric, device="cpu"),
            )
        )
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][0], results[2][0])
    assert results[0][1] == results[1][1] == results[2][1]


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros((2, 3)),
        torch.zeros((1, 1, 1, 2, 3, 4)),
        torch.zeros((2, 2, 3, 4)),
        torch.zeros((1, 2, 2, 3, 4)),
    ],
)
def test_invalid_rank_or_non_singleton_leading_axes_fail_closed(tensor):
    with pytest.raises(ValueError, match="rank|non-singleton"):
        audit.adapt_full_volume_layout(tensor, role="synthetic invalid volume")


def test_nonfinite_volume_fails_and_singleton_spatial_axis_is_preserved():
    malformed = torch.zeros((1, 1, 2, 3, 4))
    malformed[..., 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        audit.adapt_full_volume_layout(malformed, role="nonfinite volume")
    singleton_spatial = torch.arange(20.0).reshape(1, 4, 5)
    adapted = audit.adapt_full_volume_layout(
        singleton_spatial, role="singleton spatial volume"
    )
    assert adapted.spatial_volume().shape == (1, 4, 5)
    assert adapted.model_batch().shape == (1, 1, 1, 4, 5)
    assert audit._volume_array(singleton_spatial, "singleton spatial").shape == (1, 4, 5)


def test_malformed_spatial_and_inconsistent_expected_shape_fail_closed():
    with pytest.raises(ValueError, match="malformed spatial"):
        audit.adapt_full_volume_layout(
            torch.zeros((1, 1, 0, 3, 4)), role="empty spatial volume"
        )
    with pytest.raises(ValueError, match="shape changed"):
        audit.adapt_full_volume_layout(
            torch.zeros((1, 1, 2, 3, 4)),
            role="inconsistent volume",
            expected_shape=(1, 1, 2, 3, 5),
        )


def test_decoder_requires_exact_one_channel_and_spatial_shape():
    source = torch.zeros((1, 1, 2, 3, 4))
    adapted = audit.adapt_full_volume_layout(source, role="source")
    for malformed in (
        torch.zeros((1, 2, 2, 3, 4)),
        torch.zeros((1, 1, 2, 3, 5)),
        torch.zeros((1, 2, 3, 4)),
    ):
        with pytest.raises(ValueError, match="one channel"):
            adapted.restore_decoder_output(malformed, role="malformed")


@dataclass
class _StreamedCase:
    case: Gate01Case
    calibrator: _CalibratorSpy
    released: bool = False

    def release(self) -> None:
        self.released = True


class _SingleIterator:
    def __init__(self, streamed: _StreamedCase) -> None:
        self.streamed = streamed
        self.closed = False
        self._used = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._used:
            raise StopIteration
        self._used = True
        return self.streamed

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((2, 1, 2, 3, 4), "non-singleton"),
        ((1, 1, 1, 2, 3, 4), "rank"),
    ],
)
def test_shape_failure_precedes_encoder_and_releases_memory_gate_case(
    monkeypatch, tmp_path: Path, shape, message
):
    bad_case = _case(shape)
    runtime, encoder, _, _, _, _ = _runtime(monkeypatch)
    streamed = _StreamedCase(bad_case, _CalibratorSpy())
    iterator = _SingleIterator(streamed)
    monkeypatch.setattr(audit, "iter_gate01_p0006_evaluation_cases", lambda path: iterator)
    with pytest.raises(ValueError, match=message):
        audit._run_or_verify_one_case_gate(
            tmp_path / "protocol.json",
            runtime,
            _plan(bad_case),
            tmp_path / "gate.json",
            initial_state=runtime.state_identity(),
            require_a100=False,
        )
    assert encoder.inputs == []
    assert streamed.released is True
    assert iterator.closed is True
    assert not (tmp_path / "gate.json").exists()


def test_production_shaped_one_case_memory_gate_completes(monkeypatch, tmp_path: Path):
    case = _case((1, 1, 2, 3, 4))
    runtime, encoder, _, _, _, _ = _runtime(monkeypatch)
    streamed = _StreamedCase(case, _CalibratorSpy())
    iterator = _SingleIterator(streamed)
    monkeypatch.setattr(audit, "iter_gate01_p0006_evaluation_cases", lambda path: iterator)
    gate = audit._run_or_verify_one_case_gate(
        tmp_path / "protocol.json",
        runtime,
        _plan(case),
        tmp_path / "gate.json",
        initial_state=runtime.state_identity(),
        require_a100=False,
    )
    assert gate["status"] == "pass"
    assert gate["contract_version"] == audit.MEMORY_GATE_CONTRACT
    assert gate["full_volume_layout_provenance"]["model_input_rank"] == 5
    assert gate["training_invoked"] is False
    assert len(encoder.inputs) == 1
    assert streamed.released is True
    assert iterator.closed is True
