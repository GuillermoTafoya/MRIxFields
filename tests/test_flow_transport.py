from __future__ import annotations

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.models.factory import build_translator
from fieldbridge.models.translators.flow_transport import FlowMatchingLatentTranslator


def _model(**overrides):
    kwargs = {
        "latent_channels": 4,
        "hidden_channels": (8, 16),
        "bottleneck_channels": 32,
        "cond_dim": 16,
        "time_embed_dim": 16,
        "spatial_dims": 3,
    }
    kwargs.update(overrides)
    return FlowMatchingLatentTranslator(
        **kwargs,
    )


def _domains(n):
    src = [Domain(0.1, Contrast.T1W), Domain(3.0, Contrast.T2W)][:n]
    tgt = [Domain(7.0, Contrast.T2_FLAIR), Domain(1.5, Contrast.T1W)][:n]
    return src, tgt


def test_velocity_has_latent_shape_and_is_finite() -> None:
    model = _model()
    z = torch.randn(2, 4, 12, 12, 12)
    src, tgt = _domains(2)
    t = torch.tensor([0.2, 0.8])
    v = model(z, src, tgt, t)
    assert v.shape == z.shape
    assert torch.isfinite(v).all()


def test_scalar_time_is_accepted() -> None:
    model = _model()
    z = torch.randn(2, 4, 8, 8, 8)
    src, tgt = _domains(2)
    v = model(z, src, tgt, 0.5)
    assert v.shape == z.shape


def test_missing_time_raises() -> None:
    model = _model()
    z = torch.randn(1, 4, 8, 8, 8)
    src, tgt = _domains(1)
    with pytest.raises(ValueError, match="requires the flow time"):
        model(z, src, tgt, None)


def test_non_multiple_spatial_size_is_padded_and_cropped() -> None:
    model = _model()  # 2 downsamples -> factor 4; 11 is not divisible by 4
    z = torch.randn(1, 4, 11, 13, 9)
    src, tgt = _domains(1)
    v = model(z, src, tgt, 0.5)
    assert v.shape == z.shape


def test_time_conditioning_changes_output() -> None:
    model = _model().eval()
    z = torch.randn(1, 4, 8, 8, 8)
    src, tgt = _domains(1)
    with torch.no_grad():
        v0 = model(z, src, tgt, 0.1)
        v1 = model(z, src, tgt, 0.9)
    assert not torch.allclose(v0, v1)


def test_factory_builds_flow_matching_latent() -> None:
    model = build_translator(
        "flow_matching_latent",
        latent_channels=4,
        hidden_channels=(8, 16),
        bottleneck_channels=32,
        cond_dim=16,
        time_embed_dim=16,
        spatial_dims=3,
    )
    assert isinstance(model, FlowMatchingLatentTranslator)


def test_historical_sb_v2_model_configuration_constructs() -> None:
    model_config = {
        "latent_channels": 4,
        "hidden_channels": [64, 128],
        "bottleneck_channels": 256,
        "cond_dim": 128,
        "time_embed_dim": 128,
        "time_scale": 1000.0,
        "spatial_dims": 3,
        "activation": "silu",
        "skip_mode": "concat",
        "zero_init_output": True,
    }

    model = build_translator("flow_matching_latent", **model_config)

    assert isinstance(model, FlowMatchingLatentTranslator)
    assert model.zero_init_output is True


def test_zero_init_output_zeros_exactly_the_initial_output_projection() -> None:
    model = _model(zero_init_output=True)

    assert torch.count_nonzero(model.output_projection.weight) == 0
    assert model.output_projection.bias is not None
    assert torch.count_nonzero(model.output_projection.bias) == 0


def test_zero_init_output_false_preserves_default_initialization() -> None:
    torch.manual_seed(19)
    default_model = _model()
    torch.manual_seed(19)
    explicit_false_model = _model(zero_init_output=False)

    assert default_model.zero_init_output is False
    assert explicit_false_model.zero_init_output is False
    for name, default_value in default_model.state_dict().items():
        assert torch.equal(default_value, explicit_false_model.state_dict()[name])
    assert torch.count_nonzero(default_model.output_projection.weight) > 0


def test_strict_checkpoint_loading_erases_constructor_initialization_difference() -> None:
    torch.manual_seed(23)
    trained_model = _model()
    with torch.no_grad():
        for index, parameter in enumerate(trained_model.parameters()):
            parameter.add_(float(index + 1) * 1e-4)
    trained_state = {
        name: value.detach().clone() for name, value in trained_model.state_dict().items()
    }

    initialized_zero = _model(zero_init_output=True).eval()
    initialized_default = _model(zero_init_output=False).eval()
    initialized_zero.load_state_dict(trained_state, strict=True)
    initialized_default.load_state_dict(trained_state, strict=True)

    for name, zero_value in initialized_zero.state_dict().items():
        assert torch.equal(zero_value, initialized_default.state_dict()[name])
    z = torch.randn(1, 4, 8, 8, 8)
    source, target = _domains(1)
    with torch.no_grad():
        zero_output = initialized_zero(z, source, target, 0.4)
        default_output = initialized_default(z, source, target, 0.4)
    torch.testing.assert_close(zero_output, default_output, rtol=0.0, atol=0.0)
