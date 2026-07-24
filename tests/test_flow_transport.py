from __future__ import annotations

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.models.factory import build_translator
from fieldbridge.models.translators.flow_transport import FlowMatchingLatentTranslator


def _model():
    return FlowMatchingLatentTranslator(
        latent_channels=4,
        hidden_channels=(8, 16),
        bottleneck_channels=32,
        cond_dim=16,
        time_embed_dim=16,
        spatial_dims=3,
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
