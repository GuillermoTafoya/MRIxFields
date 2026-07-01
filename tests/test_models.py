import pytest
import torch

from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.conditioning import DomainConditioner
from fieldbridge.models.factory import build_decoder, build_encoder, build_translator
from fieldbridge.models.film import FiLMLayer
from fieldbridge.models.translators.identity import IdentityTranslator


def test_identity_model_interfaces_preserve_shape() -> None:
    source = Domain(3.0, "T1w")
    target = Domain(1.5, "T2w")
    x = torch.randn(2, 1, 4, 4, 4)

    encoder = IdentityEncoder()
    translator = IdentityTranslator()
    decoder = IdentityDecoder()

    z = encoder.encode(x, [source, source])
    translated = translator(z, [source, source], [target, target])
    y = decoder.decode(translated, [target, target])

    assert z.shape == x.shape
    assert translated.shape == x.shape
    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_domain_conditioner_shape() -> None:
    conditioner = DomainConditioner(conditioning_dim=16)
    sources = [Domain(3.0, "T1w"), Domain(0.1, "T2w")]
    targets = [Domain(7.0, "T2-FLAIR"), Domain(0.1, "T2w")]
    output = conditioner(sources, targets)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_domain_conditioner_identity_pair_has_zero_log_ratio() -> None:
    conditioner = DomainConditioner(conditioning_dim=16)
    same = Domain(1.5, "T1w")

    output = conditioner([same], [same])

    assert output.shape == (1, 16)
    assert torch.isfinite(output).all()


def test_film_layer_preserves_shape() -> None:
    film = FiLMLayer(conditioning_dim=16, num_channels=4)
    x = torch.randn(2, 4, 8, 8)
    conditioning = torch.randn(2, 16)

    modulated = film(x, conditioning)

    assert modulated.shape == x.shape
    assert torch.isfinite(modulated).all()


def test_factory_builds_identity_models_by_name() -> None:
    assert isinstance(build_encoder("identity"), IdentityEncoder)
    assert isinstance(build_decoder("identity"), IdentityDecoder)
    assert isinstance(build_translator("identity", learnable_scale=True), IdentityTranslator)


def test_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_encoder("does-not-exist")

