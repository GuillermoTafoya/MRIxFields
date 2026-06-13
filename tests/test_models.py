import torch

from clbfield.data.domains import Domain
from clbfield.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from clbfield.models.conditioning import DomainConditioner
from clbfield.models.translators.identity import IdentityTranslator


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
    domains = [Domain(3.0, "T1w"), Domain(7.0, "T2-FLAIR")]
    output = conditioner(domains)
    assert output.shape == (2, 16)

