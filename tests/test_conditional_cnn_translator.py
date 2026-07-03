import pytest
import torch

from fieldbridge.data.domains import Domain
from fieldbridge.models.conditioning import DomainEmbedding
from fieldbridge.models.factory import build_translator
from fieldbridge.models.film import FiLMGroupNorm
from fieldbridge.models.translators.conditional_cnn import ConditionalCNNFieldTranslator


def test_domain_embedding_broadcasts_single_domain_to_batch() -> None:
    embedding = DomainEmbedding(cond_dim=12)
    source = Domain(0.1, "T2-FLAIR")
    target = Domain(7.0, "T2-FLAIR")

    conditioning = embedding(source, target, batch_size=3)

    assert conditioning.shape == (3, 12)
    assert torch.isfinite(conditioning).all()


def test_domain_embedding_accepts_domain_sequences() -> None:
    embedding = DomainEmbedding(cond_dim=10)
    sources = [Domain(0.1, "T2-FLAIR"), Domain(5.0, "T1w")]
    targets = [Domain(7.0, "T2-FLAIR"), Domain(1.5, "T1w")]

    conditioning = embedding(sources, targets)

    assert conditioning.shape == (2, 10)
    assert torch.isfinite(conditioning).all()


def test_domain_embedding_rejects_bad_sequence_length() -> None:
    embedding = DomainEmbedding(cond_dim=8)
    sources = [Domain(0.1, "T2-FLAIR"), Domain(5.0, "T1w")]
    target = Domain(7.0, "T2-FLAIR")

    with pytest.raises(ValueError, match="source_domain sequence length"):
        embedding(sources, target, batch_size=3)


def test_domain_embedding_changes_when_target_domain_changes() -> None:
    torch.manual_seed(1)
    embedding = DomainEmbedding(cond_dim=16)
    source = Domain(3.0, "T1w")
    target_a = Domain(3.0, "T1w")
    target_b = Domain(7.0, "T2-FLAIR")

    conditioning_a = embedding(source, target_a, batch_size=2)
    conditioning_b = embedding(source, target_b, batch_size=2)

    assert not torch.allclose(conditioning_a, conditioning_b)


def test_film_group_norm_works_for_2d_tensor() -> None:
    film = FiLMGroupNorm(conditioning_dim=12, num_channels=4)
    x = torch.randn(2, 4, 8, 8)
    conditioning = torch.randn(2, 12)

    output = film(x, conditioning)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_film_group_norm_works_for_3d_tensor() -> None:
    film = FiLMGroupNorm(conditioning_dim=12, num_channels=4)
    x = torch.randn(2, 4, 4, 8, 8)
    conditioning = torch.randn(2, 12)

    output = film(x, conditioning)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_film_group_norm_rejects_bad_conditioning_batch_size() -> None:
    film = FiLMGroupNorm(conditioning_dim=12, num_channels=4)
    x = torch.randn(2, 4, 8, 8)
    conditioning = torch.randn(1, 12)

    with pytest.raises(ValueError, match="batch size"):
        film(x, conditioning)


def test_conditional_translator_preserves_2d_input_shape() -> None:
    model = ConditionalCNNFieldTranslator(
        hidden_channels=(4, 8),
        latent_channels=8,
        cond_dim=16,
        spatial_dims=2,
    )
    x = torch.randn(2, 1, 32, 32)
    source = Domain(0.1, "T2-FLAIR")
    target = Domain(7.0, "T2-FLAIR")

    output = model(x, source, target)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_conditional_translator_preserves_3d_input_shape() -> None:
    model = ConditionalCNNFieldTranslator(
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=3,
    )
    x = torch.randn(2, 1, 8, 16, 16)
    source = Domain(5.0, "T1w")
    target = Domain(1.5, "T1w")

    output = model(x, [source, source], [target, target])

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_conditional_translator_same_domain_call_works() -> None:
    model = ConditionalCNNFieldTranslator(
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=2,
    )
    x = torch.randn(2, 1, 16, 16)
    domain = Domain(3.0, "T2w")

    output = model(x, domain, domain)

    assert output.shape == x.shape


def test_conditional_translator_cross_domain_call_changes_output() -> None:
    torch.manual_seed(2)
    model = ConditionalCNNFieldTranslator(
        hidden_channels=(4, 8),
        latent_channels=8,
        cond_dim=16,
        spatial_dims=2,
    )
    model.eval()
    x = torch.randn(2, 1, 32, 32)
    source = Domain(3.0, "T1w")
    target_a = Domain(3.0, "T1w")
    target_b = Domain(7.0, "T2-FLAIR")

    with torch.no_grad():
        output_a = model(x, source, target_a)
        output_b = model(x, source, target_b)

    assert not torch.allclose(output_a, output_b)


def test_conditional_translator_uses_target_domain_in_decoder_conditioning() -> None:
    torch.manual_seed(7)
    model = ConditionalCNNFieldTranslator(
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=2,
    )
    model.eval()
    x = torch.randn(2, 1, 16, 16)
    source = Domain(3.0, "T1w")
    target_a = Domain(3.0, "T1w")
    target_b = Domain(7.0, "T2-FLAIR")

    cond_a = model.domain_embedding(
        source,
        target_a,
        batch_size=int(x.shape[0]),
        device=x.device,
        dtype=x.dtype,
    )
    cond_b = model.domain_embedding(
        source,
        target_b,
        batch_size=int(x.shape[0]),
        device=x.device,
        dtype=x.dtype,
    )
    assert not torch.allclose(cond_a, cond_b)

    decoder_conditioning: list[torch.Tensor] = []

    def _capture_conditioning(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del module, output
        decoder_conditioning.append(inputs[1].detach().clone())

    handle = model.decoder_blocks[0].modulation.register_forward_hook(_capture_conditioning)
    try:
        with torch.no_grad():
            output_a = model(x, source, target_a)
            output_b = model(x, source, target_b)
    finally:
        handle.remove()

    assert len(decoder_conditioning) == 2
    assert torch.allclose(decoder_conditioning[0], cond_a)
    assert torch.allclose(decoder_conditioning[1], cond_b)
    assert not torch.equal(output_a, output_b)


def test_factory_builds_conditional_translator() -> None:
    translator = build_translator(
        "conditional_cnn_field_translator",
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=2,
    )

    assert isinstance(translator, ConditionalCNNFieldTranslator)
