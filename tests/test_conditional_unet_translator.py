from pathlib import Path

import pytest
import torch
from torch import nn

from fieldbridge.config import load_yaml_config
from fieldbridge.data.domains import Domain
from fieldbridge.models.factory import build_translator
from fieldbridge.models.translators.conditional_unet import ConditionalUNetFieldTranslator


def _domain_pair() -> tuple[Domain, Domain]:
    return Domain(3.0, "T1w"), Domain(7.0, "T2-FLAIR")


def _small_unet(**kwargs: object) -> ConditionalUNetFieldTranslator:
    defaults: dict[str, object] = {
        "hidden_channels": (4, 8),
        "latent_channels": 8,
        "cond_dim": 16,
        "spatial_dims": 2,
    }
    defaults.update(kwargs)
    return ConditionalUNetFieldTranslator(**defaults)


def test_conditional_unet_preserves_2d_input_shape() -> None:
    model = _small_unet()
    source, target = _domain_pair()
    x = torch.randn(2, 1, 32, 32)

    output = model(x, source, target)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_conditional_unet_preserves_3d_input_shape() -> None:
    model = ConditionalUNetFieldTranslator(
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=3,
    )
    source, target = _domain_pair()
    x = torch.randn(2, 1, 8, 32, 32)

    output = model(x, [source, source], [target, target])

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_conditional_unet_pads_and_crops_non_divisible_2d_shape() -> None:
    model = _small_unet(pad_to_multiple=True)
    source, target = _domain_pair()
    x = torch.randn(2, 1, 31, 35)

    output = model(x, source, target)

    assert output.shape == x.shape


def test_conditional_unet_pads_and_crops_non_divisible_3d_shape() -> None:
    model = ConditionalUNetFieldTranslator(
        hidden_channels=(4, 8),
        latent_channels=8,
        cond_dim=16,
        spatial_dims=3,
        pad_to_multiple=True,
    )
    source, target = _domain_pair()
    x = torch.randn(2, 1, 7, 15, 17)

    output = model(x, source, target)

    assert output.shape == x.shape


def test_conditional_unet_requires_divisible_shape_when_padding_disabled() -> None:
    model = _small_unet(pad_to_multiple=False)
    source, target = _domain_pair()
    x = torch.randn(2, 1, 31, 35)

    with pytest.raises(ValueError, match="pad_to_multiple=False"):
        model(x, source, target)


def test_conditional_unet_same_domain_call_works() -> None:
    model = _small_unet()
    domain = Domain(3.0, "T2w")
    x = torch.randn(2, 1, 32, 32)

    output = model(x, domain, domain)

    assert output.shape == x.shape


def test_conditional_unet_cross_domain_call_works() -> None:
    model = _small_unet()
    source, target = _domain_pair()
    x = torch.randn(2, 1, 32, 32)

    output = model(x, source, target)

    assert output.shape == x.shape


def test_conditional_unet_passes_source_and_target_to_domain_embedding() -> None:
    class _DomainEmbeddingSpy(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped
            self.calls: list[tuple[object, object]] = []

        def forward(
            self,
            source_domain: object,
            target_domain: object,
            **kwargs: object,
        ) -> torch.Tensor:
            self.calls.append((source_domain, target_domain))
            return self.wrapped(source_domain, target_domain, **kwargs)

    model = _small_unet()
    spy = _DomainEmbeddingSpy(model.domain_embedding)
    model.domain_embedding = spy
    source, target = _domain_pair()
    x = torch.randn(2, 1, 32, 32)

    output = model(x, source, target)

    assert output.shape == x.shape
    assert spy.calls == [(source, target)]


def test_conditional_unet_uses_target_domain_in_decoder_conditioning() -> None:
    torch.manual_seed(7)
    model = _small_unet(hidden_channels=(4,), latent_channels=4, cond_dim=12)
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

    hook_target = model.decoder_blocks[0].conv_block.modulation1
    handle = hook_target.register_forward_hook(_capture_conditioning)
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


def test_conditional_unet_gated_skip_uses_source_target_conditioning() -> None:
    torch.manual_seed(5)
    model = _small_unet(hidden_channels=(4,), latent_channels=4, cond_dim=12, skip_mode="gated")
    model.eval()
    x = torch.randn(2, 1, 16, 16)
    source = Domain(3.0, "T1w")
    target_a = Domain(3.0, "T1w")
    target_b = Domain(7.0, "T2-FLAIR")
    cond_a = model.domain_embedding(source, target_a, batch_size=2, device=x.device, dtype=x.dtype)
    cond_b = model.domain_embedding(source, target_b, batch_size=2, device=x.device, dtype=x.dtype)
    assert not torch.allclose(cond_a, cond_b)

    skip_gate_conditioning: list[torch.Tensor] = []

    def _capture_skip_gate_conditioning(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del module, output
        skip_gate_conditioning.append(inputs[1].detach().clone())

    skip_gate = model.decoder_blocks[0].skip_gate
    assert skip_gate is not None
    handle = skip_gate.register_forward_hook(_capture_skip_gate_conditioning)
    try:
        with torch.no_grad():
            model(x, source, target_a)
            model(x, source, target_b)
    finally:
        handle.remove()

    assert len(skip_gate_conditioning) == 2
    assert torch.allclose(skip_gate_conditioning[0], cond_a)
    assert torch.allclose(skip_gate_conditioning[1], cond_b)


@pytest.mark.parametrize("skip_mode", ["gated", "concat", "none"])
def test_conditional_unet_skip_modes_work(skip_mode: str) -> None:
    model = _small_unet(skip_mode=skip_mode)
    source, target = _domain_pair()
    x = torch.randn(2, 1, 32, 32)

    output = model(x, source, target)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_conditional_unet_uses_expected_2d_and_3d_operations() -> None:
    model_2d = _small_unet(spatial_dims=2, upsample_mode="interpolate")
    model_3d = ConditionalUNetFieldTranslator(
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=3,
        upsample_mode="interpolate",
    )
    model_2d_transpose = _small_unet(spatial_dims=2, upsample_mode="transpose")

    assert any(isinstance(module, nn.Conv2d) for module in model_2d.modules())
    assert not any(isinstance(module, nn.Conv3d) for module in model_2d.modules())
    assert model_2d.decoder_blocks[0].upsample is None
    assert isinstance(model_2d.decoder_blocks[0].post_upsample_conv, nn.Conv2d)

    assert any(isinstance(module, nn.Conv3d) for module in model_3d.modules())
    assert not any(isinstance(module, nn.Conv2d) for module in model_3d.modules())
    assert model_3d.decoder_blocks[0].upsample is None
    assert isinstance(model_3d.decoder_blocks[0].post_upsample_conv, nn.Conv3d)

    assert isinstance(model_2d_transpose.decoder_blocks[0].upsample, nn.ConvTranspose2d)
    assert model_2d_transpose.decoder_blocks[0].post_upsample_conv is None


def test_conditional_unet_tiny_same_domain_overfit_reduces_loss() -> None:
    torch.manual_seed(11)
    model = _small_unet(hidden_channels=(4, 8), latent_channels=8, cond_dim=16)
    model.train()
    domain = Domain(3.0, "T1w")
    x = torch.randn(1, 1, 16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(model(x, domain, domain), x)

    final_loss = initial_loss
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, domain, domain)
        loss = torch.nn.functional.mse_loss(prediction, x)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()

    assert final_loss < initial_loss


def test_factory_builds_conditional_unet_translator_from_config() -> None:
    config = load_yaml_config(Path("configs/model/conditional_unet_translator.yaml"))
    translator_config = dict(config["translator"])
    name = translator_config.pop("name")

    translator_from_config = build_translator(name, **translator_config)
    translator = build_translator(
        "conditional_unet_field_translator",
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=2,
    )

    assert isinstance(translator, ConditionalUNetFieldTranslator)
    assert isinstance(translator_from_config, ConditionalUNetFieldTranslator)
