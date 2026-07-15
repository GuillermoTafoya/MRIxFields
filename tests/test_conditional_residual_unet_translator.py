from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import SliceGeometry
from fieldbridge.data.pseudo_pairs import (
    PseudoPairSliceSample,
    collate_pseudo_pair_slices,
)
from fieldbridge.evaluation.pseudo_pairs import PseudoPairEvalConfig, evaluate_pseudo_pairs
from fieldbridge.models.factory import build_translator
from fieldbridge.models.translators.conditional_residual_unet import (
    ConditionalResidualUNetFieldTranslator,
)
from fieldbridge.models.translators.conditional_unet import ConditionalUNetFieldTranslator
from fieldbridge.training.checkpoints import load_checkpoint, save_checkpoint


TARGET_FIELDS = (1.5, 3.0, 5.0, 7.0)


def _small_residual(
    *, model_range: str = "minus_one_one"
) -> ConditionalResidualUNetFieldTranslator:
    return ConditionalResidualUNetFieldTranslator(
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=2,
        model_range=model_range,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("target_field", TARGET_FIELDS)
def test_residual_unet_is_exact_identity_at_initialization_for_every_target(
    target_field: float,
) -> None:
    model = _small_residual()
    x = torch.linspace(-1.0, 1.0, 2 * 1 * 15 * 17).reshape(2, 1, 15, 17)

    prediction = model(
        x,
        Domain(0.1, "T2-FLAIR"),
        Domain(target_field, "T2-FLAIR"),
    )

    assert torch.equal(prediction, x)
    assert torch.count_nonzero(model.backbone.output_projection.weight) == 0
    assert torch.count_nonzero(model.backbone.output_projection.bias) == 0


@pytest.mark.parametrize(
    ("model_range", "lower", "upper"),
    [("zero_one", 0.0, 1.0), ("minus_one_one", -1.0, 1.0)],
)
def test_residual_unet_preserves_shape_and_bounds_output(
    model_range: str,
    lower: float,
    upper: float,
) -> None:
    model = _small_residual(model_range=model_range)
    source = Domain(0.1, "T2-FLAIR")
    target = Domain(7.0, "T2-FLAIR")
    x = torch.full((2, 1, 15, 17), (lower + upper) * 0.5)

    with torch.no_grad():
        model.backbone.output_projection.bias.fill_(10.0)
        upper_prediction = model(x, source, target)
        model.backbone.output_projection.bias.fill_(-10.0)
        lower_prediction = model(x, source, target)

    assert upper_prediction.shape == lower_prediction.shape == x.shape
    assert torch.all(upper_prediction == upper)
    assert torch.all(lower_prediction == lower)


def test_residual_unet_rejects_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="matching input and output channels"):
        ConditionalResidualUNetFieldTranslator(in_channels=1, out_channels=2)


def test_residual_and_conditioning_paths_receive_gradients_after_head_update() -> None:
    torch.manual_seed(23)
    model = _small_residual()
    model.train()
    source = Domain(0.1, "T2-FLAIR")
    target_domain = Domain(7.0, "T2-FLAIR")
    x = torch.linspace(-0.5, 0.5, 2 * 1 * 16 * 16).reshape(2, 1, 16, 16)
    target = x + 0.2
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    first_loss = torch.nn.functional.mse_loss(model(x, source, target_domain), target)
    first_loss.backward()
    head = model.backbone.output_projection
    assert head.weight.grad is not None
    assert torch.count_nonzero(head.weight.grad) > 0
    optimizer.step()
    assert torch.count_nonzero(head.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    second_loss = torch.nn.functional.mse_loss(model(x, source, target_domain), target)
    second_loss.backward()

    conditioning_gradients = [
        parameter.grad
        for parameter in model.backbone.domain_embedding.parameters()
        if parameter.grad is not None
    ]
    residual_gradients = [
        parameter.grad
        for name, parameter in model.backbone.named_parameters()
        if not name.startswith("domain_embedding") and parameter.grad is not None
    ]
    assert any(torch.count_nonzero(gradient) > 0 for gradient in conditioning_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in residual_gradients)


def test_legacy_unet_factory_and_checkpoint_roundtrip_are_unchanged(tmp_path: Path) -> None:
    torch.manual_seed(29)
    kwargs = {
        "hidden_channels": (4,),
        "latent_channels": 4,
        "cond_dim": 12,
        "spatial_dims": 2,
        "final_activation": "tanh",
    }
    legacy = build_translator("conditional_unet_field_translator", **kwargs)
    assert type(legacy) is ConditionalUNetFieldTranslator
    legacy_state_keys = tuple(legacy.state_dict())
    assert legacy_state_keys
    assert all(not key.startswith("backbone.") for key in legacy_state_keys)

    checkpoint = tmp_path / "legacy-unet.pt"
    save_checkpoint(
        checkpoint,
        {"model": legacy.state_dict(), "model_class": type(legacy).__name__},
    )
    loaded = load_checkpoint(checkpoint)
    restored = ConditionalUNetFieldTranslator(**kwargs)
    incompatible = restored.load_state_dict(loaded["model"], strict=True)

    x = torch.randn(2, 1, 16, 16)
    source = Domain(0.1, "T2-FLAIR")
    target = Domain(7.0, "T2-FLAIR")
    legacy.eval()
    restored.eval()
    with torch.no_grad():
        expected = legacy(x, source, target)
        actual = restored(x, source, target)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(actual, expected)

    residual = build_translator(
        "conditional_residual_unet_field_translator",
        hidden_channels=(4,),
        latent_channels=4,
        cond_dim=12,
        spatial_dims=2,
        model_range="minus_one_one",
    )
    assert type(residual) is ConditionalResidualUNetFieldTranslator
    assert all(key.startswith("backbone.") for key in residual.state_dict())


class _StepZeroDataset(Dataset[PseudoPairSliceSample]):
    def __len__(self) -> int:
        return len(TARGET_FIELDS)

    def __getitem__(self, index: int) -> PseudoPairSliceSample:
        target_field = TARGET_FIELDS[index]
        x_low = torch.linspace(-0.8, 0.8, 64).reshape(1, 8, 8)
        x_high = (x_low * 0.5 + 0.1).clamp(-1.0, 1.0)
        return PseudoPairSliceSample(
            x_low=x_low,
            x_high=x_high,
            mask=torch.ones_like(x_low),
            source_domain=Domain(0.1, "T2-FLAIR"),
            target_domain=Domain(target_field, "T2-FLAIR"),
            record_id=f"record-{index}",
            subject_id=f"subject-{index}",
            volume_path=f"volume-{index}.nii.gz",
            slice_index=index,
            degradation_seed=100 + index,
            degradation_strength=0.5,
            geometry=SliceGeometry(
                slice_index=index,
                original_height=8,
                original_width=8,
                resized_height=8,
                resized_width=8,
                output_height=8,
                output_width=8,
            ),
        )


def test_step_zero_metrics_equal_degraded_baseline() -> None:
    loader = DataLoader(
        _StepZeroDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_pseudo_pair_slices,
    )
    payload = evaluate_pseudo_pairs(
        _small_residual(),
        loader,
        PseudoPairEvalConfig(
            model_range="minus_one_one",
            lpips="off",
            target_fields=TARGET_FIELDS,
        ),
    )

    assert payload["aggregate"]["predicted"] == payload["aggregate"]["degraded"]
    assert payload["macro_average"]["predicted"] == payload["macro_average"]["degraded"]
    volume_summary = payload["sampled_slice_per_volume"]
    assert volume_summary["macro_average"]["predicted"] == volume_summary[
        "macro_average"
    ]["degraded"]
    for field_payload in volume_summary["per_target_field"].values():
        assert field_payload["predicted"] == field_payload["degraded"]
