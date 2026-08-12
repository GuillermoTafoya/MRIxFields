from __future__ import annotations

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.models.discriminators import DomainProjectionDiscriminator, domain_labels
from fieldbridge.training.stage2_transport import Stage2TransportConfig, _validate_config


def test_projection_discriminator_supports_latents_and_images() -> None:
    domains = [Domain(0.1, Contrast.T1W), Domain(7.0, Contrast.T2_FLAIR)]
    for channels in (4, 1):
        model = DomainProjectionDiscriminator(channels, (8, 16))
        score, logits = model(torch.randn(2, channels, 16, 16, 16), domains)
        assert score.shape == (2,)
        assert logits.shape == (2, 15)


def test_projection_discriminator_bounds_large_image_inputs() -> None:
    model = DomainProjectionDiscriminator(1, (2,))
    domain = Domain(3.0, Contrast.T2W)
    score, _ = model(torch.randn(1, 1, 65, 65, 65), domain)
    assert score.shape == (1,)


def test_domain_labels_cover_joint_contrast_field_grid() -> None:
    domains = [Domain(0.1, Contrast.T1W), Domain(7.0, Contrast.T2_FLAIR)]
    assert domain_labels(domains, 2, torch.device("cpu")).tolist() == [0, 14]


@pytest.mark.parametrize("space", ["none", "latent", "image"])
def test_adversarial_space_is_validated(space: str) -> None:
    weights = {"adversarial": 0.1} if space != "none" else {}
    _validate_config(Stage2TransportConfig(adversarial_space=space, loss_weights=weights))


def test_adversarial_weight_requires_a_discriminator() -> None:
    with pytest.raises(ValueError, match="adversarial_space"):
        _validate_config(
            Stage2TransportConfig(
                adversarial_space="none",
                loss_weights={"flow": 1.0, "adversarial": 0.1},
            )
        )


def test_adversarial_training_requires_contrast_constrained_batches() -> None:
    with pytest.raises(ValueError, match="same_contrast"):
        _validate_config(
            Stage2TransportConfig(
                same_contrast=False,
                adversarial_space="latent",
                loss_weights={"adversarial": 0.1},
            )
        )
