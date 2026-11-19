from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Contrast, Domain
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.data.photometry_factored_latent_bank import (
    derive_encoder_local_support_rule,
    propagate_encoder_local_valid_core_support,
)
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SOURCE_MODULES,
    PhotometryFitVolume,
    SourceCanonicalizedVolume,
    canonical_tensor_sha256,
    fit_frozen_photometry,
    sha256_text,
)
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256
from fieldbridge.evaluation import zero_transport_diagnostic as diagnostic
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder


class SyntheticPhotometry:
    artifact_sha256 = "a" * 64

    def normalize_source(self, source, domain):
        # Some real foreground values map to canonical zero. The source mask survives.
        return SourceCanonicalizedVolume(
            values=(source / 2 - 0.125).clamp_min(0),
            support_mask=source != 0,
            source_domain=domain,
            artifact_sha256=self.artifact_sha256,
        )


class PosteriorEncoder(nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.register_buffer("mean", mean)
        self.inputs = []

    def encode_dist(self, volume, domain):
        assert not torch.is_autocast_enabled("cpu")
        assert volume.dtype == torch.float32
        self.inputs.append(volume.clone())
        return self.mean, torch.full_like(self.mean, 7.0)


class FullVolumeDecoder(nn.Module):
    output_activation = "none"

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([1.0, 0.25]).view(1, 2, 1, 1, 1))
        self.latents = []
        self.images = []
        self.domains = []

    def decode(self, latent, domain):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        self.domains.append(domain if isinstance(domain, Domain) else list(domain)[0])
        self.latents.append(latent.clone())
        coarse = F.conv3d(latent, self.weight)
        assert coarse.dtype == torch.bfloat16
        # Full-volume coupling lets changes outside the core affect its interior.
        image = F.interpolate(coarse + coarse.mean() / 1024, scale_factor=2, mode="nearest")
        self.images.append(image.float().clone())
        return image


def rmse(delta):
    return delta.double().square().mean().sqrt().item()


@pytest.mark.parametrize(("native_exterior", "sign"), [(0.125, 1), (9.0, -1)])
def test_zero_transport_matches_unclipped_full_foreground_rmse(
    monkeypatch, native_exterior, sign
):
    domain = Domain(3.0, Contrast.T2W)
    source = torch.full((1, 1, 4, 4, 4), 0.75)
    source[..., 0, :, 0] = 0
    source[..., 1, :, 1] = 0.25
    photometry = SyntheticPhotometry()
    canonical = photometry.normalize_source(source, domain).values
    core = torch.zeros((2, 2, 2), dtype=torch.bool)
    core[0, 0, 0] = core[1, 0, 1] = True
    native = torch.full((1, 2, 2, 2, 2), -0.5)
    # A BF16 midpoint makes the small FP32 round-trip error visible after decoding.
    native[:, 0] = torch.where(core, 0.12548828125, native_exterior)
    encoder = PosteriorEncoder(native.clone()).eval()
    decoder = FullVolumeDecoder().eval()
    rule_hash = "b" * 64

    def support_rule(mask, model, *, expected_rule_sha256):
        assert model is encoder
        assert expected_rule_sha256 == rule_hash
        assert torch.equal(mask.squeeze(), source.squeeze() != 0)
        return core.clone()

    monkeypatch.setattr(diagnostic, "propagate_encoder_local_valid_core_support", support_rule)
    stats = FactoredLatentStats(
        mean=torch.tensor([2.89, -2.0]),
        std=torch.tensor([0.625, 0.75]),
        supported_count=torch.tensor([100, 100]),
        artifact_sha256=photometry.artifact_sha256,
    )
    result = diagnostic.run_zero_transport_scan(
        source, domain, artifact=photometry, encoder=encoder, decoder=decoder,
        stats=stats, support_rule_sha256=rule_hash, device=torch.device("cpu"),
    )

    assert len(encoder.inputs) == 1
    assert torch.equal(encoder.inputs[0], canonical)
    assert torch.equal(encoder.mean, native)
    assert decoder.domains == [domain, domain, domain]
    a, b, c = decoder.latents
    assert torch.equal(a, native)
    mean = stats.mean.view(1, 2, 1, 1, 1)
    std = stats.std.view(1, 2, 1, 1, 1)
    assert torch.equal(b, torch.where(core, ((native - mean) / std) * std + mean, mean))
    assert torch.equal(c, torch.where(core, b, native))
    # The affine round trip introduces a small FP32 difference *inside* the core.
    # This detects an implementation that silently makes arm C a copy of arm A.
    assert (c - a).abs().max().item() > 0
    expanded_core = core.expand_as(native)
    for name, region in {
        "all": torch.ones_like(expanded_core),
        "core": expanded_core,
        "outside_core": ~expanded_core,
    }.items():
        delta = (c - a)[region]
        metrics = result["latent_c_vs_a"][name]
        assert metrics["count"] == delta.numel()
        assert metrics["rmse"] == pytest.approx(rmse(delta))
        assert metrics["max_abs"] == pytest.approx(delta.abs().max().item())

    a_image, b_image, c_image = decoder.images
    foreground = source != 0
    projected = F.interpolate(core[None, None].float(), scale_factor=2, mode="nearest").bool()
    assert torch.any(foreground & (canonical == 0))
    assert b_image[foreground].max() > 1
    for name, region in {
        "foreground": foreground,
        "projected_core_foreground": foreground & projected,
        "remaining_foreground": foreground & ~projected,
    }.items():
        metrics = result["image_strata"][name]
        errors = [rmse((image - canonical)[region]) for image in decoder.images]
        assert metrics["count"] == int(region.sum())
        for arm, error in zip(("a", "b", "c"), errors, strict=True):
            assert metrics[f"{arm}_rmse"] == pytest.approx(error)
        assert metrics["b_minus_a_rmse"] == pytest.approx(errors[1] - errors[0])
        assert metrics["c_minus_a_rmse"] == pytest.approx(errors[2] - errors[0])
        restored_delta = (c_image - a_image)[region]
        assert metrics["c_vs_a_rmse"] == pytest.approx(rmse(restored_delta))
        assert metrics["c_vs_a_max_abs"] == pytest.approx(restored_delta.abs().max().item())
    full_image_delta = c_image - a_image
    assert full_image_delta.abs().max() > 0
    full_image = result["full_image_c_vs_a"]
    assert full_image["count"] == source.numel()
    assert full_image["max_abs"] == pytest.approx(full_image_delta.abs().max().item())
    assert full_image["rmse"] == pytest.approx(rmse(full_image_delta))
    expected = rmse((b_image - canonical)[foreground]) - rmse((a_image - canonical)[foreground])
    assert result["primary_b_minus_a_rmse"] == pytest.approx(expected)
    assert sign * expected > 0


def test_full_volume_klvae_with_fitted_photometry_and_real_support():
    # All 15 training templates are fit using tiny synthetic intensity ramps.
    samples = []
    for contrast in CONTRASTS:
        for field in FIELD_STRENGTHS_T:
            domain = Domain(field, contrast)
            identity = f"R_train_{domain.label}"
            samples.append(PhotometryFitVolume(
                volume=torch.linspace(0.05, 0.5 + field / 20, 8).reshape(2, 2, 2),
                domain=domain,
                record_identity=identity,
                subject_identity=identity,
                metadata_prefix="R",
                source_path_identity=f"synthetic/{identity}",
                source_file_sha256=sha256_text(identity),
            ))
    artifact = fit_frozen_photometry(
        samples,
        source_split_file_sha256="a" * 64,
        source_membership_fingerprint="synthetic-membership",
        source_recovery_fingerprint="synthetic-recovery",
        code_commit="synthetic-commit",
        code_provenance={
            "git_head": "synthetic-commit",
            "checkout_clean": True,
            "module_sha256": {name: sha256_text(name) for name in PHOTOMETRY_SOURCE_MODULES},
        },
        resolved_config={"contract": "stage2-photometry-variant-a-config-v1"},
        num_quantiles=5,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = dict(base_channels=2, latent_channels=2, spatial_dims=3, num_res_blocks=1)
        encoder = KLVAEEncoder(**model).requires_grad_(False)
        decoder = KLVAEDecoder(**model).requires_grad_(False)
    source = torch.linspace(0, 1, 24**3).reshape(1, 1, 24, 24, 24)
    domain = Domain(3.0, Contrast.T2W)
    canonical = artifact.normalize_source(source, domain)
    rule_hash = derive_encoder_local_support_rule(encoder)["rule_sha256"]
    support = propagate_encoder_local_valid_core_support(
        canonical.support_mask, encoder, expected_rule_sha256=rule_hash
    )
    sidecar = {
        "source_shape": list(source.shape),
        "source_loaded_array_sha256": canonical_tensor_sha256(source),
        "canonical_shape": list(canonical.values.shape),
        "canonical_tensor_sha256": canonical_tensor_sha256(canonical.values),
        "source_support_shape": list(canonical.support_mask.shape),
        "source_support_tensor_sha256": storage_tensor_sha256(canonical.support_mask),
        "local_valid_core_support_shape": list(support.shape),
        "local_valid_core_support_mask_sha256": storage_tensor_sha256(support),
    }
    stats = FactoredLatentStats(
        torch.tensor([0.25, -0.25]), torch.tensor([0.7, 0.9]),
        torch.tensor([100, 100]), artifact.artifact_sha256,
    )

    def run(seal):
        return diagnostic.run_zero_transport_scan(
            source, domain, artifact=artifact, encoder=encoder, decoder=decoder,
            stats=stats, support_rule_sha256=rule_hash, device="cpu", expected_sidecar=seal,
        )

    with patch.object(encoder, "encode_dist", wraps=encoder.encode_dist) as encode:
        result = run(sidecar)
        for key in (
            "canonical_tensor_sha256", "source_support_tensor_sha256",
            "local_valid_core_support_mask_sha256",
        ):
            with pytest.raises(ValueError, match=key):
                run({**sidecar, key: "0" * 64})
        encode.assert_called_once()
    strata = result["image_strata"]
    core_count = strata["projected_core_foreground"]["count"]
    remaining_count = strata["remaining_foreground"]["count"]
    assert core_count > 0 and remaining_count > 0
    assert core_count + remaining_count == strata["foreground"]["count"] == 24**3 - 1
    assert all(math.isfinite(value) for metrics in strata.values() for value in metrics.values())
    assert math.isfinite(result["primary_b_minus_a_rmse"])
    latent = result["latent_c_vs_a"]
    assert latent["outside_core"]["count"] > 0
    assert latent["outside_core"]["max_abs"] == 0
    assert latent["outside_core"]["rmse"] == 0
    raw_foreground = result["metrics"]["native_image_vs_measured/foreground"]
    assert raw_foreground["count"] == strata["foreground"]["count"]
    assert raw_foreground["rmse"] == strata["foreground"]["a_rmse"]
    assert 0 <= raw_foreground["mae"] <= raw_foreground["max_abs"]
