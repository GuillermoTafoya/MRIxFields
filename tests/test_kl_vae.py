import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import collate_raw_batches
from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.models.factory import build_decoder, build_encoder
from fieldbridge.training.losses import kl_divergence
from fieldbridge.training.stage1_vae import Stage1VAEConfig, run_stage1_vae_train


def _domain_pair(batch_size: int) -> list[Domain]:
    return [Domain(3.0, "T1w") for _ in range(batch_size)]


class _Synthetic3DDataset(Dataset[RawBatch]):
    """Small 3D (C, D, H, W) dummy dataset — matches the real manifest path, whose
    NIfTI volumes are full 3D volumes with no slice-extraction step."""

    def __init__(self, *, num_samples: int = 4, volume_shape: tuple[int, int, int, int] = (1, 8, 8, 8), seed: int = 13) -> None:
        self.num_samples = num_samples
        self.volume_shape = volume_shape
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> RawBatch:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(self.volume_shape, generator=generator)
        domain = Domain(3.0, "T1w")
        return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"s{index}"})


def _make_synthetic_3d_loader(*, num_samples: int = 4, batch_size: int = 2) -> DataLoader[RawBatch]:
    dataset = _Synthetic3DDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_batches)


class _Synthetic2DDataset(Dataset[RawBatch]):
    """Small 2D (C, H, W) dummy dataset for stage1_vae smoke tests.

    Deliberately NOT SyntheticVolumeDataset, which is fixed to a 3D (C, D, H, W)
    convention (see docs/plans/fase-b-vae.md §3) — building a local, test-scoped 2D
    dataset avoids touching that shape convention.
    """

    def __init__(self, *, num_samples: int = 4, image_shape: tuple[int, int, int] = (1, 16, 16), seed: int = 13) -> None:
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> RawBatch:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(self.image_shape, generator=generator)
        domain = Domain(3.0, "T1w")
        return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"s{index}"})


def _make_synthetic_2d_loader(*, num_samples: int = 4, batch_size: int = 2) -> DataLoader[RawBatch]:
    dataset = _Synthetic2DDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_batches)


def test_encode_dist_shape_and_finite() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    x = torch.randn(2, 1, 32, 32)

    mean, logvar = encoder.encode_dist(x)

    assert mean.shape == (2, 6, 8, 8)
    assert logvar.shape == (2, 6, 8, 8)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(logvar).all()


def test_encoder_rejects_spatial_dims_not_divisible_by_downsample_factor() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    x = torch.randn(2, 1, 30, 30)

    with pytest.raises(ValueError):
        encoder.encode_dist(x)


def test_encode_decode_roundtrip_preserves_shape() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    x = torch.randn(2, 1, 32, 32)
    domain = _domain_pair(2)

    z = encoder.encode(x, domain)
    y = decoder.decode(z, domain)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_decoder_output_is_bounded_by_tanh() -> None:
    decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    # Large-magnitude latent, to confirm Tanh() actually clamps the range rather than
    # just happening to land in [-1, 1] for small inputs.
    z = torch.randn(2, 6, 8, 8) * 100.0
    domain = _domain_pair(2)

    y = decoder.decode(z, domain)

    assert y.min() >= -1.0
    assert y.max() <= 1.0


def test_decoder_rejects_wrong_latent_channels() -> None:
    decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    z = torch.randn(2, 4, 8, 8)

    with pytest.raises(ValueError):
        decoder.decode(z, _domain_pair(2))


def test_kl_divergence_on_encoder_output_is_finite_and_nonnegative() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    x = torch.randn(2, 1, 32, 32)

    mean, logvar = encoder.encode_dist(x)
    loss = kl_divergence(mean, logvar)

    assert torch.isfinite(loss)
    assert loss >= -1e-5


def test_encode_decode_roundtrip_3d_volume() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    x = torch.randn(1, 1, 8, 8, 8)
    domain = _domain_pair(1)

    z = encoder.encode(x, domain)
    y = decoder.decode(z, domain)

    assert z.shape == (1, 3, 2, 2, 2)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.min() >= -1.0
    assert y.max() <= 1.0


def test_encoder_3d_rejects_4d_input() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    x = torch.randn(1, 1, 8, 8)

    with pytest.raises(ValueError):
        encoder.encode_dist(x)


def test_factory_builds_kl_vae_by_name() -> None:
    assert isinstance(build_encoder("kl_vae", base_channels=8, latent_channels=6), KLVAEEncoder)
    assert isinstance(build_decoder("kl_vae", base_channels=8, latent_channels=6), KLVAEDecoder)


def test_run_stage1_vae_train_smoke() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    loader = _make_synthetic_2d_loader()
    config = Stage1VAEConfig(
        steps=3,
        batch_size=2,
        loss_weights={"ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4},
    )

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert result.steps == 3
    assert len(result.losses) == 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_run_stage1_vae_train_smoke_3d_volume() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    loader = _make_synthetic_3d_loader(num_samples=4, batch_size=2)
    config = Stage1VAEConfig(
        steps=2,
        batch_size=2,
        loss_weights={"ssim": 0.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4},
    )

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert result.steps == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_run_stage1_vae_train_3d_volume_raises_if_ssim_weight_nonzero() -> None:
    # evaluation.metrics.ssim is 2D-only by design (avg_pool2d-based) — confirms the
    # guard in _compute_vae_loss is load-bearing, not just an optimization, and that
    # spatial_dims=3 configs MUST keep ssim weight at 0 (nrmse+lpips+kl instead).
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    loader = _make_synthetic_3d_loader(num_samples=2, batch_size=2)
    config = Stage1VAEConfig(
        steps=1, batch_size=2, loss_weights={"ssim": 1.0, "nrmse": 0.0, "lpips": 0.0, "kl": 0.0}
    )

    with pytest.raises(ValueError):
        run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)


def test_run_stage1_vae_train_smoke_with_lpips() -> None:
    pytest.importorskip("lpips")
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    loader = _make_synthetic_2d_loader()
    config = Stage1VAEConfig(
        steps=2,
        batch_size=2,
        loss_weights={"ssim": 1.0, "nrmse": 1.0, "lpips": 1.0, "kl": 1e-4},
    )

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert result.steps == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_run_stage1_vae_train_checkpoint_and_resume(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    loader = _make_synthetic_2d_loader()
    config = Stage1VAEConfig(
        steps=2,
        batch_size=2,
        loss_weights={"ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4},
        checkpoint_dir=tmp_path,
        checkpoint_at_end=True,
    )

    run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1

    resumed_encoder = KLVAEEncoder(base_channels=8, latent_channels=6)
    resumed_decoder = KLVAEDecoder(base_channels=8, latent_channels=6)
    resume_config = Stage1VAEConfig(
        steps=1,
        batch_size=2,
        loss_weights={"ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4},
        resume_from=checkpoints[0],
    )
    result = run_stage1_vae_train(resume_config, encoder=resumed_encoder, decoder=resumed_decoder, loader=loader)
    assert result.steps == 1
