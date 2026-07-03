import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import collate_raw_batches
from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.kl_vae import KLVAEEncoder
from fieldbridge.models.diffusion.denoising_unet import DenoisingUNet
from fieldbridge.models.diffusion.field_conditioner import FieldStrengthConditioner
from fieldbridge.models.diffusion.schedule import make_schedule, q_sample
from fieldbridge.models.diffusion.timestep_embedding import sinusoidal_timestep_embedding
from fieldbridge.training.stage2_diffuser import Stage2DiffuserConfig, run_stage2_diffuser_train


class _Synthetic2DDataset(Dataset[RawBatch]):
    """Small 2D (C, H, W) dummy dataset, test-scoped — see test_kl_vae.py for why this
    doesn't reuse SyntheticVolumeDataset (fixed to a 3D convention)."""

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


class _Synthetic3DDataset(Dataset[RawBatch]):
    """Small 3D (C, D, H, W) dummy dataset — matches the real manifest path (full
    volumes, no slice-extraction step)."""

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


def test_timestep_embedding_shape_and_finite() -> None:
    embedding = sinusoidal_timestep_embedding(torch.arange(5), embedding_dim=16)

    assert embedding.shape == (5, 16)
    assert torch.isfinite(embedding).all()


def test_timestep_embedding_distinct_for_distinct_timesteps() -> None:
    embedding = sinusoidal_timestep_embedding(torch.arange(5), embedding_dim=16)

    for i in range(1, 5):
        assert not torch.allclose(embedding[0], embedding[i])


def test_timestep_embedding_handles_odd_dim() -> None:
    embedding = sinusoidal_timestep_embedding(torch.arange(3), embedding_dim=15)

    assert embedding.shape == (3, 15)
    assert torch.isfinite(embedding).all()


def test_timestep_embedding_rejects_nonpositive_dim() -> None:
    with pytest.raises(ValueError):
        sinusoidal_timestep_embedding(torch.arange(3), embedding_dim=0)


def test_field_strength_conditioner_shape_and_finite() -> None:
    conditioner = FieldStrengthConditioner(conditioning_dim=16)
    domains = [Domain(0.1, "T2-FLAIR"), Domain(7.0, "T1w")]

    output = conditioner(domains)

    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_field_strength_conditioner_accepts_single_domain_with_batch_size() -> None:
    conditioner = FieldStrengthConditioner(conditioning_dim=16)

    output = conditioner(Domain(3.0, "T1w"), batch_size=3)

    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()


def test_q_sample_at_t_zero_is_close_to_x0() -> None:
    schedule = make_schedule(num_timesteps=100)
    x0 = torch.randn(4, 4, 8, 8)
    t = torch.zeros(4, dtype=torch.long)

    x_t, noise = q_sample(x0, t, schedule)

    # At t=0, alpha_bar_0 = 1 - beta_start ~= 0.9999, so x_t should be very close to x0.
    assert torch.allclose(x_t, x0, atol=0.05)
    assert noise.shape == x0.shape


def test_q_sample_at_large_t_is_dominated_by_noise() -> None:
    schedule = make_schedule(num_timesteps=100)
    x0 = torch.randn(4, 4, 8, 8)
    t = torch.full((4,), 99, dtype=torch.long)

    x_t, noise = q_sample(x0, t, schedule)

    # At t close to T, x_t should correlate much more with the injected noise than x0.
    corr_with_noise = torch.corrcoef(torch.stack([x_t.flatten(), noise.flatten()]))[0, 1]
    corr_with_x0 = torch.corrcoef(torch.stack([x_t.flatten(), x0.flatten()]))[0, 1]
    assert corr_with_noise.abs() > corr_with_x0.abs()


def test_q_sample_broadcasts_per_sample_timestep_correctly() -> None:
    schedule = make_schedule(num_timesteps=100)
    x0 = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 99], dtype=torch.long)

    x_t, _ = q_sample(x0, t, schedule, noise=noise)
    x_t_each = torch.stack(
        [q_sample(x0[i : i + 1], t[i : i + 1], schedule, noise=noise[i : i + 1])[0][0] for i in range(2)]
    )

    assert torch.allclose(x_t, x_t_each, atol=1e-5)


def test_q_sample_returns_finite_output() -> None:
    schedule = make_schedule(num_timesteps=100)
    x0 = torch.randn(3, 4, 8, 8)
    t = torch.randint(0, 100, (3,))

    x_t, noise = q_sample(x0, t, schedule)

    assert torch.isfinite(x_t).all()
    assert torch.isfinite(noise).all()


def test_denoising_unet_forward_shape_and_finite() -> None:
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    z_t = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 100, (2,))
    domains = [Domain(0.1, "T2-FLAIR"), Domain(3.0, "T1w")]

    eps_hat = unet(z_t, t, domains)

    assert eps_hat.shape == z_t.shape
    assert torch.isfinite(eps_hat).all()


def test_denoising_unet_forward_shape_num_levels_two() -> None:
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=2)
    z_t = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 100, (2,))
    domains = [Domain(0.1, "T2-FLAIR"), Domain(3.0, "T1w")]

    eps_hat = unet(z_t, t, domains)

    assert eps_hat.shape == z_t.shape
    assert torch.isfinite(eps_hat).all()


def test_denoising_unet_gradients_reach_conditioner_and_blocks() -> None:
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    z_t = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 100, (2,))
    domains = [Domain(0.1, "T2-FLAIR"), Domain(3.0, "T1w")]

    eps_hat = unet(z_t, t, domains)
    eps_hat.sum().backward()

    assert unet.field_conditioner.projection[0].weight.grad is not None
    assert torch.isfinite(unet.field_conditioner.projection[0].weight.grad).all()
    assert unet.blocks_level0[0].film.projection.weight.grad is not None
    assert torch.isfinite(unet.blocks_level0[0].film.projection.weight.grad).all()


def test_denoising_unet_forward_shape_and_finite_3d() -> None:
    unet = DenoisingUNet(latent_channels=3, base_channels=6, spatial_dims=3, num_levels=1)
    z_t = torch.randn(2, 3, 4, 4, 4)
    t = torch.randint(0, 100, (2,))
    domains = [Domain(0.1, "T2-FLAIR"), Domain(3.0, "T1w")]

    eps_hat = unet(z_t, t, domains)

    assert eps_hat.shape == z_t.shape
    assert torch.isfinite(eps_hat).all()


def test_denoising_unet_rejects_wrong_latent_channels() -> None:
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    z_t = torch.randn(2, 6, 8, 8)
    t = torch.randint(0, 100, (2,))

    with pytest.raises(ValueError):
        unet(z_t, t, [Domain(0.1, "T2-FLAIR"), Domain(3.0, "T1w")])


def test_run_stage2_diffuser_train_smoke_frozen_vae_blocks_encoder_gradients() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=4)
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    loader = _make_synthetic_2d_loader()
    config = Stage2DiffuserConfig(steps=3, batch_size=2, num_timesteps=10, train_vae_jointly=False)

    result = run_stage2_diffuser_train(config, unet=unet, encoder=encoder, loader=loader)

    assert result.steps == 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert all(p.grad is None for p in encoder.parameters())


def test_run_stage2_diffuser_train_smoke_joint_vae_reaches_encoder_gradients() -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=4)
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    loader = _make_synthetic_2d_loader()
    config = Stage2DiffuserConfig(steps=1, batch_size=2, num_timesteps=10, train_vae_jointly=True)

    result = run_stage2_diffuser_train(config, unet=unet, encoder=encoder, loader=loader)

    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert any(p.grad is not None for p in encoder.parameters())


def test_run_stage2_diffuser_train_smoke_3d_volume() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    unet = DenoisingUNet(latent_channels=3, base_channels=6, spatial_dims=3, num_levels=1)
    loader = _make_synthetic_3d_loader()
    config = Stage2DiffuserConfig(steps=2, batch_size=2, num_timesteps=10, train_vae_jointly=False)

    result = run_stage2_diffuser_train(config, unet=unet, encoder=encoder, loader=loader)

    assert result.steps == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_run_stage2_diffuser_train_checkpoint_and_resume(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=8, latent_channels=4)
    unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    loader = _make_synthetic_2d_loader()
    config = Stage2DiffuserConfig(
        steps=2, batch_size=2, num_timesteps=10, checkpoint_dir=tmp_path, checkpoint_at_end=True
    )

    run_stage2_diffuser_train(config, unet=unet, encoder=encoder, loader=loader)
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1

    resumed_unet = DenoisingUNet(latent_channels=4, base_channels=8, num_levels=1)
    resume_config = Stage2DiffuserConfig(steps=1, batch_size=2, num_timesteps=10, resume_from=checkpoints[0])
    result = run_stage2_diffuser_train(resume_config, unet=resumed_unet, encoder=encoder, loader=loader)
    assert result.steps == 1
