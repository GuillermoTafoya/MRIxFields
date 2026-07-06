import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import collate_raw_batches
from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.kl_vae import (
    _LOGVAR_MAX,
    _LOGVAR_MIN,
    KLVAEDecoder,
    KLVAEEncoder,
)
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


def test_vae_res_blocks_3d_forward_backward_no_nan() -> None:
    # The real config: latent_channels=4 at /4 spatial on a 64^3 patch => 16^3 x 4 latent
    # (16x compression), with residual blocks. Confirms shapes and a finite backward pass
    # before spending GPU-hrs.
    encoder = KLVAEEncoder(base_channels=8, latent_channels=4, spatial_dims=3, num_res_blocks=2)
    decoder = KLVAEDecoder(base_channels=8, latent_channels=4, spatial_dims=3, num_res_blocks=2)
    x = torch.randn(1, 1, 64, 64, 64)
    domain = _domain_pair(1)

    mean, logvar = encoder.encode_dist(x, domain)
    assert mean.shape == (1, 4, 16, 16, 16)
    assert torch.isfinite(mean).all() and torch.isfinite(logvar).all()

    z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
    y = decoder.decode(z, domain)
    assert y.shape == x.shape
    assert y.min() >= -1.0 and y.max() <= 1.0

    loss = torch.nn.functional.mse_loss(y, x)
    loss.backward()
    grads = [p.grad for p in list(encoder.parameters()) + list(decoder.parameters()) if p.grad is not None]
    assert grads, "expected gradients to flow"
    assert all(torch.isfinite(g).all() for g in grads)


def test_encode_dist_clamps_logvar() -> None:
    # Force a huge log-variance via the to_dist bias and confirm encode_dist clamps it,
    # so exp(logvar) can't overflow / the KL term can't run away.
    encoder = KLVAEEncoder(base_channels=8, latent_channels=4, spatial_dims=3, num_res_blocks=1)
    with torch.no_grad():
        encoder.to_dist.bias[encoder.latent_channels :] = 1e4  # logvar half of the output
        encoder.to_dist.bias[: encoder.latent_channels] = -1e4  # mean half, unaffected by clamp
    x = torch.randn(1, 1, 16, 16, 16)

    _, logvar = encoder.encode_dist(x)

    assert torch.isfinite(logvar).all()
    assert logvar.max() <= _LOGVAR_MAX + 1e-4
    assert logvar.min() >= _LOGVAR_MIN - 1e-4
    assert torch.isclose(logvar.max(), torch.tensor(_LOGVAR_MAX), atol=1e-3)


def test_grad_clip_norm_plumbs_and_trains() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    loader = _make_synthetic_3d_loader(num_samples=2, batch_size=2)
    config = Stage1VAEConfig(
        steps=2, batch_size=2, grad_clip_norm=0.5, loss_weights={"nrmse": 1.0, "kl": 1e-4}
    )
    assert config.grad_clip_norm == 0.5

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_early_stopping_halts_run_and_persists_state(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    loader = _make_synthetic_3d_loader(num_samples=4, batch_size=2)
    # min_delta=10.0 makes any real improvement fail the (1 - min_delta) threshold, so every
    # checkpoint after the baseline counts as "bad" -> deterministic stop after patience,
    # regardless of the actual loss trajectory. Verifies the loop actually breaks + persists.
    config = Stage1VAEConfig(
        steps=50,
        batch_size=2,
        loss_weights={"nrmse": 1.0, "kl": 1e-4},
        early_stopping=True,
        early_stopping_min_delta=10.0,
        early_stopping_patience=2,
        early_stopping_ema_decay=0.0,
        checkpoint_dir=tmp_path,
        checkpoint_every_steps=1,
        checkpoint_max_bytes=200_000_000,
    )

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert result.stopped_early is True
    assert result.steps < 50  # stopped well before the step budget
    # The final checkpoint carries the tracker state so resume_from can continue the count.
    last_ckpt = max(tmp_path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    state = torch.load(last_ckpt, weights_only=False)
    assert "early_stop" in state
    assert state["early_stop"]["num_bad_checkpoints"] >= 2


def test_num_res_blocks_flows_through_factory() -> None:
    encoder = build_encoder("kl_vae", base_channels=8, latent_channels=4, spatial_dims=3, num_res_blocks=3)
    assert isinstance(encoder, KLVAEEncoder)
    assert encoder.num_res_blocks == 3


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


def test_run_stage1_vae_train_3d_volume_with_ssim3d() -> None:
    # ssim_loss now dispatches to ssim3d (avg_pool3d) for 5D volumes — the 2D-only
    # limitation is gone, so a spatial_dims=3 config with ssim weight > 0 trains fine.
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    loader = _make_synthetic_3d_loader(num_samples=2, batch_size=2)
    config = Stage1VAEConfig(
        steps=1, batch_size=2, loss_weights={"ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4}
    )

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert result.steps == 1
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_run_stage1_vae_train_3d_volume_with_slice_lpips() -> None:
    # lpips now runs on 5D volumes via the slice-averaged variant (lpips_loss_3d) inside
    # _compute_vae_loss — a spatial_dims=3 config with lpips weight > 0 trains fine.
    pytest.importorskip("lpips")
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3)
    loader = _make_synthetic_3d_loader(num_samples=2, batch_size=2)
    config = Stage1VAEConfig(
        steps=1,
        batch_size=2,
        loss_weights={"ssim": 1.0, "nrmse": 1.0, "lpips": 1.0, "kl": 1e-4},
        lpips_num_slices=4,
    )

    result = run_stage1_vae_train(config, encoder=encoder, decoder=decoder, loader=loader)

    assert result.steps == 1
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


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
