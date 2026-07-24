"""Contract tests for the official [0, 1] intensity format and stratified cropping.

Covers the migration off the old percentile-clip [-1, 1] convention: the official
MRIxFields2026 format ships [0, 1] volumes and forbids rescaling in training or
evaluation, so metrics stay comparable to the challenge leaderboard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fieldbridge.data.transforms import (
    StratifiedCropConfig,
    assert_official_unit_range,
    stratified_crop,
)
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder


def _brain_volume(
    shape: tuple[int, int, int] = (48, 56, 48), radii: tuple[int, int, int] = (12, 15, 13)
) -> tuple[torch.Tensor, torch.Tensor]:
    """A [0, 1] volume with a centered ellipsoid of nonzero intensity, plus its mask."""

    gz, gy, gx = torch.meshgrid(*[torch.arange(d) for d in shape], indexing="ij")
    inside = (
        ((gz - shape[0] / 2) / radii[0]) ** 2
        + ((gy - shape[1] / 2) / radii[1]) ** 2
        + ((gx - shape[2] / 2) / radii[2]) ** 2
    ) < 1.0
    volume = torch.zeros(1, *shape)
    volume[0][inside] = torch.rand(int(inside.sum())).clamp(0.05, 1.0)
    return volume, inside.float().unsqueeze(0)


def test_assert_official_unit_range_passes_through_unchanged() -> None:
    volume, _ = _brain_volume()
    result = assert_official_unit_range(volume)
    assert torch.equal(result, volume), "the official transform must not rescale intensity"


def test_assert_official_unit_range_rejects_rescaled_volume() -> None:
    volume, _ = _brain_volume()
    with pytest.raises(ValueError, match=r"\[0, 1\] intensity contract"):
        assert_official_unit_range(volume * 2.0 - 1.0)


def test_stratified_crop_shapes_and_dtype_preserved() -> None:
    volume, mask = _brain_volume()
    patch = (16, 16, 16)
    crop = stratified_crop(
        volume,
        patch_size=patch,
        mask=mask,
        config=StratifiedCropConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    assert tuple(crop.shape) == (1, *patch)
    assert crop.dtype == volume.dtype
    assert torch.isfinite(crop).all()


def test_stratified_crop_beats_uniform_on_foreground_rate() -> None:
    """The whole point of the sampler: most crops must actually contain brain."""

    volume, mask = _brain_volume()
    patch = (16, 16, 16)
    generator = torch.Generator().manual_seed(0)
    config = StratifiedCropConfig()
    crops = [
        stratified_crop(volume, patch_size=patch, mask=mask, config=config, generator=generator)
        for _ in range(200)
    ]
    with_foreground = sum(float((crop > 0).float().mean()) >= 0.1 for crop in crops) / len(crops)
    assert with_foreground > 0.5, f"stratified sampling left only {with_foreground:.2f} usable crops"


def test_stratified_crop_still_yields_pure_air_patches() -> None:
    """Air must stay represented — the challenge metrics score the whole volume.

    Uses a brain-to-FOV ratio close to the real data (~2% of a 364^3 volume). Not `all`:
    `stratified_crop` deliberately returns its last draw when a stratum can't be satisfied
    within `max_attempts`, so a run of off-quota patches is correct behavior, not a bug.
    """

    volume, mask = _brain_volume(radii=(8, 10, 9))
    generator = torch.Generator().manual_seed(0)
    config = StratifiedCropConfig(foreground=0.0, border=0.0, air=1.0)
    crops = [
        stratified_crop(volume, patch_size=(16, 16, 16), mask=mask, config=config, generator=generator)
        for _ in range(50)
    ]
    pure_air = sum(float(crop.max()) == 0.0 for crop in crops) / len(crops)
    assert pure_air > 0.9, f"air stratum only produced {pure_air:.2f} pure-air crops"


def test_stratified_crop_falls_back_instead_of_raising_when_stratum_unsatisfiable() -> None:
    """A volume with no air at all must not kill a multi-hour training run."""

    volume = torch.ones(1, 24, 24, 24)
    mask = torch.ones(1, 24, 24, 24)
    crop = stratified_crop(
        volume,
        patch_size=(8, 8, 8),
        mask=mask,
        config=StratifiedCropConfig(foreground=0.0, border=0.0, air=1.0, max_attempts=5),
        generator=torch.Generator().manual_seed(0),
    )
    assert tuple(crop.shape) == (1, 8, 8, 8)


def test_stratified_crop_rejects_mismatched_mask() -> None:
    volume, _ = _brain_volume()
    with pytest.raises(ValueError, match="does not match image spatial shape"):
        stratified_crop(
            volume,
            patch_size=(16, 16, 16),
            mask=torch.ones(1, 8, 8, 8),
            config=StratifiedCropConfig(),
        )


@pytest.mark.parametrize("bad", [{"foreground": -1.0}, {"min_foreground_fraction": 0.0}, {"max_attempts": 0}])
def test_stratified_crop_config_validates(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StratifiedCropConfig(**bad)


def test_stratified_crop_config_from_mapping_none_means_uniform() -> None:
    assert StratifiedCropConfig.from_mapping(None) is None
    assert StratifiedCropConfig.from_mapping({}) is None
    assert StratifiedCropConfig.from_mapping({"air": 0.3}).air == pytest.approx(0.3)


def test_decoder_linear_head_can_represent_exact_zero_background() -> None:
    """A saturating head cannot emit exactly 0; the linear default must be able to."""

    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    assert decoder.output_activation == "none"
    out = decoder.decode(torch.randn(1, 3, 4, 4, 4), None)
    # Drive the head's bias to a large negative value: a linear head reaches <= 0, a
    # sigmoid/tanh head never does.
    with torch.no_grad():
        decoder.to_image.bias.fill_(-50.0)
    driven = decoder.decode(torch.zeros(1, 3, 4, 4, 4), None)
    assert torch.isfinite(out).all()
    assert float(driven.max()) <= 0.0


def test_decoder_bounded_heads_available_for_ablation() -> None:
    for name, lower in (("sigmoid", 0.0), ("clamp", 0.0)):
        decoder = KLVAEDecoder(
            base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1, output_activation=name
        )
        out = decoder.decode(torch.randn(1, 3, 4, 4, 4), None)
        assert float(out.min()) >= lower
        assert float(out.max()) <= 1.0


def test_decoder_rejects_unknown_output_activation() -> None:
    with pytest.raises(ValueError, match="output_activation must be one of"):
        KLVAEDecoder(base_channels=4, latent_channels=3, output_activation="relu")


def test_l1_term_is_active_by_default_and_finite() -> None:
    """L1 is the intensity anchor added to the recipe — it must be in the default weights
    and contribute a finite term to the loss."""

    from fieldbridge.training.stage1_vae import (
        DEFAULT_VAE_LOSS_WEIGHTS,
        Stage1VAEConfig,
        _compute_vae_loss,
    )

    assert DEFAULT_VAE_LOSS_WEIGHTS.get("l1", 0.0) > 0.0

    volume, _ = _brain_volume(shape=(16, 16, 16), radii=(5, 5, 5))
    batch = SimpleNamespace(image=volume.unsqueeze(0), source_domain=None)
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)

    with_l1 = Stage1VAEConfig(loss_weights={"l1": 1.0, "nrmse": 0.0, "ssim": 0.0, "lpips": 0.0, "kl": 0.0})
    without_l1 = Stage1VAEConfig(loss_weights={"l1": 0.0, "nrmse": 0.0, "ssim": 0.0, "lpips": 0.0, "kl": 0.0})

    torch.manual_seed(0)
    loss_with = _compute_vae_loss(encoder, decoder, batch, with_l1, lpips_net=None)
    torch.manual_seed(0)
    loss_without = _compute_vae_loss(encoder, decoder, batch, without_l1, lpips_net=None)

    assert torch.isfinite(loss_with).all()
    assert float(loss_with) > 0.0
    assert float(loss_without) == 0.0, "with every weight zero the loss must be exactly zero"


def test_compute_vae_loss_components_breaks_down_and_sums_to_total() -> None:
    """The per-term dict (for logging/validation) must contain each active term and its
    weighted sum must equal the scalar total used for backward."""

    from fieldbridge.training.stage1_vae import Stage1VAEConfig, _compute_vae_loss_components

    volume, _ = _brain_volume(shape=(16, 16, 16), radii=(5, 5, 5))
    batch = SimpleNamespace(image=volume.unsqueeze(0), source_domain=None)
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    weights = {"l1": 1.0, "ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4}
    cfg = Stage1VAEConfig(loss_weights=weights)

    torch.manual_seed(0)
    components = _compute_vae_loss_components(encoder, decoder, batch, cfg, lpips_net=None)

    for term in ("l1", "ssim", "nrmse", "kl", "total"):
        assert term in components and torch.isfinite(components[term]).all()
    assert "lpips" not in components, "a zero-weight term must not be computed"
    manual = sum(weights.get(name, 0.0) * components[name] for name in components if name != "total")
    assert torch.allclose(components["total"], manual, atol=1e-6)


def test_per_epoch_validation_writes_history_and_best_checkpoint(tmp_path) -> None:
    """Wiring a val loader must produce a per-epoch history.jsonl (train+val term
    breakdown) and a best-by-validation checkpoint."""

    import json

    from torch.utils.data import DataLoader
    from fieldbridge.data.contracts import RawBatch
    from fieldbridge.data.datasets import collate_raw_batches
    from fieldbridge.data.domains import Domain
    from fieldbridge.training.stage1_vae import Stage1VAEConfig, run_stage1_vae_train

    class _Vols(torch.utils.data.Dataset):
        def __init__(self, n: int, seed: int) -> None:
            self.n, self.seed = n, seed

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, index: int) -> RawBatch:
            gen = torch.Generator().manual_seed(self.seed * 100 + index)
            image = torch.rand(1, 16, 16, 16, generator=gen)
            domain = Domain(3.0, "T1w")
            return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"c{index}"})

    train = DataLoader(_Vols(8, 1), batch_size=2, collate_fn=collate_raw_batches)
    val = DataLoader(_Vols(4, 2), batch_size=2, collate_fn=collate_raw_batches)
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    cfg = Stage1VAEConfig(
        steps=8,
        batch_size=2,
        steps_per_epoch=4,
        device="cpu",
        precision="fp32",
        loss_weights={"l1": 1.0, "ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 1e-4},
        checkpoint_dir=tmp_path,
        checkpoint_max_bytes=50_000_000,
        # This legacy instrumentation assertion exercises promoted persistence;
        # the scientific v3 configurations retain the stricter 3/4 gate.
        promotion_min_active_channels=1,
    )

    run_stage1_vae_train(cfg, encoder=encoder, decoder=decoder, loader=train, val_loader=val)

    history = tmp_path / "history.jsonl"
    assert history.exists()
    lines = [json.loads(line) for line in history.read_text().strip().splitlines()]
    assert len(lines) == 2  # steps / steps_per_epoch
    for entry in lines:
        assert {"epoch", "step", "train", "validation", "best"} <= set(entry)
        assert "total" in entry["train"] and "total" in entry["validation"]
        assert "ssim3d_proxy" in entry["validation"]
    assert (tmp_path / "vae_kl_vae_best.pt").exists()


def test_no_validation_without_val_loader(tmp_path) -> None:
    """Manifest/patch-bank training (no split) keeps the original no-validation behavior."""

    from torch.utils.data import DataLoader
    from fieldbridge.data.contracts import RawBatch
    from fieldbridge.data.datasets import collate_raw_batches
    from fieldbridge.data.domains import Domain
    from fieldbridge.training.stage1_vae import Stage1VAEConfig, run_stage1_vae_train

    class _Vols(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> RawBatch:
            image = torch.rand(1, 16, 16, 16, generator=torch.Generator().manual_seed(index))
            domain = Domain(3.0, "T1w")
            return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"c{index}"})

    loader = DataLoader(_Vols(), batch_size=2, collate_fn=collate_raw_batches)
    cfg = Stage1VAEConfig(steps=4, batch_size=2, steps_per_epoch=4, device="cpu", precision="fp32",
                          loss_weights={"nrmse": 1.0, "kl": 1e-4}, checkpoint_dir=tmp_path,
                          checkpoint_max_bytes=50_000_000)
    run_stage1_vae_train(cfg, encoder=KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1),
                         decoder=KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1),
                         loader=loader, val_loader=None)
    assert not (tmp_path / "history.jsonl").exists()


def test_vae_forward_on_unit_range_input_is_finite() -> None:
    """Dummy forward sanity check before any real run: shapes line up, nothing is NaN."""

    volume, _ = _brain_volume(shape=(16, 16, 16), radii=(5, 5, 5))
    batch = volume.unsqueeze(0)
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    mean, logvar = encoder.encode_dist(batch)
    recon = decoder.decode(mean, None)
    assert tuple(recon.shape) == tuple(batch.shape)
    for name, tensor in (("mean", mean), ("logvar", logvar), ("recon", recon)):
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"
