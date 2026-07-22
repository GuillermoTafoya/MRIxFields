import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import collate_raw_batches
from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.losses import foreground_weighted_l1_loss
from fieldbridge.training.stage1_vae import (
    Stage1VAEConfig,
    _lr_at_step,
    _weighted_terms,
    run_stage1_vae_train,
)


def _loader(num_samples: int = 4, batch_size: int = 2) -> DataLoader[RawBatch]:
    domain = Domain(3.0, "T1w")

    class _DS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return num_samples

        def __getitem__(self, index: int) -> RawBatch:
            generator = torch.Generator().manual_seed(index)
            image = torch.rand(1, 8, 8, 8, generator=generator)  # [0,1] contract
            return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"c{index}"})

    return DataLoader(_DS(), batch_size=batch_size, shuffle=False, collate_fn=collate_raw_batches)


# --- LR schedule --------------------------------------------------------------------


def test_constant_schedule_returns_base_lr_everywhere() -> None:
    cfg = Stage1VAEConfig(lr=1e-3, lr_schedule="constant")
    assert _lr_at_step(1, cfg, total_steps=1000) == 1e-3
    assert _lr_at_step(999, cfg, total_steps=1000) == 1e-3


def test_cosine_schedule_warms_up_then_decays_to_floor() -> None:
    cfg = Stage1VAEConfig(lr=1e-3, lr_schedule="cosine", lr_warmup_steps=10, lr_min_factor=0.1)
    total = 100
    warm = _lr_at_step(5, cfg, total_steps=total)  # mid-warmup
    peak = _lr_at_step(10, cfg, total_steps=total)  # end of warmup
    mid = _lr_at_step(55, cfg, total_steps=total)
    end = _lr_at_step(total, cfg, total_steps=total)

    assert warm < peak  # warmup ramps up
    assert abs(peak - 1e-3) < 1e-9  # reaches base lr at end of warmup
    assert peak > mid > end  # cosine decays after warmup
    assert abs(end - 1e-3 * 0.1) < 1e-6  # floors at lr * lr_min_factor


# --- weighted term logging ----------------------------------------------------------


def test_weighted_terms_multiplies_only_loss_terms() -> None:
    means = {"l1": 0.5, "kl": 2.0, "nrmse": 0.3, "total": 9.0, "num_batches": 3, "ssim3d": 0.8}
    weights = {"l1": 1.0, "kl": 1e-4, "nrmse": 1.0}
    weighted = _weighted_terms(means, weights)

    assert weighted == {"l1": 0.5, "kl": 2.0 * 1e-4, "nrmse": 0.3}
    assert "total" not in weighted and "num_batches" not in weighted and "ssim3d" not in weighted


# --- foreground-weighted L1 ---------------------------------------------------------


def test_foreground_weight_one_equals_plain_l1() -> None:
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    fg = foreground_weighted_l1_loss(pred, target, threshold=0.0, foreground_weight=1.0)
    assert torch.allclose(fg, F.l1_loss(pred, target), atol=1e-6)


def test_foreground_weight_upweights_foreground_and_stays_finite() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[..., :2, :] = 0.8  # top half foreground
    pred = torch.zeros_like(target)  # miss everywhere
    weighted = foreground_weighted_l1_loss(target * 0 + pred, target, threshold=0.0, foreground_weight=5.0)
    plain = F.l1_loss(pred, target)
    # Error lives only in the foreground, which is now up-weighted => higher than plain mean.
    assert torch.isfinite(weighted)
    assert float(weighted) > float(plain)


# --- end-to-end: validation history carries per-term-weighted + latent + best ckpt ---


def test_validation_history_has_weighted_latent_and_best_checkpoint(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    config = Stage1VAEConfig(
        steps=4,
        batch_size=2,
        steps_per_epoch=2,  # => 2 epochs
        loss_weights={"l1": 1.0, "nrmse": 1.0, "ssim": 0.0, "lpips": 0.0, "kl": 1e-4},
        checkpoint_dir=tmp_path,
        checkpoint_max_bytes=200_000_000,
        recon_dump_every_epochs=1,
        val_every_epochs=1,
    )

    run_stage1_vae_train(
        config, encoder=encoder, decoder=decoder, loader=_loader(), val_loader=_loader()
    )

    history = (tmp_path / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 2
    entry = json.loads(history[0])
    assert "train_weighted" in entry and "validation_weighted" in entry
    # weighted == raw * weight (l1 weight is 1.0, so they match; kl weight scales it down).
    assert abs(entry["train_weighted"]["l1"] - entry["train"]["l1"]) < 1e-9
    assert entry["latent"]["num_dims"] == 3
    assert "active_units" in entry["latent"]

    assert (tmp_path / "vae_kl_vae_best.pt").exists()
    assert (tmp_path / "recon_epoch0001.png").exists()  # recon hook fired


def test_lr_schedule_default_is_constant_and_preserves_behavior() -> None:
    # A default config must not touch the optimizer lr curve — cfg.lr straight through.
    cfg = Stage1VAEConfig()
    assert cfg.lr_schedule == "constant"
    assert _lr_at_step(1, cfg, total_steps=100) == cfg.lr
    assert cfg.foreground_loss_weighting is False


# --- val-based early stopping (item 5) -----------------------------------------------


def test_val_early_stopping_config_parsed_from_training() -> None:
    cfg = Stage1VAEConfig.from_mapping(
        {"training": {"val_early_stopping": True, "val_early_stopping_patience": 7}}
    )
    assert cfg.val_early_stopping is True
    assert cfg.val_early_stopping_patience == 7
    # Default preserved when unset.
    assert Stage1VAEConfig().val_early_stopping is False


def test_val_early_stopping_does_not_stop_when_patience_exceeds_validations(tmp_path) -> None:
    # Wiring guard: with patience larger than the number of validations the run can never trip,
    # so it completes the full schedule (no spurious early stop).
    encoder = KLVAEEncoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=3, spatial_dims=3, num_res_blocks=1)
    config = Stage1VAEConfig(
        steps=6,
        batch_size=2,
        steps_per_epoch=2,  # => 3 epochs / 3 validations
        loss_weights={"l1": 1.0, "nrmse": 1.0, "ssim": 0.0, "lpips": 0.0, "kl": 1e-4},
        checkpoint_dir=tmp_path,
        checkpoint_max_bytes=200_000_000,
        recon_dump_every_epochs=0,
        val_every_epochs=1,
        val_early_stopping=True,
        val_early_stopping_patience=100,
    )

    result = run_stage1_vae_train(
        config, encoder=encoder, decoder=decoder, loader=_loader(), val_loader=_loader()
    )
    assert result.steps == 6
    assert result.stopped_early is False
