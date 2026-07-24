from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.datasets import StreamingPatchDataset, collate_raw_batches
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.sampling import (
    exposure_report,
    joint_domain_subject_balanced_indices,
)
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.losses import ssim_loss
from fieldbridge.training.ssim import stable_training_ssim3d
from fieldbridge.training.stage1_vae import (
    Stage1VAEConfig,
    _compute_vae_loss_components,
    _kl_warmup_factor,
    _run_validation,
    run_stage1_vae_train,
)


def _joint_records() -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    for field in FIELD_STRENGTHS_T:
        for contrast in CONTRASTS:
            for subject_index, volume_count in ((0, 1), (1, 2)):
                for volume_index in range(volume_count):
                    case = f"{field:g}-{contrast.value}-s{subject_index}-v{volume_index}"
                    records.append(
                        VolumeRecord(
                            case_id=case,
                            image_path=f"{case}.nii.gz",
                            domain=Domain(field, contrast),
                            subject_id=f"s{subject_index}",
                        )
                    )
    return records


def test_joint_domain_schedule_is_exact_reproducible_and_subject_fair() -> None:
    records = _joint_records()
    first = joint_domain_subject_balanced_indices(records, seed=13, pass_index=0)
    second = joint_domain_subject_balanced_indices(records, seed=13, pass_index=0)
    report = exposure_report(records, first)

    assert first == second
    assert len(report["by_domain"]) == 15
    assert max(report["by_domain"].values()) - min(report["by_domain"].values()) == 0
    for subjects in report["by_domain_subject"].values():
        assert max(subjects.values()) - min(subjects.values()) <= 1


def test_joint_domain_schedule_requires_all_domains() -> None:
    with pytest.raises(ValueError, match="all 15"):
        joint_domain_subject_balanced_indices(_joint_records()[:-3])


def test_streaming_joint_balance_reports_expected_and_observed_exposure() -> None:
    records = _joint_records()
    dataset = StreamingPatchDataset(
        records,
        image_loader=lambda path, record: torch.zeros(1, 4, 4, 4),
        patch_size=None,
        patches_per_volume=2,
        seed=7,
        joint_domain_balance=True,
    )
    list(dataset)

    assert dataset.last_exposure_report is not None
    assert len(dataset.last_exposure_report["by_domain"]) == 15
    expected = dataset.expected_exposure_report(pass_index=0)
    assert expected["total_draws"] == 2 * len(records)


@pytest.mark.parametrize(
    "prediction_factory",
    [
        lambda target: target.clone(),
        lambda target: torch.zeros_like(target),
        lambda target: target + 0.5,
        lambda target: target - 0.5,
        lambda target: target * 4.0 - 1.0,
    ],
)
def test_ssim_range_and_nonnegative_loss_for_edge_cases(prediction_factory) -> None:
    target = torch.zeros(1, 1, 8, 8, 8)
    target[..., 2:6, 2:6, 2:6] = 0.7
    prediction = prediction_factory(target).requires_grad_(True)

    similarity = stable_training_ssim3d(
        prediction, target, window_size=3
    )
    loss = ssim_loss(prediction, target, window_size=3)

    assert torch.isfinite(similarity)
    assert -1.0 <= float(similarity.detach()) <= 1.0
    assert torch.isfinite(loss)
    assert 0.0 <= float(loss.detach()) <= 2.0
    if torch.equal(prediction.detach(), target):
        assert float(similarity) == pytest.approx(1.0, abs=1e-6)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_ssim_is_bf16_autocast_compatible() -> None:
    target = torch.rand(1, 1, 8, 8, 8)
    prediction = (target + 0.1).requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = ssim_loss(prediction, target, window_size=3)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_ssim_fails_fast_on_nonfinite_input() -> None:
    prediction = torch.zeros(1, 1, 8, 8, 8)
    prediction[..., 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ssim_loss(prediction, torch.zeros_like(prediction), window_size=3)


def test_deterministic_arm_has_no_sampling_or_effective_kl() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=2, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=2, spatial_dims=3)
    domain = Domain(3.0, "T1w")
    batch = RawBatch(
        image=torch.rand(1, 1, 8, 8, 8),
        source_domain=domain,
        target_domain=domain,
    )
    cfg = Stage1VAEConfig(
        latent_mode="deterministic",
        loss_weights={"masked_l1": 1.0, "background": 0.1, "kl": 1.0},
    )

    components = _compute_vae_loss_components(
        encoder, decoder, batch, cfg, lpips_net=None
    )

    assert float(components["kl"]) == 0.0
    assert torch.isfinite(components["raw_kl"])
    assert torch.isfinite(components["total"])


def test_kl_warmup_is_explicit_and_linear() -> None:
    cfg = Stage1VAEConfig(latent_mode="stochastic", kl_warmup_steps=100)
    assert _kl_warmup_factor(0, cfg) == 0.0
    assert _kl_warmup_factor(50, cfg) == 0.5
    assert _kl_warmup_factor(100, cfg) == 1.0
    assert _kl_warmup_factor(200, cfg) == 1.0


def test_target_conditioned_decoder_is_one_shared_network_and_changes_output() -> None:
    decoder = KLVAEDecoder(
        base_channels=4,
        latent_channels=2,
        spatial_dims=3,
        domain_conditioning_dim=8,
    )
    latent = torch.randn(1, 2, 2, 2, 2)
    low = decoder.decode(latent, Domain(0.1, "T1w"))
    high = decoder.decode(latent, Domain(7.0, "T2-FLAIR"))

    assert decoder.domain_conditioner is not None
    assert not hasattr(decoder, "routers")
    assert not torch.allclose(low, high)


def test_training_fails_fast_on_nonfinite_values() -> None:
    domain = Domain(3.0, "T1w")

    class _NonfiniteDataset(Dataset[RawBatch]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> RawBatch:
            image = torch.zeros(1, 8, 8, 8)
            image[..., 0, 0, 0] = float("nan")
            return RawBatch(image=image, source_domain=domain, target_domain=domain)

    loader = DataLoader(_NonfiniteDataset(), batch_size=1, collate_fn=collate_raw_batches)
    encoder = KLVAEEncoder(base_channels=4, latent_channels=2, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=2, spatial_dims=3)
    cfg = Stage1VAEConfig(steps=1, loss_weights={"nrmse": 1.0, "kl": 0.0})

    with pytest.raises(FloatingPointError, match="non-finite"):
        run_stage1_vae_train(cfg, encoder=encoder, decoder=decoder, loader=loader)


def test_validation_uses_equal_volume_then_equal_domain_aggregation() -> None:
    domains = [Domain(3.0, "T1w")] * 3 + [Domain(7.0, "T2w")]
    values = [1.0, 1.0, 1.0, 0.5]

    class _ValidationDataset(Dataset[RawBatch]):
        def __len__(self) -> int:
            return len(domains)

        def __getitem__(self, index: int) -> RawBatch:
            image = torch.full((1, 4, 4, 4), values[index])
            return RawBatch(
                image=image,
                source_domain=domains[index],
                target_domain=domains[index],
                metadata={"case_id": f"case-{index}"},
            )

    class _Encoder(torch.nn.Module):
        latent_channels = 1

        def encode_dist(self, image, domain):
            return torch.zeros_like(image), torch.zeros_like(image)

    class _Decoder(torch.nn.Module):
        def decode(self, latent, domain):
            return torch.zeros_like(latent)

    loader = DataLoader(
        _ValidationDataset(), batch_size=2, shuffle=False, collate_fn=collate_raw_batches
    )
    cfg = Stage1VAEConfig(
        ssim_window_size=3,
        loss_weights={"masked_l1": 1.0, "background": 0.1, "kl": 0.0},
        latent_mode="deterministic",
        latent_activity_rule="std",
    )

    metrics, _ = _run_validation(
        _Encoder(),
        _Decoder(),
        loader,
        cfg,
        torch.device("cpu"),
        None,
        nullcontext(),
    )

    # Domain means are 1.0 and 0.5; macro is 0.75, not pooled-volume 0.875.
    assert metrics["masked_mae"] == pytest.approx(0.75)
    assert metrics["num_volumes"] == 4
    assert metrics["num_domains"] == 2


def test_validation_can_fail_closed_when_any_joint_domain_is_missing() -> None:
    domain = Domain(3.0, "T1w")

    class _Dataset(Dataset[RawBatch]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> RawBatch:
            return RawBatch(
                image=torch.ones(1, 4, 4, 4),
                source_domain=domain,
                target_domain=domain,
                metadata={"case_id": "only-domain"},
            )

    class _Encoder(torch.nn.Module):
        latent_channels = 1

        def encode_dist(self, image, domain):
            return image, torch.zeros_like(image)

    class _Decoder(torch.nn.Module):
        def decode(self, latent, domain):
            return latent

    loader = DataLoader(_Dataset(), batch_size=1, collate_fn=collate_raw_batches)
    cfg = Stage1VAEConfig(
        ssim_window_size=3,
        loss_weights={"masked_l1": 1.0, "kl": 0.0},
        latent_mode="deterministic",
        require_all_validation_domains=True,
    )

    with pytest.raises(ValueError, match="all 15"):
        _run_validation(
            _Encoder(),
            _Decoder(),
            loader,
            cfg,
            torch.device("cpu"),
            None,
            nullcontext(),
        )
