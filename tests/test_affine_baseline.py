from __future__ import annotations

import json

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.models.translators.affine_baseline import (
    AFFINE_BASELINE_CONTRACT_VERSION,
    AffineLatentBaseline,
    fit_affine_baselines,
    fit_paired_affine,
    latent_foreground_mask,
)

CHANNELS = 3
SHAPE = (CHANNELS, 6, 6, 6)
SRC = Domain(0.1, Contrast.T1W)
TGT = Domain(3.0, Contrast.T1W)


def _latent(mean: torch.Tensor, std: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(SHAPE, generator=generator)
    # Center/scale exactly so the fitted moments are the requested ones up to the pooled fit.
    noise = (noise - noise.mean(dim=(1, 2, 3), keepdim=True)) / noise.std(
        dim=(1, 2, 3), keepdim=True, unbiased=False
    )
    return noise * std.reshape(-1, 1, 1, 1) + mean.reshape(-1, 1, 1, 1)


def _pool(seed: int = 0) -> list[tuple[torch.Tensor, Domain]]:
    src_mean = torch.tensor([0.0, 1.0, -2.0])
    src_std = torch.tensor([1.0, 2.0, 0.5])
    tgt_mean = torch.tensor([3.0, -1.0, 0.0])
    tgt_std = torch.tensor([2.0, 1.0, 1.5])
    return [
        (_latent(src_mean, src_std, seed + 1), SRC),
        (_latent(tgt_mean, tgt_std, seed + 2), TGT),
    ]


def test_fit_recovers_the_closed_form_moment_matching_coefficients() -> None:
    baselines = fit_affine_baselines(_pool(), channels=CHANNELS)
    baseline = baselines["all"]
    a, b = baseline.coefficients(SRC, TGT)

    src_moments = baseline.moments[SRC.label]
    tgt_moments = baseline.moments[TGT.label]
    torch.testing.assert_close(a, tgt_moments.std / src_moments.std)
    torch.testing.assert_close(b, tgt_moments.mean - a * src_moments.mean)


def test_transported_latent_matches_the_target_marginal() -> None:
    pool = _pool()
    baseline = fit_affine_baselines(pool, channels=CHANNELS)["all"]
    z_source = pool[0][0]
    moved = baseline.transport(z_source, SRC, TGT)

    target_moments = baseline.moments[TGT.label]
    torch.testing.assert_close(
        moved.mean(dim=(1, 2, 3)), target_moments.mean, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        moved.std(dim=(1, 2, 3), unbiased=False), target_moments.std, rtol=1e-4, atol=1e-4
    )


def test_same_domain_transport_is_the_identity() -> None:
    pool = _pool()
    baseline = fit_affine_baselines(pool, channels=CHANNELS)["all"]
    a, b = baseline.coefficients(SRC, SRC)
    torch.testing.assert_close(a, torch.ones(CHANNELS))
    torch.testing.assert_close(b, torch.zeros(CHANNELS), atol=1e-6, rtol=0)
    z = pool[0][0]
    torch.testing.assert_close(baseline.transport(z, SRC, SRC), z, rtol=1e-5, atol=1e-5)


def test_transport_is_invertible() -> None:
    pool = _pool()
    baseline = fit_affine_baselines(pool, channels=CHANNELS)["all"]
    z = pool[0][0]
    there = baseline.transport(z, SRC, TGT)
    back = baseline.transport(there, TGT, SRC)
    torch.testing.assert_close(back, z, rtol=1e-4, atol=1e-4)


def test_transport_accepts_batched_and_unbatched_latents() -> None:
    baseline = fit_affine_baselines(_pool(), channels=CHANNELS)["all"]
    z = torch.randn(SHAPE)
    unbatched = baseline.transport(z, SRC, TGT)
    batched = baseline.transport(z.unsqueeze(0), SRC, TGT)
    assert unbatched.shape == SHAPE
    assert batched.shape == (1, *SHAPE)
    torch.testing.assert_close(batched[0], unbatched)


def test_transport_rejects_wrong_rank() -> None:
    baseline = fit_affine_baselines(_pool(), channels=CHANNELS)["all"]
    with pytest.raises(ValueError, match="transport expects"):
        baseline.transport(torch.randn(CHANNELS, 4), SRC, TGT)


def test_unfitted_domain_raises_with_the_fitted_domains_listed() -> None:
    baseline = fit_affine_baselines(_pool(), channels=CHANNELS)["all"]
    with pytest.raises(KeyError, match="No fitted moments"):
        baseline.coefficients(SRC, Domain(7.0, Contrast.T2W))


def test_foreground_table_is_fitted_on_fewer_voxels_than_the_all_table() -> None:
    baselines = fit_affine_baselines(_pool(), channels=CHANNELS, foreground_percentile=50.0)
    assert set(baselines) == {"all", "foreground"}
    for label in baselines["all"].moments:
        assert (
            baselines["foreground"].moments[label].voxels
            < baselines["all"].moments[label].voxels
        )


def test_foreground_mask_selects_the_requested_quantile() -> None:
    latent = torch.randn(CHANNELS, 8, 8, 8)
    mask = latent_foreground_mask(latent, percentile=75.0)
    assert mask.shape == (8, 8, 8)
    assert mask.dtype == torch.bool
    assert 0.20 < mask.float().mean().item() < 0.30


def test_foreground_mask_rejects_an_out_of_range_percentile() -> None:
    with pytest.raises(ValueError, match="percentile"):
        latent_foreground_mask(torch.randn(CHANNELS, 4, 4, 4), percentile=100.0)


def test_channel_count_mismatch_is_caught_before_a_long_pass() -> None:
    with pytest.raises(ValueError, match="but 3 were declared"):
        fit_affine_baselines([(torch.randn(2, 4, 4, 4), SRC)], channels=CHANNELS)


def test_json_round_trip_preserves_the_coefficients(tmp_path) -> None:
    baseline = fit_affine_baselines(
        _pool(), channels=CHANNELS, provenance={"pool": "unit-test"}
    )["foreground"]
    path = baseline.save(tmp_path / "affine.json")
    reloaded = AffineLatentBaseline.load(path)

    a, b = baseline.coefficients(SRC, TGT)
    a2, b2 = reloaded.coefficients(SRC, TGT)
    torch.testing.assert_close(a, a2)
    torch.testing.assert_close(b, b2)
    assert reloaded.space == "foreground"
    assert reloaded.provenance["pool"] == "unit-test"
    assert reloaded.provenance["pool_volumes"] == 2


def test_contract_version_mismatch_is_refused(tmp_path) -> None:
    baseline = fit_affine_baselines(_pool(), channels=CHANNELS)["all"]
    payload = baseline.to_dict()
    assert payload["contract_version"] == AFFINE_BASELINE_CONTRACT_VERSION
    payload["contract_version"] = "affine-latent-baseline-v0"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        AffineLatentBaseline.load(path)


def test_paired_least_squares_recovers_an_exact_affine_relation() -> None:
    z_source = torch.randn(CHANNELS, 5, 5, 5)
    a_true = torch.tensor([2.0, -1.0, 0.5])
    b_true = torch.tensor([1.0, 0.0, -3.0])
    z_target = a_true.reshape(-1, 1, 1, 1) * z_source + b_true.reshape(-1, 1, 1, 1)

    a, b = fit_paired_affine([(z_source, z_target)], channels=CHANNELS)
    torch.testing.assert_close(a, a_true, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b, b_true, rtol=1e-4, atol=1e-4)


def test_paired_least_squares_needs_pairs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_paired_affine([], channels=CHANNELS)
