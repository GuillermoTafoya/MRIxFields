import torch

from fieldbridge.training.latent_stats import (
    LatentStatsAccumulator,
    summarize_latent_stats,
)


def test_collapsed_channel_has_zero_std_and_is_inactive() -> None:
    # A channel that is constant everywhere (posterior mean == 0, logvar == 0 => KL 0)
    # must read as dead: std 0, KL 0, not counted as active.
    mean = torch.zeros(4, 3, 8, 8)
    logvar = torch.zeros_like(mean)
    acc = LatentStatsAccumulator(latent_channels=3)
    acc.update(mean, logvar)

    stats = acc.compute(active_threshold=0.01)

    assert stats["num_dims"] == 3
    assert stats["active_units"] == 0
    assert stats["dead_units"] == 3
    assert stats["per_dim_std"] == [0.0, 0.0, 0.0]
    assert all(kl < 1e-6 for kl in stats["per_dim_kl"])


def test_active_channels_are_counted() -> None:
    torch.manual_seed(0)
    # Channels 0 and 2 carry a real posterior (nonzero mean, informative => positive KL);
    # channel 1 is collapsed to the prior.
    mean = torch.randn(8, 3, 6, 6)
    mean[:, 1] = 0.0
    logvar = torch.zeros(8, 3, 6, 6)
    logvar[:, 1] = 0.0  # channel 1 == prior
    acc = LatentStatsAccumulator(latent_channels=3)
    acc.update(mean, logvar)

    stats = acc.compute(active_threshold=0.01)

    assert stats["active_units"] == 2
    assert stats["dead_units"] == 1
    assert stats["per_dim_kl"][1] < stats["per_dim_kl"][0]
    assert stats["per_dim_kl"][1] < stats["per_dim_kl"][2]


def test_unit_variance_latent_reads_global_std_near_one() -> None:
    torch.manual_seed(1)
    mean = torch.randn(64, 4, 8, 8)  # ~N(0, 1) per dim
    logvar = torch.zeros_like(mean)
    acc = LatentStatsAccumulator(latent_channels=4)
    acc.update(mean, logvar)

    stats = acc.compute()

    assert abs(stats["global_std"] - 1.0) < 0.05
    assert all(abs(std - 1.0) < 0.1 for std in stats["per_dim_std"])


def test_online_accumulation_matches_single_pass() -> None:
    torch.manual_seed(2)
    a_mean, a_logvar = torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4) * 0.1
    b_mean, b_logvar = torch.randn(5, 2, 4, 4), torch.randn(5, 2, 4, 4) * 0.1

    online = LatentStatsAccumulator(latent_channels=2)
    online.update(a_mean, a_logvar)
    online.update(b_mean, b_logvar)

    single = LatentStatsAccumulator(latent_channels=2)
    single.update(torch.cat([a_mean, b_mean]), torch.cat([a_logvar, b_logvar]))

    online_stats = online.compute()
    single_stats = single.compute()
    for key in ("per_dim_std", "per_dim_kl", "global_std"):
        assert torch.allclose(
            torch.tensor(online_stats[key]), torch.tensor(single_stats[key]), atol=1e-6
        )


def test_summarize_latent_stats_is_a_single_line() -> None:
    acc = LatentStatsAccumulator(latent_channels=4)
    acc.update(torch.randn(2, 4, 4, 4), torch.zeros(2, 4, 4, 4))
    line = summarize_latent_stats(acc.compute())
    assert "active_units=" in line and "\n" not in line


def test_3d_and_2d_latents_both_supported() -> None:
    acc3d = LatentStatsAccumulator(latent_channels=4)
    acc3d.update(torch.randn(2, 4, 3, 3, 3), torch.zeros(2, 4, 3, 3, 3))
    acc2d = LatentStatsAccumulator(latent_channels=4)
    acc2d.update(torch.randn(2, 4, 5, 5), torch.zeros(2, 4, 5, 5))

    assert acc3d.compute()["num_dims"] == 4
    assert acc2d.compute()["num_dims"] == 4
