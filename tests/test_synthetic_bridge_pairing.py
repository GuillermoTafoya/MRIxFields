"""Scientific controls for the synthetic Brownian endpoint-pairing experiment."""

import pytest
import torch

from fieldbridge.evaluation.synthetic_bridge_phantoms import TISSUE_INTENSITY, make_cohort, measure
from fieldbridge.evaluation.synthetic_bridge_training import batch_indices, sample_bridge_target, train_arm
from fieldbridge.training.stage2_unified import UnifiedStage2Config, _bridge_sample


def test_current_velocity_is_fixed_noise_path_derivative_and_drift_replays_production():
    generator = torch.Generator().manual_seed(191)
    source, target, noise = torch.randn(3, 3, 1, 2, 3, 4, generator=generator, dtype=torch.float64)
    times = torch.tensor([0.18, 0.5, 0.83], dtype=torch.float64)
    config = UnifiedStage2Config(bridge="schrodinger", sigma=0.24, time_eps=0.02)

    state, historical = _bridge_sample(source, target, times, config, noise=noise)
    drift_state, drift = sample_bridge_target(source, target, times, noise, objective="forward_drift", config=config)
    current_state, velocity = sample_bridge_target(
        source, target, times, noise, objective="current_velocity", config=config
    )
    assert torch.equal(drift_state, state)
    assert torch.equal(current_state, state)
    assert torch.equal(drift, historical)

    def path(time):
        t = time.view(-1, 1, 1, 1, 1)
        return torch.lerp(source, target, t) + noise * config.sigma * torch.sqrt(t * (1 - t))

    step = 1e-5
    derivative = (path(times + step) - path(times - step)) / (2 * step)
    torch.testing.assert_close(velocity, derivative, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(velocity[1], (target - source)[1])
    assert not torch.allclose(drift[1], velocity[1])


def test_subject_split_preserves_tissue_boundaries_across_fields():
    train, heldout = make_cohort(n_train=10, n_test=6, side=16, seed=437)
    assert len(set(train.ids) | set(heldout.ids)) == 16
    volumes = torch.cat([train.source, heldout.source]).flatten(1)
    assert torch.unique(volumes, dim=0).shape[0] == volumes.shape[0]
    for group in (train, heldout):
        assert group.source.shape == group.tissue.shape
        assert group.tissue.flatten(1).any(1).all()
        assert (~group.tissue).flatten(1).any(1).all()
        for field in (1.5, 7.0):
            known = group.targets[field]
            assert torch.equal(known != group.source, group.tissue)
            torch.testing.assert_close(known[group.tissue], torch.full_like(known[group.tissue], TISSUE_INTENSITY[field]))


def test_shuffled_endpoint_schedule_preserves_each_fields_batch_and_has_no_fixed_points():
    generator = torch.Generator().manual_seed(395)
    for _ in range(12):
        source, shuffled = batch_indices(19, 6, generator)
        assert source.shape == shuffled.shape == (2, 6)
        assert torch.all(source != shuffled)
        assert torch.equal(source.sort(1).values, shuffled.sort(1).values)
        for row in source:
            assert len(torch.unique(row)) == 6


def test_population_scores_ignore_subject_order_but_image_distance_preserves_geometry():
    # Three subjects differ by the position and intensity of a tissue.
    target = torch.zeros(3, 1, 4, 8, 8)
    for subject in range(len(target)):
        target[subject, :, :, subject, 1:4] = 0.4 + 0.2 * subject
    shifted = target.roll(1, 0)
    scores = measure(shifted, target)
    direct = (shifted - target).flatten(1).square().mean(1).sqrt()
    assert scores["subject_rmse"] == pytest.approx(direct.tolist())
    assert scores["mean_subject_rmse"] > 0
    assert scores["population_image_w2"] == pytest.approx(0.0, abs=1e-8)
    assert scores["population_intensity_w1"] == pytest.approx(0.0, abs=1e-8)
    spatially_shifted = measure(target.roll(4, -1), target)
    assert spatially_shifted["population_image_w2"] > 0
    assert spatially_shifted["population_intensity_w1"] == pytest.approx(0.0, abs=1e-8)


def test_all_training_arms_replay_the_same_draws_and_initialization():
    train, _ = make_cohort(n_train=6, n_test=2, side=8, seed=418)
    receipts = {}
    for pairing in ("paired", "shuffled"):
        for objective in ("forward_drift", "current_velocity"):
            model, receipt = train_arm(
                train, pairing=pairing, objective=objective, seed=71, updates=2,
                per_field=3, sigma=0.10, time_eps=0.02, lr=0.001,
            )
            assert all(p.device.type == "cpu" for p in model.parameters())
            receipts[pairing, objective] = receipt
    for shared in ("initial_state_sha256", "draws_sha256", "parameters"):
        assert len({receipt[shared] for receipt in receipts.values()}) == 1
    for pairing in ("paired", "shuffled"):
        assert receipts[pairing, "forward_drift"]["pairing_indices_sha256"] == (
            receipts[pairing, "current_velocity"]["pairing_indices_sha256"]
        )
    assert receipts["paired", "current_velocity"]["pairing_indices_sha256"] != (
        receipts["shuffled", "current_velocity"]["pairing_indices_sha256"]
    )
