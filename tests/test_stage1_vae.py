from fieldbridge.training.stage1_vae import Stage1VAEConfig, _EarlyStopTracker, _epoch_label


def test_early_stop_tracker_does_not_stop_while_improving() -> None:
    tracker = _EarlyStopTracker(decay=0.0, min_delta=0.01, patience=3)  # decay 0 => ema == last loss
    # Strictly decreasing loss, each step a clear improvement.
    for loss in [10.0, 8.0, 6.0, 4.0, 2.0]:
        tracker.update_step(loss)
        assert tracker.should_stop() is False
    assert tracker.best == 2.0
    assert tracker.num_bad_checkpoints == 0


def test_early_stop_tracker_stops_after_patience_on_plateau() -> None:
    tracker = _EarlyStopTracker(decay=0.0, min_delta=0.01, patience=3)
    tracker.update_step(5.0)
    assert tracker.should_stop() is False  # first checkpoint sets the baseline
    # Flat loss: no improvement beyond min_delta for `patience` checks -> stop on the 3rd.
    tracker.update_step(5.0)
    assert tracker.should_stop() is False  # bad 1
    tracker.update_step(5.0)
    assert tracker.should_stop() is False  # bad 2
    tracker.update_step(5.0)
    assert tracker.should_stop() is True  # bad 3 == patience


def test_early_stop_tracker_min_delta_requires_meaningful_improvement() -> None:
    tracker = _EarlyStopTracker(decay=0.0, min_delta=0.10, patience=2)
    tracker.update_step(1.0)
    assert tracker.should_stop() is False  # baseline best=1.0
    tracker.update_step(0.95)  # only 5% better, below the 10% min_delta => counts as bad
    assert tracker.should_stop() is False  # bad 1
    tracker.update_step(0.94)
    assert tracker.should_stop() is True  # bad 2 == patience


def test_early_stop_tracker_state_round_trips() -> None:
    tracker = _EarlyStopTracker(decay=0.9, min_delta=0.01, patience=3)
    for loss in [3.0, 2.5, 2.4]:
        tracker.update_step(loss)
    tracker.should_stop()
    state = tracker.state_dict()

    restored = _EarlyStopTracker(decay=0.9, min_delta=0.01, patience=3)
    restored.load_state_dict(state)
    assert restored.ema == tracker.ema
    assert restored.best == tracker.best
    assert restored.num_bad_checkpoints == tracker.num_bad_checkpoints


def test_epoch_label_maps_steps_to_epochs() -> None:
    # 10 steps per epoch: step 1 -> epoch 1 [1/10]; step 10 -> epoch 1 [10/10];
    # step 11 -> epoch 2 [1/10].
    assert _epoch_label(1, 10) == "epoch=1 [1/10]"
    assert _epoch_label(10, 10) == "epoch=1 [10/10]"
    assert _epoch_label(11, 10) == "epoch=2 [1/10]"
    assert _epoch_label(5, 0) == "epoch=?"  # unknown epoch size


def test_stage1_config_parses_early_stopping_and_steps_per_epoch() -> None:
    cfg = Stage1VAEConfig.from_mapping(
        {
            "training": {
                "early_stopping": True,
                "early_stopping_patience": 7,
                "early_stopping_min_delta": 0.02,
                "early_stopping_ema_decay": 0.95,
                "steps_per_epoch": 1234,
            }
        }
    )
    assert cfg.early_stopping is True
    assert cfg.early_stopping_patience == 7
    assert cfg.early_stopping_min_delta == 0.02
    assert cfg.early_stopping_ema_decay == 0.95
    assert cfg.steps_per_epoch == 1234
