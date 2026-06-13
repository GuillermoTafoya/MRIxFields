import math

from clbfield.training.smoke_train import SmokeTrainConfig, run_smoke_train


def test_smoke_train_runs_on_cpu() -> None:
    result = run_smoke_train(
        SmokeTrainConfig(
            steps=2,
            batch_size=2,
            num_samples=4,
            volume_shape=(1, 4, 4, 4),
            seed=3,
        )
    )
    assert result.steps == 2
    assert len(result.losses) == 2
    assert math.isfinite(result.final_loss)

