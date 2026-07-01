import pytest
import torch

from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.translators.identity import IdentityTranslator
from fieldbridge.training.train_loop import TrainLoopConfig, assert_frozen, run_train_loop


def _models(*, learnable_scale: bool = True) -> tuple[IdentityEncoder, IdentityDecoder, IdentityTranslator]:
    return IdentityEncoder(), IdentityDecoder(), IdentityTranslator(learnable_scale=learnable_scale)


def test_run_train_loop_produces_finite_losses() -> None:
    encoder, decoder, translator = _models()
    config = TrainLoopConfig(steps=3, batch_size=2, num_samples=4, seed=13)

    result = run_train_loop(config, encoder=encoder, decoder=decoder, translator=translator)

    assert result.steps == 3
    assert len(result.losses) == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in result.losses)


def test_run_train_loop_with_cycle_and_identity_terms() -> None:
    encoder, decoder, translator = _models()
    config = TrainLoopConfig(
        steps=2,
        batch_size=2,
        num_samples=4,
        seed=13,
        loss_weights={"reconstruction": 1.0, "transport_cost": 0.1, "cycle": 0.1, "identity": 0.1},
    )

    result = run_train_loop(config, encoder=encoder, decoder=decoder, translator=translator)

    assert all(torch.isfinite(torch.tensor(value)) for value in result.losses)


def test_assert_frozen_rejects_trainable_params() -> None:
    trainable_translator = IdentityTranslator(learnable_scale=True)

    with pytest.raises(RuntimeError):
        assert_frozen(trainable_translator)

    assert_frozen(IdentityEncoder())


def test_run_train_loop_checkpoint_and_resume_restore_state(tmp_path) -> None:
    encoder, decoder, translator = _models()
    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainLoopConfig(
        steps=2,
        batch_size=2,
        num_samples=4,
        seed=13,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_steps=2,
    )

    run_train_loop(config, encoder=encoder, decoder=decoder, translator=translator)
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) == 1

    resumed_translator = IdentityTranslator(learnable_scale=True)
    resumed_config = TrainLoopConfig(
        steps=0,
        batch_size=2,
        num_samples=4,
        seed=13,
        resume_from=checkpoints[0],
    )
    run_train_loop(resumed_config, encoder=IdentityEncoder(), decoder=IdentityDecoder(), translator=resumed_translator)

    assert torch.equal(resumed_translator.scale, translator.scale)
