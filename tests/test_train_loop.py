import pytest
import torch

from fieldbridge.models.autoencoders.cnn_autoencoder import CNNDecoder, CNNEncoder
from fieldbridge.models.autoencoders.identity import IdentityDecoder, IdentityEncoder
from fieldbridge.models.translators.identity import IdentityTranslator
from fieldbridge.training.checkpoints import load_checkpoint
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


def test_run_train_loop_autoencoder_checkpoint_contains_model_weights(tmp_path) -> None:
    encoder = CNNEncoder(hidden_channels=(2,), latent_channels=3, spatial_dims=3)
    decoder = CNNDecoder(hidden_channels=(2,), latent_channels=3, spatial_dims=3)
    translator = IdentityTranslator()
    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainLoopConfig(
        steps=1,
        batch_size=1,
        num_samples=2,
        volume_shape=(1, 8, 8, 8),
        seed=13,
        stage="autoencoder",
        variant="cnn3d-test",
        checkpoint_dir=checkpoint_dir,
        checkpoint_at_end=True,
    )

    run_train_loop(config, encoder=encoder, decoder=decoder, translator=translator)

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) == 1
    state = load_checkpoint(checkpoints[0])
    assert state["step"] == 1
    assert state["encoder"]
    assert state["decoder"]
    assert "translator" in state
    assert "optimizer" in state


def test_run_train_loop_autoencoder_updates_encoder_and_decoder_params() -> None:
    encoder = CNNEncoder(hidden_channels=(2,), latent_channels=3, spatial_dims=3)
    decoder = CNNDecoder(hidden_channels=(2,), latent_channels=3, spatial_dims=3)
    translator = IdentityTranslator()
    encoder_before = [param.detach().clone() for param in encoder.parameters()]
    decoder_before = [param.detach().clone() for param in decoder.parameters()]
    config = TrainLoopConfig(
        steps=1,
        batch_size=1,
        num_samples=2,
        volume_shape=(1, 8, 8, 8),
        seed=13,
        lr=0.01,
        stage="autoencoder",
    )

    run_train_loop(config, encoder=encoder, decoder=decoder, translator=translator)

    assert any(not torch.equal(before, after) for before, after in zip(encoder_before, encoder.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(decoder_before, decoder.parameters()))
