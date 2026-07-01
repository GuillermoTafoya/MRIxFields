import json

from fieldbridge.cli import main


def test_train_cli_runs_with_default_smoke_config(capsys) -> None:
    exit_code = main(["train", "--steps", "2", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["steps"] == 2
    assert len(payload["losses"]) == 2


def test_train_cli_runs_with_cnn_autoencoder_config(tmp_path, capsys) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "autoencoder.yaml"
    config_path.write_text(
        f"""
seed: 3
data:
  name: synthetic
  num_samples: 2
  volume_shape: [1, 8, 8, 8]
model:
  name: cnn_autoencoder
  variant: cnn3d-test
  spatial_dims: 3
  in_channels: 1
  out_channels: 1
  hidden_channels: [2]
  latent_channels: 3
  translator:
    name: identity
training:
  stage: autoencoder
  steps: 1
  batch_size: 1
  lr: 0.001
  checkpoint_dir: {checkpoint_dir.as_posix()}
  checkpoint_at_end: true
""",
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["steps"] == 1
    assert len(list(checkpoint_dir.glob("*.pt"))) == 1
