import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from fieldbridge.cli import _build_manifest_loader, main
from fieldbridge.data.transforms import normalize_percentile_clip_to_unit_range

nibabel = pytest.importorskip("nibabel")


def _write_synthetic_2d_manifest(tmp_path: Path, *, num_records: int = 4) -> Path:
    records = []
    for index in range(num_records):
        image_path = tmp_path / f"case_{index}.nii.gz"
        array = np.random.default_rng(index).normal(size=(16, 16)).astype("float32")
        nibabel.save(nibabel.Nifti1Image(array, affine=np.eye(4)), str(image_path))
        records.append(
            {
                "case_id": f"case_{index}",
                "image_path": str(image_path),
                "domain": {"field_strength_t": 3.0, "contrast": "T1w"},
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"name": "test-manifest", "records": records}), encoding="utf-8")
    return manifest_path


def test_build_manifest_loader_defaults_to_percentile_clip_transform() -> None:
    default_transform = inspect.signature(_build_manifest_loader).parameters["transform"].default

    assert default_transform is normalize_percentile_clip_to_unit_range


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


def test_train_stage1_vae_cli_runs_against_a_small_real_manifest(tmp_path, capsys) -> None:
    manifest_path = _write_synthetic_2d_manifest(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "stage1_vae.yaml"
    config_path.write_text(
        f"""
seed: 3
model:
  name: kl_vae
  in_channels: 1
  base_channels: 4
  latent_channels: 4
training:
  steps: 1
  batch_size: 2
  lr: 0.001
  loss_weights:
    ssim: 1.0
    nrmse: 1.0
    lpips: 0.0
    kl: 0.0001
  checkpoint_dir: {checkpoint_dir.as_posix()}
  checkpoint_at_end: true
""",
        encoding="utf-8",
    )

    exit_code = main(["train-stage1-vae", "--config", str(config_path), "--manifest", str(manifest_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["steps"] == 1
    assert len(list(checkpoint_dir.glob("*.pt"))) == 1


def test_train_stage2_diffuser_cli_runs_against_a_small_real_manifest(tmp_path, capsys) -> None:
    manifest_path = _write_synthetic_2d_manifest(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "stage2_diffuser.yaml"
    config_path.write_text(
        f"""
seed: 3
model:
  name: field_conditioned_unet
  latent_channels: 4
  base_channels: 8
  num_levels: 1
  num_blocks_per_level: 1
  timestep_embedding_dim: 16
  field_conditioning_dim: 16
vae_model:
  in_channels: 1
  base_channels: 4
  latent_channels: 4
training:
  steps: 1
  batch_size: 2
  lr: 0.001
  num_timesteps: 10
  checkpoint_dir: {checkpoint_dir.as_posix()}
  checkpoint_at_end: true
""",
        encoding="utf-8",
    )

    exit_code = main(
        ["train-stage2-diffuser", "--config", str(config_path), "--manifest", str(manifest_path), "--json"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["steps"] == 1
    assert len(list(checkpoint_dir.glob("*.pt"))) == 1
