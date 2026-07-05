import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from fieldbridge.cli import _build_manifest_loader, _load_loss_curve, main
from fieldbridge.data.transforms import normalize_percentile_clip_to_unit_range


def test_load_loss_curve_tolerates_empty_and_invalid_json(tmp_path: Path) -> None:
    # A training run whose stdout never reached the redirect leaves an empty metrics file;
    # the loss-curve overlay is optional and must not crash eval-stage1-vae.
    empty = tmp_path / "empty.json"
    empty.write_text("")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json{")
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"losses": [3.0, 2.0, 1.0]}))

    assert _load_loss_curve(empty) is None
    assert _load_loss_curve(invalid) is None
    assert _load_loss_curve(tmp_path / "missing.json") is None
    assert _load_loss_curve(valid) == [3.0, 2.0, 1.0]


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


def _write_synthetic_3d_manifest(
    tmp_path: Path, *, num_records: int = 4, volume_shape: tuple[int, int, int] = (8, 8, 8)
) -> Path:
    records = []
    for index in range(num_records):
        image_path = tmp_path / f"volume_{index}.nii.gz"
        array = np.random.default_rng(index).normal(size=volume_shape).astype("float32")
        nibabel.save(nibabel.Nifti1Image(array, affine=np.eye(4)), str(image_path))
        records.append(
            {
                "case_id": f"volume_{index}",
                "image_path": str(image_path),
                "domain": {"field_strength_t": 3.0, "contrast": "T1w"},
            }
        )
    manifest_path = tmp_path / "manifest_3d.json"
    manifest_path.write_text(json.dumps({"name": "test-manifest-3d", "records": records}), encoding="utf-8")
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


def test_train_stage1_vae_cli_crops_volumes_larger_than_patch_size(tmp_path, capsys) -> None:
    # Proves the crop actually engages, not just that config wiring parses: the raw
    # volume shape (30) is NOT divisible by 4 (KLVAEEncoder's downsample factor) and
    # would fail shape validation if fed through uncropped — only succeeds because
    # patch_size=[8,8,8] (divisible by 4) is what actually reaches the encoder.
    manifest_path = _write_synthetic_3d_manifest(tmp_path, volume_shape=(30, 30, 30))
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "stage1_vae_patch.yaml"
    config_path.write_text(
        f"""
seed: 3
data:
  patch_size: [8, 8, 8]
model:
  name: kl_vae
  in_channels: 1
  base_channels: 4
  latent_channels: 3
  spatial_dims: 3
training:
  steps: 1
  batch_size: 2
  lr: 0.001
  loss_weights:
    ssim: 0.0
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


def test_train_stage1_vae_cli_runs_against_a_small_real_3d_manifest(tmp_path, capsys) -> None:
    # Mirrors the actual Colab scenario: real NIfTI files are full 3D volumes, no
    # slice-extraction step. ssim MUST stay 0 (evaluation.metrics.ssim is 2D-only).
    manifest_path = _write_synthetic_3d_manifest(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "stage1_vae_3d.yaml"
    config_path.write_text(
        f"""
seed: 3
model:
  name: kl_vae
  in_channels: 1
  base_channels: 4
  latent_channels: 3
  spatial_dims: 3
training:
  steps: 1
  batch_size: 2
  lr: 0.001
  loss_weights:
    ssim: 0.0
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


def test_train_stage1_vae_cli_patches_per_volume_flag_runs_multiple_steps(tmp_path, capsys) -> None:
    # 4 volumes x --patches-per-volume 3 = 12 patches per pass, enough for 4 steps at
    # batch 2 without re-iterating; exercises the flag -> data.patches_per_volume override.
    manifest_path = _write_synthetic_3d_manifest(tmp_path)
    config_path = tmp_path / "stage1_vae_3d.yaml"
    config_path.write_text(
        """
seed: 3
model:
  name: kl_vae
  in_channels: 1
  base_channels: 4
  latent_channels: 3
  spatial_dims: 3
training:
  steps: 1
  batch_size: 2
  lr: 0.001
  loss_weights:
    ssim: 0.0
    nrmse: 1.0
    lpips: 0.0
    kl: 0.0001
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "train-stage1-vae",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--steps",
            "4",
            "--patches-per-volume",
            "3",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["steps"] == 4


def test_train_stage2_diffuser_cli_runs_against_a_small_real_3d_manifest(tmp_path, capsys) -> None:
    manifest_path = _write_synthetic_3d_manifest(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "stage2_diffuser_3d.yaml"
    config_path.write_text(
        f"""
seed: 3
model:
  name: field_conditioned_unet
  latent_channels: 3
  base_channels: 6
  spatial_dims: 3
  num_levels: 1
  num_blocks_per_level: 1
  timestep_embedding_dim: 16
  field_conditioning_dim: 16
vae_model:
  in_channels: 1
  base_channels: 4
  latent_channels: 3
  spatial_dims: 3
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
