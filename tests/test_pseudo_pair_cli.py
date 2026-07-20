import json
from pathlib import Path

import pytest
import torch

from fieldbridge.cli import main
from fieldbridge.training.checkpoints import save_checkpoint


def _write_manifest(tmp_path: Path) -> Path:
    records = []
    for field in (1.5, 3.0):
        for index in range(2):
            records.append(
                {
                    "case_id": f"{field:g}T-case-{index}",
                    "image_path": f"{field:g}T-case-{index}.nii.gz",
                    "domain": {"field_strength_t": field, "contrast": "T2-FLAIR"},
                    "subject_id": f"{field:g}T-subject-{index}",
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"name": "pseudo-test", "records": records}), encoding="utf-8")
    return manifest


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "pseudo.yaml"
    config.write_text(
        f"""
seed: 2
data:
  sequence: T2-FLAIR
  source_field: 0.1
  target_fields: [1.5, 3.0]
  train_volumes_per_field: 1
  val_volumes_per_field: 1
  test_volumes_per_field: 0
  split_json: {(tmp_path / "splits.json").as_posix()}
  preprocessing:
    slice_start: 0
    slice_end: 2
    slices_per_volume: 1
    normalization: official_01
    model_range: zero_one
    resize_mode: native
    slice_axis: x
model:
  name: conditional_unet_field_translator
  in_channels: 1
  out_channels: 1
  hidden_channels: [2]
  latent_channels: 4
  cond_dim: 8
  spatial_dims: 2
  final_activation: sigmoid
training:
  epochs: 1
  batch_size: 2
  num_workers: 0
  lr: 0.001
  weight_decay: 0.0
  grad_clip_norm: 1.0
  scheduler:
    name: none
  checkpoint_dir: {(tmp_path / "checkpoints").as_posix()}
  log_every_steps: 0
  loss_weights:
    masked_l1: 1.0
    gradient: 0.0
    background: 0.0
evaluation:
  lpips: off
""",
        encoding="utf-8",
    )
    return config


def test_train_and_eval_pseudo_pairs_cli_with_injected_loader(tmp_path, monkeypatch, capsys) -> None:
    manifest = _write_manifest(tmp_path)
    config = _write_config(tmp_path)

    def synthetic_loader(path, record):  # type: ignore[no-untyped-def]
        del path
        # Slice plane must be large enough to survive the translator's downsampling
        # (a 4x4 plane collapsed a pooled feature map to size 0). Volume is (C, X, Y, Z);
        # slice_axis=x keeps X=2 (two source slices) and gives 32x32 model inputs.
        base = torch.linspace(0.0, 1.0, 1 * 2 * 32 * 32, dtype=torch.float32).reshape(1, 2, 32, 32)
        return (base * (record.domain.field_strength_t / 7.0)).clamp(0.0, 1.0)

    monkeypatch.setattr("fieldbridge.cli.nifti_image_loader", synthetic_loader)

    preflight_code = main(
        [
            "train-pseudo-pairs",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--preflight",
            "--json",
        ]
    )
    preflight_payload = json.loads(capsys.readouterr().out)

    assert preflight_code == 0
    assert preflight_payload["steps_per_epoch"] == 1
    assert preflight_payload["datasets"]["train"]["samples"] == 2
    assert preflight_payload["datasets"]["validation"]["samples"] == 2
    assert preflight_payload["leakage_audit"]["ok"] is True
    assert Path(preflight_payload["split_json"]).exists()
    assert preflight_payload["preprocessing"]["raw_volume_order"] == "C,X,Y,Z"
    assert preflight_payload["preprocessing"]["slice_axis"] == "x"
    assert preflight_payload["preprocessing"]["slice_plane"] == "Y,Z"

    train_code = main(
        [
            "train-pseudo-pairs",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--json",
        ]
    )
    train_payload = json.loads(capsys.readouterr().out)

    assert train_code == 0
    assert train_payload["steps_per_epoch"] == 1
    assert Path(train_payload["best_checkpoint"]).exists()
    assert Path(train_payload["last_checkpoint"]).exists()

    eval_code = main(
        [
            "eval-pseudo-pairs",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--checkpoint",
            train_payload["best_checkpoint"],
            "--split",
            "validation",
            "--json",
        ]
    )
    eval_payload = json.loads(capsys.readouterr().out)

    assert eval_code == 0
    assert eval_payload["num_samples"] == 2
    assert "degraded" in eval_payload["aggregate"]
    assert "predicted" in eval_payload["aggregate"]


def test_eval_pseudo_pairs_rejects_v1_checkpoint(tmp_path) -> None:
    manifest = _write_manifest(tmp_path)
    config = _write_config(tmp_path)
    checkpoint = tmp_path / "pseudo-pair-v1.pt"
    save_checkpoint(
        checkpoint,
        {"trainer": "pseudo_pair_epochs", "pseudo_pair_pipeline_version": 1},
    )

    with pytest.raises(ValueError, match="rerun train-pseudo-pairs from scratch"):
        main(
            [
                "eval-pseudo-pairs",
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "validation",
            ]
        )
