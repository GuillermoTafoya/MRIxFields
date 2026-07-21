import json

import pytest
import torch

from fieldbridge.data.mrixfields_adapter import adapt_mrixfields_manifest
from fieldbridge.data.patch_bank import build_patch_bank
from fieldbridge.evaluation.stage1_diagnostics import (
    Stage1DiagnosticSpec,
    _official_metrics,
    identity_tiler_contract,
    minus_one_one_foreground_mask,
    run_stage1_reconstruction_diagnostics,
    seam_gradient_metric,
    validate_minus_one_one_background_threshold,
)
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.official.data_manifest import parse_mrixfields_data_path
from fieldbridge.training.checkpoints import save_checkpoint


def _official_records():
    return [
        parse_mrixfields_data_path(
            "Training_retrospective/T1W/1.5T/R_T1W_1.5T_0001.nii.gz"
        ),
        parse_mrixfields_data_path(
            "Training_retrospective/T2W/3T/R_T2W_3T_0001.nii.gz"
        ),
    ]


def _resolved_config() -> dict:
    return {
        "seed": 13,
        "data": {"patch_size": [8, 8, 8], "patches_per_volume": 2},
        "model": {
            "name": "kl_vae",
            "in_channels": 1,
            "base_channels": 4,
            "latent_channels": 2,
            "num_res_blocks": 1,
            "spatial_dims": 3,
            "activation": "silu",
        },
        "training": {
            "steps": 1,
            "batch_size": 1,
            "lr": 0.0001,
            "device": "cpu",
            "precision": "fp32",
            "loss_weights": {"ssim": 1.0, "nrmse": 1.0, "lpips": 0.0, "kl": 0.0001},
            "ssim_window_size": 3,
            "lpips_num_slices": 0,
            "grad_clip_norm": 1.0,
            "steps_per_epoch": 4,
            "early_stopping": True,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.005,
            "early_stopping_ema_decay": 0.98,
            "checkpoint_at_end": True,
            "log_every_steps": 0,
        },
    }


def _raw_volume(path, record):  # type: ignore[no-untyped-def]
    del path
    offset = 0.02 if record.domain.field_strength_t > 2.0 else 0.0
    return (torch.linspace(0.0, 1.0, 1_000).reshape(1, 10, 10, 10) + offset).clamp(0, 1)


def _write_checkpoint(tmp_path, config: dict):
    model = config["model"]
    shared = {
        "base_channels": model["base_channels"],
        "latent_channels": model["latent_channels"],
        "num_res_blocks": model["num_res_blocks"],
        "spatial_dims": model["spatial_dims"],
        "activation": model["activation"],
    }
    encoder = KLVAEEncoder(in_channels=1, **shared)
    decoder = KLVAEDecoder(**shared)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=0.0001)
    path = tmp_path / "synthetic-stage1.pt"
    save_checkpoint(
        path,
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 4,
            "early_stop": {"ema": 1.2, "best": 1.1, "num_bad_checkpoints": 2},
        },
        seed=13,
        config={
            "steps": 1,
            "batch_size": 1,
            "seed": 13,
            "lr": 0.0001,
            "device": "cpu",
            "precision": "fp32",
            "loss_weights": {
                "ssim": 1.0,
                "nrmse": 1.0,
                "lpips": 0.0,
                "kl": 0.0001,
            },
            "ssim_window_size": 3,
            "lpips_num_slices": 0,
            "grad_clip_norm": 1.0,
            "steps_per_epoch": 4,
            "early_stopping": True,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.005,
            "early_stopping_ema_decay": 0.98,
            "checkpoint_at_end": True,
            "log_every_steps": 0,
        },
        git_commit="a" * 40,
    )
    return path


def _rewrite_bank_case_ids_as_legacy_subject_ids(bank_dir, manifest) -> None:
    index_path = bank_dir / "bank_index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    for row, record in zip(rows, manifest.records):
        row["case_id"] = record.subject_id
    index_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_minus_one_one_mask_uses_explicit_background_threshold() -> None:
    target = torch.full((1, 1, 7, 7, 7), -1.0)
    target[..., 2:5, 2:5, 2:5] = -0.5

    mask = minus_one_one_foreground_mask(target, threshold=-0.95)

    assert mask.shape == target.shape
    assert mask[..., 3, 3, 3].item() == 1.0
    assert mask[..., 0, 0, 0].item() == 0.0
    with pytest.raises(ValueError, match="strictly between -1 and 0"):
        validate_minus_one_one_background_threshold(0.0)
    with pytest.raises(ValueError, match=r"must be in \[-1,1\]"):
        minus_one_one_foreground_mask(target + 3.0, threshold=-0.95)


def test_identity_tiler_contract_covers_declared_overlap_sweep() -> None:
    report = identity_tiler_contract()

    assert report["passed"] is True
    assert set(report["by_overlap"]) == {"0.25", "0.50", "0.75"}
    assert all(item["max_abs_error"] <= 1e-5 for item in report["by_overlap"].values())


def test_seam_metric_detects_regular_boundary_jump() -> None:
    smooth = torch.zeros(1, 1, 16, 16, 16)
    grid = smooth.clone()
    grid[..., 8:, :, :] = 1.0

    smooth_metric = seam_gradient_metric(smooth, patch_size=(8, 8, 8), overlap=0.0)
    grid_metric = seam_gradient_metric(grid, patch_size=(8, 8, 8), overlap=0.0)

    assert smooth_metric["boundary_gradient_mean_abs"] == 0.0
    assert grid_metric["boundary_gradient_mean_abs"] > 0.0
    assert grid_metric["boundary_to_overall_ratio"] > 1.0


class _FakeLPIPS(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(prediction - target).mean(dim=(1, 2, 3), keepdim=True)


def test_official_metrics_preserve_unmasked_values_and_optional_lpips() -> None:
    target = torch.zeros(1, 1, 8, 8, 8)
    reconstruction = torch.full_like(target, 0.25)

    computed = _official_metrics(
        reconstruction,
        target,
        lpips_net=_FakeLPIPS(),
        lpips_status="computed",
        lpips_num_slices=2,
    )
    skipped = _official_metrics(
        reconstruction,
        target,
        lpips_net=None,
        lpips_status="skipped_optional_dependency_unavailable",
        lpips_num_slices=2,
    )

    assert computed["data_range"] == 2.0
    assert computed["nrmse"] == pytest.approx(0.125)
    assert computed["lpips"]["status"] == "computed"
    assert computed["lpips"]["value"] == pytest.approx(0.25)
    assert skipped["lpips"] == {
        "status": "skipped_optional_dependency_unavailable",
        "value": None,
    }


def test_stage1_diagnostic_runs_end_to_end_on_synthetic_cpu(tmp_path) -> None:
    config = _resolved_config()
    adapted = adapt_mrixfields_manifest(_official_records())
    bank_dir = tmp_path / "bank"
    build_patch_bank(
        adapted.manifest.records,
        image_loader=_raw_volume,
        out_dir=bank_dir,
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        seed=13,
    )
    _rewrite_bank_case_ids_as_legacy_subject_ids(bank_dir, adapted.manifest)
    checkpoint = _write_checkpoint(tmp_path, config)
    spec = Stage1DiagnosticSpec(
        fixed_patch_index=0,
        fixed_volume_index=0,
        sampled_latent_seed=13,
        histogram_bins=8,
        lpips_num_slices=0,
    )

    inference_mode_observations: list[bool] = []

    def diagnostic_loader(path, record):  # type: ignore[no-untyped-def]
        inference_mode_observations.append(torch.is_inference_mode_enabled())
        return _raw_volume(path, record)

    report = run_stage1_reconstruction_diagnostics(
        checkpoint_path=checkpoint,
        patch_bank_dir=bank_dir,
        manifest=adapted.manifest,
        resolved_config=config,
        diagnostic_spec=spec,
        checkpoint_sweep_paths=(checkpoint,),
        image_loader=diagnostic_loader,
        device=torch.device("cpu"),
    )

    assert report["training_performed"] is False
    assert report["stage2_started"] is False
    assert report["held_out"] is False
    assert report["checkpoint"]["git_commit"] == "a" * 40
    assert report["checkpoint"]["step"] == 4
    assert report["checkpoint"]["early_stop"] == {
        "ema": 1.2,
        "best": 1.1,
        "num_bad_checkpoints": 2,
    }
    assert report["checkpoint"]["checkpoint_schema"] == "stage1_vae_unversioned"
    assert report["patch_bank"]["compatibility_ok"] is True
    assert report["patch_bank"]["volumes"] == 2
    assert report["patch_bank"]["patches"] == 4
    assert report["patch_bank"]["identity_alignment"] == "legacy_subject_id_order_only"
    assert report["patch_bank"]["case_ids_unique"] is False
    assert report["patch_bank"]["duplicate_case_id_values"] == 1
    assert len(report["patch_bank"]["fingerprint_sha256"]) == 64
    assert set(report["manifest"]["coverage_by_field_contrast"]) == {"1.5T/T1w", "3T/T2w"}

    patch = report["fixed_patch"]
    assert 0.0 <= patch["foreground_occupancy"] <= 1.0
    assert set(patch) >= {
        "reconstruction_from_latent_mean",
        "reconstruction_from_sampled_latent",
        "training_loss_components_on_sampled_reconstruction",
    }
    assert patch["training_loss_components_on_sampled_reconstruction"]["kl"] >= 0.0
    assert patch["reconstruction_from_latent_mean"]["official_metrics"]["lpips"][
        "status"
    ] == "skipped_disabled"
    assert report["direct_vs_tiled_fixed_patch"]["passed"] is True
    assert report["identity_tiler_contract"]["passed"] is True

    full_volume = report["fixed_full_volume"]
    assert full_volume["complete_volume"] is True
    assert set(full_volume["overlap_sweep"]) == {"0.25", "0.50", "0.75"}
    for overlap in full_volume["overlap_sweep"].values():
        assert set(overlap["official_full_volume_metrics"]) >= {
            "nrmse",
            "ssim3d",
            "lpips",
            "mae",
            "mse",
        }
        assert set(overlap["masked_diagnostics"]) >= {
            "foreground_mae",
            "outside_mae",
        }
        assert "boundary_to_overall_ratio" in overlap["seam_metric"]
    assert report["checkpoint_step_sweep"]["best_checkpoint_selected"] is False
    assert [item["step"] for item in report["checkpoint_step_sweep"]["results"]] == [4]
    assert report["recommendation"]["status"] == "NO_NEXT_TRAINING_EXPERIMENT"
    assert inference_mode_observations and all(inference_mode_observations)

    serialized = json.dumps(report)
    for private_key in ('"case_id":', '"subject_id":', '"sample_id":', '"image_path":'):
        assert private_key not in serialized


def test_stage1_diagnostic_rejects_patch_bank_config_mismatch(tmp_path) -> None:
    config = _resolved_config()
    adapted = adapt_mrixfields_manifest(_official_records())
    bank_dir = tmp_path / "bank"
    build_patch_bank(
        adapted.manifest.records,
        image_loader=_raw_volume,
        out_dir=bank_dir,
        patch_size=(8, 8, 8),
        patches_per_volume=2,
        seed=13,
    )
    checkpoint = _write_checkpoint(tmp_path, config)
    config["data"]["patches_per_volume"] = 3

    with pytest.raises(ValueError, match="compatibility failed"):
        run_stage1_reconstruction_diagnostics(
            checkpoint_path=checkpoint,
            patch_bank_dir=bank_dir,
            manifest=adapted.manifest,
            resolved_config=config,
            diagnostic_spec=Stage1DiagnosticSpec(fixed_patch_index=0, lpips_num_slices=0),
            image_loader=_raw_volume,
            device=torch.device("cpu"),
        )
