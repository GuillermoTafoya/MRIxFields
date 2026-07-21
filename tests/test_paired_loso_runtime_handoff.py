import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import fieldbridge.training.paired_loso as paired_training
from fieldbridge.config import load_yaml_config
from fieldbridge.data.paired_loso import RealPairedSliceDataset
from fieldbridge.data.preprocessing import SlicePreprocessingSpec
from fieldbridge.data.pseudo_pairs import collate_pseudo_pair_slices
from fieldbridge.evaluation.paired_loso import reconstruct_complete_candidate
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.paired_loso import (
    NEURAL_INITIALIZATION_ARMS,
    SanitizedRunProgress,
    initialize_residual_arm,
    resolve_arm_recovery,
    train_fixed_endpoint,
)
from fieldbridge.training.pseudo_pair_epochs import PseudoPairEpochConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/prospective_paired_loso_residual_v1.yaml"
RUNNER_PATH = PROJECT_ROOT / "notebooks/prospective_paired_loso_residual_runner.py"
FROZEN_CONFIG_SHA256 = "27ce8b3f37b08af419473b672a3f260ed9299a212ac0955e81d48c2a7b118b01"


def _spec(*, depth: int = 2) -> SlicePreprocessingSpec:
    return SlicePreprocessingSpec(
        slice_start=0,
        slice_end=depth,
        slices_per_volume=None,
        normalization="official_01",
        model_range="minus_one_one",
        resize_mode="fit_pad",
        output_height=6,
        output_width=8,
        slice_axis="z",
    )


def _volume(*, depth: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.rand((1, 4, 5, depth), generator=generator) * 0.9 + 0.05


def _dataset() -> RealPairedSliceDataset:
    source = _volume()
    volumes = {
        "0007": {
            0.1: source,
            1.5: (source + 0.01).clamp(0.0, 1.0),
            3.0: (source + 0.02).clamp(0.0, 1.0),
            5.0: (source + 0.03).clamp(0.0, 1.0),
            7.0: (source + 0.04).clamp(0.0, 1.0),
        }
    }
    return RealPairedSliceDataset(
        volumes,
        case_ids=("0007",),
        preprocessing=_spec(),
        slice_indices=(0, 1),
    )


def _loader(dataset: RealPairedSliceDataset, *, seed: int = 13) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_pseudo_pair_slices,
    )


def _model_config() -> dict[str, object]:
    return {
        "name": "conditional_residual_unet_field_translator",
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": [4],
        "latent_channels": 8,
        "cond_dim": 8,
        "spatial_dims": 2,
        "upsample_mode": "interpolate",
        "skip_mode": "gated",
        "pad_to_multiple": True,
        "model_range": "minus_one_one",
    }


def _interrupt_after_first_epoch(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PseudoPairEpochConfig, RealPairedSliceDataset]:
    dataset = _dataset()
    cfg = PseudoPairEpochConfig(
        epochs=2,
        batch_size=4,
        seed=13,
        amp=False,
        log_every_steps=1,
    )
    original = paired_training._train_epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(paired_training, "_train_epoch", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        train_fixed_endpoint(
            cfg,
            model=initialize_residual_arm(
                _model_config(), arm="identity_initialization"
            ),
            train_loader=_loader(dataset),
            checkpoint_dir=directory,
            fold_slot="fold_01",
            case_slot="case_01",
            arm="identity_initialization",
            experiment_fingerprint="a" * 64,
            expected_steps_per_epoch=2,
            expected_global_step=4,
        )
    monkeypatch.setattr(paired_training, "_train_epoch", original)
    return cfg, dataset


def _recovery(directory: Path, cfg: PseudoPairEpochConfig, *, resume: bool = True):
    return resolve_arm_recovery(
        directory,
        resume=resume,
        cfg=cfg,
        fold_slot="fold_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=cfg.epochs * 2,
    )


def test_interrupted_first_arm_resumes_and_future_arms_start_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, dataset = _interrupt_after_first_epoch(tmp_path / "identity", monkeypatch)
    plan = _recovery(tmp_path / "identity", cfg)
    assert (plan.action, plan.epoch, plan.global_step) == ("resume", 1, 2)

    result = train_fixed_endpoint(
        cfg,
        model=initialize_residual_arm(_model_config(), arm="identity_initialization"),
        train_loader=_loader(dataset, seed=999),
        checkpoint_dir=tmp_path / "identity",
        fold_slot="fold_01",
        case_slot="case_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=4,
        resume_from=plan.resume_path,
    )
    assert result.global_step == 4
    assert _recovery(tmp_path / "identity", cfg).action == "endpoint"

    future = resolve_arm_recovery(
        tmp_path / "synthetic",
        resume=True,
        cfg=cfg,
        fold_slot="fold_01",
        arm="synthetic_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=4,
    )
    assert future.action == "fresh"


def test_invalid_endpoint_or_resume_fingerprint_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, _ = _interrupt_after_first_epoch(tmp_path / "resume", monkeypatch)
    resume_path = tmp_path / "resume" / "resume.pt"
    state = load_checkpoint(resume_path)
    state["experiment_fingerprint"] = "b" * 64
    torch.save(state, resume_path)
    with pytest.raises(ValueError, match="experiment_fingerprint"):
        _recovery(tmp_path / "resume", cfg)

    cfg_endpoint = PseudoPairEpochConfig(epochs=1, batch_size=4, seed=13, amp=False)
    endpoint_dir = tmp_path / "endpoint"
    train_fixed_endpoint(
        cfg_endpoint,
        model=initialize_residual_arm(_model_config(), arm="identity_initialization"),
        train_loader=_loader(_dataset()),
        checkpoint_dir=endpoint_dir,
        fold_slot="fold_01",
        case_slot="case_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=2,
    )
    endpoint_path = endpoint_dir / "endpoint.pt"
    state = load_checkpoint(endpoint_path)
    state["experiment_fingerprint"] = "b" * 64
    torch.save(state, endpoint_path)
    with pytest.raises(ValueError, match="experiment_fingerprint"):
        _recovery(endpoint_dir, cfg_endpoint)


def test_partial_artifacts_fail_closed_and_non_resume_refuses_existing_run(
    tmp_path: Path,
) -> None:
    cfg = PseudoPairEpochConfig(epochs=2, batch_size=4, seed=13, amp=False)
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / "history.jsonl").write_text(
        json.dumps({"epoch": 1, "global_step": 2}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="without a valid paired checkpoint"):
        _recovery(directory, cfg)
    with pytest.raises(FileExistsError, match="require --resume"):
        _recovery(directory, cfg, resume=False)


def test_sanitized_progress_transitions_use_atomic_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(paired_training.os, "replace", recording_replace)
    path = tmp_path / "run_progress_sanitized.json"
    progress = SanitizedRunProgress(
        path,
        fold_case_slots={"fold_01": "case_01"},
        resume=False,
    )
    progress.update(
        fold_slot="fold_01",
        case_slot="case_01",
        arm="identity_initialization",
        state="running",
        epoch=0,
        global_step=0,
        elapsed_seconds=0.0,
    )
    progress.validate_recovery("fold_01", "identity_initialization", "fresh")
    for state, epoch, step in (
        ("running", 1, 2),
        ("endpoint_complete", 2, 4),
        ("evaluating", 2, 4),
        ("complete", 2, 4),
    ):
        progress.update(
            fold_slot="fold_01",
            case_slot="case_01",
            arm="identity_initialization",
            state=state,
            epoch=epoch,
            global_step=step,
            elapsed_seconds=float(step),
        )
    payload_text = path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    status = payload["folds"][0]["arms"]["identity_initialization"]
    assert status["state"] == "complete"
    assert len(replacements) == 6
    assert all(
        source.name.endswith(".tmp") and destination == path
        for source, destination in replacements
    )
    assert not path.with_name(path.name + ".tmp").exists()
    assert "0006" not in payload_text and str(tmp_path) not in payload_text
    with pytest.raises(ValueError, match="partially started arm"):
        progress.validate_recovery("fold_01", "identity_initialization", "fresh")
    with pytest.raises(FileExistsError, match="requires --resume"):
        SanitizedRunProgress(
            path,
            fold_case_slots={"fold_01": "case_01"},
            resume=False,
        )


def test_progress_console_is_sanitized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = PseudoPairEpochConfig(
        epochs=1,
        batch_size=4,
        seed=13,
        amp=False,
        log_every_steps=1,
    )
    train_fixed_endpoint(
        cfg,
        model=initialize_residual_arm(_model_config(), arm="identity_initialization"),
        train_loader=_loader(_dataset()),
        checkpoint_dir=tmp_path / "arm",
        fold_slot="fold_01",
        case_slot="case_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=2,
    )
    output = capsys.readouterr().out
    assert "fold=fold_01 case=case_01 arm=identity_initialization" in output
    assert "step=2/2" in output and "global_step=2/2" in output
    assert all(name in output for name in ("total=", "masked_l1=", "gradient=", "background="))
    assert "steps_per_second=" in output and "0007" not in output and str(tmp_path) not in output


def test_raw_and_roundtrip_native_baselines_are_distinct_and_aggregated() -> None:
    source = _volume(depth=3)
    result = reconstruct_complete_candidate(
        source_volume=source,
        target_volume=source,
        preprocessing=_spec(depth=3),
        candidate=lambda image: image,
    )
    raw = result["raw_native_source_baseline"]
    roundtrip = result["roundtrip_native_source_baseline"]
    assert raw["nrmse"] == pytest.approx(0.0)
    assert roundtrip["nrmse"] > raw["nrmse"]

    runner = _load_runner()
    rows = []
    for fold in range(1, 4):
        for field in ("1.5T", "3T", "5T", "7T"):
            for arm in ("source", "affine", *NEURAL_INITIALIZATION_ARMS):
                rows.append(
                    {
                        "fold_slot": f"fold_{fold:02d}",
                        "case_slot": f"case_{fold:02d}",
                        "target_field": field,
                        "arm": arm,
                        "complete_volume": True,
                        "model_grid": {"nrmse": 0.2},
                        "reconstructed_native_grid": {"nrmse": 0.3},
                        "raw_native_source_baseline": {"nrmse": 0.1},
                        "roundtrip_native_source_baseline": {"nrmse": 0.15},
                    }
                )
    aggregate = runner._aggregate_complete_rows(rows)
    baselines = aggregate["native_source_baselines"]["macro"]
    assert baselines["raw_native_source_baseline"]["nrmse"] == pytest.approx(0.1)
    assert baselines["roundtrip_native_source_baseline"]["nrmse"] == pytest.approx(0.15)


def test_frozen_experiment_config_and_viability_are_unchanged() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == FROZEN_CONFIG_SHA256
    rules = load_yaml_config(CONFIG_PATH)["viability_rules"]
    assert rules == {
        "neural_macro_nrmse_below_source": True,
        "neural_macro_ssim_not_below_source": True,
        "min_fields_improved_nrmse": 3,
        "min_held_out_cases_improved_nrmse": 2,
        "min_case_field_units_correct_best_nrmse": 9,
        "require_positive_mean_conditioning_margin_every_field": True,
        "max_absolute_nrmse_regression_per_field": 0.005,
        "neural_must_beat_affine_macro_nrmse": True,
        "neural_macro_ssim_not_below_affine": True,
        "retain_synthetic_initialization_only_if_better_than_identity": True,
        "synthetic_must_not_weaken_conditioning": True,
    }


def _load_runner():
    spec = importlib.util.spec_from_file_location("paired_loso_runtime_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
