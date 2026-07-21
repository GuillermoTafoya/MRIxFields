import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from fieldbridge.config import load_yaml_config
from fieldbridge.data.paired_loso import (
    RealPairedSliceDataset,
    build_loso_folds,
    fit_train_only_affine_calibrations,
    reconstruct_native_grid_volume,
    validate_loso_folds,
    verify_full_slice_coverage,
)
from fieldbridge.data.preprocessing import (
    SlicePreprocessingSpec,
    preprocess_volume_slice,
    selected_slice_indices,
)
from fieldbridge.data.pseudo_pairs import collate_pseudo_pair_slices
from fieldbridge.evaluation.paired_loso import (
    aggregate_selected_rows,
    evaluate_viability,
    reconstruct_complete_candidate,
    sanitized_loso_handoff,
)
from fieldbridge.models.factory import build_translator
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.paired_loso import (
    PAIRED_LOSO_PIPELINE_VERSION,
    initialize_residual_arm,
    train_fixed_endpoint,
    validate_endpoint_checkpoint,
)
from fieldbridge.training.pseudo_pair_epochs import PseudoPairEpochConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiment/prospective_paired_loso_residual_v1.yaml"
ZERO_SHOT_CONFIG_PATH = PROJECT_ROOT / "configs/experiment/prospective_paired_zero_shot_v1.yaml"
RUNNER_PATH = PROJECT_ROOT / "notebooks/prospective_paired_loso_residual_runner.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/prospective_paired_loso_residual_colab.ipynb"


def _spec(*, depth: int = 4) -> SlicePreprocessingSpec:
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


def _volume(offset: float = 0.0, *, depth: int = 4) -> torch.Tensor:
    base = torch.linspace(0.05, 0.35, 4 * 5 * depth).reshape(1, 4, 5, depth)
    return (base + offset).clamp(0.0, 1.0)


def _volumes(case_ids=("0006", "0007", "0009"), *, depth: int = 4):
    return {
        case: {
            0.1: _volume(0.00, depth=depth),
            1.5: _volume(0.02, depth=depth),
            3.0: _volume(0.04, depth=depth),
            5.0: _volume(0.06, depth=depth),
            7.0: _volume(0.08, depth=depth),
        }
        for case in case_ids
    }


def _tiny_model_config():
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


def test_frozen_config_preregisters_exact_folds_arms_endpoint_and_viability() -> None:
    config = load_yaml_config(CONFIG_PATH)
    zero_shot = load_yaml_config(ZERO_SHOT_CONFIG_PATH)
    experiment = config["experiment"]

    assert experiment["case_ids"] == ["0006", "0007", "0009"]
    assert experiment["folds"] == [
        {"fold": 1, "train_cases": ["0007", "0009"], "held_out_case": "0006"},
        {"fold": 2, "train_cases": ["0006", "0009"], "held_out_case": "0007"},
        {"fold": 3, "train_cases": ["0006", "0007"], "held_out_case": "0009"},
    ]
    assert config["model"] == zero_shot["model"]
    expected_preprocessing = dict(zero_shot["preprocessing"])
    expected_preprocessing["slices_per_volume"] = None
    assert config["preprocessing"] == expected_preprocessing
    assert selected_slice_indices(
        SlicePreprocessingSpec.from_mapping(config["preprocessing"])
    ) == tuple(range(72, 292))
    assert config["evaluation"]["selected_slice_indices"] == [72, 103, 135, 166, 197, 228, 260, 291]
    assert config["training"]["validation_loader"] is None
    assert config["training"]["optimizer"] == "adamw"
    assert config["training"]["scheduler"] == {"name": "none"}
    assert config["training"]["steps_per_epoch"] == 220
    assert config["training"]["endpoint_global_step"] == 2200
    assert experiment["synthetic_examples_during_paired_training"] is False
    assert config["viability_rules"] == {
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


def test_loso_folds_are_subject_first_disjoint_and_exact() -> None:
    folds = build_loso_folds(("0006", "0007", "0009"))
    assert [(fold.train_case_ids, fold.held_out_case_id) for fold in folds] == [
        (("0007", "0009"), "0006"),
        (("0006", "0009"), "0007"),
        (("0006", "0007"), "0009"),
    ]
    validate_loso_folds(folds, ("0006", "0007", "0009"))
    with pytest.raises(ValueError, match="leakage"):
        validate_loso_folds(
            [type(folds[0])(1, ("0006", "0007"), "0006"), *folds[1:]],
            ("0006", "0007", "0009"),
        )


def test_dataset_expands_only_after_training_subject_selection() -> None:
    dataset = RealPairedSliceDataset(
        _volumes(),
        case_ids=("0007", "0009"),
        preprocessing=_spec(),
        slice_indices=(0, 1, 2, 3),
    )
    assert len(dataset) == 2 * 4 * 4
    assert {case for case, _, _ in dataset.samples} == {"0007", "0009"}
    assert "0006" not in {case for case, _, _ in dataset.samples}
    sample = dataset[0]
    assert sample.degradation_strength == 0.0
    assert sample.x_low.shape == sample.x_high.shape == (1, 6, 8)


def test_affine_calibration_uses_training_cases_only() -> None:
    volumes = _volumes()
    source = volumes["0007"][0.1]
    for case in ("0007", "0009"):
        for field in (1.5, 3.0, 5.0, 7.0):
            volumes[case][field] = (source * 1.5 + 0.1).clamp(0.0, 1.0)
    for field in (1.5, 3.0, 5.0, 7.0):
        volumes["0006"][field] = torch.ones_like(source)

    fitted = fit_train_only_affine_calibrations(
        volumes,
        train_case_ids=("0007", "0009"),
        preprocessing=_spec(),
        slice_indices=(0, 1, 2, 3),
    )
    assert all(value.fitted_cases == 2 for value in fitted.values())
    assert fitted[1.5].scale == pytest.approx(1.5, rel=1e-5)
    assert fitted[1.5].bias == pytest.approx(0.1, rel=1e-5)


def test_initialization_arms_isolate_only_checkpoint_state() -> None:
    config = _tiny_model_config()
    identity = initialize_residual_arm(config, arm="identity_initialization")
    source_model = build_translator(
        config["name"],
        **{key: value for key, value in config.items() if key != "name"},
    )
    with torch.no_grad():
        source_model.backbone.output_projection.bias.fill_(0.125)
    synthetic = initialize_residual_arm(
        config,
        arm="synthetic_initialization",
        synthetic_checkpoint={"model": source_model.state_dict()},
    )
    x = torch.zeros(1, 1, 6, 8)
    source_domain = {"field_strength_t": 0.1, "contrast": "T2-FLAIR"}
    target_domain = {"field_strength_t": 7.0, "contrast": "T2-FLAIR"}
    from fieldbridge.data.domains import Domain

    with torch.inference_mode():
        identity_output = identity(
            x,
            Domain.from_dict(source_domain),
            Domain.from_dict(target_domain),
        )
        synthetic_output = synthetic(
            x,
            Domain.from_dict(source_domain),
            Domain.from_dict(target_domain),
        )
    assert torch.equal(identity_output, x)
    assert not torch.equal(synthetic_output, x)


def test_fixed_endpoint_checkpoint_and_resume_contract(tmp_path: Path) -> None:
    dataset = RealPairedSliceDataset(
        _volumes(case_ids=("0007",)),
        case_ids=("0007",),
        preprocessing=_spec(depth=2),
        slice_indices=(0, 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(13),
        collate_fn=collate_pseudo_pair_slices,
    )
    cfg = PseudoPairEpochConfig(epochs=1, batch_size=4, seed=13, amp=False)
    model = initialize_residual_arm(_tiny_model_config(), arm="identity_initialization")
    result = train_fixed_endpoint(
        cfg,
        model=model,
        train_loader=loader,
        checkpoint_dir=tmp_path,
        fold_slot="fold_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=2,
    )
    state = load_checkpoint(result.endpoint_checkpoint)
    assert state["paired_loso_pipeline_version"] == PAIRED_LOSO_PIPELINE_VERSION == 1
    assert state["endpoint"] is True
    assert state["optimizer_name"] == "AdamW"
    assert state["global_step"] == 2
    assert "data_loader_generator_state" in state
    validate_endpoint_checkpoint(
        state,
        cfg=cfg,
        fold_slot="fold_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=2,
    )

    resumed_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(999),
        collate_fn=collate_pseudo_pair_slices,
    )
    resumed_model = initialize_residual_arm(
        _tiny_model_config(),
        arm="identity_initialization",
    )
    resumed = train_fixed_endpoint(
        cfg,
        model=resumed_model,
        train_loader=resumed_loader,
        checkpoint_dir=tmp_path,
        fold_slot="fold_01",
        arm="identity_initialization",
        experiment_fingerprint="a" * 64,
        expected_steps_per_epoch=2,
        expected_global_step=2,
        resume_from=tmp_path / "resume.pt",
    )
    assert resumed.global_step == 2
    assert resumed.endpoint_checkpoint.is_file()


def _metric(nrmse_value: float, ssim_value: float = 0.9):
    return {"nrmse": nrmse_value, "ssim": ssim_value}


def test_conditioning_sweep_aggregation_and_viability_are_hierarchical() -> None:
    rows = []
    fields = ("1.5T", "3T", "5T", "7T")
    for case_index in range(3):
        for field_index, field in enumerate(fields):
            arms = {
                "source": _metric(0.10),
                "affine": _metric(0.09),
                "identity_initialization": _metric(0.08),
                "synthetic_initialization": _metric(0.07),
            }
            conditioning = {}
            arm_values = (
                ("identity_initialization", 0.08),
                ("synthetic_initialization", 0.07),
            )
            for arm, correct in arm_values:
                requested = {label: _metric(correct + 0.02) for label in fields}
                requested[field] = _metric(correct)
                conditioning[arm] = {"requested": requested}
            rows.append(
                {
                    "fold_slot": f"fold_{case_index + 1:02d}",
                    "case_slot": f"case_{case_index + 1:02d}",
                    "target_field": field,
                    "slice_index": field_index,
                    "arms": arms,
                    "conditioning": conditioning,
                }
            )
    aggregate = aggregate_selected_rows(rows)
    assert aggregate["macro"]["synthetic_initialization"]["nrmse"] == pytest.approx(0.07)
    assert len(aggregate["case_field_units"]) == 12
    rules = load_yaml_config(CONFIG_PATH)["viability_rules"]
    viability = evaluate_viability(aggregate, rules)
    assert viability["identity_initialization"]["viable"] is True
    assert viability["synthetic_initialization_retention"]["retain"] is True


def test_every_z_slice_and_inverse_geometry_are_required() -> None:
    volume = _volume(depth=3)
    spec = _spec(depth=3)
    slices = []
    geometries = []
    for index in range(3):
        image, geometry = preprocess_volume_slice(volume, index, spec, apply_model_range=False)
        slices.append(image)
        geometries.append(geometry)
    verify_full_slice_coverage((0, 1, 2), 3)
    with pytest.raises(ValueError, match="every z slice"):
        verify_full_slice_coverage((0, 2), 3)
    restored = reconstruct_native_grid_volume(slices, geometries, depth=3)
    assert restored.shape == volume.shape

    report = reconstruct_complete_candidate(
        source_volume=volume,
        target_volume=volume,
        preprocessing=spec,
        candidate=lambda source: source,
    )
    assert report["complete_volume"] is True
    assert report["processed_slices"] == 3
    assert set(report) >= {"model_grid", "reconstructed_native_grid"}
    assert report["model_grid"]["nrmse"] == pytest.approx(0.0)


def test_sanitized_handoff_rejects_identities_paths_images_and_checkpoints() -> None:
    payload = sanitized_loso_handoff(
        audit_commit="a" * 40,
        training_checkpoint_commit="b" * 40,
        experiment_commit="c" * 40,
        aggregate={"case_field_units": [{"case_slot": "case_01"}]},
        full_volume={"complete_volume": True},
        viability={"identity_initialization": {"viable": False}},
        provenance={"config_sha256": "d" * 64},
    )
    assert payload["complete_volume_evidence"]["complete_volume"] is True
    with pytest.raises(ValueError, match="forbidden"):
        sanitized_loso_handoff(
            audit_commit="a" * 40,
            training_checkpoint_commit="b" * 40,
            experiment_commit="c" * 40,
            aggregate={"subject_id": "private"},
            full_volume={},
            viability={},
            provenance={},
        )


def test_runner_and_notebook_preregister_preflight_dry_run_resume_scratch_and_telemetry() -> None:
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "--preflight" in runner_source
    assert "--dry-run" in runner_source
    assert "--resume" in runner_source
    assert "NvidiaSmiTelemetry" in runner_source
    assert "shutil.copy2" in runner_source
    assert "scratch_dir" in runner_source
    assert "validation_loader" in runner_source

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"loso_cell_{index}", "exec")
    assert (
        source.index("nvidia-smi")
        < source.index("git', 'clone")
        < source.index("'pip', 'install'")
    )
    assert "EXPECTED_EXPERIMENT_COMMIT" in source
    assert "fieldbridge_loso_scratch" in source
    assert "--preflight" in source and "--dry-run" in source and "--resume" in source


def test_runner_config_preflight_contract_loads_without_private_inputs() -> None:
    spec = importlib.util.spec_from_file_location("paired_loso_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preprocessing, folds = module._validate_config(load_yaml_config(CONFIG_PATH))
    assert len(folds) == 3
    assert selected_slice_indices(preprocessing) == tuple(range(72, 292))

    dataset = RealPairedSliceDataset(
        _volumes(case_ids=("0007",)),
        case_ids=("0007",),
        preprocessing=_spec(depth=2),
        slice_indices=(0, 1),
    )
    train_cfg = PseudoPairEpochConfig(batch_size=4, seed=13)
    first = next(iter(module._build_train_loader(dataset, train_cfg=train_cfg, num_workers=0)))
    second = next(iter(module._build_train_loader(dataset, train_cfg=train_cfg, num_workers=0)))
    assert torch.equal(first.slice_index, second.slice_index)
    assert first.record_id == second.record_id
