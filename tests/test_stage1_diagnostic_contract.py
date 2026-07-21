import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from fieldbridge.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment"
    / "stage1_vae_reconstruction_diagnostic_v1.yaml"
)
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "stage1_vae_reconstruction_diagnostic_colab.ipynb"
)
RUNNER_PATH = (
    PROJECT_ROOT / "notebooks" / "stage1_vae_reconstruction_diagnostic_runner.py"
)
STATUS_PATH = PROJECT_ROOT / "docs" / "STATUS.md"
HISTORICAL_TRAINING_COMMIT = "c9ee9dd738f8d9fee7acf9340dc4325c47a639cd"


def _load_runner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("stage1_diagnostic_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage1_diagnostic_config_is_inference_only_and_frozen() -> None:
    config = load_yaml_config(CONFIG_PATH)
    diagnostic = config["diagnostic"]

    assert set(config) == {"diagnostic"}
    assert diagnostic["contract_version"] == 1
    assert diagnostic["evidence_scope"] == "stage1_reconstruction_engineering_diagnostic"
    assert diagnostic["held_out"] is False
    assert diagnostic["confirmatory"] is False
    assert diagnostic["fixed_patch_index"] == 13
    assert diagnostic["fixed_volume_index"] == 0
    assert diagnostic["sampled_latent_seed"] == 13
    assert diagnostic["overlap_sweep"] == [0.25, 0.5, 0.75]
    assert diagnostic["reference_overlap"] == 0.5
    assert diagnostic["background_threshold_minus_one_one"] == -0.95
    assert diagnostic["collapse_std_ratio_threshold"] == 0.5
    assert diagnostic["overlap_nrmse_span_threshold"] == 0.01
    assert diagnostic["seam_ratio_span_threshold"] == 0.1
    assert diagnostic["checkpoint_step_sweep_policy"] == (
        "report_all_chronologically_no_best_selection"
    )
    assert diagnostic["training_allowed"] is False
    assert diagnostic["stage2_allowed"] is False
    assert "model" not in config
    assert "training" not in config


def test_stage1_diagnostic_notebook_is_unexecuted_and_accepts_external_inputs() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"stage1_diagnostic_cell_{index}", "exec")

    for prompt in (
        "External Stage-1 checkpoint path",
        "External Stage-1 patch-bank directory",
        "External official JSONL manifest path",
        "Historical training/checkpoint commit SHA",
        "New diagnostic output directory",
    ):
        assert prompt in source
    assert "EXPECTED_CODE_COMMIT = input" in source
    assert "stage1_vae_reconstruction_diagnostic_runner.py" in source
    assert "run_stage1_diagnostic" in source
    assert "CHECKPOINT_SWEEP_PATHS" in source
    assert "nvidia-smi" in source
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "train-stage1-vae" not in source
    assert "train-stage2-diffuser" not in source
    assert "run_stage1_vae_train" not in source
    assert "build_patch_bank" not in source

    preflight_index = source.index('subprocess.run(["nvidia-smi"]')
    clone_index = source.index('"clone", "--no-checkout"')
    install_index = source.index('"pip", "install"')
    source_path_index = source.index("sys.path.insert(0, SOURCE_DIR)")
    invalidate_index = source.index("importlib.invalidate_caches()")
    package_import_index = source.index("import fieldbridge as installed_fieldbridge")
    assert preflight_index < clone_index < install_index
    assert install_index < source_path_index < invalidate_index < package_import_index
    assert 'if not REPO_DIR.exists():' in source
    assert '["git", "fetch", "origin", EXPECTED_CODE_COMMIT]' in source
    assert "Use a fresh Colab runtime" not in source
    assert 'module_name.startswith("fieldbridge.")' in source
    assert "REPO_DIR not in PACKAGE_FILE.parents" in source
    assert "reconstruct_resolved_training_yaml" in source


def test_stage1_diagnostic_runner_has_no_training_or_selection_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert "run_stage1_diagnostic" in function_names
    assert "reconstruct_resolved_training_yaml" in function_names
    assert "load_adapted_mrixfields_manifest" in source
    assert "strict_paths=True" in source
    assert "run_stage1_reconstruction_diagnostics" in source
    assert "stage1_diagnostic_handoff.json" in source
    assert "torch.device(\"cuda\")" in source
    assert "train-stage1-vae" not in source
    assert "train-stage2-diffuser" not in source
    assert "run_stage1_vae_train" not in source
    assert "build_patch_bank" not in source
    assert "best.pt" not in source
    assert "diagnostics.png" not in source


def test_resolved_training_yaml_uses_historical_commit_checkpoint_and_bank(tmp_path) -> None:
    runner = _load_runner()
    recorded_config = {
        "steps": 54_000,
        "batch_size": 8,
        "seed": 13,
        "lr": 0.0001,
        "device": "cuda",
        "precision": "bf16",
        "loss_weights": {"ssim": 1.0, "nrmse": 1.0, "lpips": 1.0, "kl": 0.0001},
        "ssim_window_size": 7,
        "lpips_num_slices": 8,
        "grad_clip_norm": 1.0,
        "steps_per_epoch": 3968,
        "early_stopping": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.005,
        "early_stopping_ema_decay": 0.98,
        "checkpoint_at_end": True,
        "log_every_steps": 1,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "_meta": {
                "git_commit": HISTORICAL_TRAINING_COMMIT,
                "seed": 13,
                "config": recorded_config,
            }
        },
        checkpoint_path,
    )
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "bank_meta.json").write_text(
        json.dumps({"patch_size": [64, 64, 64], "patches_per_volume": 32, "seed": 13}),
        encoding="utf-8",
    )
    output_path = tmp_path / "resolved.yaml"

    result = runner.reconstruct_resolved_training_yaml(
        repo_dir=PROJECT_ROOT,
        checkpoint_path=checkpoint_path,
        patch_bank_dir=bank_dir,
        expected_training_commit=HISTORICAL_TRAINING_COMMIT,
        output_path=output_path,
    )

    assert result == output_path
    resolved = load_yaml_config(output_path)
    assert resolved["seed"] == 13
    assert resolved["data"]["patch_size"] == [64, 64, 64]
    assert resolved["data"]["patches_per_volume"] == 32
    assert resolved["training"]["steps"] == 54_000
    assert resolved["training"]["steps_per_epoch"] == 3968
    assert resolved["training"]["loss_weights"] == recorded_config["loss_weights"]
    assert "l1" not in resolved["training"]["loss_weights"]
    assert "stratified_crop" not in resolved["data"]

    with pytest.raises(ValueError, match="does not equal"):
        runner.reconstruct_resolved_training_yaml(
            repo_dir=PROJECT_ROOT,
            checkpoint_path=checkpoint_path,
            patch_bank_dir=bank_dir,
            expected_training_commit="a" * 40,
            output_path=output_path,
        )


def test_status_records_supplied_negative_engineering_evidence_without_overclaim() -> None:
    status = STATUS_PATH.read_text(encoding="utf-8")

    assert "user-supplied evidence" in status
    assert "have not independently verified" in status
    assert "1,984 volumes, 63,488 patches, 32 patches per volume" in status
    assert "54,000 steps" in status
    assert "1,939 retrospective and 45 prospective" in status
    assert "nRMSE | 0.48881893" in status
    assert "SSIM3D | -0.00149328" in status
    assert "LPIPS | 0.62425638" in status
    assert "MAE | 0.90241840" in status
    assert "MSE | 0.95595611" in status
    assert "not held out or confirmatory" in status
    assert "overlap `0.25`, despite notebook prose stating `0.5`" in status
    assert "did not select a\nbest checkpoint post hoc" in status
    assert "9b071cc17c545e126891ae77f7e0dd27c2815b1c" in status
    assert HISTORICAL_TRAINING_COMMIT in status
    assert "held_out: false" in status
    assert "confirmatory: false" in status
    assert "complete_volume: true" in status
    assert "0.50438815" in status
    assert "0.05191850" in status
    assert "possible variance-channel information leakage" in status
    assert "Stage 2 was not started\nand remains blocked" in status
    assert "diagnostic v1 pending" not in status
