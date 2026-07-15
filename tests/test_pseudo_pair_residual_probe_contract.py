import ast
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

from fieldbridge.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "configs" / "experiment"
BASELINE_CONFIG_PATH = (
    EXPERIMENT_DIR / "pseudo_pair_t2flair_duration_probe_10epoch.yaml"
)
PROBE_CONFIG_PATH = (
    EXPERIMENT_DIR / "pseudo_pair_t2flair_residual_probe_10epoch.yaml"
)
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "pseudo_pair_residual_probe_colab.ipynb"
RUNNER_PATH = PROJECT_ROOT / "notebooks" / "pseudo_pair_residual_probe_runner.py"
EXPECTED_SPLIT_SHA256 = (
    "17f00411ab04331fa0380526b2d8f0cd0173e4ff6f8978f72c61053fa7385dbe"
)


def _without(mapping: dict, *keys: str) -> dict:
    value = deepcopy(mapping)
    for key in keys:
        value.pop(key)
    return value


def _load_runner():
    spec = importlib.util.spec_from_file_location("residual_probe_contract_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residual_probe_changes_only_model_parameterization_and_output_namespace() -> None:
    baseline = load_yaml_config(BASELINE_CONFIG_PATH)
    probe = load_yaml_config(PROBE_CONFIG_PATH)

    assert probe["seed"] == baseline["seed"] == 13
    assert _without(probe["data"], "split_json") == _without(
        baseline["data"], "split_json"
    )
    assert _without(probe["training"], "checkpoint_dir") == _without(
        baseline["training"], "checkpoint_dir"
    )
    assert probe["evaluation"] == baseline["evaluation"]

    baseline_model = _without(baseline["model"], "name", "final_activation")
    probe_model = _without(probe["model"], "name", "model_range")
    assert probe_model == baseline_model
    assert baseline["model"]["name"] == "conditional_unet_field_translator"
    assert baseline["model"]["final_activation"] == "tanh"
    assert probe["model"]["name"] == "conditional_residual_unet_field_translator"
    assert probe["model"]["model_range"] == "minus_one_one"

    assert probe["data"]["split_json"] != baseline["data"]["split_json"]
    assert probe["training"]["checkpoint_dir"] != baseline["training"]["checkpoint_dir"]
    assert "residual_probe_10epoch" in probe["data"]["split_json"]
    assert "residual_probe_10epoch" in probe["training"]["checkpoint_dir"]


def test_residual_probe_freezes_endpoint_split_losses_and_thresholds() -> None:
    probe = load_yaml_config(PROBE_CONFIG_PATH)
    contract = probe["probe"]

    assert probe["data"]["preprocessing"] == {
        "slice_start": 72,
        "slice_end": 292,
        "slices_per_volume": 8,
        "normalization": "official_01",
        "model_range": "minus_one_one",
        "resize_mode": "fit_pad",
        "output_height": 128,
        "output_width": 160,
        "slice_axis": "z",
    }
    assert probe["training"]["epochs"] == 10
    assert probe["training"]["resume_from"] is None
    assert probe["training"]["loss_weights"] == {
        "masked_l1": 1.0,
        "gradient": 0.2,
        "background": 0.5,
    }
    assert probe["evaluation"]["evaluation_after_epoch"] == 10
    assert contract["prior_split_sha256"] == EXPECTED_SPLIT_SHA256
    assert contract["split_evidence_role"] == "development_reuse_not_confirmatory"
    assert contract["fresh_initialization_required"] is True
    assert contract["exact_identity_at_initialization"] is True
    assert contract["endpoint_evaluation_only"] is True
    assert contract["expected_steps_per_epoch"] == 16
    assert contract["expected_global_steps"] == 160
    assert contract["scaled_pilot_blocked"] is True

    restoration = contract["gate_groups"]["restoration"]
    conditioning = contract["gate_groups"]["conditioning"]
    assert restoration == [
        "min_macro_relative_nrmse_improvement",
        "min_macro_absolute_ssim_improvement",
        "min_fields_with_nrmse_improvement",
        "max_macro_outside_mask_mean_abs",
    ]
    assert conditioning == [
        "min_fraction_volumes_correct_best_nrmse",
        "min_mean_margin_vs_best_wrong_nrmse",
        "min_relative_correct_vs_wrong_nrmse_improvement",
        "min_relative_correct_vs_permuted_nrmse_improvement",
    ]
    assert set(restoration).isdisjoint(conditioning)
    assert contract["scientific_gate_rule"] == "restoration_and_conditioning"


def test_residual_probe_notebook_is_unexecuted_and_repairs_same_kernel_imports() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"residual_probe_cell_{index}", "exec")

    assert EXPECTED_SPLIT_SHA256 in source
    assert "pseudo_pair_t2flair_residual_probe_10epoch.yaml" in source
    assert "EXPECTED_CODE_COMMIT = input" in source
    assert "Prior Drive volume_splits.json path" in source
    assert "New residual-probe run directory" in source
    assert "run_residual_probe" in source
    assert "pseudo_pair_residual_probe_runner.py" in source
    assert "pseudo_pair_t2flair_pilot.yaml" not in source
    assert "--resume-checkpoint" not in source

    install_index = source.index('"pip", "install"')
    source_path_index = source.index("sys.path.insert(0, SOURCE_DIR)")
    invalidate_index = source.index("importlib.invalidate_caches()")
    package_import_index = source.index("import fieldbridge as installed_fieldbridge")
    assert install_index < source_path_index < invalidate_index < package_import_index
    assert 'module_name.startswith("fieldbridge.")' in source
    assert "REPO_DIR not in PACKAGE_FILE.parents" in source


def test_residual_runner_enforces_fresh_endpoint_only_execution() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "EXPECTED_EPOCHS = 10" in source
    assert "EXPECTED_STEPS_PER_EPOCH = 16" in source
    assert "EXPECTED_GLOBAL_STEPS = 160" in source
    assert EXPECTED_SPLIT_SHA256 in source
    assert "conditional_residual_unet_field_translator" in source
    assert "ConditionalResidualUNetFieldTranslator" in source
    assert "torch.equal(prediction, x_low)" in source
    assert "for field in TARGET_FIELDS" in source
    assert "--resume-checkpoint" not in source
    assert source.count('"eval-pseudo-pairs"') == 1
    assert 'checkpoint_dir / "last.pt"' in source
    assert '"epoch": EXPECTED_EPOCHS' in source
    assert '"global_step": EXPECTED_GLOBAL_STEPS' in source
    assert '"scaled_pilot": "BLOCKED_PENDING_REVIEW"' in source
    assert '"rule": "restoration_and_conditioning"' in source
    assert "restoration_status == \"PASS\" and conditioning_status == \"PASS\"" in source

    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "run_residual_probe",
        "summarize_nvidia_smi",
        "_validate_step_zero_identity",
        "_run_endpoint_evaluation",
        "_build_sanitized_handoff",
    } <= function_names


def test_residual_probe_telemetry_summary_is_aggregated_and_sanitized() -> None:
    runner = _load_runner()
    rows = [
        {
            "utilization.gpu [%]": "10 %",
            "memory.used [MiB]": "100 MiB",
            "power.draw [W]": "50 W",
        },
        {
            "utilization.gpu [%]": "20 %",
            "memory.used [MiB]": "400 MiB",
            "power.draw [W]": "60 W",
        },
        {
            "utilization.gpu [%]": "30 %",
            "memory.used [MiB]": "300 MiB",
            "power.draw [W]": "70 W",
        },
        {
            "utilization.gpu [%]": "40 %",
            "memory.used [MiB]": "200 MiB",
            "power.draw [W]": "80 W",
        },
    ]

    assert runner.summarize_nvidia_smi(rows) == {
        "gpu_utilization_percent": {
            "mean": 25.0,
            "median": 25.0,
            "p95": 38.5,
            "max": 40.0,
        },
        "memory_used_mib": {"max": 400.0},
        "power_draw_watts": {"mean": 65.0, "max": 80.0},
    }


def test_sanitized_handoff_literal_excludes_private_identity_and_artifact_fields() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    handoff_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_sanitized_handoff"
    )
    constants = [
        node.value.lower()
        for node in ast.walk(handoff_function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    forbidden = (
        "subject_id",
        "volume_path",
        "record_id",
        "slice_index",
        ".nii",
        "/content/drive",
        "last.pt",
        "best.pt",
        "image",
    )

    assert not any(term in value for term in forbidden for value in constants)
    assert "restoration" in constants
    assert "conditioning" in constants
    assert "scientific" in constants
    assert "development_reuse_not_confirmatory" in constants
    assert "sampled_slice_per_volume_exploratory" in constants
