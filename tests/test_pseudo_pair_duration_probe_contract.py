import ast
import json
import re
import statistics
from copy import deepcopy
from pathlib import Path

from fieldbridge.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MICRO_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiment" / "pseudo_pair_t2flair_micro.yaml"
)
PROBE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment"
    / "pseudo_pair_t2flair_duration_probe_10epoch.yaml"
)
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "pseudo_pair_duration_probe_colab.ipynb"
STATUS_PATH = PROJECT_ROOT / "docs" / "STATUS.md"
EXPECTED_SPLIT_SHA256 = (
    "17f00411ab04331fa0380526b2d8f0cd0173e4ff6f8978f72c61053fa7385dbe"
)


def _without(mapping: dict, *keys: str) -> dict:
    value = deepcopy(mapping)
    for key in keys:
        value.pop(key)
    return value


def test_duration_probe_changes_only_duration_and_output_namespace() -> None:
    micro = load_yaml_config(MICRO_CONFIG_PATH)
    probe = load_yaml_config(PROBE_CONFIG_PATH)

    assert probe["seed"] == micro["seed"] == 13
    assert probe["model"] == micro["model"]
    assert _without(probe["data"], "split_json") == _without(
        micro["data"], "split_json"
    )
    assert _without(probe["training"], "epochs", "checkpoint_dir") == _without(
        micro["training"], "epochs", "checkpoint_dir"
    )
    assert _without(probe["evaluation"], "evaluation_after_epoch") == _without(
        micro["evaluation"], "evaluation_after_epoch"
    )

    assert probe["training"]["epochs"] == 10
    assert probe["training"]["resume_from"] is None
    assert probe["evaluation"]["evaluation_after_epoch"] == 10
    assert probe["data"]["split_json"] != micro["data"]["split_json"]
    assert probe["training"]["checkpoint_dir"] != micro["training"]["checkpoint_dir"]

    contract = probe["probe"]
    assert contract == {
        "kind": "duration_only",
        "changed_variable": "training.epochs",
        "prior_effective_epochs": 2,
        "expected_epochs": 10,
        "expected_steps_per_epoch": 16,
        "expected_global_steps": 160,
        "prior_split_sha256": EXPECTED_SPLIT_SHA256,
        "split_evidence_role": "development_reuse_not_confirmatory",
        "fresh_initialization_required": True,
        "endpoint_evaluation_only": True,
        "scaled_pilot_blocked": True,
    }


def test_duration_probe_notebook_is_unexecuted_and_enforces_run_contract() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"duration_probe_cell_{index}", "exec")

    assert EXPECTED_SPLIT_SHA256 in source
    assert "pseudo_pair_t2flair_duration_probe_10epoch.yaml" in source
    assert "EXPECTED_CODE_COMMIT = input" in source
    assert "Prior Drive volume_splits.json path" in source
    assert "if RUN_DIR.exists():" in source
    assert "raise FileExistsError" in source
    assert "volume_splits_fingerprint(PRIOR_SPLITS)" in source
    assert "shutil.copy2(PRIOR_SPLIT_PATH, RUN_SPLIT_PATH)" in source
    assert '"--epochs"' in source
    assert "str(EXPECTED_EPOCHS)" in source
    assert "--resume-checkpoint" not in source
    assert source.count('"eval-pseudo-pairs"') == 1
    assert '"--split"' in source and '"test"' in source
    assert 'CHECKPOINT_DIR / "last.pt"' in source
    assert 'STATE.get("epoch", -1)' in source
    assert 'STATE.get("global_step", -1)' in source
    assert "EXPECTED_GLOBAL_STEPS / TRAIN_WALL_SECONDS" in source
    install_index = source.index('"pip", "install"')
    source_path_index = source.index("sys.path.insert(0, SOURCE_DIR)")
    invalidate_index = source.index("importlib.invalidate_caches()")
    package_import_index = source.index("import fieldbridge as installed_fieldbridge")
    assert install_index < source_path_index < invalidate_index < package_import_index
    assert 'module_name.startswith("fieldbridge.")' in source
    assert "REPO_DIR not in PACKAGE_FILE.parents" in source

    for telemetry_field in (
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
    ):
        assert telemetry_field in source
    assert '"--loop=5"' in source
    assert '"gpu_utilization_percent"' in source
    assert '"memory_used_mib"' in source
    assert '"power_draw_watts"' in source
    assert '"p95"' in source
    assert '"gpu_telemetry": GPU_TELEMETRY_SUMMARY' in source
    assert "codex_handoff_sanitized.json" in source
    assert '"scaled_pilot": "BLOCKED_PENDING_REVIEW"' in source
    assert "pseudo_pair_t2flair_pilot.yaml" not in source


def test_duration_probe_handoff_literal_contains_no_private_identity_or_artifact() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    handoff_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "HANDOFF = {" in "".join(cell["source"])
    )
    tree = ast.parse(handoff_cell)
    handoff_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "HANDOFF" for target in node.targets)
    )
    constants = [
        node.value.lower()
        for node in ast.walk(handoff_assignment.value)
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
    assert "pipeline_version" in constants
    assert "split_sha256" in constants
    assert "development_reuse_not_confirmatory" in constants
    assert "sampled_slice_per_volume_exploratory" in constants
    assert "gpu_telemetry" in constants


def test_duration_probe_telemetry_summary_is_numeric_and_sanitized() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    telemetry_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def summarize_nvidia_smi" in "".join(cell["source"])
    )
    tree = ast.parse(telemetry_cell)
    functions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"re": re, "statistics": statistics}
    exec(compile(functions, "duration_probe_telemetry", "exec"), namespace)

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
    summary = namespace["summarize_nvidia_smi"](rows)

    assert summary == {
        "gpu_utilization_percent": {
            "mean": 25.0,
            "median": 25.0,
            "p95": 38.5,
            "max": 40.0,
        },
        "memory_used_mib": {"max": 400.0},
        "power_draw_watts": {"mean": 65.0, "max": 80.0},
    }
    assert not any("path" in key for key in summary)


def test_status_records_user_supplied_negative_result_without_overclaiming() -> None:
    status = STATUS_PATH.read_text(encoding="utf-8")

    assert "user-supplied evidence" in status
    assert "independently verified" in status
    assert EXPECTED_SPLIT_SHA256 in status
    assert "Macro nRMSE | 0.054048 | 0.104325" in status
    assert "Macro SSIM | 0.874324 | 0.167202" in status
    assert "No target field improved nRMSE (`0/4`)" in status
    assert "valid negative evidence" in status
    assert "scaled pilot remains blocked" in status
    assert "cannot satisfy promotion or\nfinal-volume gates" in status
    assert "Macro nRMSE | 0.05404807 | 0.03838583" in status
    assert "Macro SSIM | 0.87432381 | 0.53409448" in status
    assert "Increasing duration rescued nRMSE" in status
    assert "did not\nrescue SSIM or target conditioning" in status
    assert "observed development split; not confirmatory evidence" in status
