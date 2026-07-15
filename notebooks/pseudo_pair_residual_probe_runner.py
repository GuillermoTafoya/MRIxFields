"""Colab orchestration for the predeclared residual pseudo-pair probe."""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data.domains import Domain
from fieldbridge.data.volume_splits import (
    audit_volume_splits,
    load_volume_splits,
    volume_splits_fingerprint,
)
from fieldbridge.models.factory import build_translator
from fieldbridge.training.checkpoints import load_checkpoint

EXPECTED_SPLIT_SHA256 = (
    "17f00411ab04331fa0380526b2d8f0cd0173e4ff6f8978f72c61053fa7385dbe"
)
EXPECTED_EPOCHS = 10
EXPECTED_STEPS_PER_EPOCH = 16
EXPECTED_GLOBAL_STEPS = 160
TARGET_FIELDS = (1.5, 3.0, 5.0, 7.0)
CONFIG_NAME = "pseudo_pair_t2flair_residual_probe_10epoch.yaml"
NVIDIA_SMI_FIELDS = (
    "timestamp,utilization.gpu,memory.used,memory.total,power.draw,power.limit"
)


def run_residual_probe(
    *,
    repo_dir: Path,
    manifest_path: Path,
    prior_split_path: Path,
    run_dir: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run one fresh residual probe and emit a sanitized decision handoff."""

    repo_dir = repo_dir.resolve()
    manifest_path = manifest_path.expanduser().resolve()
    prior_split_path = prior_split_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    _validate_external_inputs(repo_dir, manifest_path, prior_split_path, run_dir)
    run_dir.mkdir(parents=True)

    config_path = repo_dir / "configs" / "experiment" / CONFIG_NAME
    config = load_yaml_config(config_path)
    _validate_probe_config(config)
    _validate_step_zero_identity(config)
    subprocess.run(["nvidia-smi"], check=True)

    splits = load_volume_splits(prior_split_path)
    audit_volume_splits(splits).raise_for_leakage()
    split_sha256 = volume_splits_fingerprint(splits)
    if split_sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"Split fingerprint mismatch: {split_sha256} != {EXPECTED_SPLIT_SHA256}"
        )

    run_split_path = run_dir / "volume_splits.json"
    shutil.copy2(prior_split_path, run_split_path)
    _require_split_fingerprint(run_split_path)
    shutil.copy2(config_path, run_dir / "declared_config.yaml")
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        raise FileExistsError("Checkpoint directory must not exist before training.")

    common_train_args = _common_train_args(
        config_path=config_path,
        manifest_path=manifest_path,
        split_path=run_split_path,
        checkpoint_dir=checkpoint_dir,
    )
    preflight = _run_preflight(common_train_args, repo_dir, run_dir)
    if preflight["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Preflight loaded a different split.")
    if int(preflight["steps_per_epoch"]) != EXPECTED_STEPS_PER_EPOCH:
        raise RuntimeError("Preflight steps_per_epoch changed from 16.")
    if not bool(preflight["leakage_audit"]["ok"]):
        raise RuntimeError("Preflight leakage audit failed.")
    _require_split_fingerprint(run_split_path)

    runtime = _run_training_with_telemetry(common_train_args, repo_dir, run_dir)
    state, last_checkpoint = _validate_endpoint_checkpoint(
        checkpoint_dir,
        split_sha256,
    )
    evaluation = _run_endpoint_evaluation(
        repo_dir=repo_dir,
        config_path=config_path,
        manifest_path=manifest_path,
        checkpoint_path=last_checkpoint,
        split_path=run_split_path,
        run_dir=run_dir,
    )
    handoff = _build_sanitized_handoff(
        evaluation=evaluation,
        config=config,
        state=state,
        runtime=runtime,
        code_commit=code_commit,
        split_sha256=split_sha256,
    )
    handoff_text = json.dumps(handoff, indent=2, sort_keys=True)
    _assert_sanitized_handoff(handoff_text)
    (run_dir / "codex_handoff_sanitized.json").write_text(
        handoff_text,
        encoding="utf-8",
    )
    print(handoff_text)
    return handoff


def _validate_external_inputs(
    repo_dir: Path,
    manifest_path: Path,
    prior_split_path: Path,
    run_dir: Path,
) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError("The external manifest path does not exist.")
    if prior_split_path.name != "volume_splits.json" or not prior_split_path.is_file():
        raise FileNotFoundError("Provide the prior Drive volume_splits.json file.")
    if run_dir.exists():
        raise FileExistsError("Fresh initialization requires a new run directory.")
    if run_dir == repo_dir or repo_dir in run_dir.parents:
        raise ValueError("Run outputs must remain outside the Git checkout.")


def _validate_probe_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    evaluation = config["evaluation"]
    model = config["model"]
    probe = config["probe"]
    if training["epochs"] != EXPECTED_EPOCHS:
        raise RuntimeError("Residual probe must declare exactly 10 epochs.")
    if training["resume_from"] is not None or not probe["fresh_initialization_required"]:
        raise RuntimeError("Residual probe must use fresh initialization.")
    if evaluation["evaluation_after_epoch"] != EXPECTED_EPOCHS:
        raise RuntimeError("Evaluation endpoint must remain epoch 10.")
    if model["name"] != "conditional_residual_unet_field_translator":
        raise RuntimeError("Residual probe selected an unexpected translator.")
    if model["model_range"] != config["data"]["preprocessing"]["model_range"]:
        raise RuntimeError("Translator and preprocessing model ranges differ.")
    if training["loss_weights"] != {
        "masked_l1": 1.0,
        "gradient": 0.2,
        "background": 0.5,
    }:
        raise RuntimeError("Residual probe loss contract changed.")
    if not training["amp"] or not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA with AMP enabled.")
    if not probe["scaled_pilot_blocked"]:
        raise RuntimeError("Scaled pilot must remain blocked.")


def _validate_step_zero_identity(config: Mapping[str, Any]) -> None:
    model_config = dict(config["model"])
    model_name = str(model_config.pop("name"))
    model = build_translator(model_name, **model_config)
    model.eval()
    x_low = torch.linspace(-1.0, 1.0, 128 * 160).reshape(1, 1, 128, 160)
    source = Domain(0.1, "T2-FLAIR")
    with torch.no_grad():
        for field in TARGET_FIELDS:
            prediction = model(x_low, source, Domain(field, "T2-FLAIR"))
            if not torch.equal(prediction, x_low):
                raise RuntimeError(
                    f"Residual translator is not exact identity for target field {field:g}T."
                )


def _require_split_fingerprint(path: Path) -> None:
    actual = volume_splits_fingerprint(load_volume_splits(path))
    if actual != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Copied split fingerprint changed: {actual}.")


def _common_train_args(
    *,
    config_path: Path,
    manifest_path: Path,
    split_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "fieldbridge.cli",
        "train-pseudo-pairs",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--split-json",
        str(split_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(EXPECTED_EPOCHS),
        "--workers",
        "0",
    ]


def _run_preflight(
    common_args: Sequence[str],
    repo_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    process = subprocess.run(
        [*common_args, "--preflight", "--json"],
        cwd=repo_dir,
        text=True,
        capture_output=True,
    )
    (run_dir / "preflight_private.log").write_text(
        process.stdout + process.stderr,
        encoding="utf-8",
    )
    print(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(process.stderr)
    return _parse_trailing_json(process.stdout)


def _parse_trailing_json(text: str) -> dict[str, Any]:
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            payload = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Command output did not end with a JSON object.")


def _run_training_with_telemetry(
    common_args: Sequence[str],
    repo_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    telemetry_path = run_dir / "nvidia_smi_training.csv"
    train_log_path = run_dir / "train_private.log"
    train_return_code: int | None = None
    with telemetry_path.open("w", encoding="utf-8") as telemetry_file:
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={NVIDIA_SMI_FIELDS}",
                "--format=csv",
                "--loop=5",
            ],
            stdout=telemetry_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = time.perf_counter()
        try:
            with train_log_path.open("w", encoding="utf-8") as train_log:
                training = subprocess.Popen(
                    [*common_args, "--json"],
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if training.stdout is None:
                    raise RuntimeError("Training process did not expose stdout.")
                for line in training.stdout:
                    print(line, end="")
                    train_log.write(line)
                train_return_code = training.wait()
        finally:
            wall_seconds = time.perf_counter() - started
            telemetry.terminate()
            try:
                telemetry.wait(timeout=15)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
    if train_return_code != 0:
        raise RuntimeError(f"Training failed with return code {train_return_code}.")
    with telemetry_path.open("r", encoding="utf-8", newline="") as telemetry_input:
        rows = list(csv.DictReader(telemetry_input))
    if not rows:
        raise RuntimeError("nvidia-smi did not record telemetry samples.")
    return {
        "cuda": True,
        "amp": True,
        "wall_seconds": wall_seconds,
        "steps_per_second": EXPECTED_GLOBAL_STEPS / wall_seconds,
        "gpu_telemetry_samples": len(rows),
        "gpu_telemetry": summarize_nvidia_smi(rows),
    }


def nvidia_smi_values(rows: Sequence[Mapping[str, str]], prefix: str) -> list[float]:
    if not rows:
        raise ValueError("nvidia-smi telemetry contains no samples.")
    key = next((name for name in rows[0] if name.startswith(prefix)), None)
    if key is None:
        raise ValueError(f"nvidia-smi telemetry is missing {prefix!r}.")
    values: list[float] = []
    for row in rows:
        match = re.search(r"-?\d+(?:\.\d+)?", row.get(key, ""))
        if match is not None:
            values.append(float(match.group(0)))
    if not values:
        raise ValueError(f"nvidia-smi telemetry has no numeric {prefix!r} samples.")
    return values


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_nvidia_smi(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    utilization = nvidia_smi_values(rows, "utilization.gpu")
    memory_used = nvidia_smi_values(rows, "memory.used")
    power_draw = nvidia_smi_values(rows, "power.draw")
    return {
        "gpu_utilization_percent": {
            "mean": statistics.fmean(utilization),
            "median": statistics.median(utilization),
            "p95": percentile(utilization, 0.95),
            "max": max(utilization),
        },
        "memory_used_mib": {"max": max(memory_used)},
        "power_draw_watts": {
            "mean": statistics.fmean(power_draw),
            "max": max(power_draw),
        },
    }


def _validate_endpoint_checkpoint(
    checkpoint_dir: Path,
    split_sha256: str,
) -> tuple[dict[str, Any], Path]:
    last_checkpoint = checkpoint_dir / "last.pt"
    if not last_checkpoint.is_file():
        raise FileNotFoundError("Training did not produce the epoch-10 last checkpoint.")
    state = load_checkpoint(last_checkpoint, map_location="cpu")
    expected = {
        "trainer": "pseudo_pair_epochs",
        "pseudo_pair_pipeline_version": 2,
        "model_class": "ConditionalResidualUNetFieldTranslator",
        "epoch": EXPECTED_EPOCHS,
        "global_step": EXPECTED_GLOBAL_STEPS,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Unexpected checkpoint {key}: {state.get(key)!r}.")
    if state.get("run_metadata", {}).get("split_sha256") != split_sha256:
        raise RuntimeError("Checkpoint metadata records a different split identity.")
    return state, last_checkpoint


def _run_endpoint_evaluation(
    *,
    repo_dir: Path,
    config_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    split_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "fieldbridge.cli",
        "eval-pseudo-pairs",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split-json",
        str(split_path),
        "--split",
        "test",
        "--workers",
        "0",
        "--json",
    ]
    process = subprocess.run(command, cwd=repo_dir, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(process.stderr)
    evaluation = json.loads(process.stdout)
    (run_dir / "evaluation_private.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if evaluation.get("complete_volume") is not False:
        raise RuntimeError("Selected-slice evaluation must report complete_volume false.")
    if evaluation.get("evidence_scope") != "sampled_slice_per_volume_exploratory":
        raise RuntimeError("Unexpected evaluation evidence scope.")
    return evaluation


def _build_sanitized_handoff(
    *,
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    runtime: Mapping[str, Any],
    code_commit: str,
    split_sha256: str,
) -> dict[str, Any]:
    volume_eval = evaluation["sampled_slice_per_volume"]
    macro = volume_eval["macro_average"]
    per_field = volume_eval["per_target_field"]
    conditioning = volume_eval["target_conditioning_audit"]
    volume_conditioning = conditioning["volume_level"]
    thresholds = config["evaluation"]

    degraded_nrmse = float(macro["degraded"]["nrmse"])
    predicted_nrmse = float(macro["predicted"]["nrmse"])
    degraded_ssim = float(macro["degraded"]["ssim"])
    predicted_ssim = float(macro["predicted"]["ssim"])
    relative_nrmse = (degraded_nrmse - predicted_nrmse) / abs(degraded_nrmse)
    absolute_ssim = predicted_ssim - degraded_ssim
    fields_improving = sum(
        float(metrics["predicted"]["nrmse"])
        < float(metrics["degraded"]["nrmse"])
        for metrics in per_field.values()
    )
    fraction_correct_best = float(
        volume_conditioning["fraction_volumes_correct_best_nrmse"]
    )
    mean_margin = float(volume_conditioning["mean_margin_vs_best_wrong_nrmse"])
    correct_vs_wrong = float(
        conditioning["correct_vs_wrong_improvement"]["relative"]["nrmse"]
    )
    correct_vs_permuted = float(
        conditioning["correct_vs_permuted_improvement"]["relative"]["nrmse"]
    )
    outside_mask = float(macro["predicted"]["outside_mask_mean_abs"])

    restoration_gates = {
        "macro_relative_nrmse_improvement": relative_nrmse
        >= float(thresholds["min_macro_relative_nrmse_improvement"]),
        "macro_absolute_ssim_improvement": absolute_ssim
        >= float(thresholds["min_macro_absolute_ssim_improvement"]),
        "fields_with_nrmse_improvement": fields_improving
        >= int(thresholds["min_fields_with_nrmse_improvement"]),
        "macro_outside_mask_mean_abs": outside_mask
        <= float(thresholds["max_macro_outside_mask_mean_abs"]),
    }
    conditioning_gates = {
        "fraction_volumes_correct_best_nrmse": fraction_correct_best
        >= float(thresholds["min_fraction_volumes_correct_best_nrmse"]),
        "mean_margin_vs_best_wrong_nrmse": mean_margin
        >= float(thresholds["min_mean_margin_vs_best_wrong_nrmse"]),
        "relative_correct_vs_wrong_nrmse": correct_vs_wrong
        >= float(thresholds["min_relative_correct_vs_wrong_nrmse_improvement"]),
        "relative_correct_vs_permuted_nrmse": correct_vs_permuted
        >= float(thresholds["min_relative_correct_vs_permuted_nrmse_improvement"]),
    }
    restoration_status = "PASS" if all(restoration_gates.values()) else "FAIL"
    conditioning_status = "PASS" if all(conditioning_gates.values()) else "FAIL"
    scientific_status = (
        "PASS"
        if restoration_status == "PASS" and conditioning_status == "PASS"
        else "FAIL"
    )

    return {
        "evidence_source": "user_executed_private_colab",
        "evidence_scope": "sampled_slice_per_volume_exploratory",
        "complete_volume": False,
        "code_commit": code_commit,
        "config_name": CONFIG_NAME,
        "model_name": config["model"]["name"],
        "split_sha256": split_sha256,
        "split_evidence_role": "development_reuse_not_confirmatory",
        "seed": int(config["seed"]),
        "pipeline_version": int(state["pseudo_pair_pipeline_version"]),
        "endpoint": {
            "epoch": int(state["epoch"]),
            "global_step": int(state["global_step"]),
        },
        "runtime": dict(runtime),
        "engineering_gate": "PASS",
        "gates": {
            "restoration": {
                "status": restoration_status,
                "details": restoration_gates,
            },
            "conditioning": {
                "status": conditioning_status,
                "details": conditioning_gates,
            },
            "scientific": {
                "status": scientific_status,
                "rule": "restoration_and_conditioning",
            },
        },
        "counts": {
            "subjects": int(volume_eval["counts"]["subjects"]),
            "volumes": int(volume_eval["counts"]["volumes"]),
            "selected_slices": int(volume_eval["counts"]["selected_slices"]),
        },
        "per_target_field": {
            field: {
                "volumes": int(metrics["volumes"]),
                "selected_slices": int(metrics["selected_slices"]),
                "degraded": dict(metrics["degraded"]),
                "predicted": dict(metrics["predicted"]),
            }
            for field, metrics in sorted(per_field.items())
        },
        "metrics": {
            "degraded_macro_nrmse": degraded_nrmse,
            "predicted_macro_nrmse": predicted_nrmse,
            "degraded_macro_ssim": degraded_ssim,
            "predicted_macro_ssim": predicted_ssim,
            "fields_with_nrmse_improvement": fields_improving,
            "fraction_volumes_correct_best_nrmse": fraction_correct_best,
            "mean_margin_vs_best_wrong_nrmse": mean_margin,
            "relative_correct_vs_wrong_nrmse": correct_vs_wrong,
            "relative_correct_vs_permuted_nrmse": correct_vs_permuted,
            "macro_outside_mask_mean_abs": outside_mask,
        },
        "scaled_pilot": "BLOCKED_PENDING_REVIEW",
        "limitations": [
            "selected slices only",
            "observed development split",
            "not confirmatory evidence",
            "not complete-volume evidence",
            "not learned real 0.1T translation evidence",
        ],
    }


def _assert_sanitized_handoff(text: str) -> None:
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
    lowered = text.lower()
    if any(term in lowered for term in forbidden):
        raise RuntimeError("Sanitized handoff contains a private identity or artifact term.")
