"""Colab orchestration for the diagnostic-only Stage-1 reconstruction contract."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data.mrixfields_adapter import load_adapted_mrixfields_manifest
from fieldbridge.evaluation.stage1_diagnostics import (
    Stage1DiagnosticSpec,
    run_stage1_reconstruction_diagnostics,
)

DIAGNOSTIC_CONFIG_NAME = "stage1_vae_reconstruction_diagnostic_v1.yaml"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def run_stage1_diagnostic(
    *,
    repo_dir: Path,
    checkpoint_path: Path,
    patch_bank_dir: Path,
    official_manifest_path: Path,
    resolved_run_config_path: Path,
    output_dir: Path,
    code_commit: str,
    checkpoint_sweep_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Run diagnostics only and write one sanitized JSON handoff."""

    repo_dir = repo_dir.resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    patch_bank_dir = patch_bank_dir.expanduser().resolve()
    official_manifest_path = official_manifest_path.expanduser().resolve()
    resolved_run_config_path = resolved_run_config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    sweep_paths = tuple(path.expanduser().resolve() for path in checkpoint_sweep_paths)
    _validate_inputs(
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        patch_bank_dir=patch_bank_dir,
        official_manifest_path=official_manifest_path,
        resolved_run_config_path=resolved_run_config_path,
        output_dir=output_dir,
        code_commit=code_commit,
        checkpoint_sweep_paths=sweep_paths,
    )
    output_dir.mkdir(parents=True)

    diagnostic_config = load_yaml_config(
        repo_dir / "configs" / "experiment" / DIAGNOSTIC_CONFIG_NAME
    )
    diagnostic_spec = Stage1DiagnosticSpec.from_mapping(diagnostic_config)
    resolved_run_config = load_yaml_config(resolved_run_config_path)

    print("stage1 diagnostic: auditing and adapting official manifest", flush=True)
    adapted = load_adapted_mrixfields_manifest(
        official_manifest_path,
        strict_paths=True,
    )
    started = time.perf_counter()
    report = run_stage1_reconstruction_diagnostics(
        checkpoint_path=checkpoint_path,
        patch_bank_dir=patch_bank_dir,
        manifest=adapted.manifest,
        resolved_config=resolved_run_config,
        diagnostic_spec=diagnostic_spec,
        checkpoint_sweep_paths=sweep_paths,
        device=torch.device("cuda"),
        logger=lambda message: print(message, flush=True),
    )
    report["diagnostic_code_commit"] = code_commit
    report["runtime"] = {
        "device": "cuda",
        "wall_seconds": time.perf_counter() - started,
    }
    report["manifest"]["official_audit"] = {
        "ok": adapted.official_audit.ok,
        "total_records": adapted.official_audit.total_records,
        "counts_by_split": adapted.official_audit.counts_by_split,
        "counts_by_modality": adapted.official_audit.counts_by_modality,
        "counts_by_field": adapted.official_audit.counts_by_field,
        "counts_by_split_modality_field": (
            adapted.official_audit.counts_by_split_modality_field
        ),
        "error_count": len(adapted.official_audit.errors),
        "warning_count": len(adapted.official_audit.warnings),
    }
    report["manifest"]["adapted_volume_audit"] = {
        "ok": bool(adapted.volume_audit["ok"]),
        "record_count": int(adapted.volume_audit["record_count"]),
        "duplicate_case_id_count": len(adapted.volume_audit["duplicate_case_ids"]),
        "missing_path_count": int(adapted.volume_audit["missing_path_count"]),
        "domain_counts": dict(adapted.volume_audit["domain_counts"]),
        "split_counts": dict(adapted.volume_audit["split_counts"]),
    }
    handoff = json.dumps(report, indent=2, sort_keys=True)
    _assert_sanitized(handoff)
    (output_dir / "stage1_diagnostic_handoff.json").write_text(
        handoff,
        encoding="utf-8",
    )
    print(handoff)
    return report


def _validate_inputs(
    *,
    repo_dir: Path,
    checkpoint_path: Path,
    patch_bank_dir: Path,
    official_manifest_path: Path,
    resolved_run_config_path: Path,
    output_dir: Path,
    code_commit: str,
    checkpoint_sweep_paths: Sequence[Path],
) -> None:
    if _SHA_PATTERN.fullmatch(code_commit) is None:
        raise ValueError("code_commit must be an exact 40-character git SHA.")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
    ).strip()
    if actual_commit != code_commit:
        raise RuntimeError(f"Checked out {actual_commit}, expected {code_commit}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("External Stage-1 checkpoint does not exist.")
    if not patch_bank_dir.is_dir():
        raise FileNotFoundError("External Stage-1 patch-bank directory does not exist.")
    if not official_manifest_path.is_file():
        raise FileNotFoundError("External official JSONL manifest does not exist.")
    if not resolved_run_config_path.is_file():
        raise FileNotFoundError("External resolved Stage-1 run config does not exist.")
    if output_dir.exists():
        raise FileExistsError("Use a new output directory for this diagnostic run.")
    if output_dir == repo_dir or repo_dir in output_dir.parents:
        raise ValueError("Diagnostic outputs must remain outside the Git checkout.")
    if not torch.cuda.is_available():
        raise RuntimeError("Full-volume Stage-1 diagnostics require a CUDA runtime.")
    for path in checkpoint_sweep_paths:
        if not path.is_file():
            raise FileNotFoundError("An optional checkpoint-sweep input does not exist.")


def _assert_sanitized(text: str) -> None:
    lowered = text.lower()
    forbidden = (
        '"subject_id":',
        '"sample_id":',
        '"case_id":',
        '"image_path":',
        '"raw_uri":',
        ".nii",
        ".pt",
        "/content/drive",
        "\\users\\",
    )
    if any(value in lowered for value in forbidden):
        raise RuntimeError("Diagnostic handoff contains a private identity or artifact path.")
