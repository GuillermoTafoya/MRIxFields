"""Private Colab runner for the preregistered prospective paired LOSO experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fieldbridge.config import load_yaml_config
from fieldbridge.data.domains import Domain
from fieldbridge.data.paired_loso import (
    RealPairedSliceDataset,
    build_loso_folds,
    fit_train_only_affine_calibrations,
    prepare_preprocessed_tensor_cache,
)
from fieldbridge.data.preprocessing import (
    SlicePreprocessingSpec,
    from_model_range,
    preprocess_volume_slice,
    selected_slice_indices,
    to_model_range,
)
from fieldbridge.data.pseudo_pairs import collate_pseudo_pair_slices
from fieldbridge.evaluation.paired_loso import (
    NEURAL_ARMS,
    aggregate_selected_rows,
    evaluate_selected_case,
    evaluate_viability,
    reconstruct_complete_candidate,
    sanitized_loso_handoff,
)
from fieldbridge.evaluation.prospective_paired import (
    ALL_FIELDS,
    CASE_IDS,
    CONTRAST,
    SELECTED_SLICE_INDICES,
    SOURCE_FIELD,
    TARGET_FIELDS,
    fixed_edge_map,
    load_nifti_acquisition,
    select_required_acquisitions,
    validate_checkpoint_contract,
    validate_paired_geometry,
)
from fieldbridge.official.data_manifest import read_manifest_jsonl
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.paired_loso import (
    SanitizedRunProgress,
    initialize_residual_arm,
    resolve_arm_recovery,
    train_fixed_endpoint,
    validate_endpoint_checkpoint,
    validate_one_batch_training_compatibility,
)
from fieldbridge.training.pseudo_pair_epochs import PseudoPairEpochConfig

AUDIT_COMMIT = "a7ac99f40dcaea4811452172d363347997c504e1"
SYNTHETIC_TRAINING_COMMIT = "e1e526ea5fa0a58f5682823f85a3957d5cc8647c"
HISTORICAL_CONFIG = "configs/experiment/pseudo_pair_t2flair_residual_probe_10epoch.yaml"
CONFIG_NAME = "prospective_paired_loso_residual_v1.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--synthetic-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment") / CONFIG_NAME,
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    payload = run_experiment(
        manifest_path=args.manifest,
        synthetic_checkpoint_path=args.synthetic_checkpoint,
        output_dir=args.output_dir,
        scratch_dir=args.scratch_dir,
        config_path=args.config,
        preflight_only=args.preflight,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_experiment(
    *,
    manifest_path: Path,
    synthetic_checkpoint_path: Path,
    output_dir: Path,
    scratch_dir: Path,
    config_path: Path,
    preflight_only: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    _log_phase("configuration", "start")
    config = load_yaml_config(config_path)
    preprocessing, folds = _validate_config(config)
    experiment_commit = _git_commit()
    config_fingerprint = _canonical_sha(config)
    experiment_fingerprint = _canonical_sha(
        {
            "config": config,
            "experiment_commit": experiment_commit,
            "synthetic_training_commit": SYNTHETIC_TRAINING_COMMIT,
        }
    )
    historical = _load_historical_yaml()
    synthetic_state = load_checkpoint(synthetic_checkpoint_path, map_location="cpu")
    validate_checkpoint_contract(
        synthetic_state,
        _mapping(config, "synthetic_checkpoint_contract"),
        historical_training_config=historical,
    )
    _log_phase("manifest_selection", "start")
    records = read_manifest_jsonl(manifest_path)
    selected = select_required_acquisitions(
        records,
        split_name=str(_mapping(config, "experiment")["manifest_split"]),
    )
    manifest_fingerprint = _file_sha(manifest_path)
    _log_phase("manifest_selection", "complete", acquisitions=len(CASE_IDS) * len(ALL_FIELDS))
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    volumes, input_fingerprints = _cache_and_load_volumes(
        selected, scratch_dir=scratch_dir
    )
    input_identity_fingerprint = _input_identity_fingerprint(input_fingerprints)
    train_indices = selected_slice_indices(preprocessing)
    alignment_fingerprint = _canonical_sha(
        {
            "manifest_sha256": manifest_fingerprint,
            "input_identity_sha256": input_identity_fingerprint,
            "preprocessing": preprocessing.to_dict(),
            "selected_slice_indices": list(SELECTED_SLICE_INDICES),
            "experiment_commit": experiment_commit,
            "config_sha256": config_fingerprint,
        }
    )
    preflight = _preflight_payload(
        config=config,
        folds=folds,
        train_indices=train_indices,
        config_fingerprint=config_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        input_identity_fingerprint=input_identity_fingerprint,
        alignment_fingerprint=alignment_fingerprint,
    )
    if preflight_only:
        _write_alignment_preflight(
            volumes,
            preprocessing=preprocessing,
            output_dir=output_dir / "alignment_preflight",
            contract_fingerprint=alignment_fingerprint,
        )
        (output_dir / "preflight_sanitized.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return preflight
    _validate_preflight_artifacts(
        output_dir=output_dir,
        expected_preflight=preflight,
        alignment_fingerprint=alignment_fingerprint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _mapping(config, "runtime").get("require_cuda") and device.type != "cuda":
        raise RuntimeError("The preregistered LOSO run requires a CUDA runtime.")
    train_cfg = PseudoPairEpochConfig.from_mapping(config)
    amp_enabled = bool(train_cfg.amp and device.type == "cuda")
    cuda_device = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    )
    _log_phase(
        "resolved_device",
        "complete",
        device=device.type,
        cuda_device=cuda_device,
        amp_enabled=str(amp_enabled).lower(),
    )
    recovery_plans = {}
    progress: SanitizedRunProgress | None = None
    if not dry_run:
        expected_steps = int(_mapping(config, "training")["steps_per_epoch"])
        expected_global = int(_mapping(config, "training")["endpoint_global_step"])
        for fold in folds:
            for arm in NEURAL_ARMS:
                arm_dir = output_dir / "checkpoints" / fold.fold_slot / arm
                recovery_plans[(fold.fold_slot, arm)] = resolve_arm_recovery(
                    arm_dir,
                    resume=resume,
                    cfg=train_cfg,
                    fold_slot=fold.fold_slot,
                    arm=arm,  # type: ignore[arg-type]
                    experiment_fingerprint=experiment_fingerprint,
                    expected_steps_per_epoch=expected_steps,
                    expected_global_step=expected_global,
                )
        progress = SanitizedRunProgress(
            output_dir / "run_progress_sanitized.json",
            fold_case_slots={
                fold.fold_slot: f"case_{index:02d}"
                for index, fold in enumerate(folds, start=1)
            },
            resume=resume,
        )
        for (fold_slot, arm), plan in recovery_plans.items():
            progress.validate_recovery(fold_slot, arm, plan.action)
    telemetry = NvidiaSmiTelemetry(device=device, amp_enabled=amp_enabled)
    telemetry.start()
    _log_phase("cache_preparation", "start")
    try:
        preprocessed_cache = prepare_preprocessed_tensor_cache(
            volumes,
            cache_root=scratch_dir,
            case_ids=CASE_IDS,
            preprocessing=preprocessing,
            slice_indices=train_indices,
            manifest_fingerprint=manifest_fingerprint,
            input_fingerprints=input_fingerprints,
            code_fingerprint=experiment_commit,
            config_fingerprint=config_fingerprint,
            progress_callback=_print_cache_progress,
        )
    except Exception:
        telemetry.stop()
        raise
    telemetry.set_cache_preparation(preprocessed_cache.stats)
    _log_phase(
        "cache_preparation",
        "complete",
        cache_hit=str(preprocessed_cache.stats.cache_hit).lower(),
        tensors=preprocessed_cache.stats.tensors,
        tensors_per_second=f"{preprocessed_cache.stats.tensors_per_second:.3f}",
    )
    if dry_run:
        try:
            dry_validation = _run_training_dry_run(
                fold=folds[0],
                case_slot="case_01",
                volumes=volumes,
                preprocessed_cache=preprocessed_cache,
                preprocessing=preprocessing,
                train_indices=train_indices,
                train_cfg=train_cfg,
                model_config=_mapping(config, "model"),
                synthetic_state=synthetic_state,
                num_workers=int(_mapping(config, "training")["num_workers"]),
                device=device,
            )
            telemetry.add_batch_preparation_rate(
                dry_validation["batch_preparation_samples_per_second"]
            )
        finally:
            telemetry.stop()
        return {
            **preflight,
            "dry_run": True,
            "training_started": False,
            "dry_run_validation": dry_validation,
            "runtime_telemetry": telemetry.summary(),
        }
    selected_rows: list[dict[str, Any]] = []
    complete_rows: list[dict[str, Any]] = []
    try:
        for fold_index, fold in enumerate(folds, start=1):
            case_slot = f"case_{fold_index:02d}"
            _log_phase("fold", "start", fold=fold.fold_slot, case=case_slot)
            affine_started = time.perf_counter()
            _log_phase("affine_fit", "start", fold=fold.fold_slot, case=case_slot)
            calibrations = fit_train_only_affine_calibrations(
                volumes,
                train_case_ids=fold.train_case_ids,
                preprocessing=preprocessing,
                slice_indices=train_indices,
                preprocessed_cache=preprocessed_cache,
            )
            _log_phase(
                "affine_fit",
                "complete",
                fold=fold.fold_slot,
                case=case_slot,
                elapsed_seconds=f"{time.perf_counter() - affine_started:.3f}",
            )
            dataset = RealPairedSliceDataset(
                volumes,
                case_ids=fold.train_case_ids,
                preprocessing=preprocessing,
                slice_indices=train_indices,
                preprocessed_cache=preprocessed_cache,
            )
            models: dict[str, torch.nn.Module] = {}
            for arm in NEURAL_ARMS:
                torch.manual_seed(train_cfg.seed)
                model = initialize_residual_arm(
                    _mapping(config, "model"),
                    arm=arm,  # type: ignore[arg-type]
                    synthetic_checkpoint=synthetic_state
                    if arm == "synthetic_initialization"
                    else None,
                )
                _validate_initialization(model, arm, preprocessing, device)
                assert progress is not None
                loader = _build_train_loader(
                    dataset,
                    train_cfg=train_cfg,
                    num_workers=int(_mapping(config, "training")["num_workers"]),
                )
                arm_dir = output_dir / "checkpoints" / fold.fold_slot / arm
                plan = recovery_plans[(fold.fold_slot, arm)]
                elapsed_offset = progress.elapsed_seconds(fold.fold_slot, arm)
                mode = {
                    "fresh": "started fresh",
                    "resume": "resumed",
                    "endpoint": "loaded completed endpoint",
                }[plan.action]
                _log_phase(
                    "arm",
                    mode.replace(" ", "_"),
                    fold=fold.fold_slot,
                    case=case_slot,
                    arm=arm,
                )
                if plan.action == "endpoint":
                    endpoint_state = plan.checkpoint_state
                    assert endpoint_state is not None
                    progress.update(
                        fold_slot=fold.fold_slot,
                        case_slot=case_slot,
                        arm=arm,  # type: ignore[arg-type]
                        state="endpoint_complete",
                        epoch=plan.epoch,
                        global_step=plan.global_step,
                        elapsed_seconds=elapsed_offset,
                    )
                else:
                    progress.update(
                        fold_slot=fold.fold_slot,
                        case_slot=case_slot,
                        arm=arm,  # type: ignore[arg-type]
                        state="running",
                        epoch=plan.epoch,
                        global_step=plan.global_step,
                        elapsed_seconds=elapsed_offset,
                    )

                    def record_epoch(epoch, global_step, elapsed, *, arm=arm):
                        assert progress is not None
                        progress.update(
                            fold_slot=fold.fold_slot,
                            case_slot=case_slot,
                            arm=arm,  # type: ignore[arg-type]
                            state="running",
                            epoch=epoch,
                            global_step=global_step,
                            elapsed_seconds=elapsed_offset + elapsed,
                        )

                    result = train_fixed_endpoint(
                        train_cfg,
                        model=model,
                        train_loader=loader,
                        checkpoint_dir=arm_dir,
                        fold_slot=fold.fold_slot,
                        arm=arm,  # type: ignore[arg-type]
                        experiment_fingerprint=experiment_fingerprint,
                        expected_steps_per_epoch=expected_steps,
                        expected_global_step=expected_global,
                        resume_from=plan.resume_path,
                        case_slot=case_slot,
                        epoch_progress=record_epoch,
                    )
                    endpoint_state = load_checkpoint(
                        result.endpoint_checkpoint, map_location=device
                    )
                    validate_endpoint_checkpoint(
                        endpoint_state,
                        cfg=train_cfg,
                        fold_slot=fold.fold_slot,
                        arm=arm,  # type: ignore[arg-type]
                        experiment_fingerprint=experiment_fingerprint,
                        expected_steps_per_epoch=expected_steps,
                        expected_global_step=expected_global,
                    )
                    progress.update(
                        fold_slot=fold.fold_slot,
                        case_slot=case_slot,
                        arm=arm,  # type: ignore[arg-type]
                        state="endpoint_complete",
                        epoch=train_cfg.epochs,
                        global_step=expected_global,
                        elapsed_seconds=progress.elapsed_seconds(fold.fold_slot, arm),
                    )
                telemetry.add_training_history(arm_dir / "history.jsonl")
                model.load_state_dict(endpoint_state["model"], strict=True)
                models[arm] = model.to(device).eval()

            held_out = fold.held_out_case_id
            assert progress is not None
            evaluation_started = time.perf_counter()
            training_elapsed = {
                arm: progress.elapsed_seconds(fold.fold_slot, arm) for arm in NEURAL_ARMS
            }
            _transition_evaluation_progress(
                progress,
                fold_slot=fold.fold_slot,
                case_slot=case_slot,
                state="evaluating",
                epoch=train_cfg.epochs,
                global_step=expected_global,
                elapsed_by_arm=training_elapsed,
                phase="selected_slices",
            )
            selected_rows.extend(
                evaluate_selected_case(
                    fold_slot=fold.fold_slot,
                    case_slot=case_slot,
                    source_volume=volumes[held_out][SOURCE_FIELD],
                    target_volumes=volumes[held_out],
                    calibrations=calibrations,
                    models=models,
                    preprocessing=preprocessing,
                    slice_indices=SELECTED_SLICE_INDICES,
                    device=device,
                    progress_callback=lambda completed, total: _print_evaluation_progress(
                        fold.fold_slot,
                        case_slot,
                        "selected_slices",
                        completed,
                        total,
                    ),
                )
            )
            current_elapsed = {
                arm: training_elapsed[arm] + time.perf_counter() - evaluation_started
                for arm in NEURAL_ARMS
            }
            _transition_evaluation_progress(
                progress,
                fold_slot=fold.fold_slot,
                case_slot=case_slot,
                state="evaluating",
                epoch=train_cfg.epochs,
                global_step=expected_global,
                elapsed_by_arm=current_elapsed,
                phase="complete_volume",
            )
            complete_rows.extend(
                _evaluate_complete_fold(
                    fold_slot=fold.fold_slot,
                    case_slot=case_slot,
                    volumes=volumes[held_out],
                    calibrations=calibrations,
                    models=models,
                    preprocessing=preprocessing,
                    device=device,
                    progress_callback=lambda completed, total: _print_evaluation_progress(
                        fold.fold_slot,
                        case_slot,
                        "complete_volume",
                        completed,
                        total,
                    ),
                )
            )
            final_elapsed = {
                arm: training_elapsed[arm] + time.perf_counter() - evaluation_started
                for arm in NEURAL_ARMS
            }
            _transition_evaluation_progress(
                progress,
                fold_slot=fold.fold_slot,
                case_slot=case_slot,
                state="complete",
                epoch=train_cfg.epochs,
                global_step=expected_global,
                elapsed_by_arm=final_elapsed,
                phase="fold_complete",
            )
            _log_phase("fold", "complete", fold=fold.fold_slot, case=case_slot)
    finally:
        telemetry.stop()

    aggregate = aggregate_selected_rows(selected_rows)
    viability = evaluate_viability(aggregate, _mapping(config, "viability_rules"))
    complete = _aggregate_complete_rows(complete_rows)
    handoff = sanitized_loso_handoff(
        audit_commit=AUDIT_COMMIT,
        training_checkpoint_commit=SYNTHETIC_TRAINING_COMMIT,
        experiment_commit=experiment_commit,
        aggregate=aggregate,
        full_volume=complete,
        viability=viability,
        provenance={
            "config_sha256": config_fingerprint,
            "experiment_sha256": experiment_fingerprint,
            "manifest_sha256": manifest_fingerprint,
            "input_identity_sha256": input_identity_fingerprint,
            "preprocessed_cache_sha256": preprocessed_cache.fingerprint,
            "seed": int(config["seed"]),
            "paired_loso_pipeline_version": int(
                _mapping(config, "paired_checkpoint_contract")["pipeline_version"]
            ),
            "gpu_telemetry": telemetry.summary(),
        },
    )
    (output_dir / "sanitized_handoff.json").write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return handoff


def _validate_config(config: Mapping[str, Any]):
    experiment = _mapping(config, "experiment")
    if (
        experiment.get("track") != "A"
        or experiment.get("evidence_scope")
        != "prospective_paired_loso_development"
    ):
        raise ValueError("Unexpected LOSO Track-A evidence contract.")
    if tuple(str(value) for value in experiment["case_ids"]) != CASE_IDS:
        raise ValueError("LOSO case set changed.")
    if tuple(float(value) for value in experiment["target_fields"]) != TARGET_FIELDS:
        raise ValueError("LOSO target fields changed.")
    if float(experiment["source_field"]) != SOURCE_FIELD:
        raise ValueError("LOSO source field changed.")
    if experiment.get("synthetic_examples_during_paired_training") is not False:
        raise ValueError("Synthetic examples must not enter paired fine-tuning.")
    folds = build_loso_folds(experiment["case_ids"])
    declared = experiment["folds"]
    for fold, declaration in zip(folds, declared, strict=True):
        if (
            int(declaration["fold"]) != fold.fold
            or tuple(declaration["train_cases"]) != fold.train_case_ids
            or declaration["held_out_case"] != fold.held_out_case_id
        ):
            raise ValueError(f"Declared {fold.fold_slot} changed.")
    preprocessing = SlicePreprocessingSpec.from_mapping(config.get("preprocessing"))
    if preprocessing.slices_per_volume is not None or preprocessing.resize_mode != "fit_pad":
        raise ValueError("Training must use every predeclared fit-pad brain-support slice.")
    if selected_slice_indices(preprocessing) != tuple(range(72, 292)):
        raise ValueError("Brain-support training slice coverage changed.")
    if tuple(_mapping(config, "evaluation")["selected_slice_indices"]) != SELECTED_SLICE_INDICES:
        raise ValueError("Frozen eight-slice evaluation contract changed.")
    training = _mapping(config, "training")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ValueError("LOSO neural arms require the preregistered AdamW optimizer.")
    scheduler = training.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("name") != "none":
        raise ValueError("LOSO neural arms require scheduler.name=none.")
    if (
        int(training["train_samples_per_fold"]) != 1760
        or int(training["steps_per_epoch"]) != 220
        or int(training["endpoint_global_step"]) != 2200
    ):
        raise ValueError("Fixed training endpoint changed.")
    if training.get("validation_loader") is not None:
        raise ValueError("Held-out validation/checkpoint selection is forbidden.")
    return preprocessing, folds


def _cache_and_load_volumes(selected, *, scratch_dir: Path):
    volumes: dict[str, dict[float, torch.Tensor]] = {}
    input_fingerprints: dict[str, dict[float, str]] = {}
    total = len(CASE_IDS) * len(ALL_FIELDS)
    completed = 0
    for case_index, case_id in enumerate(CASE_IDS, start=1):
        loaded = {}
        volumes[case_id] = {}
        input_fingerprints[case_id] = {}
        for field in ALL_FIELDS:
            case_slot = f"case_{case_index:02d}"
            field_label = f"{field:g}T"
            _log_phase(
                "staging",
                "start",
                case=case_slot,
                field=field_label,
                progress=f"{completed + 1}/{total}",
            )
            source_path = Path(selected[case_id][field].raw_uri)
            filename_field = str(field).replace(".", "p")
            destination = (
                scratch_dir / f"case_{case_index:02d}_field_{filename_field}T.nii.gz"
            )
            shutil.copy2(source_path, destination)
            actual_sha = _file_sha(destination)
            declared_sha = selected[case_id][field].sha256
            if declared_sha is not None and actual_sha != str(declared_sha).lower():
                raise ValueError("Staged acquisition checksum differs from the manifest.")
            input_fingerprints[case_id][field] = actual_sha
            _log_phase(
                "staging",
                "complete",
                case=case_slot,
                field=field_label,
                progress=f"{completed + 1}/{total}",
            )
            _log_phase(
                "loading",
                "start",
                case=case_slot,
                field=field_label,
                progress=f"{completed + 1}/{total}",
            )
            acquisition = load_nifti_acquisition(destination)
            loaded[field] = acquisition
            volumes[case_id][field] = acquisition.volume
            completed += 1
            _log_phase(
                "loading",
                "complete",
                case=case_slot,
                field=field_label,
                progress=f"{completed}/{total}",
            )
        validate_paired_geometry(loaded)
    return volumes, input_fingerprints


def _write_alignment_preflight(
    volumes,
    *,
    preprocessing,
    output_dir: Path,
    contract_fingerprint: str,
) -> None:
    _log_phase(
        "alignment",
        "start",
        panels=len(CASE_IDS) * len(TARGET_FIELDS) * len(SELECTED_SLICE_INDICES),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(CASE_IDS) * len(TARGET_FIELDS) * len(SELECTED_SLICE_INDICES)
    completed = 0
    for case_index, case_id in enumerate(CASE_IDS, start=1):
        for field in TARGET_FIELDS:
            for slice_index in SELECTED_SLICE_INDICES:
                source, source_geometry = preprocess_volume_slice(
                    volumes[case_id][SOURCE_FIELD],
                    slice_index,
                    preprocessing,
                    apply_model_range=False,
                )
                target, target_geometry = preprocess_volume_slice(
                    volumes[case_id][field], slice_index, preprocessing, apply_model_range=False
                )
                if source_geometry != target_geometry:
                    raise ValueError("Alignment preflight found unequal fit-pad geometry.")
                source_edge = fixed_edge_map(source.unsqueeze(0))[0, 0].numpy()
                target_edge = fixed_edge_map(target.unsqueeze(0))[0, 0].numpy()
                overlay = torch.stack(
                    [
                        torch.from_numpy(source_edge),
                        torch.from_numpy(target_edge),
                        torch.from_numpy(target_edge),
                    ]
                ).permute(1, 2, 0).numpy()
                overlay /= max(float(overlay.max()), 1e-8)
                figure, axis = plt.subplots(figsize=(4, 4))
                axis.imshow(overlay, vmin=0.0, vmax=1.0)
                axis.axis("off")
                axis.set_title("source red / target cyan")
                stem = f"case_{case_index:02d}_{str(field).replace('.', 'p')}T_{slice_index:03d}"
                figure.savefig(output_dir / f"{stem}.png", dpi=100, bbox_inches="tight")
                plt.close(figure)
                completed += 1
                if completed % len(SELECTED_SLICE_INDICES) == 0 or completed == total:
                    _log_phase(
                        "alignment",
                        "running",
                        case=f"case_{case_index:02d}",
                        field=f"{field:g}T",
                        progress=f"{completed}/{total}",
                    )
    (output_dir / "contract_sanitized.json").write_text(
        json.dumps(
            {"alignment_contract_sha256": contract_fingerprint},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _log_phase("alignment", "complete", panels=completed)


def _validate_preflight_artifacts(
    *,
    output_dir: Path,
    expected_preflight: Mapping[str, Any],
    alignment_fingerprint: str,
) -> None:
    _log_phase("alignment", "validate_start")
    preflight_path = output_dir / "preflight_sanitized.json"
    contract_path = output_dir / "alignment_preflight" / "contract_sanitized.json"
    try:
        observed_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        observed_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Preflight artifacts are missing or invalid; rerun --preflight first."
        ) from exc
    if observed_preflight != dict(expected_preflight):
        raise ValueError("Preflight contract changed; rerun --preflight first.")
    if observed_contract != {"alignment_contract_sha256": alignment_fingerprint}:
        raise ValueError("Alignment contract changed; rerun --preflight first.")
    expected_panels = {
        f"case_{case_offset:02d}_{str(field).replace('.', 'p')}T_{slice_index:03d}.png"
        for case_offset in range(1, len(CASE_IDS) + 1)
        for field in TARGET_FIELDS
        for slice_index in SELECTED_SLICE_INDICES
    }
    observed_panels = {
        path.name
        for path in (output_dir / "alignment_preflight").glob("*.png")
    }
    if observed_panels != expected_panels:
        raise ValueError("Alignment panel coverage changed; rerun --preflight first.")
    _log_phase("alignment", "validated", panels=len(observed_panels))


def _validate_initialization(model, arm, preprocessing, device) -> None:
    source = torch.linspace(-1.0, 1.0, 128 * 160).reshape(1, 1, 128, 160).to(device)
    model = model.to(device).eval()
    with torch.inference_mode():
        output = model(source, Domain(SOURCE_FIELD, CONTRAST), Domain(7.0, CONTRAST))
    if arm == "identity_initialization" and not torch.equal(output, source):
        raise RuntimeError("Identity-initialized arm is not exact identity at step zero.")
    if output.shape != source.shape or preprocessing.model_range != "minus_one_one":
        raise RuntimeError("Initialization model/preprocessing contract changed.")


def _build_train_loader(dataset, *, train_cfg, num_workers):
    generator = torch.Generator().manual_seed(train_cfg.seed)
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(num_workers),
        collate_fn=collate_pseudo_pair_slices,
    )


def _run_training_dry_run(
    *,
    fold,
    case_slot,
    volumes,
    preprocessed_cache,
    preprocessing,
    train_indices,
    train_cfg,
    model_config,
    synthetic_state,
    num_workers,
    device,
):
    _log_phase("dry_run", "start", fold=fold.fold_slot, case=case_slot)
    dataset = RealPairedSliceDataset(
        volumes,
        case_ids=fold.train_case_ids,
        preprocessing=preprocessing,
        slice_indices=train_indices,
        preprocessed_cache=preprocessed_cache,
    )
    loader = _build_train_loader(
        dataset,
        train_cfg=train_cfg,
        num_workers=num_workers,
    )
    batch_started = time.perf_counter()
    batch = next(iter(loader))
    batch_elapsed = max(time.perf_counter() - batch_started, 1e-12)
    batch_throughput = int(batch.x_low.shape[0]) / batch_elapsed
    _log_phase(
        "dry_run_batch",
        "complete",
        fold=fold.fold_slot,
        case=case_slot,
        batch_size=int(batch.x_low.shape[0]),
        samples_per_second=f"{batch_throughput:.3f}",
    )
    arm_results = {}
    for arm in NEURAL_ARMS:
        _log_phase(
            "dry_run_arm",
            "start",
            fold=fold.fold_slot,
            case=case_slot,
            arm=arm,
        )
        torch.manual_seed(train_cfg.seed)
        model = initialize_residual_arm(
            model_config,
            arm=arm,
            synthetic_checkpoint=synthetic_state
            if arm == "synthetic_initialization"
            else None,
        )
        _validate_initialization(model, arm, preprocessing, device)
        arm_results[arm] = validate_one_batch_training_compatibility(
            train_cfg,
            model=model,
            batch=batch,
            device=device,
        )
        _log_phase(
            "dry_run_arm",
            "complete",
            fold=fold.fold_slot,
            case=case_slot,
            arm=arm,
            cuda="true",
            amp_enabled=str(bool(train_cfg.amp)).lower(),
            forward="true",
            loss="true",
            backward="true",
            optimizer_step="true",
        )
    _log_phase("dry_run", "complete", fold=fold.fold_slot, case=case_slot)
    return {
        "fold_slot": fold.fold_slot,
        "case_slot": case_slot,
        "affine_fit_executed": False,
        "real_batch": True,
        "batch_preparation_samples_per_second": batch_throughput,
        "arms": arm_results,
    }


def _print_cache_progress(case_slot, field, completed, total, tensors_per_second):
    _log_phase(
        "cache_preparation",
        "running",
        case=case_slot,
        field=f"{field:g}T",
        progress=f"{completed}/{total}",
        tensors_per_second=f"{tensors_per_second:.3f}",
    )


def _log_phase(phase: str, state: str, **values: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in values.items())
    suffix = f" {details}" if details else ""
    print(f"paired_loso phase={phase} state={state}{suffix}", flush=True)


def _input_identity_fingerprint(
    input_fingerprints: Mapping[str, Mapping[float, str]],
) -> str:
    anonymous = [
        {
            "case_slot": f"case_{case_offset:02d}",
            "fields": {
                f"{field:g}T": input_fingerprints[case_id][field]
                for field in ALL_FIELDS
            },
        }
        for case_offset, case_id in enumerate(CASE_IDS, start=1)
    ]
    return _canonical_sha(anonymous)


def _transition_evaluation_progress(
    progress,
    *,
    fold_slot,
    case_slot,
    state,
    epoch,
    global_step,
    elapsed_by_arm,
    phase,
):
    for arm in NEURAL_ARMS:
        progress.update(
            fold_slot=fold_slot,
            case_slot=case_slot,
            arm=arm,
            state=state,
            epoch=epoch,
            global_step=global_step,
            elapsed_seconds=elapsed_by_arm[arm],
        )
        _log_phase(
            "evaluation",
            state,
            fold=fold_slot,
            case=case_slot,
            arm=arm,
            evaluation_phase=phase,
        )


def _print_evaluation_progress(fold_slot, case_slot, phase, completed, total):
    _log_phase(
        "evaluation",
        "running",
        fold=fold_slot,
        case=case_slot,
        arms="identity_initialization,synthetic_initialization",
        evaluation_phase=phase,
        progress=f"{completed}/{total}",
    )


def _evaluate_complete_fold(
    *,
    fold_slot,
    case_slot,
    volumes,
    calibrations,
    models,
    preprocessing,
    device,
    progress_callback=None,
):
    rows = []
    total = len(TARGET_FIELDS) * (2 + len(NEURAL_ARMS))
    completed = 0
    source_volume = volumes[SOURCE_FIELD]
    for field in TARGET_FIELDS:
        target_volume = volumes[field]
        candidates = {
            "source": lambda source: source,
            "affine": lambda source, field=field: calibrations[field].apply(source),
        }
        for arm in NEURAL_ARMS:
            model = models[arm]

            def predict(source, *, model=model, field=field):
                source_device = source.to(device)
                output = model(
                    to_model_range(source_device, preprocessing.model_range),
                    Domain(SOURCE_FIELD, CONTRAST),
                    Domain(field, CONTRAST),
                )
                return from_model_range(output, preprocessing.model_range).cpu()

            candidates[arm] = predict
        for arm, candidate in candidates.items():
            result = reconstruct_complete_candidate(
                source_volume=source_volume,
                target_volume=target_volume,
                preprocessing=preprocessing,
                candidate=candidate,
            )
            rows.append(
                {
                    "fold_slot": fold_slot,
                    "case_slot": case_slot,
                    "target_field": f"{field:g}T",
                    "arm": arm,
                    **result,
                }
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
    return rows


def _aggregate_complete_rows(rows):
    expected = 3 * 4 * 4
    if len(rows) != expected or not all(row["complete_volume"] for row in rows):
        raise ValueError("Complete-volume evidence is incomplete.")
    grids = ("model_grid", "reconstructed_native_grid")
    output = {"complete_volume": True, "case_field_arm_units": len(rows), "grids": {}}
    for grid in grids:
        per_field = {}
        for field in sorted({row["target_field"] for row in rows}):
            field_rows = [row for row in rows if row["target_field"] == field]
            per_field[field] = {
                arm: _mean_metrics([row[grid] for row in field_rows if row["arm"] == arm])
                for arm in ("source", "affine", *NEURAL_ARMS)
            }
        output["grids"][grid] = {
            "per_target_field": per_field,
            "macro": {
                arm: _mean_metrics([per_field[field][arm] for field in per_field])
                for arm in ("source", "affine", *NEURAL_ARMS)
            },
        }
    source_rows = [row for row in rows if row["arm"] == "source"]
    baseline_names = (
        "raw_native_source_baseline",
        "roundtrip_native_source_baseline",
    )
    per_field_baselines = {}
    for field in sorted({row["target_field"] for row in source_rows}):
        field_rows = [row for row in source_rows if row["target_field"] == field]
        per_field_baselines[field] = {
            name: _mean_metrics([row[name] for row in field_rows])
            for name in baseline_names
        }
    output["native_source_baselines"] = {
        "per_target_field": per_field_baselines,
        "macro": {
            name: _mean_metrics(
                [per_field_baselines[field][name] for field in per_field_baselines]
            )
            for name in baseline_names
        },
    }
    return output


def _preflight_payload(
    *,
    config,
    folds,
    train_indices,
    config_fingerprint,
    experiment_fingerprint,
    manifest_fingerprint,
    input_identity_fingerprint,
    alignment_fingerprint,
):
    return {
        "ok": True,
        "evidence_scope": "prospective_paired_loso_development",
        "folds": [
            {
                "fold_slot": fold.fold_slot,
                "training_cases": 2,
                "held_out_cases": 1,
                "subject_disjoint": True,
            }
            for fold in folds
        ],
        "training_slices_per_volume": len(train_indices),
        "selected_evaluation_slices": len(SELECTED_SLICE_INDICES),
        "full_volume_coverage": "every_z_slice",
        "fixed_endpoint": {
            "epochs": int(_mapping(config, "training")["epochs"]),
            "steps_per_epoch": int(_mapping(config, "training")["steps_per_epoch"]),
            "global_step": int(_mapping(config, "training")["endpoint_global_step"]),
        },
        "config_sha256": config_fingerprint,
        "experiment_sha256": experiment_fingerprint,
        "manifest_sha256": manifest_fingerprint,
        "input_identity_sha256": input_identity_fingerprint,
        "alignment_contract_sha256": alignment_fingerprint,
        "held_out_checkpoint_selection": False,
        "synthetic_examples_in_paired_training": False,
    }


class NvidiaSmiTelemetry:
    def __init__(
        self,
        *,
        device: torch.device,
        amp_enabled: bool,
        period_seconds: float = 5.0,
    ) -> None:
        self.device = device
        self.amp_enabled = bool(amp_enabled)
        self.period_seconds = period_seconds
        self.rows: list[tuple[float, float, float, float]] = []
        self.batch_preparation_rates: list[float] = []
        self.training_step_rates: list[float] = []
        self.cache_preparation: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.period_seconds + 1.0)

    def _sample(self) -> None:
        while not self._stop.is_set():
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if process.returncode == 0 and process.stdout.strip():
                values = process.stdout.splitlines()[0].split(",")
                utilization, memory, power = (
                    float(value.strip()) for value in values
                )
                elapsed = time.perf_counter() - self._started_at
                self.rows.append((elapsed, utilization, memory, power))
                print(
                    "paired_loso phase=gpu_telemetry state=sample "
                    f"elapsed_seconds={elapsed:.1f} "
                    f"utilization_percent={utilization:.1f} "
                    f"memory_used_mib={memory:.1f}",
                    flush=True,
                )
            self._stop.wait(self.period_seconds)

    def set_cache_preparation(self, stats) -> None:
        self.cache_preparation = {
            "cache_hit": bool(stats.cache_hit),
            "tensors": int(stats.tensors),
            "elapsed_seconds": float(stats.elapsed_seconds),
            "tensors_per_second": float(stats.tensors_per_second),
        }

    def add_batch_preparation_rate(self, value: float) -> None:
        self.batch_preparation_rates.append(float(value))

    def add_training_history(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            train = record.get("train", {})
            if "batch_preparation_samples_per_second" in train:
                self.batch_preparation_rates.append(
                    float(train["batch_preparation_samples_per_second"])
                )
            if "training_steps_per_second" in train:
                self.training_step_rates.append(
                    float(train["training_steps_per_second"])
                )

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": bool(self.rows),
            "cuda_device": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "unavailable"
            ),
            "amp_enabled": self.amp_enabled,
            "periodic_samples": [
                {
                    "elapsed_seconds": elapsed,
                    "utilization_percent": utilization,
                    "memory_used_mib": memory,
                    "power_draw_watts": power,
                }
                for elapsed, utilization, memory, power in self.rows
            ],
            "cache_preparation": self.cache_preparation,
            "batch_preparation_samples_per_second": _throughput_summary(
                self.batch_preparation_rates
            ),
            "training_steps_per_second": _throughput_summary(
                self.training_step_rates
            ),
        }
        if self.device.type == "cuda":
            payload["torch_gpu_memory_allocated_mib_max"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            )
            payload["torch_gpu_memory_reserved_mib_max"] = (
                torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
            )
        if self.rows:
            utilization = [row[1] for row in self.rows]
            memory = [row[2] for row in self.rows]
            power = [row[3] for row in self.rows]
            payload.update(
                {
                    "samples": len(self.rows),
                    "gpu_utilization_percent_mean": statistics.fmean(utilization),
                    "memory_used_mib_max": max(memory),
                    "power_draw_watts_mean": statistics.fmean(power),
                }
            )
        else:
            payload["samples"] = 0
        return payload


def _throughput_summary(values):
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_historical_yaml():
    process = subprocess.run(
        ["git", "show", f"{SYNTHETIC_TRAINING_COMMIT}:{HISTORICAL_CONFIG}"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = yaml.safe_load(process.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Historical residual YAML is invalid.")
    return payload


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mean_metrics(rows):
    values = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            values[name].append(float(value))
    return {name: statistics.fmean(items) for name, items in sorted(values.items())}


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping {key!r}.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
