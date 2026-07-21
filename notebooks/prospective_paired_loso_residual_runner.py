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
    initialize_residual_arm,
    train_fixed_endpoint,
    validate_endpoint_checkpoint,
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
    records = read_manifest_jsonl(manifest_path)
    selected = select_required_acquisitions(
        records,
        split_name=str(_mapping(config, "experiment")["manifest_split"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    volumes = _cache_and_load_volumes(selected, scratch_dir=scratch_dir)
    _write_alignment_preflight(
        volumes,
        preprocessing=preprocessing,
        output_dir=output_dir / "alignment_preflight",
    )
    train_indices = selected_slice_indices(preprocessing)
    preflight = _preflight_payload(
        config=config,
        folds=folds,
        train_indices=train_indices,
        config_fingerprint=config_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
    )
    (output_dir / "preflight_sanitized.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if preflight_only:
        return preflight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _mapping(config, "runtime").get("require_cuda") and device.type != "cuda":
        raise RuntimeError("The preregistered LOSO run requires a CUDA runtime.")
    train_cfg = PseudoPairEpochConfig.from_mapping(config)
    telemetry = NvidiaSmiTelemetry()
    telemetry.start()
    selected_rows: list[dict[str, Any]] = []
    complete_rows: list[dict[str, Any]] = []
    try:
        for fold_index, fold in enumerate(folds, start=1):
            case_slot = f"case_{fold_index:02d}"
            calibrations = fit_train_only_affine_calibrations(
                volumes,
                train_case_ids=fold.train_case_ids,
                preprocessing=preprocessing,
                slice_indices=train_indices,
            )
            dataset = RealPairedSliceDataset(
                volumes,
                case_ids=fold.train_case_ids,
                preprocessing=preprocessing,
                slice_indices=train_indices,
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
                if dry_run:
                    models[arm] = model.to(device).eval()
                    continue
                loader = _build_train_loader(
                    dataset,
                    train_cfg=train_cfg,
                    num_workers=int(_mapping(config, "training")["num_workers"]),
                )
                arm_dir = output_dir / "checkpoints" / fold.fold_slot / arm
                resume_path = arm_dir / "resume.pt" if resume else None
                if resume_path is not None and not resume_path.is_file():
                    raise FileNotFoundError(
                        "Resume requested but no checkpoint exists for "
                        f"{fold.fold_slot}/{arm}."
                    )
                checkpoint_names = ("resume.pt", "endpoint.pt")
                if not resume and any(
                    (arm_dir / name).exists() for name in checkpoint_names
                ):
                    raise FileExistsError(
                        f"Existing checkpoints require --resume for {fold.fold_slot}/{arm}."
                    )
                result = train_fixed_endpoint(
                    train_cfg,
                    model=model,
                    train_loader=loader,
                    checkpoint_dir=arm_dir,
                    fold_slot=fold.fold_slot,
                    arm=arm,  # type: ignore[arg-type]
                    experiment_fingerprint=experiment_fingerprint,
                    expected_steps_per_epoch=int(_mapping(config, "training")["steps_per_epoch"]),
                    expected_global_step=int(_mapping(config, "training")["endpoint_global_step"]),
                    resume_from=resume_path,
                )
                endpoint_state = load_checkpoint(result.endpoint_checkpoint, map_location=device)
                validate_endpoint_checkpoint(
                    endpoint_state,
                    cfg=train_cfg,
                    fold_slot=fold.fold_slot,
                    arm=arm,  # type: ignore[arg-type]
                    experiment_fingerprint=experiment_fingerprint,
                    expected_steps_per_epoch=result.steps_per_epoch,
                    expected_global_step=result.global_step,
                )
                model.load_state_dict(endpoint_state["model"], strict=True)
                models[arm] = model.to(device).eval()
            if dry_run:
                _dry_run_forward(models, dataset, device)
                continue

            held_out = fold.held_out_case_id
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
                )
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
                )
            )
    finally:
        telemetry.stop()

    if dry_run:
        return {**preflight, "dry_run": True, "training_started": False}
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
            "manifest_sha256": _file_sha(manifest_path),
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
    for case_index, case_id in enumerate(CASE_IDS, start=1):
        loaded = {}
        volumes[case_id] = {}
        for field in ALL_FIELDS:
            source_path = Path(selected[case_id][field].raw_uri)
            field_label = str(field).replace(".", "p")
            destination = (
                scratch_dir / f"case_{case_index:02d}_field_{field_label}T.nii.gz"
            )
            shutil.copy2(source_path, destination)
            acquisition = load_nifti_acquisition(destination)
            loaded[field] = acquisition
            volumes[case_id][field] = acquisition.volume
        validate_paired_geometry(loaded)
    return volumes


def _write_alignment_preflight(volumes, *, preprocessing, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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


def _validate_initialization(model, arm, preprocessing, device) -> None:
    source = torch.linspace(-1.0, 1.0, 128 * 160).reshape(1, 1, 128, 160).to(device)
    model = model.to(device).eval()
    with torch.inference_mode():
        output = model(source, Domain(SOURCE_FIELD, CONTRAST), Domain(7.0, CONTRAST))
    if arm == "identity_initialization" and not torch.equal(output, source):
        raise RuntimeError("Identity-initialized arm is not exact identity at step zero.")
    if output.shape != source.shape or preprocessing.model_range != "minus_one_one":
        raise RuntimeError("Initialization model/preprocessing contract changed.")


def _dry_run_forward(models, dataset, device) -> None:
    sample = dataset[0]
    source = sample.x_low.unsqueeze(0).to(device)
    for arm, model in models.items():
        with torch.inference_mode():
            output = model(source, sample.source_domain, sample.target_domain)
        if output.shape != source.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"Dry-run forward failed for {arm}.")


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


def _evaluate_complete_fold(
    *, fold_slot, case_slot, volumes, calibrations, models, preprocessing, device
):
    rows = []
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
    return output


def _preflight_payload(*, config, folds, train_indices, config_fingerprint, experiment_fingerprint):
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
        "held_out_checkpoint_selection": False,
        "synthetic_examples_in_paired_training": False,
    }


class NvidiaSmiTelemetry:
    def __init__(self, period_seconds: float = 5.0) -> None:
        self.period_seconds = period_seconds
        self.rows: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
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
                self.rows.append(tuple(float(value.strip()) for value in values))
            self._stop.wait(self.period_seconds)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"available": False, "samples": 0}
        utilization, memory, power = zip(*self.rows, strict=True)
        return {
            "available": True,
            "samples": len(self.rows),
            "gpu_utilization_percent_mean": statistics.fmean(utilization),
            "memory_used_mib_max": max(memory),
            "power_draw_watts_mean": statistics.fmean(power),
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
