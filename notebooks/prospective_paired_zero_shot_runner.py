"""Private-data runner for the Track-A prospective paired zero-shot audit.

The runner writes only anonymous case slots and aggregate metrics.  It never
trains, updates, or saves model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

from fieldbridge.config import load_yaml_config
from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import (
    SlicePreprocessingSpec,
    from_model_range,
    preprocess_volume_slice,
    selected_slice_indices,
    to_model_range,
)
from fieldbridge.evaluation.prospective_paired import (
    ALL_FIELDS,
    CASE_IDS,
    CONTRAST,
    EVIDENCE_SCOPE,
    SELECTED_SLICE_INDICES,
    SOURCE_FIELD,
    TARGET_FIELDS,
    aggregate_rows,
    assert_sanitized_handoff,
    compute_paired_metrics,
    conditioning_margins,
    error_improvement_map,
    fixed_edge_map,
    foreground_and_outside_masks,
    load_nifti_acquisition,
    sanitized_handoff,
    select_required_acquisitions,
    validate_checkpoint_contract,
    validate_paired_geometry,
    validate_preprocessed_geometry,
)
from fieldbridge.models.factory import build_translator
from fieldbridge.official.data_manifest import MRIxFieldsDataRecord, read_manifest_jsonl
from fieldbridge.training.checkpoints import load_checkpoint

FROZEN_TRAINING_COMMIT = "e1e526ea5fa0a58f5682823f85a3957d5cc8647c"
HISTORICAL_CONFIG = "configs/experiment/pseudo_pair_t2flair_residual_probe_10epoch.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/prospective_paired_zero_shot_v1.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    run_audit(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        device_name=args.device,
    )
    return 0


def run_audit(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config_path: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    historical = _load_historical_training_yaml(FROZEN_TRAINING_COMMIT)
    preprocessing = _validate_audit_config(config, historical)
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_contract = _mapping(config, "checkpoint_contract")
    validate_checkpoint_contract(
        state,
        checkpoint_contract,
        historical_training_config=historical,
    )

    records = read_manifest_jsonl(manifest_path)
    selected = select_required_acquisitions(
        records,
        split_name=str(_mapping(config, "audit")["split_name"]),
    )
    device = _resolve_device(device_name)
    model = _build_frozen_model(historical, state, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = output_dir / "alignment_and_improvement_maps"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    geometry_fingerprints: list[str] = []
    for case_index, case_id in enumerate(CASE_IDS, start=1):
        case_slot = f"case_{case_index:02d}"
        case_records = selected[case_id]
        loaded = {
            field: load_nifti_acquisition(case_records[field].raw_uri)
            for field in ALL_FIELDS
        }
        for field, acquisition in loaded.items():
            _validate_manifest_payload(case_records[field], acquisition)
        validate_paired_geometry(loaded)
        geometry_fingerprints.append(_geometry_fingerprint(loaded[SOURCE_FIELD].geometry))

        for true_field in TARGET_FIELDS:
            source_volume = loaded[SOURCE_FIELD].volume
            target_volume = loaded[true_field].volume
            for slice_index in SELECTED_SLICE_INDICES:
                source_01, source_geometry = preprocess_volume_slice(
                    source_volume,
                    slice_index,
                    preprocessing,
                    apply_model_range=False,
                )
                target_01, target_geometry = preprocess_volume_slice(
                    target_volume,
                    slice_index,
                    preprocessing,
                    apply_model_range=False,
                )
                validate_preprocessed_geometry(source_geometry, target_geometry)
                source_01 = source_01.unsqueeze(0)
                target_01 = target_01.unsqueeze(0)
                source_model = to_model_range(source_01, preprocessing.model_range).to(device)
                target_01_device = target_01.to(device)
                source_01_device = source_01.to(device)
                foreground, outside = foreground_and_outside_masks(
                    target_01_device,
                    target_geometry,
                )
                source_metrics = compute_paired_metrics(
                    source_01_device,
                    target_01_device,
                    source_01_device,
                    foreground,
                    outside,
                )
                predictions: dict[float, torch.Tensor] = {}
                prediction_metrics: dict[float, dict[str, float]] = {}
                with torch.inference_mode():
                    for requested_field in TARGET_FIELDS:
                        output = model(
                            source_model,
                            Domain(SOURCE_FIELD, CONTRAST),
                            Domain(requested_field, CONTRAST),
                        )
                        prediction = from_model_range(
                            output,
                            preprocessing.model_range,
                        ).clamp(0.0, 1.0)
                        predictions[requested_field] = prediction
                        prediction_metrics[requested_field] = compute_paired_metrics(
                            prediction,
                            target_01_device,
                            source_01_device,
                            foreground,
                            outside,
                        )
                correct = prediction_metrics[true_field]
                wrong_by_field = {
                    _field_label(field): metrics
                    for field, metrics in prediction_metrics.items()
                    if field != true_field
                }
                wrong_mean = _mean_metrics(list(wrong_by_field.values()))
                margins_by_field = {
                    field: conditioning_margins(correct, metrics)
                    for field, metrics in wrong_by_field.items()
                }
                margins_mean = _mean_metrics(list(margins_by_field.values()))
                rows.append(
                    {
                        "case_slot": case_slot,
                        "target_field": _field_label(true_field),
                        "slice_index": slice_index,
                        "source": source_metrics,
                        "correct": correct,
                        "wrong_mean": wrong_mean,
                        "wrong_by_requested_target": wrong_by_field,
                        "margins_by_requested_target": margins_by_field,
                        "margins_mean": margins_mean,
                    }
                )
                table_rows.extend(
                    _metric_table_rows(
                        case_slot=case_slot,
                        true_field=true_field,
                        slice_index=slice_index,
                        source_metrics=source_metrics,
                        prediction_metrics=prediction_metrics,
                    )
                )
                _write_visuals(
                    asset_dir=asset_dir,
                    case_slot=case_slot,
                    true_field=true_field,
                    slice_index=slice_index,
                    source=source_01_device,
                    target=target_01_device,
                    prediction=predictions[true_field],
                )

    aggregate = aggregate_rows(rows)
    sweep = _aggregate_conditioning_sweep(rows)
    handoff = sanitized_handoff(
        checkpoint_contract=checkpoint_contract,
        aggregate=aggregate,
        target_conditioning_sweep=sweep,
        counts={
            "cases": len(CASE_IDS),
            "target_fields": len(TARGET_FIELDS),
            "selected_slices_per_case_field": len(SELECTED_SLICE_INDICES),
            "paired_slice_comparisons": len(rows),
            "conditioned_predictions": len(rows) * len(TARGET_FIELDS),
            "geometry_contracts_verified": len(geometry_fingerprints),
        },
    )
    assert_sanitized_handoff(handoff)
    (output_dir / "sanitized_handoff.json").write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_metric_table(output_dir / "baseline_vs_prediction_metrics.csv", table_rows)
    _write_summary_table(output_dir / "per_case_and_field_summary.csv", aggregate)
    _write_sweep_table(output_dir / "target_conditioning_sweep.csv", sweep)
    return handoff


def _validate_audit_config(
    config: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> SlicePreprocessingSpec:
    audit = _mapping(config, "audit")
    if audit.get("evidence_scope") != EVIDENCE_SCOPE or audit.get("complete_volume") is not False:
        raise ValueError("Audit evidence scope or complete_volume contract changed.")
    if tuple(str(value) for value in audit.get("case_ids", ())) != CASE_IDS:
        raise ValueError("Prospective case set changed.")
    configured_indices = tuple(
        int(value) for value in audit.get("selected_slice_indices", ())
    )
    if configured_indices != SELECTED_SLICE_INDICES:
        raise ValueError("Frozen selected slice indices changed.")
    if tuple(float(value) for value in audit.get("target_fields", ())) != TARGET_FIELDS:
        raise ValueError("Target field set changed.")
    if float(audit.get("source_field")) != SOURCE_FIELD:
        raise ValueError("Source field changed.")
    if config.get("model") != historical.get("model"):
        raise ValueError("Audit model config differs from the exact historical training YAML.")
    historical_data = _mapping(historical, "data")
    if config.get("preprocessing") != historical_data.get("preprocessing"):
        raise ValueError("Audit preprocessing differs from the exact historical training YAML.")
    spec = SlicePreprocessingSpec.from_mapping(config.get("preprocessing"))
    if selected_slice_indices(spec) != SELECTED_SLICE_INDICES:
        raise ValueError("SlicePreprocessingSpec no longer resolves to the frozen indices.")
    if spec.resize_mode != "fit_pad" or (spec.output_height, spec.output_width) != (128, 160):
        raise ValueError("Audit must preserve the historical fit_pad geometry.")
    return spec


def _load_historical_training_yaml(commit: str) -> dict[str, Any]:
    process = subprocess.run(
        ["git", "show", f"{commit}:{HISTORICAL_CONFIG}"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = yaml.safe_load(process.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Historical training YAML is not a mapping.")
    return payload


def _build_frozen_model(
    historical: Mapping[str, Any],
    state: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model_config = dict(_mapping(historical, "model"))
    name = str(model_config.pop("name"))
    model = build_translator(name, **model_config)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _validate_manifest_payload(record: MRIxFieldsDataRecord, acquisition: Any) -> None:
    if record.shape is not None and tuple(record.shape) != acquisition.geometry.shape:
        raise ValueError("Manifest shape does not match loaded NIfTI shape.")
    if record.affine_hash is not None:
        affine = torch.tensor(acquisition.geometry.affine, dtype=torch.float64).numpy()
        digest = hashlib.sha256(affine.astype("float64").tobytes()).hexdigest()
        if digest != record.affine_hash:
            raise ValueError("Manifest affine fingerprint does not match loaded NIfTI affine.")


def _geometry_fingerprint(geometry: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "shape": geometry.shape,
                "affine": geometry.affine,
                "orientation": geometry.orientation,
                "voxel_sizes": geometry.voxel_sizes,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _metric_table_rows(
    *,
    case_slot: str,
    true_field: float,
    slice_index: int,
    source_metrics: Mapping[str, float],
    prediction_metrics: Mapping[float, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "case_slot": case_slot,
            "true_target_field": _field_label(true_field),
            "slice_index": slice_index,
            "evaluation": "source_baseline",
            "requested_target_field": "source",
            **source_metrics,
        }
    ]
    for requested, metrics in sorted(prediction_metrics.items()):
        rows.append(
            {
                "case_slot": case_slot,
                "true_target_field": _field_label(true_field),
                "slice_index": slice_index,
                "evaluation": "correct" if requested == true_field else "wrong",
                "requested_target_field": _field_label(requested),
                **metrics,
            }
        )
    return rows


def _write_visuals(
    *,
    asset_dir: Path,
    case_slot: str,
    true_field: float,
    slice_index: int,
    source: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    source_np = source[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()
    edge_source = fixed_edge_map(source)[0, 0].detach().cpu().numpy()
    edge_target = fixed_edge_map(target)[0, 0].detach().cpu().numpy()
    improvement = error_improvement_map(source, prediction, target)[0, 0].detach().cpu().numpy()
    limit = max(float(abs(improvement).max()), 1e-8)
    stem = f"{case_slot}_target_{str(true_field).replace('.', 'p')}T_slice_{slice_index:03d}"
    figure, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    axes[0].imshow(source_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("real 0.1T")
    axes[1].imshow(target_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("actual target")
    intensity_overlay = 0.5 * source_np + 0.5 * target_np
    axes[2].imshow(intensity_overlay, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("fixed 50/50 overlay")
    edge_overlay = torch.stack(
        [
            torch.from_numpy(edge_source),
            torch.from_numpy(edge_target),
            torch.from_numpy(edge_target),
        ]
    ).permute(1, 2, 0).numpy()
    edge_overlay /= max(float(edge_overlay.max()), 1e-8)
    axes[3].imshow(edge_overlay, vmin=0.0, vmax=1.0)
    axes[3].set_title("edges: source red, target cyan")
    axes[4].imshow(improvement, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[4].set_title("error improvement (+ better)")
    for axis in axes:
        axis.axis("off")
    figure.savefig(asset_dir / f"{stem}.png", dpi=120)
    plt.close(figure)


def _aggregate_conditioning_sweep(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, float]]] = defaultdict(list)
    margin_groups: dict[tuple[str, str], list[Mapping[str, float]]] = defaultdict(list)
    for row in rows:
        true_field = str(row["target_field"])
        for requested, metrics in row["wrong_by_requested_target"].items():
            grouped[(true_field, str(requested))].append(metrics)
            margin_groups[(true_field, str(requested))].append(
                row["margins_by_requested_target"][requested]
            )
        grouped[(true_field, true_field)].append(row["correct"])
    return {
        true_field: {
            requested: {
                "metrics": _mean_metrics(group),
                "correct_vs_requested_margin": (
                    None
                    if requested == true_field
                    else _mean_metrics(margin_groups[(field, requested)])
                ),
            }
            for (field, requested), group in sorted(grouped.items())
            if field == true_field
        }
        for true_field in sorted({field for field, _ in grouped})
    }


def _write_metric_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_table(path: Path, aggregate: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for case_slot, fields in aggregate["per_case"].items():
        for field, payload in fields.items():
            for evaluation, metrics in payload.items():
                rows.append(
                    {
                        "level": "case",
                        "case_slot": case_slot,
                        "target_field": field,
                        "evaluation": evaluation,
                        **metrics,
                    }
                )
    for field, payload in aggregate["per_target_field"].items():
        for evaluation, metrics in payload.items():
            rows.append(
                {
                    "level": "field",
                    "case_slot": "macro_cases",
                    "target_field": field,
                    "evaluation": evaluation,
                    **metrics,
                }
            )
    for evaluation, metrics in aggregate["macro"].items():
        rows.append(
            {
                "level": "macro",
                "case_slot": "macro_cases",
                "target_field": "macro_fields",
                "evaluation": evaluation,
                **metrics,
            }
        )
    identity_columns = {"level", "case_slot", "target_field", "evaluation"}
    metric_names = sorted(
        {key for row in rows for key in row if key not in identity_columns}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level",
                "case_slot",
                "target_field",
                "evaluation",
                *metric_names,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_sweep_table(path: Path, sweep: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for true_field, requests in sweep.items():
        for requested, payload in requests.items():
            margins = payload["correct_vs_requested_margin"] or {}
            rows.append(
                {
                    "true_target_field": true_field,
                    "requested_target_field": requested,
                    **payload["metrics"],
                    **{f"margin_{name}": value for name, value in margins.items()},
                }
            )
    identity_columns = {"true_target_field", "requested_target_field"}
    fieldnames = ["true_target_field", "requested_target_field"] + sorted(
        {key for row in rows for key in row if key not in identity_columns}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            values[str(name)].append(float(value))
    return {name: sum(items) / len(items) for name, items in sorted(values.items())}


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration requires mapping {key!r}.")
    return value


def _field_label(field: float) -> str:
    return f"{field:g}T"


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


if __name__ == "__main__":
    raise SystemExit(main())
