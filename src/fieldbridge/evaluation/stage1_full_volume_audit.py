"""Reproducible, domain-balanced full-volume audit for Stage-1 KL-VAE checkpoints.

This module hardens :mod:`fieldbridge.evaluation.stage1_report`: it reuses the same
posterior-mean sliding-window reconstruction, but freezes selection, metric, aggregation,
provenance, and recovery contracts for long private evaluations.  Repository tests use
synthetic tensors only; private selections, NIfTIs, checkpoints, and run directories stay
outside Git.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.datasets import ALL_DOMAINS
from fieldbridge.data.vae_splits import VaeSplits, audit_vae_splits, vae_splits_fingerprint
from fieldbridge.evaluation.metrics import gradient_mae, ssim3d
from fieldbridge.evaluation.stage1_report import _central_first_spatial_axis_slice, _matplotlib_pyplot
from fieldbridge.evaluation.stage1_report import sliding_window_reconstruct
from fieldbridge.training.checkpoints import resolve_git_commit
from fieldbridge.training.latent_stats import DEFAULT_ACTIVE_KL_THRESHOLD, LatentStatsAccumulator

SELECTION_ALGORITHM_VERSION = "stage1-domain-balanced-sha256-v1"
SELECTION_SCHEMA_VERSION = 1
AUDIT_CONTRACT_VERSION = "stage1-full-volume-audit-v1"
METRIC_CONTRACT_VERSION = "stage1-full-volume-metrics-v1"
HISTOGRAM_BINS = 256
TORCH_QUANTILE_MAX_ELEMENTS = 2**24
DEFAULT_SELECTION_SEED = 13
VOLUMES_PER_DOMAIN = 4
EXPECTED_DOMAIN_COUNT = 15

Precision = Literal["float32", "amp-bfloat16"]
VolumeLoader = Callable[[VolumeRecord], torch.Tensor]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _domain_order() -> tuple[str, ...]:
    return tuple(domain.label for domain in ALL_DOMAINS)


def _record_identity(record: VolumeRecord) -> str:
    """Stable volume identity. Official records use sample_id as case_id."""

    identity = str(record.case_id).strip()
    if not identity:
        raise ValueError("Every audit-eligible record must have a non-empty case_id/sample_id.")
    return identity


def _record_contract(record: VolumeRecord) -> dict[str, Any]:
    return {
        "record_id": _record_identity(record),
        "subject_id": record.subject_id,
        "image_path": str(record.image_path),
        "domain": record.domain.to_dict(),
    }


def _all_records_fingerprint(splits: VaeSplits) -> str:
    payload: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        records = splits.records_for(split_name)
        payload[split_name] = sorted(
            (_record_contract(record) for record in records),
            key=lambda item: (str(item["record_id"]), _canonical_json(item)),
        )
    return _sha256_payload(payload)


def _validate_unique_identities(splits: VaeSplits) -> None:
    seen: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for record in splits.records_for(split_name):
            identity = _record_identity(record)
            previous = seen.get(identity)
            if previous is not None:
                raise ValueError(
                    f"Duplicate stable record identity {identity!r} appears in {previous} and "
                    f"{split_name}; audit selection fails closed."
                )
            seen[identity] = split_name
            path_identity = str(record.image_path)
            previous_path = seen_paths.get(path_identity)
            if previous_path is not None:
                raise ValueError(
                    f"Duplicate acquisition path appears in {previous_path} and {split_name}; "
                    "audit selection fails closed."
                )
            seen_paths[path_identity] = split_name


def _selection_contract(splits: VaeSplits, *, seed: int) -> dict[str, Any]:
    leakage = audit_vae_splits(splits)
    leakage.raise_for_leakage()
    _validate_unique_identities(splits)
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "seed": int(seed),
        "split_fingerprint": vae_splits_fingerprint(splits),
        "record_fingerprint": _all_records_fingerprint(splits),
        "source_split": "test",
        "source_split_seed": int(splits.seed),
        "source_split_fractions": [float(value) for value in splits.fractions],
        "domains": list(_domain_order()),
        "volumes_per_domain": VOLUMES_PER_DOMAIN,
        "total_volumes": EXPECTED_DOMAIN_COUNT * VOLUMES_PER_DOMAIN,
    }


def _rank_record(record: VolumeRecord, *, contract: Mapping[str, Any]) -> str:
    value = (
        f"{contract['algorithm_version']}|{contract['seed']}|"
        f"{contract['split_fingerprint']}|{_record_identity(record)}|{record.domain.label}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def freeze_stage1_audit_selection(
    splits: VaeSplits,
    *,
    private_path: str | Path,
    sanitized_path: str | Path,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    """Freeze exactly four deterministic test records in each of the 15 domains.

    Existing private selections are validated byte-for-contract against the supplied split
    and seed.  They are never silently regenerated.
    """

    private = Path(private_path)
    sanitized = Path(sanitized_path)
    if private.resolve() == sanitized.resolve():
        raise ValueError("Private and sanitized selection paths must be different.")
    expected_contract = _selection_contract(splits, seed=seed)
    if private.exists():
        payload = json.loads(private.read_text(encoding="utf-8"))
        _validate_frozen_selection(payload, splits=splits, expected_contract=expected_contract)
        _write_json_atomic(sanitized, _sanitized_selection(payload))
        return payload

    by_domain: dict[str, list[VolumeRecord]] = {label: [] for label in _domain_order()}
    for record in splits.test:
        label = record.domain.label
        if label not in by_domain:
            raise ValueError(f"Test split contains unsupported canonical domain {label!r}.")
        by_domain[label].append(record)

    selected: list[dict[str, Any]] = []
    for domain_index, label in enumerate(_domain_order(), start=1):
        eligible = by_domain[label]
        if len(eligible) < VOLUMES_PER_DOMAIN:
            raise ValueError(
                f"Domain {label} has {len(eligible)} eligible test records; "
                f"exactly {VOLUMES_PER_DOMAIN} are required."
            )
        ranked = sorted(
            eligible,
            key=lambda record: (
                _rank_record(record, contract=expected_contract),
                _record_identity(record),
            ),
        )[:VOLUMES_PER_DOMAIN]
        for case_index, record in enumerate(ranked, start=1):
            selected.append(
                {
                    "domain": label,
                    "domain_slot": f"domain-{domain_index:02d}",
                    "case_slot": f"domain-{domain_index:02d}-case-{case_index:02d}",
                    "record_id": _record_identity(record),
                    "rank_sha256": _rank_record(record, contract=expected_contract),
                    "record": record.to_dict(),
                    "is_exemplar": case_index == 1,
                }
            )
    payload = {
        **expected_contract,
        "selected": selected,
    }
    payload["selection_fingerprint"] = _sha256_payload(
        {
            "contract": expected_contract,
            "selected": [
                {key: item[key] for key in ("domain", "case_slot", "record_id", "rank_sha256")}
                for item in selected
            ],
        }
    )
    _write_json_atomic(private, payload)
    _write_json_atomic(sanitized, _sanitized_selection(payload))
    return payload


def _validate_frozen_selection(
    payload: Mapping[str, Any],
    *,
    splits: VaeSplits,
    expected_contract: Mapping[str, Any],
) -> None:
    for key, expected in expected_contract.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Existing selection is incompatible at {key!r}: "
                f"expected {expected!r}, found {payload.get(key)!r}."
            )
    selected = payload.get("selected")
    if not isinstance(selected, list) or len(selected) != EXPECTED_DOMAIN_COUNT * VOLUMES_PER_DOMAIN:
        raise ValueError("Existing selection does not contain exactly 60 records.")
    current = {_record_identity(record): record for record in splits.test}
    seen: set[str] = set()
    per_domain = {label: 0 for label in _domain_order()}
    for item in selected:
        if not isinstance(item, Mapping):
            raise ValueError("Existing selection contains a malformed record entry.")
        identity = str(item.get("record_id", ""))
        if identity in seen or identity not in current:
            raise ValueError("Existing selection contains a duplicate or non-test record identity.")
        seen.add(identity)
        record = current[identity]
        if item.get("domain") != record.domain.label or item.get("record") != record.to_dict():
            raise ValueError("Existing selection record metadata differs from the current split.")
        per_domain[record.domain.label] += 1
    if any(count != VOLUMES_PER_DOMAIN for count in per_domain.values()):
        raise ValueError("Existing selection is not exactly balanced at four records per domain.")
    fingerprint = _sha256_payload(
        {
            "contract": dict(expected_contract),
            "selected": [
                {key: item[key] for key in ("domain", "case_slot", "record_id", "rank_sha256")}
                for item in selected
            ],
        }
    )
    if payload.get("selection_fingerprint") != fingerprint:
        raise ValueError("Existing selection fingerprint is invalid.")


def _sanitized_selection(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "algorithm_version",
            "seed",
            "split_fingerprint",
            "record_fingerprint",
            "source_split",
            "source_split_seed",
            "source_split_fractions",
            "domains",
            "volumes_per_domain",
            "total_volumes",
            "selection_fingerprint",
        )
    } | {
        "selected": [
            {
                "domain": item["domain"],
                "domain_slot": item["domain_slot"],
                "case_slot": item["case_slot"],
                "is_exemplar": bool(item["is_exemplar"]),
            }
            for item in payload["selected"]
        ]
    }


def load_and_validate_stage1_audit_selection(
    path: str | Path, *, splits: VaeSplits, seed: int | None = None
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selection_seed = int(payload.get("seed", -1) if seed is None else seed)
    expected = _selection_contract(splits, seed=selection_seed)
    _validate_frozen_selection(payload, splits=splits, expected_contract=expected)
    return payload


@dataclass(frozen=True, slots=True)
class AuditRuntime:
    patch_size: tuple[int, int, int]
    overlap: float = 0.5
    foreground_threshold: float = 0.0
    precision: Precision = "float32"
    seed: int = DEFAULT_SELECTION_SEED
    latent_active_kl_threshold: float = DEFAULT_ACTIVE_KL_THRESHOLD

    def __post_init__(self) -> None:
        if len(self.patch_size) != 3 or any(int(value) <= 0 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive integers.")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1).")
        if self.precision not in ("float32", "amp-bfloat16"):
            raise ValueError(f"Unsupported audit precision {self.precision!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_size": list(self.patch_size),
            "overlap": self.overlap,
            "foreground_threshold": self.foreground_threshold,
            "precision": self.precision,
            "seed": self.seed,
            "latent_active_kl_threshold": self.latent_active_kl_threshold,
        }


def compute_full_volume_metrics(
    *,
    target: torch.Tensor,
    raw_reconstruction: torch.Tensor,
    foreground_threshold: float = 0.0,
    histogram_bins: int = HISTOGRAM_BINS,
    latent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the frozen full-volume metric contract on a clamped reconstruction.

    Foreground is ``target > foreground_threshold``. Empty foreground and non-finite
    tensors are fatal. Empty background and constant-foreground correlation are reported
    explicitly with status fields instead of emitting NaN.
    """

    if target.shape != raw_reconstruction.shape or target.ndim != 5:
        raise ValueError("target and reconstruction must be equal-shape (B,C,X,Y,Z) tensors.")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(raw_reconstruction).all()):
        raise ValueError("Audit metrics require finite target and reconstruction tensors.")
    target_min, target_max = float(target.min()), float(target.max())
    if target_min < -1e-6 or target_max > 1.0 + 1e-6:
        raise ValueError(
            f"Target violates the official [0,1] contract: [{target_min}, {target_max}]."
        )
    reconstruction = raw_reconstruction.clamp(0.0, 1.0)
    mask = target > float(foreground_threshold)
    if not bool(mask.any()):
        raise ValueError("Target foreground mask is empty under the frozen threshold contract.")
    foreground_target = target[mask]
    foreground_recon = reconstruction[mask]
    difference = foreground_recon - foreground_target
    foreground_mae = float(difference.abs().mean())
    foreground_nrmse = float(difference.square().mean().sqrt())  # data range is exactly 1

    correlation, correlation_status = _explicit_correlation(foreground_recon, foreground_target)
    outside = ~mask
    if bool(outside.any()):
        background_leakage: float | None = float(reconstruction[outside].abs().mean())
        background_status = "ok"
    else:
        background_leakage = None
        background_status = "not_available_no_background"

    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99)
    target_q = _linear_quantiles(foreground_target, quantiles)
    recon_q = _linear_quantiles(foreground_recon, quantiles)
    q_metrics: dict[str, float] = {}
    for name, target_value, recon_value in zip(("q01", "q05", "q50", "q95", "q99"), target_q, recon_q):
        signed = float(recon_value - target_value)
        q_metrics[f"target_{name}"] = float(target_value)
        q_metrics[f"reconstruction_{name}"] = float(recon_value)
        q_metrics[f"quantile_{name}_signed_error"] = signed
        q_metrics[f"quantile_{name}_absolute_error"] = abs(signed)

    tail_mask = mask & (target >= target_q[-1])
    tail_difference = reconstruction[tail_mask] - target[tail_mask]
    if tail_difference.numel() == 0:
        raise RuntimeError("Frozen q99 tail contract selected no voxels.")

    metrics: dict[str, Any] = {
        "foreground_mae": foreground_mae,
        "foreground_nrmse": foreground_nrmse,
        "ssim3d": float(ssim3d(reconstruction, target, data_range=1.0)),
        "correlation": correlation,
        "correlation_status": correlation_status,
        "gradient_mae": float(gradient_mae(reconstruction, target, mask.to(target.dtype))),
        "background_leakage": background_leakage,
        "background_leakage_status": background_status,
        "signed_foreground_bias": float(difference.mean()),
        "prediction_minus_source_residual_magnitude": foreground_mae,
        "high_intensity_tail_mae": float(tail_difference.abs().mean()),
        "high_intensity_tail_signed_bias": float(tail_difference.mean()),
        "foreground_histogram_wasserstein_cdf": _histogram_cdf_distance(
            foreground_recon, foreground_target, bins=histogram_bins
        ),
        "histogram_bins": int(histogram_bins),
        "raw_reconstruction_min": float(raw_reconstruction.min()),
        "raw_reconstruction_max": float(raw_reconstruction.max()),
        "raw_fraction_below_zero": float((raw_reconstruction < 0.0).float().mean()),
        "raw_fraction_above_one": float((raw_reconstruction > 1.0).float().mean()),
        "foreground_voxels": int(mask.sum()),
        "background_voxels": int(outside.sum()),
        **q_metrics,
    }
    if latent is not None:
        metrics.update(
            {
                "latent_posterior_kl": float(latent["mean_per_dim_kl"]),
                "latent_mean": float(latent["global_mean"]),
                "latent_std": float(latent["global_std"]),
                "latent_active_channels": int(latent["active_units"]),
                "latent_total_channels": int(latent["num_dims"]),
                "latent_per_channel_kl": list(latent["per_dim_kl"]),
                "latent_per_channel_std": list(latent["per_dim_std"]),
            }
        )
    _validate_metric_finiteness(metrics)
    return metrics


def _linear_quantiles(values: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """Match float32 ``torch.quantile`` without its 2^24-element hard limit."""

    flattened = values.detach().float().reshape(-1)
    q = torch.tensor(tuple(quantiles), dtype=torch.float32, device=flattened.device)
    if flattened.numel() <= TORCH_QUANTILE_MAX_ELEMENTS:
        return torch.quantile(flattened, q)

    # NumPy's linear method uses q * (n - 1), matching torch.quantile's default
    # interpolation. Cast the result back to float32 before returning it on the input
    # device so downstream tail selection retains the frozen float32 contract.
    cpu_values = flattened.cpu().numpy()
    numpy_q = np.asarray(tuple(quantiles), dtype=np.float32)
    result = np.asarray(np.quantile(cpu_values, numpy_q, method="linear"), dtype=np.float32)
    return torch.from_numpy(result).to(device=flattened.device)


def _validate_metric_finiteness(metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None or isinstance(item, (str, bool)):
                continue
            if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                raise ValueError(f"Metric {key!r} is non-finite; audit fails closed.")


def _explicit_correlation(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, str]:
    pred_centered = prediction.double() - prediction.double().mean()
    target_centered = target.double() - target.double().mean()
    pred_scale = pred_centered.square().mean().sqrt()
    target_scale = target_centered.square().mean().sqrt()
    eps = torch.finfo(torch.float64).eps
    if float(target_scale) <= eps or float(pred_scale) <= eps:
        if bool(torch.equal(prediction, target)):
            return 1.0, "constant_equal"
        return 0.0, "constant_undefined_reported_zero"
    value = (pred_centered * target_centered).mean() / (pred_scale * target_scale)
    return float(value), "ok"


def _histogram_cdf_distance(
    prediction: torch.Tensor, target: torch.Tensor, *, bins: int = HISTOGRAM_BINS
) -> float:
    if bins <= 0:
        raise ValueError("histogram bins must be positive.")
    # CPU histograms avoid device-specific atomic accumulation order while keeping the
    # expensive SSIM3D and gradient computations on the requested audit device.
    pred_hist = torch.histc(prediction.detach().float().cpu(), bins=bins, min=0.0, max=1.0)
    target_hist = torch.histc(target.detach().float().cpu(), bins=bins, min=0.0, max=1.0)
    pred_hist = pred_hist / pred_hist.sum()
    target_hist = target_hist / target_hist.sum()
    bin_width = 1.0 / bins
    return float(torch.abs(torch.cumsum(pred_hist, 0) - torch.cumsum(target_hist, 0)).sum() * bin_width)


def _numeric_metric_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    excluded = {
        "histogram_bins",
        "foreground_voxels",
        "background_voxels",
        "latent_total_channels",
    }
    keys: set[str] = set()
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if key not in excluded and (value is None or isinstance(value, (int, float))):
                keys.add(str(key))
    return sorted(keys)


def _mean_available(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float | None, int]:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics", {})
        value = metrics.get(key) if isinstance(metrics, Mapping) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def aggregate_domain_balanced(
    rows: Sequence[Mapping[str, Any]], *, expected_domains: Sequence[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate equal-volume domain means, then equal-domain macro means.

    ``micro_metrics`` is reported only as a secondary pooled-volume summary.
    """

    domains = tuple(expected_domains or _domain_order())
    grouped: dict[str, list[Mapping[str, Any]]] = {label: [] for label in domains}
    for row in rows:
        label = str(row.get("domain"))
        if label not in grouped:
            raise ValueError(f"Unexpected domain {label!r} in audit metrics.")
        grouped[label].append(row)
    missing = [label for label, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"Cannot form domain-balanced macro result; missing domains: {missing}.")

    metric_keys = _numeric_metric_keys(rows)
    per_domain: dict[str, Any] = {}
    for label in domains:
        domain_rows = grouped[label]
        domain_metrics: dict[str, float | None] = {}
        available: dict[str, int] = {}
        for key in metric_keys:
            value, count = _mean_available(domain_rows, key)
            domain_metrics[key] = value
            available[key] = count
        per_domain[label] = {
            "volume_count": len(domain_rows),
            "metrics": domain_metrics,
            "available_volume_counts": available,
        }

    macro_metrics: dict[str, float | None] = {}
    macro_counts: dict[str, int] = {}
    micro_metrics: dict[str, float | None] = {}
    micro_counts: dict[str, int] = {}
    for key in metric_keys:
        domain_values = [
            per_domain[label]["metrics"][key]
            for label in domains
            if per_domain[label]["metrics"][key] is not None
        ]
        macro_metrics[key] = (
            sum(float(value) for value in domain_values) / len(domain_values)
            if domain_values
            else None
        )
        macro_counts[key] = len(domain_values)
        micro_metrics[key], micro_counts[key] = _mean_available(rows, key)
    macro = {
        "primary_aggregation": "equal_volume_within_domain_then_equal_domain_macro",
        "domain_count": len(domains),
        "volume_count": len(rows),
        "macro_metrics": macro_metrics,
        "available_domain_counts": macro_counts,
        "micro_metrics_secondary": micro_metrics,
        "micro_available_volume_counts": micro_counts,
    }
    return per_domain, macro


def _safe_label(label: str) -> str:
    if any(separator in label for separator in ("/", "\\", ":")):
        raise ValueError("Checkpoint labels must be descriptive names, not paths.")
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-.").lower()
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Checkpoint label {label!r} cannot be sanitized safely.")
    return safe


def checkpoint_public_metadata(
    state: Mapping[str, Any], *, encoder: Any, decoder: Any
) -> dict[str, Any]:
    meta = state.get("_meta", {})
    if not isinstance(meta, Mapping):
        meta = {}
    training_commit = meta.get("git_commit", state.get("git_commit"))
    epoch = state.get("epoch", meta.get("epoch"))
    global_step = state.get("global_step", state.get("step", meta.get("global_step")))
    checkpoint_version = state.get("checkpoint_version", meta.get("checkpoint_version"))
    checkpoint_config = meta.get("config")
    return {
        "training_commit": str(training_commit) if training_commit is not None else None,
        "epoch": int(epoch) if isinstance(epoch, (int, float)) else None,
        "global_step": int(global_step) if isinstance(global_step, (int, float)) else None,
        "seed": int(meta["seed"]) if isinstance(meta.get("seed"), (int, float)) else None,
        "checkpoint_version": checkpoint_version,
        "encoder_class": type(encoder).__name__,
        "decoder_class": type(decoder).__name__,
        "checkpoint_recorded_config_sha256": (
            _sha256_payload(checkpoint_config) if checkpoint_config is not None else None
        ),
    }


def _audit_root_contract(
    *,
    selection: Mapping[str, Any],
    audit_commit: str,
    config_sha256: str,
    runtime: AuditRuntime,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "selection_fingerprint": selection["selection_fingerprint"],
        "split_fingerprint": selection["split_fingerprint"],
        "record_fingerprint": selection["record_fingerprint"],
        "audit_commit": audit_commit,
        "config_sha256": config_sha256,
        "device": str(device),
        "runtime": runtime.to_dict(),
        "input_contract": "official_[0,1]_no_rescaling",
        "reconstruction_contract": {
            "complete_native_tensor": True,
            "latent": "posterior_mean_z_equals_mu",
            "random_crops_or_slices": False,
            "augmentation": False,
            "window_traversal": "first_spatial_axis_then_second_then_third_ascending_edge_clamped",
            "blending": "separable_hann_clamped_min_1e-3_normalized_weight_sum",
            "raw_range_recorded_before_clamp": True,
            "official_metrics_input": "clamp(raw_reconstruction,0,1)",
        },
        "foreground_contract": {
            "definition": f"target > {runtime.foreground_threshold:g}",
            "empty_foreground": "fatal",
            "empty_background": "null_with_explicit_status",
            "constant_correlation": "one_if_equal_else_zero_with_explicit_status",
        },
        "histogram_contract": {
            "range": [0.0, 1.0],
            "bins": HISTOGRAM_BINS,
            "formula": "sum(abs(CDF_pred-CDF_target))*bin_width",
        },
        "aggregation_contract": "equal volumes within domain; equal 15 domain means for macro",
        "scientific_gate": None,
        "claim_scope": "descriptive full-volume reconstruction evidence; no automatic viability declaration",
    }


def prepare_audit_root(
    out_dir: str | Path,
    *,
    selection: Mapping[str, Any],
    audit_commit: str,
    config_sha256: str,
    runtime: AuditRuntime,
    device: torch.device,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contract = _audit_root_contract(
        selection=selection,
        audit_commit=audit_commit,
        config_sha256=config_sha256,
        runtime=runtime,
        device=device,
    )
    path = out / "audit_contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("Existing audit_contract.json is incompatible; refusing recovery.")
    else:
        _write_json_atomic(path, contract)
    _write_json_atomic(
        out / "selection_fingerprint.json",
        {
            "selection_fingerprint": selection["selection_fingerprint"],
            "split_fingerprint": selection["split_fingerprint"],
            "record_fingerprint": selection["record_fingerprint"],
            "algorithm_version": selection["algorithm_version"],
            "seed": selection["seed"],
            "source_split_seed": selection["source_split_seed"],
            "domain_count": EXPECTED_DOMAIN_COUNT,
            "volume_count": EXPECTED_DOMAIN_COUNT * VOLUMES_PER_DOMAIN,
        },
    )
    return contract


def _ensure_official_input(volume: torch.Tensor) -> torch.Tensor:
    if volume.ndim != 4:
        raise ValueError(f"Audit volume loader must return (C,X,Y,Z), got {tuple(volume.shape)}.")
    volume = volume.to(dtype=torch.float32)
    if not bool(torch.isfinite(volume).all()):
        raise ValueError("Audit input contains non-finite values.")
    low, high = float(volume.min()), float(volume.max())
    if low < -1e-6 or high > 1.0 + 1e-6:
        raise ValueError(f"Audit input violates official [0,1] contract: [{low}, {high}].")
    return volume


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "float32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("amp-bfloat16 audit precision requires CUDA.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _checkpoint_contract(
    *,
    checkpoint_label: str,
    checkpoint_sha256: str,
    public_metadata: Mapping[str, Any],
    root_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_label": _safe_label(checkpoint_label),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": dict(public_metadata),
        "root_contract_sha256": _sha256_payload(root_contract),
    }


def audit_stage1_checkpoint(
    *,
    encoder: Any,
    decoder: Any,
    volume_loader: VolumeLoader,
    selection: Mapping[str, Any],
    out_dir: str | Path,
    checkpoint_slot: str,
    checkpoint_label: str,
    checkpoint_sha256: str,
    checkpoint_metadata: Mapping[str, Any],
    root_contract: Mapping[str, Any],
    runtime: AuditRuntime,
    device: torch.device,
    resume: bool = False,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint with validated per-volume recovery."""

    checkpoint_dir = Path(out_dir)
    state_dir = checkpoint_dir / ".audit_state"
    panels_dir = checkpoint_dir / "diagnostic_panels"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(
        checkpoint_label=checkpoint_label,
        checkpoint_sha256=checkpoint_sha256,
        public_metadata=checkpoint_metadata,
        root_contract=root_contract,
    )
    contract_path = checkpoint_dir / "checkpoint_contract.json"
    existing_artifacts = list(checkpoint_dir.iterdir())
    if existing_artifacts and not resume:
        raise FileExistsError(
            f"Checkpoint audit directory for {checkpoint_slot} is not empty; pass --resume "
            "to validate and recover it."
        )
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise ValueError("Checkpoint audit recovery contract is incompatible.")
    elif existing_artifacts:
        raise ValueError("Partial checkpoint audit artifacts exist without a valid contract.")
    else:
        _write_json_atomic(contract_path, contract)

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    torch.manual_seed(runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime.seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    selected = list(selection["selected"])
    expected_state_names = {f"{entry['case_slot']}.json" for entry in selected}
    if state_dir.exists():
        unexpected = sorted(path.name for path in state_dir.iterdir() if path.name not in expected_state_names)
        if unexpected:
            raise ValueError(f"Unexpected recovery artifacts in .audit_state: {unexpected}.")
    expected_panel_names = {
        f"{entry['domain_slot']}.png" for entry in selected if bool(entry["is_exemplar"])
    }
    if panels_dir.exists():
        unexpected_panels = sorted(
            path.name for path in panels_dir.iterdir() if path.name not in expected_panel_names
        )
        if unexpected_panels:
            raise ValueError(f"Unexpected diagnostic panel artifacts: {unexpected_panels}.")
    completed_rows: dict[str, dict[str, Any]] = {}
    for entry in selected:
        case_slot = str(entry["case_slot"])
        result_path = state_dir / f"{case_slot}.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("checkpoint_contract_sha256") != _sha256_payload(contract):
            raise ValueError(f"Recovered result {case_slot} has an incompatible fingerprint.")
        if result.get("case_slot") != case_slot or result.get("domain") != entry["domain"]:
            raise ValueError(f"Recovered result {case_slot} has incompatible anonymous identity.")
        panel = panels_dir / f"{entry['domain_slot']}.png"
        if entry["is_exemplar"] and not panel.exists():
            raise ValueError(f"Recovered exemplar {case_slot} is missing its diagnostic panel.")
        completed_rows[case_slot] = result

    progress_file = Path(progress_path) if progress_path is not None else checkpoint_dir / "run_progress_sanitized.json"
    for index, entry in enumerate(selected, start=1):
        case_slot = str(entry["case_slot"])
        if case_slot in completed_rows:
            _update_progress(progress_file, checkpoint_slot, case_slot, index, len(selected), "validated_complete")
            continue
        _update_progress(progress_file, checkpoint_slot, case_slot, index, len(selected), "reconstructing")
        record = VolumeRecord(**entry["record"])
        volume = _ensure_official_input(volume_loader(record)).unsqueeze(0).to(device)
        latent_channels = getattr(encoder, "latent_channels", int(volume.shape[1]))
        latent_accumulator = LatentStatsAccumulator(int(latent_channels))
        with torch.inference_mode(), _autocast_context(device, runtime.precision):
            raw_reconstruction = sliding_window_reconstruct(
                encoder,
                decoder,
                volume,
                patch_size=runtime.patch_size,
                domain=record.domain,
                overlap=runtime.overlap,
                clamp_output=False,
                latent_accumulator=latent_accumulator,
            )
        latent = latent_accumulator.compute(active_threshold=runtime.latent_active_kl_threshold)
        metrics = compute_full_volume_metrics(
            target=volume.float(),
            raw_reconstruction=raw_reconstruction.float(),
            foreground_threshold=runtime.foreground_threshold,
            latent=latent,
        )
        row = {
            "case_slot": case_slot,
            "domain_slot": entry["domain_slot"],
            "domain": entry["domain"],
            "complete_volume": True,
            "input_shape_cxyz": list(volume.shape[1:]),
            "metrics": metrics,
            "checkpoint_contract_sha256": _sha256_payload(contract),
        }
        if entry["is_exemplar"]:
            panels_dir.mkdir(parents=True, exist_ok=True)
            render_audit_panel(
                volume.detach().cpu(),
                raw_reconstruction.clamp(0.0, 1.0).detach().cpu(),
                path=panels_dir / f"{entry['domain_slot']}.png",
                domain_slot=str(entry["domain_slot"]),
            )
        _write_json_atomic(state_dir / f"{case_slot}.json", row)
        completed_rows[case_slot] = row
        _update_progress(progress_file, checkpoint_slot, case_slot, index, len(selected), "volume_complete")

    ordered_rows = [completed_rows[str(item["case_slot"])] for item in selected]
    per_domain, macro = aggregate_domain_balanced(ordered_rows)
    _write_jsonl_atomic(checkpoint_dir / "per_volume_metrics.jsonl", ordered_rows)
    _write_json_atomic(checkpoint_dir / "per_domain_metrics.json", per_domain)
    _write_json_atomic(checkpoint_dir / "macro_metrics.json", macro)
    report = _checkpoint_report(checkpoint_slot, checkpoint_label, checkpoint_sha256, ordered_rows, per_domain, macro)
    (checkpoint_dir / "report.md").write_text(report, encoding="utf-8")
    _update_progress(progress_file, checkpoint_slot, None, len(selected), len(selected), "complete")
    return {
        "checkpoint_slot": checkpoint_slot,
        "checkpoint_label": _safe_label(checkpoint_label),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": dict(checkpoint_metadata),
        "volume_count": len(ordered_rows),
        "domain_count": len(per_domain),
        "macro_metrics": macro["macro_metrics"],
        "relative_artifacts": {
            "per_volume_metrics": f"checkpoints/{checkpoint_slot}/per_volume_metrics.jsonl",
            "per_domain_metrics": f"checkpoints/{checkpoint_slot}/per_domain_metrics.json",
            "macro_metrics": f"checkpoints/{checkpoint_slot}/macro_metrics.json",
            "report": f"checkpoints/{checkpoint_slot}/report.md",
            "diagnostic_panels": f"checkpoints/{checkpoint_slot}/diagnostic_panels",
        },
    }


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _update_progress(
    path: Path,
    checkpoint_slot: str,
    case_slot: str | None,
    completed: int,
    total: int,
    state: str,
) -> None:
    payload = {
        "state": state,
        "checkpoint_slot": checkpoint_slot,
        "case_slot": case_slot,
        "completed_volumes": int(completed if state in {"volume_complete", "validated_complete", "complete"} else max(completed - 1, 0)),
        "total_volumes": int(total),
    }
    _write_json_atomic(path, payload)
    print(
        f"stage1_audit checkpoint={checkpoint_slot} case={case_slot or '-'} "
        f"progress={payload['completed_volumes']}/{total} state={state}",
        flush=True,
    )


def render_audit_panel(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    path: Path,
    domain_slot: str,
) -> None:
    """Write target/reconstruction/error/foreground-histogram for a frozen exemplar."""

    import numpy as np

    plt = _matplotlib_pyplot()
    target_slice = _central_first_spatial_axis_slice(target)
    recon_slice = _central_first_spatial_axis_slice(reconstruction)
    error = np.abs(target_slice - recon_slice)
    foreground = target > 0.0
    target_values = target[foreground].numpy()
    recon_values = reconstruction[foreground].numpy()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), squeeze=False)
    axes[0, 0].imshow(target_slice, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(f"Target - {domain_slot}")
    axes[0, 1].imshow(recon_slice, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Posterior-mean reconstruction")
    image = axes[0, 2].imshow(error, cmap="hot", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Absolute error")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 3].hist(target_values, bins=HISTOGRAM_BINS, range=(0, 1), alpha=0.5, label="target")
    axes[0, 3].hist(recon_values, bins=HISTOGRAM_BINS, range=(0, 1), alpha=0.5, label="reconstruction")
    axes[0, 3].set_title("Foreground intensity histogram")
    axes[0, 3].legend()
    for axis in axes[0, :3]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _checkpoint_report(
    checkpoint_slot: str,
    checkpoint_label: str,
    checkpoint_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    per_domain: Mapping[str, Any],
    macro: Mapping[str, Any],
) -> str:
    lines = [
        f"# Stage-1 Full-Volume Audit - {checkpoint_slot}",
        "",
        f"Checkpoint label: `{_safe_label(checkpoint_label)}`  ",
        f"Checkpoint SHA-256: `{checkpoint_sha256}`  ",
        f"Complete volumes: `{len(rows)}` across `{len(per_domain)}` canonical domains.",
        "",
        "Primary aggregation gives every volume equal weight within its domain and every domain equal weight in the 15-domain macro.",
        "No scientific viability threshold is applied by this audit.",
        "",
        "| Domain | n | foreground nRMSE | foreground MAE | SSIM3D | correlation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, summary in per_domain.items():
        metrics = summary["metrics"]
        lines.append(
            f"| {domain} | {summary['volume_count']} | {_fmt(metrics.get('foreground_nrmse'))} | "
            f"{_fmt(metrics.get('foreground_mae'))} | {_fmt(metrics.get('ssim3d'))} | "
            f"{_fmt(metrics.get('correlation'))} |"
        )
    metrics = macro["macro_metrics"]
    lines.extend(
        [
            "",
            "## Equal-domain macro",
            "",
            f"- Foreground nRMSE: `{_fmt(metrics.get('foreground_nrmse'))}`",
            f"- Foreground MAE: `{_fmt(metrics.get('foreground_mae'))}`",
            f"- SSIM3D: `{_fmt(metrics.get('ssim3d'))}`",
            f"- Correlation: `{_fmt(metrics.get('correlation'))}`",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.8f}"


def write_audit_comparison(
    out_dir: str | Path,
    *,
    checkpoint_summaries: Sequence[Mapping[str, Any]],
    root_contract: Mapping[str, Any],
) -> dict[str, Any]:
    out = Path(out_dir)
    summaries = [dict(summary) for summary in checkpoint_summaries]
    comparison: dict[str, Any] = {
        "comparison_contract": "independently evaluated compatible checkpoints; first listed is reference",
        "root_contract_sha256": _sha256_payload(root_contract),
        "checkpoints": summaries,
        "deltas_vs_first": [],
        "scientific_gate": None,
    }
    if summaries:
        reference = summaries[0]["macro_metrics"]
        for summary in summaries:
            current = summary["macro_metrics"]
            deltas = {
                key: float(current[key]) - float(reference[key])
                for key in sorted(set(reference) & set(current))
                if isinstance(reference[key], (int, float)) and isinstance(current[key], (int, float))
            }
            comparison["deltas_vs_first"].append(
                {"checkpoint_slot": summary["checkpoint_slot"], "metric_deltas": deltas}
            )
    _write_json_atomic(out / "checkpoint_comparison.json", comparison)
    handoff = {
        "complete": bool(summaries) and all(int(item["volume_count"]) == 60 for item in summaries),
        "complete_volume": True,
        "domain_balanced": True,
        "selection_fingerprint": root_contract["selection_fingerprint"],
        "split_fingerprint": root_contract["split_fingerprint"],
        "audit_commit": root_contract["audit_commit"],
        "config_sha256": root_contract["config_sha256"],
        "metric_contract_version": root_contract["metric_contract_version"],
        "checkpoints": summaries,
        "checkpoint_comparison": comparison,
        "relative_artifacts": {
            "audit_contract": "audit_contract.json",
            "selection_fingerprint": "selection_fingerprint.json",
            "progress": "run_progress_sanitized.json",
            "checkpoint_comparison": "checkpoint_comparison.json",
            "report": "report.md",
        },
        "scientific_viability": "not_declared_no_preregistered_gate",
    }
    _write_json_atomic(out / "sanitized_handoff.json", handoff)
    (out / "report.md").write_text(_comparison_report(summaries), encoding="utf-8")
    return comparison


def _comparison_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Stage-1 Domain-Balanced Full-Volume Audit",
        "",
        "Each checkpoint was evaluated independently on the same frozen 60-volume selection.",
        "Primary metrics are equal-volume within each domain and equal-domain across all 15 domains.",
        "This report is descriptive and does not invent a scientific pass/fail threshold.",
        "",
        "| Checkpoint slot | Label | volumes | macro nRMSE | macro SSIM3D | macro MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        metrics = summary["macro_metrics"]
        lines.append(
            f"| {summary['checkpoint_slot']} | {summary['checkpoint_label']} | {summary['volume_count']} | "
            f"{_fmt(metrics.get('foreground_nrmse'))} | {_fmt(metrics.get('ssim3d'))} | "
            f"{_fmt(metrics.get('foreground_mae'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def resolve_audit_commit() -> str:
    return resolve_git_commit()


class _SyntheticIdentityEncoder(torch.nn.Module):
    latent_channels = 1

    def encode_dist(self, image: torch.Tensor, domain: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return image, torch.zeros_like(image)


class _SyntheticIdentityDecoder(torch.nn.Module):
    def decode(self, latent: torch.Tensor, domain: Any) -> torch.Tensor:
        return latent


def run_synthetic_stage1_audit_smoke(out_dir: str | Path) -> dict[str, Any]:
    """Run all 15 domains through the real selection/audit/report orchestration on CPU."""

    out = Path(out_dir)
    records: list[VolumeRecord] = []
    for domain_index, domain in enumerate(ALL_DOMAINS):
        for record_index in range(VOLUMES_PER_DOMAIN):
            identity = f"synthetic-d{domain_index:02d}-v{record_index:02d}"
            records.append(
                VolumeRecord(
                    case_id=identity,
                    image_path=f"synthetic/{identity}.nii.gz",
                    domain=domain,
                    subject_id=identity,
                )
            )
    splits = VaeSplits(
        train=(),
        validation=(),
        test=tuple(records),
        seed=DEFAULT_SELECTION_SEED,
        fractions=(0.0, 0.0, 1.0),
        metadata={"synthetic": True},
    )
    selection = freeze_stage1_audit_selection(
        splits,
        private_path=out / "selection_private_synthetic.json",
        sanitized_path=out / "selection_sanitized.json",
    )
    runtime = AuditRuntime(patch_size=(4, 5, 6), overlap=0.5)
    device = torch.device("cpu")
    root_contract = prepare_audit_root(
        out,
        selection=selection,
        audit_commit="synthetic-smoke",
        config_sha256=hashlib.sha256(b"synthetic-stage1-audit-config-v1").hexdigest(),
        runtime=runtime,
        device=device,
    )

    def load_volume(record: VolumeRecord) -> torch.Tensor:
        digest = hashlib.sha256(record.case_id.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:2], "big") / 65535.0 * 0.05
        x = torch.linspace(-1.0, 1.0, 9)
        y = torch.linspace(-1.0, 1.0, 10)
        z = torch.linspace(-1.0, 1.0, 11)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        support = xx.square() + yy.square() + zz.square() <= 0.75
        signal = ((xx + 1.0) * 0.25 + (yy + 1.0) * 0.15 + (zz + 1.0) * 0.1 + offset).clamp(0, 1)
        return torch.where(support, signal, torch.zeros_like(signal)).unsqueeze(0)

    summary = audit_stage1_checkpoint(
        encoder=_SyntheticIdentityEncoder(),
        decoder=_SyntheticIdentityDecoder(),
        volume_loader=load_volume,
        selection=selection,
        out_dir=out / "checkpoints" / "checkpoint-01",
        checkpoint_slot="checkpoint-01",
        checkpoint_label="synthetic-identity",
        checkpoint_sha256=hashlib.sha256(b"synthetic-identity-checkpoint").hexdigest(),
        checkpoint_metadata={
            "training_commit": "synthetic",
            "epoch": 0,
            "global_step": 0,
            "checkpoint_version": "synthetic-v1",
            "encoder_class": "_SyntheticIdentityEncoder",
            "decoder_class": "_SyntheticIdentityDecoder",
        },
        root_contract=root_contract,
        runtime=runtime,
        device=device,
        progress_path=out / "run_progress_sanitized.json",
    )
    comparison = write_audit_comparison(out, checkpoint_summaries=[summary], root_contract=root_contract)
    return {"ok": True, "volume_count": summary["volume_count"], "domain_count": summary["domain_count"], "comparison": comparison}
