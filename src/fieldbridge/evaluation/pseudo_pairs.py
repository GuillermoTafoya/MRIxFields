"""Evaluation for pseudo-pair degraded and predicted baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import ModelRange, from_model_range
from fieldbridge.data.pseudo_pairs import PseudoPairSliceBatch
from fieldbridge.evaluation.metrics import (
    gradient_mae,
    lpips_metric,
    masked_mae,
    normalized_cross_correlation,
    nrmse,
    outside_mask_mean_abs,
    psnr,
    ssim,
)
from fieldbridge.models.translators.base import BaseTranslator

Device = Literal["auto", "cpu", "cuda"]
LPIPSMode = Literal["auto", "off", "on"]

_LOWER_BETTER = {"nrmse", "masked_mae", "gradient_mae", "outside_mask_mean_abs", "lpips"}
_HIGHER_BETTER = {"ssim", "psnr", "correlation"}


@dataclass(frozen=True, slots=True)
class PseudoPairEvalConfig:
    device: Device = "auto"
    model_range: ModelRange = "minus_one_one"
    lpips: LPIPSMode = "auto"
    target_fields: tuple[float, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PseudoPairEvalConfig":
        defaults = cls()
        evaluation = dict(data.get("evaluation", {})) if isinstance(data.get("evaluation", {}), Mapping) else {}
        data_config = dict(data.get("data", {})) if isinstance(data.get("data", {}), Mapping) else {}
        preproc = data_config.get("preprocessing", {})
        model_range = defaults.model_range
        if isinstance(preproc, Mapping):
            model_range = preproc.get("model_range", model_range)
        target_fields = data_config.get("target_fields", evaluation.get("target_fields", ()))
        return cls(
            device=evaluation.get("device", data.get("device", defaults.device)),
            model_range=evaluation.get("model_range", model_range),
            lpips=evaluation.get("lpips", defaults.lpips),
            target_fields=tuple(float(field) for field in target_fields),
        )


def evaluate_pseudo_pairs(
    model: BaseTranslator,
    loader: DataLoader[PseudoPairSliceBatch],
    config: PseudoPairEvalConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    device = _resolve_device(cfg.device)
    model = model.to(device)
    model.eval()
    lpips_net, lpips_status = _build_optional_lpips(cfg.lpips, device)

    rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            supported_fields = _supported_target_fields(batch.target_domain, cfg.target_fields)
            prediction = model(batch.x_low, batch.source_domain, batch.target_domain)
            wrong_predictions_01: dict[float, torch.Tensor] = {}
            for wrong_field in supported_fields:
                wrong_domains = [Domain(wrong_field, domain.contrast) for domain in batch.target_domain]
                wrong_prediction = model(batch.x_low, batch.source_domain, wrong_domains)
                wrong_predictions_01[wrong_field] = from_model_range(
                    wrong_prediction,
                    cfg.model_range,
                ).clamp(0.0, 1.0)
            permuted_domains = _permuted_target_domains(batch.target_domain, cfg.target_fields)
            permuted_prediction = model(batch.x_low, batch.source_domain, permuted_domains)

            x_low = from_model_range(batch.x_low, cfg.model_range).clamp(0.0, 1.0)
            x_high = from_model_range(batch.x_high, cfg.model_range).clamp(0.0, 1.0)
            x_pred = from_model_range(prediction, cfg.model_range).clamp(0.0, 1.0)
            x_permuted = from_model_range(permuted_prediction, cfg.model_range).clamp(0.0, 1.0)
            for index, target_domain in enumerate(batch.target_domain):
                target = x_high[index : index + 1]
                mask = batch.mask[index : index + 1].to(device)
                target_field = float(target_domain.field_strength_t)
                target_field_label = _field_label(target_field)
                degraded_metrics = _compute_metrics(
                    x_low[index : index + 1],
                    target,
                    mask,
                    lpips_net=lpips_net,
                )
                predicted_metrics = _compute_metrics(
                    x_pred[index : index + 1],
                    target,
                    mask,
                    lpips_net=lpips_net,
                )
                wrong_metrics_by_field: dict[str, dict[str, float]] = {}
                for wrong_field, wrong_prediction_01 in wrong_predictions_01.items():
                    if float(wrong_field) == target_field:
                        continue
                    wrong_metrics = _compute_metrics(
                        wrong_prediction_01[index : index + 1],
                        target,
                        mask,
                        lpips_net=lpips_net,
                    )
                    wrong_field_label = _field_label(wrong_field)
                    wrong_metrics_by_field[wrong_field_label] = wrong_metrics
                    wrong_rows.append(
                        {
                            "target_field": target_field_label,
                            "wrong_target_field": wrong_field_label,
                            "predicted": predicted_metrics,
                            "wrong_conditioned": wrong_metrics,
                        }
                    )
                wrong_summary = (
                    _mean_metric_dict(wrong_metrics_by_field.values())
                    if wrong_metrics_by_field
                    else dict(predicted_metrics)
                )
                wrong_nrmse_values = [metrics["nrmse"] for metrics in wrong_metrics_by_field.values()]
                best_wrong_nrmse = min(wrong_nrmse_values) if wrong_nrmse_values else None
                correct_nrmse = predicted_metrics["nrmse"]
                rows.append(
                    {
                        "target_field": target_field_label,
                        "degraded": degraded_metrics,
                        "predicted": predicted_metrics,
                        "wrong_conditioned": wrong_summary,
                        "wrong_conditioned_by_target_field": wrong_metrics_by_field,
                        "permuted_conditioned": _compute_metrics(
                            x_permuted[index : index + 1],
                            target,
                            mask,
                            lpips_net=lpips_net,
                        ),
                        "conditioning": {
                            "correct_nrmse": correct_nrmse,
                            "best_wrong_nrmse": best_wrong_nrmse,
                            "margin_vs_best_wrong_nrmse": None
                            if best_wrong_nrmse is None
                            else best_wrong_nrmse - correct_nrmse,
                            "correct_has_best_nrmse": None
                            if best_wrong_nrmse is None
                            else correct_nrmse <= best_wrong_nrmse,
                        },
                    }
                )

    aggregate = {
        "degraded": _aggregate(rows, "degraded"),
        "predicted": _aggregate(rows, "predicted"),
        "wrong_conditioned": _aggregate(rows, "wrong_conditioned"),
        "permuted_conditioned": _aggregate(rows, "permuted_conditioned"),
    }
    per_field = _aggregate_by_field(rows)
    macro_average = {
        "degraded": _macro_average(per_field, "degraded"),
        "predicted": _macro_average(per_field, "predicted"),
        "wrong_conditioned": _macro_average(per_field, "wrong_conditioned"),
        "permuted_conditioned": _macro_average(per_field, "permuted_conditioned"),
    }
    improvement = _metric_delta(aggregate["degraded"], aggregate["predicted"])
    conditioning_effect = _metric_effect(aggregate["wrong_conditioned"], aggregate["predicted"])
    permuted_effect = _metric_effect(aggregate["permuted_conditioned"], aggregate["predicted"])
    return {
        "num_samples": len(rows),
        "aggregate": aggregate,
        "macro_average": macro_average,
        "per_target_field": per_field,
        "improvement_over_degraded": improvement,
        "target_conditioning_audit": {
            "metric": "nrmse",
            "correct_vs_wrong_improvement": conditioning_effect,
            "correct_vs_permuted_improvement": permuted_effect,
            "sample_level": _conditioning_sample_summary(rows),
            "by_true_target_field": _conditioning_summary_by_true_field(rows),
            "by_wrong_target_field": _conditioning_summary_by_wrong_field(wrong_rows),
        },
        "lpips": lpips_status,
    }


def _compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    lpips_net: nn.Module | None,
) -> dict[str, float]:
    metrics = {
        "nrmse": float(nrmse(prediction, target, data_range=1.0).detach().cpu()),
        "ssim": float(ssim(prediction, target, data_range=1.0).detach().cpu()),
        "psnr": float(psnr(prediction, target, data_range=1.0).detach().cpu()),
        "masked_mae": float(masked_mae(prediction, target, mask).detach().cpu()),
        "gradient_mae": float(gradient_mae(prediction, target, mask).detach().cpu()),
        "outside_mask_mean_abs": float(outside_mask_mean_abs(prediction, mask).detach().cpu()),
        "correlation": float(normalized_cross_correlation(prediction, target, mask).detach().cpu()),
    }
    if lpips_net is not None:
        metrics["lpips"] = float(
            lpips_metric(prediction * 2.0 - 1.0, target * 2.0 - 1.0, net=lpips_net).detach().cpu()
        )
    return metrics


def _aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row[key]
        for metric_name, metric_value in metrics.items():
            values[metric_name].append(float(metric_value))
    return {
        metric_name: sum(metric_values) / len(metric_values)
        for metric_name, metric_values in sorted(values.items())
        if metric_values
    }


def _aggregate_by_field(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_field"])].append(row)
    payload: dict[str, Any] = {}
    for field, field_rows in sorted(grouped.items()):
        degraded = _aggregate(field_rows, "degraded")
        predicted = _aggregate(field_rows, "predicted")
        wrong_conditioned = _aggregate(field_rows, "wrong_conditioned")
        permuted_conditioned = _aggregate(field_rows, "permuted_conditioned")
        payload[field] = {
            "samples": len(field_rows),
            "degraded": degraded,
            "predicted": predicted,
            "wrong_conditioned": wrong_conditioned,
            "permuted_conditioned": permuted_conditioned,
            "improvement_over_degraded": _metric_delta(degraded, predicted),
            "target_conditioning_audit": {
                "correct_vs_wrong_improvement": _metric_effect(wrong_conditioned, predicted),
                "correct_vs_permuted_improvement": _metric_effect(permuted_conditioned, predicted),
                "sample_level": _conditioning_sample_summary(field_rows),
            },
        }
    return payload


def _macro_average(per_field: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for field_payload in per_field.values():
        metrics = field_payload[key]
        for metric_name, metric_value in metrics.items():
            values[metric_name].append(float(metric_value))
    return {
        metric_name: sum(metric_values) / len(metric_values)
        for metric_name, metric_values in sorted(values.items())
        if metric_values
    }


def _metric_delta(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for metric_name, baseline_value in baseline.items():
        if metric_name not in candidate:
            continue
        if metric_name in _LOWER_BETTER:
            delta[metric_name] = float(baseline_value) - float(candidate[metric_name])
        elif metric_name in _HIGHER_BETTER:
            delta[metric_name] = float(candidate[metric_name]) - float(baseline_value)
    return delta


def _metric_effect(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, Any]:
    absolute = _metric_delta(baseline, candidate)
    return {
        "absolute": absolute,
        "relative": _metric_delta_relative(baseline, absolute),
    }


def _metric_delta_relative(
    baseline: Mapping[str, float],
    absolute: Mapping[str, float],
) -> dict[str, float | None]:
    relative: dict[str, float | None] = {}
    for metric_name, improvement in absolute.items():
        baseline_value = float(baseline[metric_name])
        if abs(baseline_value) < 1e-12:
            relative[metric_name] = None
        else:
            relative[metric_name] = float(improvement) / abs(baseline_value)
    return relative


def _mean_metric_dict(metrics: Iterable[Mapping[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        for metric_name, metric_value in metric.items():
            values[metric_name].append(float(metric_value))
    return {
        metric_name: sum(metric_values) / len(metric_values)
        for metric_name, metric_values in sorted(values.items())
        if metric_values
    }


def _conditioning_sample_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flags: list[bool] = []
    margins: list[float] = []
    for row in rows:
        conditioning = row.get("conditioning", {})
        if not isinstance(conditioning, Mapping):
            continue
        flag = conditioning.get("correct_has_best_nrmse")
        margin = conditioning.get("margin_vs_best_wrong_nrmse")
        if flag is not None:
            flags.append(bool(flag))
        if margin is not None:
            margins.append(float(margin))
    return {
        "samples_with_wrong_targets": len(flags),
        "fraction_correct_best_nrmse": None if not flags else sum(flags) / len(flags),
        "mean_margin_vs_best_wrong_nrmse": _mean_or_none(margins),
        "median_margin_vs_best_wrong_nrmse": _median_or_none(margins),
    }


def _conditioning_summary_by_true_field(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_field"])].append(row)
    return {
        field: {
            "samples": len(field_rows),
            **_conditioning_sample_summary(field_rows),
        }
        for field, field_rows in sorted(grouped.items())
    }


def _conditioning_summary_by_wrong_field(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["wrong_target_field"])].append(row)
    payload: dict[str, Any] = {}
    for wrong_field, field_rows in sorted(grouped.items()):
        predicted = _aggregate(field_rows, "predicted")
        wrong_conditioned = _aggregate(field_rows, "wrong_conditioned")
        payload[wrong_field] = {
            "samples": len(field_rows),
            "predicted": predicted,
            "wrong_conditioned": wrong_conditioned,
            "correct_vs_wrong_improvement": _metric_effect(wrong_conditioned, predicted),
        }
    return payload


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _supported_target_fields(domains: Sequence[Domain], target_fields: Sequence[float]) -> tuple[float, ...]:
    fields = tuple(float(field) for field in target_fields)
    if fields:
        return fields
    return tuple(sorted({float(domain.field_strength_t) for domain in domains}))


def _field_label(field: float) -> str:
    return f"{float(field):g}T"


def _wrong_target_domains(domains: Sequence[Domain], target_fields: Sequence[float]) -> list[Domain]:
    fields = tuple(float(field) for field in target_fields)
    if len(fields) < 2:
        fields = tuple(sorted({float(domain.field_strength_t) for domain in domains}))
    if len(fields) < 2:
        return list(domains)
    wrong: list[Domain] = []
    for domain in domains:
        try:
            index = fields.index(float(domain.field_strength_t))
        except ValueError:
            index = -1
        wrong_field = fields[(index + 1) % len(fields)]
        wrong.append(Domain(wrong_field, domain.contrast))
    return wrong


def _permuted_target_domains(domains: Sequence[Domain], target_fields: Sequence[float]) -> list[Domain]:
    if len(domains) < 2:
        return _wrong_target_domains(domains, target_fields)
    fields = tuple(float(field) for field in target_fields)
    if len(fields) >= 2:
        shift = 2 if len(fields) > 2 else 1
        return [
            Domain(fields[(fields.index(float(domain.field_strength_t)) + shift) % len(fields)], domain.contrast)
            if float(domain.field_strength_t) in fields
            else domain
            for domain in domains
        ]
    return [Domain(domains[(index + 1) % len(domains)].field_strength_t, domain.contrast) for index, domain in enumerate(domains)]


def _build_optional_lpips(mode: LPIPSMode, device: torch.device) -> tuple[nn.Module | None, dict[str, Any]]:
    if mode == "off":
        return None, {"enabled": False, "skipped": True, "reason": "disabled"}
    try:
        from fieldbridge.training.losses import build_lpips_net

        net = build_lpips_net(device)
    except Exception as exc:
        if mode == "on":
            raise
        return None, {"enabled": False, "skipped": True, "reason": str(exc)}
    return net.eval(), {"enabled": True, "skipped": False, "reason": None}


def _move_batch(batch: PseudoPairSliceBatch, device: torch.device) -> PseudoPairSliceBatch:
    return PseudoPairSliceBatch(
        x_low=batch.x_low.to(device),
        x_high=batch.x_high.to(device),
        mask=batch.mask.to(device),
        source_domain=batch.source_domain,
        target_domain=batch.target_domain,
        record_id=batch.record_id,
        volume_path=batch.volume_path,
        slice_index=batch.slice_index.to(device),
        degradation_seed=batch.degradation_seed,
        degradation_strength=batch.degradation_strength.to(device),
        geometry=batch.geometry,
    )


def _coerce_config(config: PseudoPairEvalConfig | Mapping[str, Any] | None) -> PseudoPairEvalConfig:
    if config is None:
        return PseudoPairEvalConfig()
    if isinstance(config, PseudoPairEvalConfig):
        return config
    return PseudoPairEvalConfig.from_mapping(config)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation.device is 'cuda', but CUDA is not available.")
    if device not in ("cpu", "cuda"):
        raise ValueError("evaluation.device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)
