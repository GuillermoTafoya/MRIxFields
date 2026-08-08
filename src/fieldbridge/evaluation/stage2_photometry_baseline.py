"""Variant-A qualification and method-neutral dual-baseline result contracts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Contrast, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_QUALIFICATION_ELIGIBILITY,
    FrozenPhotometryArtifact,
    canonical_tensor_sha256,
    deterministic_quantiles,
    sha256_file,
    sha256_json,
    sha256_text,
    validate_qualification_role,
    write_json_atomic,
)
from fieldbridge.evaluation.mrixfields2026_official import (
    official_task3_nrmse,
    official_task3_ssim,
)

VARIANT_A_QUALIFICATION_CONTRACT_VERSION = "stage2-photometry-variant-a-qualification-v1"
VARIANT_A_CONTINUITY_REFERENCE_VERSION = "stage2-photometry-continuity-reference-v1"
VARIANT_A_DUAL_BASELINE_RESULT_VERSION = "stage2-photometry-dual-baseline-result-v1"
VARIANT_A_EVALUATION_SEMANTICS = "separate-fixed-map-and-gate01-posthoc-method-identities-v1"

FIXED_MAP_METHOD = "fixed_map_factorized_identity"
GATE01_POSTHOC_METHOD = "gate01_posthoc_calibrated_identity"
RAW_IDENTITY_METHOD = "raw_identity"
STAGE1_CEILING_METHOD = "stage1_reconstruction_ceiling"
CONTINUITY_METHODS = (
    GATE01_POSTHOC_METHOD,
    RAW_IDENTITY_METHOD,
    STAGE1_CEILING_METHOD,
)

RoundTrip = Callable[[torch.Tensor, Domain], torch.Tensor]
MetricFunction = Callable[[torch.Tensor, torch.Tensor], Mapping[str, float]]
ProgressFunction = Callable[[int, int | None, str], None]


@dataclass(frozen=True, slots=True)
class VariantAQualificationThresholds:
    """Versioned proposed defaults; these are not universal physical constants."""

    monotonic_tolerance_fraction: float = 1e-7
    roundtrip_macro_nrmse_max: float = 0.01
    roundtrip_worst_domain_nrmse_max: float = 0.02
    roundtrip_p99_range_fraction_max: float = 0.03
    histogram_macro_distance_max: float = 0.02
    histogram_worst_field_distance_max: float = 0.03
    spearman_min: float = 0.995
    contrast_macro_f1_drop_max: float = 0.02
    contrast_recall_drop_max: float = 0.05
    scaling_factors: tuple[float, ...] = (0.9, 1.1)
    scaling_histogram_distance_max: float = 0.03
    scaling_ssim_min: float = 0.98
    vae_nrmse_absolute_increase_max: float = 0.02
    vae_nrmse_relative_increase_max: float = 0.15
    vae_ssim_decrease_max: float = 0.01
    vae_lpips_increase_max: float = 0.005
    vae_domain_nrmse_increase_max: float = 0.03
    vae_domain_ssim_decrease_max: float = 0.015
    vae_domain_lpips_increase_max: float = 0.008

    def __post_init__(self) -> None:
        values = self.to_dict()
        for name, value in values.items():
            if name == "scaling_factors":
                if not value or any(
                    not math.isfinite(float(item)) or float(item) <= 0
                    for item in value
                ):
                    raise ValueError("Variant-A scaling factors must be finite and positive.")
            elif not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Variant-A threshold {name} must be finite and non-negative.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "VariantAQualificationThresholds":
        qualification = config.get("qualification", {})
        if not isinstance(qualification, Mapping):
            raise ValueError("Variant-A config qualification section must be a mapping.")
        thresholds = qualification.get("thresholds", {})
        if not isinstance(thresholds, Mapping):
            raise ValueError("Variant-A qualification thresholds must be a mapping.")
        scaling = qualification.get("scaling_factors", (0.9, 1.1))
        if not isinstance(scaling, Sequence) or isinstance(scaling, (str, bytes)):
            raise ValueError("Variant-A scaling_factors must be a sequence.")
        defaults = cls()
        values = defaults.to_dict()
        values.update({str(key): value for key, value in thresholds.items()})
        values["scaling_factors"] = tuple(float(value) for value in scaling)
        unknown = sorted(set(values) - set(defaults.to_dict()))
        if unknown:
            raise ValueError(f"Unknown Variant-A qualification thresholds: {unknown}.")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monotonic_tolerance_fraction": float(self.monotonic_tolerance_fraction),
            "roundtrip_macro_nrmse_max": float(self.roundtrip_macro_nrmse_max),
            "roundtrip_worst_domain_nrmse_max": float(self.roundtrip_worst_domain_nrmse_max),
            "roundtrip_p99_range_fraction_max": float(self.roundtrip_p99_range_fraction_max),
            "histogram_macro_distance_max": float(self.histogram_macro_distance_max),
            "histogram_worst_field_distance_max": float(self.histogram_worst_field_distance_max),
            "spearman_min": float(self.spearman_min),
            "contrast_macro_f1_drop_max": float(self.contrast_macro_f1_drop_max),
            "contrast_recall_drop_max": float(self.contrast_recall_drop_max),
            "scaling_factors": [float(value) for value in self.scaling_factors],
            "scaling_histogram_distance_max": float(self.scaling_histogram_distance_max),
            "scaling_ssim_min": float(self.scaling_ssim_min),
            "vae_nrmse_absolute_increase_max": float(self.vae_nrmse_absolute_increase_max),
            "vae_nrmse_relative_increase_max": float(self.vae_nrmse_relative_increase_max),
            "vae_ssim_decrease_max": float(self.vae_ssim_decrease_max),
            "vae_lpips_increase_max": float(self.vae_lpips_increase_max),
            "vae_domain_nrmse_increase_max": float(self.vae_domain_nrmse_increase_max),
            "vae_domain_ssim_decrease_max": float(self.vae_domain_ssim_decrease_max),
            "vae_domain_lpips_increase_max": float(self.vae_domain_lpips_increase_max),
        }


@dataclass(frozen=True, slots=True)
class QualificationVolume:
    volume: torch.Tensor
    domain: Domain
    record_identity: str
    subject_identity: str | None
    source_path_identity: str
    source_file_sha256: str
    split: str = "validation"
    cohort: str = "R"


@dataclass(frozen=True, slots=True)
class BaselineEvaluationCase:
    case_identity: str
    selection_identity: str
    source: torch.Tensor
    target: torch.Tensor
    source_domain: Domain
    target_domain: Domain


@dataclass(frozen=True, slots=True)
class ContinuityReference:
    """Hash-bound Gate 0.1/raw/ceiling metrics supplied without rerunning Gate 0.1."""

    evaluation_identity: str
    source_result_sha256: str
    methods: Mapping[str, Mapping[str, float]]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.evaluation_identity:
            raise ValueError("Continuity reference requires an evaluation identity.")
        _require_sha256(self.source_result_sha256, "continuity source result")
        if set(self.methods) != set(CONTINUITY_METHODS):
            raise ValueError(
                "Continuity reference must contain Gate 0.1 calibrated identity, raw identity, "
                "and Stage 1 reconstruction ceiling as separate methods."
            )
        for label, metrics in self.methods.items():
            _validate_metric_mapping(metrics, label)
        if not self.provenance:
            raise ValueError("Continuity reference requires source provenance.")

    @property
    def artifact_sha256(self) -> str:
        return str(self.to_dict()["artifact_sha256"])

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": VARIANT_A_CONTINUITY_REFERENCE_VERSION,
            "evaluation_identity": self.evaluation_identity,
            "source_result_sha256": self.source_result_sha256,
            "methods": {
                label: {key: float(value) for key, value in sorted(metrics.items())}
                for label, metrics in sorted(self.methods.items())
            },
            "provenance": dict(self.provenance),
            "calibration_semantics": {
                GATE01_POSTHOC_METHOD: "unchanged Gate 0.1 prediction-CDF diagnostic",
                RAW_IDENTITY_METHOD: "no calibration",
                STAGE1_CEILING_METHOD: "frozen Stage 1 reconstruction reference",
            },
        }
        payload["artifact_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContinuityReference":
        if payload.get("contract_version") != VARIANT_A_CONTINUITY_REFERENCE_VERSION:
            raise ValueError("Unsupported Variant-A continuity-reference contract.")
        methods_raw = payload.get("methods", {})
        if not isinstance(methods_raw, Mapping):
            raise ValueError("Continuity reference methods must be a mapping.")
        reference = cls(
            evaluation_identity=str(payload.get("evaluation_identity", "")),
            source_result_sha256=str(payload.get("source_result_sha256", "")),
            methods={
                str(label): {str(key): float(value) for key, value in dict(metrics).items()}
                for label, metrics in methods_raw.items()
            },
            provenance=dict(payload.get("provenance", {})),
        )
        if payload.get("artifact_sha256") != reference.artifact_sha256:
            raise ValueError("Continuity-reference artifact hash mismatch.")
        return reference

    @classmethod
    def load(
        cls, path: str | Path, *, source_result_path: str | Path
    ) -> "ContinuityReference":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load continuity reference {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Continuity reference root must be a JSON object.")
        reference = cls.from_dict(payload)
        actual_source_hash = sha256_file(source_result_path)
        if actual_source_hash != reference.source_result_sha256:
            raise ValueError("Continuity source-result SHA-256 mismatch.")
        return reference


def qualify_variant_a(
    artifact: FrozenPhotometryArtifact,
    volumes: Iterable[QualificationVolume],
    *,
    thresholds: VariantAQualificationThresholds,
    resolved_config: Mapping[str, Any],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    vae_roundtrip: RoundTrip,
    vae_provenance: Mapping[str, Any],
    continuity: ContinuityReference,
    metric_function: MetricFunction,
    progress: ProgressFunction | None = None,
) -> dict[str, Any]:
    """Run all training-independent Variant-A qualification arithmetic."""

    if source_split_file_sha256 != artifact.provenance["source_split_file_sha256"]:
        raise ValueError("Qualification split-file SHA-256 does not match the artifact.")
    if source_membership_fingerprint != artifact.split_fingerprint:
        raise ValueError("Qualification membership fingerprint does not match the artifact.")
    if source_recovery_fingerprint != artifact.recovery_fingerprint:
        raise ValueError("Qualification recovery fingerprint does not match the artifact.")
    config_payload = _json_safe_mapping(resolved_config)
    if config_payload.get("contract") != "stage2-photometry-variant-a-config-v1":
        raise ValueError("Qualification config contract is incompatible.")
    if artifact.provenance["resolved_config_sha256"] != sha256_json(config_payload):
        raise ValueError("Qualification config does not match the fitted artifact.")
    vae_payload = _json_safe_mapping(vae_provenance)
    _require_sha256(str(vae_payload.get("checkpoint_sha256", "")), "VAE checkpoint")
    _require_sha256(str(vae_payload.get("config_file_sha256", "")), "VAE config file")

    rows: list[dict[str, Any]] = []
    raw_features: list[torch.Tensor] = []
    canonical_features: list[torch.Tensor] = []
    contrast_labels: list[str] = []
    identities: set[str] = set()
    for index, item in enumerate(volumes, start=1):
        validate_qualification_role(
            record_identity=item.record_identity,
            subject_identity=item.subject_identity,
            cohort=item.cohort,
            split=item.split,
        )
        if item.record_identity in identities:
            raise ValueError("Qualification received a duplicate record identity.")
        identities.add(item.record_identity)
        _require_sha256(item.source_file_sha256, "qualification source file")
        source = _validate_volume(item.volume, item.record_identity)
        canonical = artifact.normalize_source(source, item.domain)
        if not bool(canonical.support_mask.any()):
            raise ValueError(f"Qualification volume {item.record_identity!r} has empty support.")
        direct = artifact.render_target(canonical, item.domain)
        exact_zero = bool((canonical.values[~canonical.support_mask] == 0).all()) and bool(
            (direct[~canonical.support_mask] == 0).all()
        )
        direct_metrics = _direct_roundtrip_metrics(
            source, direct, canonical.support_mask, artifact, item.domain
        )
        histogram_distance = _canonical_histogram_distance(
            canonical.values, canonical.support_mask, artifact, item.domain
        )
        spearman = _spearman(
            source[canonical.support_mask], canonical.values[canonical.support_mask]
        )
        scaling = _scaling_measurements(
            source,
            canonical.values,
            canonical.support_mask,
            artifact,
            item.domain,
            thresholds,
        )

        raw_reconstruction = _validate_reconstruction(
            vae_roundtrip(source, item.domain), source, "raw VAE reconstruction"
        )
        canonical_decoded = _validate_reconstruction(
            vae_roundtrip(canonical.values, item.domain), source, "canonical VAE reconstruction"
        )
        factorized_reconstruction = artifact.render_target(
            canonical.with_values(canonical_decoded), item.domain
        )
        raw_metrics = _finite_metrics(
            metric_function(raw_reconstruction, source), "raw reconstruction"
        )
        factorized_metrics = _finite_metrics(
            metric_function(factorized_reconstruction, source), "factorized reconstruction"
        )
        vae_deltas = _vae_deltas(raw_metrics, factorized_metrics)

        raw_features.append(_contrast_features(source, canonical.support_mask))
        canonical_features.append(_contrast_features(canonical.values, canonical.support_mask))
        contrast_labels.append(item.domain.contrast.value)
        rows.append(
            {
                "record_identity": item.record_identity,
                "record_identity_sha256": sha256_text(item.record_identity),
                "subject_identity": item.subject_identity,
                "source_path_identity_sha256": sha256_text(item.source_path_identity),
                "source_file_sha256": item.source_file_sha256,
                "canonical_loaded_array_sha256": canonical_tensor_sha256(source),
                "domain": item.domain.label,
                "cohort": item.cohort,
                "split": item.split,
                "exact_zero_support": exact_zero,
                "direct_roundtrip": direct_metrics,
                "canonical_histogram_distance": histogram_distance,
                "spearman": spearman,
                "scaling_sensitivity": scaling,
                "vae": {
                    "raw_reconstruction": raw_metrics,
                    "factorized_reconstruction": factorized_metrics,
                    "deltas": vae_deltas,
                },
            }
        )
        if progress is not None:
            progress(index, None, item.record_identity)

    _require_qualification_domains(rows)
    contrast_control = _contrast_preservation_control(
        raw_features, canonical_features, contrast_labels
    )
    interpolation = _interpolation_qualification(artifact, thresholds)
    aggregate = _aggregate_qualification(
        rows,
        contrast_control,
        interpolation,
        thresholds,
    )
    stage1_continuity = _stage1_continuity_comparison(aggregate, continuity)
    failures: list[str] = []
    if not aggregate["photometry_factorization_pass"]:
        failures.append("photometry_factorization_failure")
    if not aggregate["canonical_vae_compatibility_pass"]:
        failures.append("canonical_vae_distribution_shift_failure")
    rows.sort(key=lambda row: (row["domain"], row["record_identity"]))
    result: dict[str, Any] = {
        "contract_version": VARIANT_A_QUALIFICATION_CONTRACT_VERSION,
        "artifact_sha256": artifact.artifact_sha256,
        "eligibility_rule": PHOTOMETRY_QUALIFICATION_ELIGIBILITY,
        "source_split": {
            "file_sha256": source_split_file_sha256,
            "membership_fingerprint": source_membership_fingerprint,
            "recovery_fingerprint": source_recovery_fingerprint,
        },
        "resolved_config": config_payload,
        "resolved_config_sha256": sha256_json(config_payload),
        "thresholds": thresholds.to_dict(),
        "threshold_status": "versioned-proposed-defaults-requiring-external-review",
        "vae_provenance": vae_payload,
        "stage1_reconstruction_ceiling_continuity": stage1_continuity,
        "records": rows,
        "record_count": len(rows),
        "eligibility_proof": {
            "all_cohort_R": all(row["cohort"] == "R" for row in rows),
            "all_split_validation": all(row["split"] == "validation" for row in rows),
            "prospective_accepted_count": 0,
            "forbidden_traveller_accepted_count": 0,
        },
        "fit_weighting_evidence": {
            "domain_volume_counts": dict(artifact.provenance["domain_volume_counts"]),
            "weighting": dict(artifact.provenance["weighting"]),
        },
        "interpolation_qualification": interpolation,
        "contrast_preservation_control": contrast_control,
        "aggregate": aggregate,
        "failure_classification": failures,
        "canonical_latent_bank_authorized": not failures,
        "gate01_substitution_forbidden": True,
    }
    result["result_sha256"] = sha256_json(result)
    return result


def evaluate_factorized_identity_case(
    artifact: FrozenPhotometryArtifact,
    case: BaselineEvaluationCase,
    *,
    continuity: ContinuityReference,
    metric_function: MetricFunction,
) -> dict[str, Any]:
    """Evaluate one directed fixed-map identity without rerunning Gate 0.1."""

    if case.selection_identity != continuity.evaluation_identity:
        raise ValueError("Fixed-map case and continuity evaluation identities do not match.")
    if case.source_domain.contrast != case.target_domain.contrast:
        raise ValueError("Variant-A evaluation supports same-contrast edges only.")
    if case.source_domain.field_strength_t == case.target_domain.field_strength_t:
        raise ValueError("Variant-A directed evaluation requires a cross-field edge.")
    source = _validate_volume(case.source, "evaluation source")
    prediction = artifact.factorized_identity(source, case.source_domain, case.target_domain)
    target = _validate_reconstruction(case.target, source, "evaluation target")
    fixed_metrics = _finite_metrics(metric_function(prediction, target), FIXED_MAP_METHOD)
    methods: dict[str, Any] = {
        FIXED_MAP_METHOD: {
            "metrics": fixed_metrics,
            "semantics": "fixed P_t(N_s(x_s)); no runtime prediction CDF",
            "prediction_sha256": canonical_tensor_sha256(prediction),
        }
    }
    for label in CONTINUITY_METHODS:
        methods[label] = {
            "metrics": dict(continuity.methods[label]),
            "semantics": (
                "unchanged Gate 0.1 posthoc prediction-CDF diagnostic"
                if label == GATE01_POSTHOC_METHOD
                else "unmodified frozen continuity reference"
            ),
            "continuity_reference_sha256": continuity.artifact_sha256,
        }
    result: dict[str, Any] = {
        "contract_version": VARIANT_A_DUAL_BASELINE_RESULT_VERSION,
        "semantics": VARIANT_A_EVALUATION_SEMANTICS,
        "case_identity": case.case_identity,
        "selection_identity": case.selection_identity,
        "source_domain": case.source_domain.to_dict(),
        "target_domain": case.target_domain.to_dict(),
        "artifact_sha256": artifact.artifact_sha256,
        "continuity_reference_sha256": continuity.artifact_sha256,
        "continuity_source_result_sha256": continuity.source_result_sha256,
        "methods": methods,
        "method_identity_invariant": (
            "fixed-map and Gate 0.1 posthoc calibrated identities are separate and "
            "must not be averaged, substituted, or described as equivalent"
        ),
    }
    result["result_sha256"] = sha256_json(result)
    return result


def write_variant_a_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    return write_json_atomic(path, result, refuse_existing=True)


class OfficialQualificationMetrics:
    """Reusable official nRMSE/SSIM/LPIPS evaluator for external qualification."""

    def __init__(
        self,
        *,
        metrics: Sequence[str] = ("nrmse", "ssim", "lpips"),
        device: str = "cuda",
    ) -> None:
        requested = tuple(dict.fromkeys(str(value) for value in metrics))
        if set(requested) - {"nrmse", "ssim", "lpips"}:
            raise ValueError(f"Unsupported qualification metrics: {requested}.")
        self.metrics = requested
        self.device = device
        self._lpips_network: Any | None = None
        self._lpips_device: str | None = None

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> Mapping[str, float]:
        pred = _spatial_numpy(prediction)
        tgt = _spatial_numpy(target)
        result: dict[str, float] = {}
        if "nrmse" in self.metrics:
            result["nrmse"] = official_task3_nrmse(pred, tgt)
        if "ssim" in self.metrics:
            result["ssim"] = official_task3_ssim(pred, tgt)
        if "lpips" in self.metrics:
            result["lpips"] = self._lpips(pred, tgt)
        return result

    def _lpips(self, prediction: np.ndarray, target: np.ndarray) -> float:
        if self._lpips_network is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "Variant-A VAE qualification requires the 'official-evaluation' "
                    "extra for LPIPS."
                ) from exc
            from fieldbridge.evaluation.mrixfields2026_official import resolve_official_lpips_device

            self._lpips_device = resolve_official_lpips_device(self.device)
            self._lpips_network = lpips.LPIPS(net="alex").to(self._lpips_device).eval()
        assert self._lpips_device is not None
        network = self._lpips_network
        pred = prediction.astype(np.float64) * 2.0 - 1.0
        tgt = target.astype(np.float64) * 2.0 - 1.0
        values: list[float] = []
        for index in range(pred.shape[2]):
            pred_slice = pred[:, :, index]
            tgt_slice = tgt[:, :, index]
            if np.abs(tgt_slice).max() < 1e-10:
                continue
            pred_tensor = (
                torch.from_numpy(pred_slice)
                .float()[None, None]
                .repeat(1, 3, 1, 1)
                .to(self._lpips_device)
            )
            target_tensor = (
                torch.from_numpy(tgt_slice)
                .float()[None, None]
                .repeat(1, 3, 1, 1)
                .to(self._lpips_device)
            )
            with torch.inference_mode():
                values.append(float(network(pred_tensor, target_tensor).item()))
        return float(np.mean(values)) if values else 0.0


def _aggregate_qualification(
    rows: Sequence[Mapping[str, Any]],
    contrast_control: Mapping[str, Any],
    interpolation: Mapping[str, Any],
    thresholds: VariantAQualificationThresholds,
) -> dict[str, Any]:
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    per_domain: dict[str, Any] = {}
    for label, items in sorted(by_domain.items()):
        per_domain[label] = {
            "count": len(items),
            "direct_roundtrip_nrmse": _mean(item["direct_roundtrip"]["nrmse"] for item in items),
            "direct_roundtrip_p99_range_fraction": _mean(
                item["direct_roundtrip"]["p99_range_fraction"] for item in items
            ),
            "canonical_histogram_distance": _mean(
                item["canonical_histogram_distance"] for item in items
            ),
            "spearman": _mean(item["spearman"] for item in items),
            "scaling_histogram_distance_worst": max(
                entry["histogram_distance"]
                for item in items
                for entry in item["scaling_sensitivity"]
            ),
            "scaling_ssim_worst": min(
                entry["ssim"] for item in items for entry in item["scaling_sensitivity"]
            ),
            "vae_deltas": {
                key: _mean(item["vae"]["deltas"][key] for item in items)
                for key in (
                    "nrmse_absolute_increase",
                    "nrmse_relative_increase",
                    "ssim_decrease",
                    "lpips_increase",
                )
            },
            "vae_raw_reconstruction": {
                key: _mean(item["vae"]["raw_reconstruction"][key] for item in items)
                for key in ("nrmse", "ssim", "lpips")
            },
            "vae_factorized_reconstruction": {
                key: _mean(
                    item["vae"]["factorized_reconstruction"][key]
                    for item in items
                )
                for key in ("nrmse", "ssim", "lpips")
            },
        }
    macro = {
        "direct_roundtrip_nrmse": _mean(
            item["direct_roundtrip_nrmse"] for item in per_domain.values()
        ),
        "canonical_histogram_distance": _mean(
            item["canonical_histogram_distance"] for item in per_domain.values()
        ),
        "vae_deltas": {
            key: _mean(item["vae_deltas"][key] for item in per_domain.values())
            for key in (
                "nrmse_absolute_increase",
                "nrmse_relative_increase",
                "ssim_decrease",
                "lpips_increase",
            )
        },
        "vae_raw_reconstruction": {
            key: _mean(item["vae_raw_reconstruction"][key] for item in per_domain.values())
            for key in ("nrmse", "ssim", "lpips")
        },
        "vae_factorized_reconstruction": {
            key: _mean(
                item["vae_factorized_reconstruction"][key]
                for item in per_domain.values()
            )
            for key in ("nrmse", "ssim", "lpips")
        },
    }
    worst = {
        "direct_roundtrip_nrmse": max(
            item["direct_roundtrip_nrmse"] for item in per_domain.values()
        ),
        "direct_roundtrip_p99_range_fraction": max(
            item["direct_roundtrip_p99_range_fraction"] for item in per_domain.values()
        ),
        "canonical_histogram_distance": max(
            item["canonical_histogram_distance"] for item in per_domain.values()
        ),
        "spearman": min(item["spearman"] for item in per_domain.values()),
        "scaling_histogram_distance": max(
            item["scaling_histogram_distance_worst"] for item in per_domain.values()
        ),
        "scaling_ssim": min(item["scaling_ssim_worst"] for item in per_domain.values()),
        "vae_domain_deltas": {
            "nrmse_absolute_increase": max(
                item["vae_deltas"]["nrmse_absolute_increase"]
                for item in per_domain.values()
            ),
            "ssim_decrease": max(
                item["vae_deltas"]["ssim_decrease"] for item in per_domain.values()
            ),
            "lpips_increase": max(
                item["vae_deltas"]["lpips_increase"] for item in per_domain.values()
            ),
        },
    }
    exact_zero = all(bool(item["exact_zero_support"]) for item in rows)
    photometry_checks = {
        "interpolation_finite_monotonic": bool(interpolation["pass"]),
        "exact_zero_support": exact_zero,
        "direct_roundtrip_macro": (
            macro["direct_roundtrip_nrmse"]
            <= thresholds.roundtrip_macro_nrmse_max
        ),
        "direct_roundtrip_worst_domain": (
            worst["direct_roundtrip_nrmse"]
            <= thresholds.roundtrip_worst_domain_nrmse_max
        ),
        "direct_roundtrip_p99": (
            worst["direct_roundtrip_p99_range_fraction"]
            <= thresholds.roundtrip_p99_range_fraction_max
        ),
        "canonical_histogram_macro": (
            macro["canonical_histogram_distance"]
            <= thresholds.histogram_macro_distance_max
        ),
        "canonical_histogram_worst_field": (
            worst["canonical_histogram_distance"]
            <= thresholds.histogram_worst_field_distance_max
        ),
        "spearman": worst["spearman"] >= thresholds.spearman_min,
        "contrast_macro_f1": (
            contrast_control["macro_f1_drop"]
            <= thresholds.contrast_macro_f1_drop_max
        ),
        "contrast_recall": (
            contrast_control["worst_recall_drop"]
            <= thresholds.contrast_recall_drop_max
        ),
        "scaling_histogram": (
            worst["scaling_histogram_distance"]
            <= thresholds.scaling_histogram_distance_max
        ),
        "scaling_ssim": worst["scaling_ssim"] >= thresholds.scaling_ssim_min,
        "per_domain_balance": set(per_domain) == set(_all_domain_labels()),
    }
    vae_checks = {
        "macro_nrmse_absolute": (
            macro["vae_deltas"]["nrmse_absolute_increase"]
            <= thresholds.vae_nrmse_absolute_increase_max
        ),
        "macro_nrmse_relative": (
            macro["vae_deltas"]["nrmse_relative_increase"]
            <= thresholds.vae_nrmse_relative_increase_max
        ),
        "macro_ssim": macro["vae_deltas"]["ssim_decrease"] <= thresholds.vae_ssim_decrease_max,
        "macro_lpips": macro["vae_deltas"]["lpips_increase"] <= thresholds.vae_lpips_increase_max,
        "worst_domain_nrmse": (
            worst["vae_domain_deltas"]["nrmse_absolute_increase"]
            <= thresholds.vae_domain_nrmse_increase_max
        ),
        "worst_domain_ssim": (
            worst["vae_domain_deltas"]["ssim_decrease"]
            <= thresholds.vae_domain_ssim_decrease_max
        ),
        "worst_domain_lpips": (
            worst["vae_domain_deltas"]["lpips_increase"]
            <= thresholds.vae_domain_lpips_increase_max
        ),
    }
    return {
        "per_domain": per_domain,
        "macro": macro,
        "worst_domain": worst,
        "photometry_checks": photometry_checks,
        "vae_checks": vae_checks,
        "photometry_factorization_pass": all(photometry_checks.values()),
        "canonical_vae_compatibility_pass": all(vae_checks.values()),
    }


def _stage1_continuity_comparison(
    aggregate: Mapping[str, Any], continuity: ContinuityReference
) -> dict[str, Any]:
    ceiling = {
        key: float(value)
        for key, value in continuity.methods[STAGE1_CEILING_METHOD].items()
    }
    raw = aggregate["macro"]["vae_raw_reconstruction"]
    factorized = aggregate["macro"]["vae_factorized_reconstruction"]
    return {
        "method_identity": STAGE1_CEILING_METHOD,
        "metrics": ceiling,
        "continuity_reference_sha256": continuity.artifact_sha256,
        "continuity_source_result_sha256": continuity.source_result_sha256,
        "raw_macro_minus_continuity": {
            key: float(raw[key]) - ceiling[key] for key in ("nrmse", "ssim", "lpips")
        },
        "factorized_macro_minus_continuity": {
            key: float(factorized[key]) - ceiling[key]
            for key in ("nrmse", "ssim", "lpips")
        },
        "interpretation": (
            "external continuity reference only; not a held-out-retrospective observation "
            "and not an additional qualification threshold"
        ),
    }


def _interpolation_qualification(
    artifact: FrozenPhotometryArtifact,
    thresholds: VariantAQualificationThresholds,
) -> dict[str, Any]:
    """Measure stored-grid finiteness and tolerance-scaled monotonicity."""

    grids: list[tuple[str, torch.Tensor]] = []
    grids.extend(
        (f"domain:{label}", template.quantiles)
        for label, template in artifact.domain_templates.items()
    )
    grids.extend(
        (f"canonical:{label}", template.quantiles)
        for label, template in artifact.canonical_templates.items()
    )
    rows: list[dict[str, Any]] = []
    for label, values in sorted(grids):
        tensor = values.detach().cpu().to(torch.float64)
        finite = bool(torch.isfinite(tensor).all())
        value_range = max(float(tensor[-1] - tensor[0]), 1e-12)
        tolerance = thresholds.monotonic_tolerance_fraction * value_range
        minimum_increment = float(torch.diff(tensor).min())
        rows.append(
            {
                "grid": label,
                "finite": finite,
                "minimum_increment": minimum_increment,
                "range": value_range,
                "tolerance": tolerance,
                "monotonic_within_tolerance": minimum_increment >= -tolerance,
            }
        )
    return {
        "threshold_fraction": thresholds.monotonic_tolerance_fraction,
        "grid_count": len(rows),
        "grids": rows,
        "pass": all(
            bool(row["finite"]) and bool(row["monotonic_within_tolerance"])
            for row in rows
        ),
    }


def _direct_roundtrip_metrics(
    source: torch.Tensor,
    direct: torch.Tensor,
    support: torch.Tensor,
    artifact: FrozenPhotometryArtifact,
    domain: Domain,
) -> dict[str, float]:
    source_values = source[support].detach().cpu().to(torch.float64)
    direct_values = direct[support].detach().cpu().to(torch.float64)
    denominator = float(torch.linalg.vector_norm(source_values))
    nrmse = float(torch.linalg.vector_norm(direct_values - source_values)) / max(denominator, 1e-12)
    robust_range = _domain_robust_range(artifact, domain)
    p99 = float(torch.quantile((direct_values - source_values).abs(), 0.99)) / robust_range
    return {"nrmse": nrmse, "p99_range_fraction": p99}


def _canonical_histogram_distance(
    values: torch.Tensor,
    support: torch.Tensor,
    artifact: FrozenPhotometryArtifact,
    domain: Domain,
) -> float:
    observed = deterministic_quantiles(values[support], artifact.probabilities)
    canonical = artifact.canonical_templates[domain.contrast.value].quantiles
    return float((observed - canonical).abs().mean()) / _canonical_robust_range(
        artifact, domain.contrast
    )


def _scaling_measurements(
    source: torch.Tensor,
    unperturbed: torch.Tensor,
    support: torch.Tensor,
    artifact: FrozenPhotometryArtifact,
    domain: Domain,
    thresholds: VariantAQualificationThresholds,
) -> list[dict[str, float]]:
    base_quantiles = deterministic_quantiles(unperturbed[support], artifact.probabilities)
    scale = _canonical_robust_range(artifact, domain.contrast)
    rows: list[dict[str, float]] = []
    for factor in thresholds.scaling_factors:
        perturbed_source = (source * factor).clamp(0.0, 1.0)
        perturbed = artifact.normalize_source(perturbed_source, domain)
        if not torch.equal(perturbed.support_mask, support):
            raise ValueError("Scaling perturbation changed the sealed nonzero source support.")
        quantiles = deterministic_quantiles(perturbed.values[support], artifact.probabilities)
        rows.append(
            {
                "factor": float(factor),
                "histogram_distance": float((quantiles - base_quantiles).abs().mean()) / scale,
                "ssim": _masked_global_ssim(perturbed.values[support], unperturbed[support]),
            }
        )
    return rows


def _contrast_preservation_control(
    raw_features: Sequence[torch.Tensor],
    canonical_features: Sequence[torch.Tensor],
    labels: Sequence[str],
) -> dict[str, Any]:
    raw = _loo_nearest_centroid(torch.stack(list(raw_features)), labels)
    canonical = _loo_nearest_centroid(torch.stack(list(canonical_features)), labels)
    recalls = {
        contrast.value: raw["recall"][contrast.value] - canonical["recall"][contrast.value]
        for contrast in CONTRASTS
    }
    return {
        "control": (
            "leave-one-out nearest-centroid on photometry-controlled "
            "value/gradient quantiles"
        ),
        "raw": raw,
        "canonical": canonical,
        "macro_f1_drop": raw["macro_f1"] - canonical["macro_f1"],
        "recall_drop_by_contrast": recalls,
        "worst_recall_drop": max(recalls.values()),
    }


def _contrast_features(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    work = values.detach().cpu().to(torch.float64)
    mask = support.detach().cpu()
    selected = work[mask]
    q = torch.quantile(selected, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64))
    scale = max(float(q[3] - q[1]), 1e-8)
    normalized_q = (q - q[2]) / scale
    gradient_features: list[torch.Tensor] = []
    for dim in range(max(0, work.ndim - 3), work.ndim):
        left = mask.narrow(dim, 0, mask.shape[dim] - 1)
        right = mask.narrow(dim, 1, mask.shape[dim] - 1)
        valid = left & right
        gradients = work.diff(dim=dim).abs()[valid] / scale
        if gradients.numel() == 0:
            gradient_features.append(torch.zeros(3, dtype=torch.float64))
        else:
            gradient_features.append(
                torch.quantile(gradients, torch.tensor([0.5, 0.75, 0.9], dtype=torch.float64))
            )
    return torch.cat([normalized_q, *gradient_features])


def _loo_nearest_centroid(features: torch.Tensor, labels: Sequence[str]) -> dict[str, Any]:
    expected = {contrast.value for contrast in CONTRASTS}
    if set(labels) != expected:
        raise ValueError("Contrast preservation requires all three contrasts.")
    predictions: list[str] = []
    for index, feature in enumerate(features):
        distances: list[tuple[float, str]] = []
        for label in sorted(expected):
            members = [
                features[j]
                for j, value in enumerate(labels)
                if value == label and j != index
            ]
            if not members:
                raise ValueError(
                    "Contrast preservation requires at least two records per contrast."
                )
            centroid = torch.stack(members).mean(dim=0)
            distances.append((float(torch.linalg.vector_norm(feature - centroid)), label))
        predictions.append(min(distances)[1])
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for label in sorted(expected):
        tp = sum(
            actual == label and predicted == label
            for actual, predicted in zip(labels, predictions)
        )
        fn = sum(
            actual == label and predicted != label
            for actual, predicted in zip(labels, predictions)
        )
        fp = sum(
            actual != label and predicted == label
            for actual, predicted in zip(labels, predictions)
        )
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        recalls[label] = recall
        f1_values.append(f1)
    return {"macro_f1": _mean(f1_values), "recall": recalls}


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    first_rank = _average_ranks(first.detach().cpu().to(torch.float64))
    second_rank = _average_ranks(second.detach().cpu().to(torch.float64))
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = float(
        torch.linalg.vector_norm(first_centered)
        * torch.linalg.vector_norm(second_centered)
    )
    if denominator <= 1e-12:
        return 1.0 if torch.equal(first, second) else 0.0
    return float(torch.dot(first_centered, second_centered)) / denominator


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values.reshape(-1), stable=True)
    unique, inverse, counts = torch.unique_consecutive(
        sorted_values, return_inverse=True, return_counts=True
    )
    del unique
    starts = torch.cumsum(counts, 0) - counts
    average = starts.to(torch.float64) + (counts.to(torch.float64) - 1.0) / 2.0
    sorted_ranks = average[inverse]
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _masked_global_ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    x = first.detach().cpu().to(torch.float64)
    y = second.detach().cpu().to(torch.float64)
    dynamic = max(float(y.max() - y.min()), 1e-6)
    c1 = (0.01 * dynamic) ** 2
    c2 = (0.03 * dynamic) ** 2
    mean_x, mean_y = x.mean(), y.mean()
    var_x = ((x - mean_x) ** 2).mean()
    var_y = ((y - mean_y) ** 2).mean()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(value.clamp(-1.0, 1.0))


def _vae_deltas(raw: Mapping[str, float], factorized: Mapping[str, float]) -> dict[str, float]:
    for key in ("nrmse", "ssim", "lpips"):
        if key not in raw or key not in factorized:
            raise ValueError("VAE qualification requires nrmse, ssim, and lpips metrics.")
    absolute = factorized["nrmse"] - raw["nrmse"]
    return {
        "nrmse_absolute_increase": absolute,
        "nrmse_relative_increase": absolute / max(abs(raw["nrmse"]), 1e-12),
        "ssim_decrease": raw["ssim"] - factorized["ssim"],
        "lpips_increase": factorized["lpips"] - raw["lpips"],
    }


def _domain_robust_range(artifact: FrozenPhotometryArtifact, domain: Domain) -> float:
    return _quantile_range(
        artifact.domain_templates[domain.label].quantiles, artifact.probabilities
    )


def _canonical_robust_range(artifact: FrozenPhotometryArtifact, contrast: Contrast) -> float:
    return _quantile_range(
        artifact.canonical_templates[contrast.value].quantiles, artifact.probabilities
    )


def _quantile_range(quantiles: torch.Tensor, probabilities: torch.Tensor) -> float:
    low = int(torch.argmin((probabilities - 0.01).abs()))
    high = int(torch.argmin((probabilities - 0.99).abs()))
    return max(float(quantiles[high] - quantiles[low]), 1e-12)


def _require_qualification_domains(rows: Sequence[Mapping[str, Any]]) -> None:
    labels = {str(row["domain"]) for row in rows}
    expected = set(_all_domain_labels())
    if labels != expected:
        raise ValueError(
            "Variant-A qualification requires all 15 retrospective validation domains; "
            f"missing={sorted(expected - labels)}, unexpected={sorted(labels - expected)}."
        )


def _all_domain_labels() -> tuple[str, ...]:
    return tuple(
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    )


def _validate_volume(volume: torch.Tensor, identity: str) -> torch.Tensor:
    if (
        not isinstance(volume, torch.Tensor)
        or volume.ndim < 3
        or not volume.dtype.is_floating_point
    ):
        raise ValueError(f"Qualification volume {identity!r} must be a floating full volume.")
    if not bool(torch.isfinite(volume).all()):
        raise ValueError(f"Qualification volume {identity!r} contains non-finite values.")
    if float(volume.min()) < 0.0 or float(volume.max()) > 1.0:
        raise ValueError(
            f"Qualification volume {identity!r} violates the official [0,1] range."
        )
    return volume


def _validate_reconstruction(
    value: torch.Tensor, reference: torch.Tensor, name: str
) -> torch.Tensor:
    value = _validate_volume(value, name)
    if tuple(value.shape) != tuple(reference.shape):
        raise ValueError(
            f"{name} shape mismatch: {tuple(value.shape)} != {tuple(reference.shape)}."
        )
    return value


def _finite_metrics(values: Mapping[str, float], name: str) -> dict[str, float]:
    result = {str(key): float(value) for key, value in values.items()}
    _validate_metric_mapping(result, name)
    return result


def _validate_metric_mapping(values: Mapping[str, float], name: str) -> None:
    if not values:
        raise ValueError(f"Metric mapping {name!r} is empty.")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"Metric mapping {name!r} contains non-finite values.")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} SHA-256 is invalid.")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if not isinstance(decoded, dict):
        raise TypeError("Resolved Variant-A config must be a mapping.")
    return decoded


def _spatial_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().to(torch.float32).numpy()
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Official Variant-A metrics require one 3-D volume, got {array.shape}.")
    return np.asarray(array, dtype=np.float32)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot average an empty measurement set.")
    return float(sum(items) / len(items))


__all__ = [
    "CONTINUITY_METHODS",
    "FIXED_MAP_METHOD",
    "GATE01_POSTHOC_METHOD",
    "RAW_IDENTITY_METHOD",
    "STAGE1_CEILING_METHOD",
    "VARIANT_A_CONTINUITY_REFERENCE_VERSION",
    "VARIANT_A_DUAL_BASELINE_RESULT_VERSION",
    "VARIANT_A_EVALUATION_SEMANTICS",
    "VARIANT_A_QUALIFICATION_CONTRACT_VERSION",
    "BaselineEvaluationCase",
    "ContinuityReference",
    "OfficialQualificationMetrics",
    "QualificationVolume",
    "VariantAQualificationThresholds",
    "evaluate_factorized_identity_case",
    "qualify_variant_a",
    "write_variant_a_result",
]
