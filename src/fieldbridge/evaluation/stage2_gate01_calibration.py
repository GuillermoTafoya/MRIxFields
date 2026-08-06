"""Leakage-safe post-hoc target-domain calibration for frozen Gate 0.1.

The calibrator implements operation B from the scientific plan: it ranks the supplied
prediction and maps those ranks onto a frozen target-domain foreground distribution.
Only retrospective training volumes define the target templates.  No paired evaluation
target, target mask, histogram, quantile, or other target-derived statistic is accepted by
the calibration API.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain

GATE01_CALIBRATOR_CONTRACT_VERSION = "stage2-gate01-target-cdf-v1"
GATE01_CALIBRATION_SEMANTICS = "prediction-cdf-to-frozen-training-target-cdf"
CalibrationMode = Literal["histogram", "robust_affine"]

STAGE1_RUN_C_CHECKPOINT_SHA256 = (
    "74132b9c514bb91b86d8eb43c63542780bce11304e31e67d3bf75c90ff5d4d79"
)
FULL_LATENT_BANK_BUILD_COMMIT = "c4b9c399baef588d9547e89100de5036e1ccfcdb"
GATE0_DIAGNOSTIC_COMMIT = "d3476b900866019b428d52d01a6d5b26b93ca65d"
SB_V2_CHECKPOINT_SHA256 = (
    "39c71b5dae702a68d9518376c2d25c13605abd985a5e74e8fb1b4c58d17a1108"
)
RESPLIT_FINGERPRINT = (
    "92187cf5f08ba00c446c08151f0658534efffa917569106a73062fdc70bcaf5f"
)


def all_domain_labels() -> tuple[str, ...]:
    """Return the frozen 5-field x 3-contrast domain set."""

    return tuple(
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    )


@dataclass(frozen=True, slots=True)
class TrainingTemplateVolume:
    """One explicitly training-only volume contributing equally to its domain."""

    volume: torch.Tensor
    domain: Domain
    record_identity: str
    split: str = "train"
    cohort: str = "R"


@dataclass(frozen=True, slots=True)
class TargetDomainTemplate:
    """Equal-volume mean foreground quantiles for one target domain."""

    quantiles: torch.Tensor
    volume_count: int
    template_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantiles": [float(value) for value in self.quantiles],
            "volume_count": int(self.volume_count),
            "template_sha256": self.template_sha256,
        }


_PROVENANCE_KEYS = {
    "split_fingerprint",
    "training_cohort_identity",
    "training_records_sha256",
    "code_commit",
    "config",
    "balancing",
    "domain_volume_counts",
}


@dataclass(frozen=True, slots=True)
class PosthocTargetCalibrator:
    """Method-agnostic prediction-CDF projection onto frozen target templates."""

    probabilities: torch.Tensor
    templates: Mapping[str, TargetDomainTemplate]
    provenance: Mapping[str, Any]
    mask_threshold: float = 0.0
    low_probability: float = 0.01
    high_probability: float = 0.99

    def __post_init__(self) -> None:
        probabilities = self.probabilities.detach().cpu().to(torch.float64)
        if probabilities.ndim != 1 or probabilities.numel() < 3:
            raise ValueError("Gate 0.1 probabilities must be a 1-D grid with at least 3 values.")
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("Gate 0.1 probabilities contain non-finite values.")
        if float(probabilities[0]) != 0.0 or float(probabilities[-1]) != 1.0:
            raise ValueError("Gate 0.1 probability grid must include exact endpoints 0 and 1.")
        if not bool((probabilities[1:] > probabilities[:-1]).all()):
            raise ValueError("Gate 0.1 probability grid must be strictly increasing.")
        if not (0.0 <= self.low_probability < self.high_probability <= 1.0):
            raise ValueError(
                "Gate 0.1 robust-affine probabilities must satisfy "
                "0 <= low < high <= 1."
            )
        if self.mask_threshold < 0 or not math.isfinite(self.mask_threshold):
            raise ValueError(
                "Gate 0.1 mask_threshold must be finite and non-negative."
            )
        object.__setattr__(self, "probabilities", probabilities)

        missing_provenance = sorted(_PROVENANCE_KEYS - set(self.provenance))
        if missing_provenance:
            raise ValueError(
                "Gate 0.1 calibrator provenance is incomplete; missing "
                f"{missing_provenance}."
            )
        expected_labels = set(all_domain_labels())
        actual_labels = set(self.templates)
        if actual_labels != expected_labels:
            raise ValueError(
                "Gate 0.1 requires all 15 domains exactly; "
                f"missing={sorted(expected_labels - actual_labels)}, "
                f"unexpected={sorted(actual_labels - expected_labels)}."
            )
        for label, template in self.templates.items():
            quantiles = template.quantiles.detach().cpu().to(torch.float64)
            if quantiles.shape != probabilities.shape:
                raise ValueError(
                    f"Template {label!r} shape {tuple(quantiles.shape)} does not match "
                    f"probability shape {tuple(probabilities.shape)}."
                )
            if not bool(torch.isfinite(quantiles).all()):
                raise ValueError(f"Template {label!r} contains non-finite quantiles.")
            if not bool((quantiles[1:] >= quantiles[:-1]).all()):
                raise ValueError(f"Template {label!r} quantiles are not monotonic.")
            if template.volume_count <= 0:
                raise ValueError(f"Template {label!r} has no training volumes.")
            expected_hash = _quantile_hash(quantiles)
            if template.template_sha256 != expected_hash:
                raise ValueError(
                    f"Template {label!r} hash mismatch: artifact is stale or altered."
                )

    @property
    def split_fingerprint(self) -> str:
        return str(self.provenance["split_fingerprint"])

    @property
    def template_sha256(self) -> str:
        return _sha256_json(self._template_payload())

    def _template_payload(self) -> dict[str, Any]:
        return {
            "probabilities": [float(value) for value in self.probabilities],
            "mask_threshold": float(self.mask_threshold),
            "low_probability": float(self.low_probability),
            "high_probability": float(self.high_probability),
            "templates": {
                label: template.to_dict()
                for label, template in sorted(self.templates.items())
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract_version": GATE01_CALIBRATOR_CONTRACT_VERSION,
            "semantics": GATE01_CALIBRATION_SEMANTICS,
            "background_contract": "prediction exact-zero voxels remain exact zero",
            "target_independence": {
                "calibration_inputs": ["method_prediction", "requested_target_domain"],
                "template_source": "retrospective training records only",
                "forbidden": [
                    "paired evaluation target image",
                    "target mask",
                    "target histogram",
                    "target quantiles",
                    "target-derived statistic",
                ],
            },
            "template_sha256": self.template_sha256,
            "provenance": dict(self.provenance),
            **self._template_payload(),
        }
        payload["artifact_sha256"] = _sha256_json(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_split_fingerprint: str | None = None,
        expected_template_sha256: str | None = None,
    ) -> "PosthocTargetCalibrator":
        version = str(payload.get("contract_version", ""))
        if version != GATE01_CALIBRATOR_CONTRACT_VERSION:
            raise ValueError(
                f"Gate 0.1 calibrator contract mismatch: {version!r} != "
                f"{GATE01_CALIBRATOR_CONTRACT_VERSION!r}."
            )
        if payload.get("semantics") != GATE01_CALIBRATION_SEMANTICS:
            raise ValueError("Gate 0.1 calibrator has incompatible projection semantics.")
        templates = {
            str(label): TargetDomainTemplate(
                quantiles=torch.tensor(
                    [float(value) for value in item["quantiles"]],
                    dtype=torch.float64,
                ),
                volume_count=int(item["volume_count"]),
                template_sha256=str(item["template_sha256"]),
            )
            for label, item in payload.get("templates", {}).items()
        }
        calibrator = cls(
            probabilities=torch.tensor(
                [float(value) for value in payload.get("probabilities", [])],
                dtype=torch.float64,
            ),
            templates=templates,
            provenance=dict(payload.get("provenance", {})),
            mask_threshold=float(payload.get("mask_threshold", 0.0)),
            low_probability=float(payload.get("low_probability", 0.01)),
            high_probability=float(payload.get("high_probability", 0.99)),
        )
        stored_hash = str(payload.get("template_sha256", ""))
        if stored_hash != calibrator.template_sha256:
            raise ValueError(
                "Gate 0.1 aggregate template hash mismatch: artifact is stale or altered."
            )
        stored_artifact_hash = str(payload.get("artifact_sha256", ""))
        computed_artifact_hash = calibrator.to_dict()["artifact_sha256"]
        if stored_artifact_hash != computed_artifact_hash:
            raise ValueError(
                "Gate 0.1 calibrator artifact hash mismatch: provenance or templates "
                "are stale or altered."
            )
        if (
            expected_template_sha256 is not None
            and calibrator.template_sha256 != expected_template_sha256
        ):
            raise ValueError(
                "Gate 0.1 template identity mismatch: "
                f"{calibrator.template_sha256} != {expected_template_sha256}."
            )
        calibrator.assert_split_fingerprint(expected_split_fingerprint)
        return calibrator

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_split_fingerprint: str | None = None,
        expected_template_sha256: str | None = None,
    ) -> "PosthocTargetCalibrator":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")),
            expected_split_fingerprint=expected_split_fingerprint,
            expected_template_sha256=expected_template_sha256,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def assert_split_fingerprint(self, expected: str | None) -> None:
        if expected is not None and self.split_fingerprint != expected:
            raise ValueError(
                "Gate 0.1 split fingerprint mismatch: calibrator is stale or incompatible: "
                f"{self.split_fingerprint} != {expected}."
            )

    def apply(
        self,
        prediction: torch.Tensor,
        requested_target: Domain,
        *,
        mode: CalibrationMode = "histogram",
    ) -> torch.Tensor:
        """Calibrate one prediction without accepting any evaluation-target argument."""

        if mode not in ("histogram", "robust_affine"):
            raise ValueError(f"Unknown Gate 0.1 calibration mode {mode!r}.")
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Gate 0.1 calibration prediction must be a torch.Tensor.")
        if prediction.ndim < 3:
            raise ValueError(
                "Gate 0.1 calibration requires a full volume with at least 3 dimensions."
            )
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Gate 0.1 prediction contains non-finite values.")
        template = self.templates.get(requested_target.label)
        if template is None:
            raise KeyError(f"Unknown Gate 0.1 target domain {requested_target.label!r}.")

        original_device = prediction.device
        original_dtype = prediction.dtype
        values = prediction.detach().cpu().to(torch.float64)
        flat = values.reshape(-1)
        foreground = flat.abs() > self.mask_threshold
        if not bool(foreground.any()):
            return torch.zeros_like(prediction)
        source_values = flat[foreground]
        source_quantiles = _deterministic_quantiles(source_values, self.probabilities)
        target_quantiles = template.quantiles.to(torch.float64)

        if mode == "robust_affine":
            mapped_values = _robust_affine_values(
                source_values,
                source_quantiles,
                target_quantiles,
                self.probabilities,
                self.low_probability,
                self.high_probability,
            )
        else:
            mapped_values = _interpolate_monotone(
                source_values, source_quantiles, target_quantiles
            )

        mapped = torch.zeros_like(flat)
        mapped[foreground] = mapped_values
        mapped = mapped.reshape(values.shape)
        if not bool(torch.isfinite(mapped).all()):
            raise ValueError("Gate 0.1 calibration produced non-finite values.")
        return mapped.to(device=original_device, dtype=original_dtype)


def fit_posthoc_target_calibrator(
    volumes: Iterable[TrainingTemplateVolume],
    *,
    split_fingerprint: str,
    training_cohort_identity: str,
    code_commit: str,
    num_quantiles: int = 256,
    mask_threshold: float = 0.0,
    low_probability: float = 0.01,
    high_probability: float = 0.99,
    expected_cohort: str = "R",
    extra_config: Mapping[str, Any] | None = None,
) -> PosthocTargetCalibrator:
    """Fit all 15 target templates from equal-weight retrospective training volumes."""

    if not split_fingerprint:
        raise ValueError("Gate 0.1 fitting requires a split fingerprint.")
    if not training_cohort_identity:
        raise ValueError("Gate 0.1 fitting requires a training cohort identity.")
    if not code_commit:
        raise ValueError("Gate 0.1 fitting requires the code commit.")
    if num_quantiles < 3:
        raise ValueError("Gate 0.1 fitting requires at least 3 quantiles.")
    if mask_threshold < 0 or not math.isfinite(mask_threshold):
        raise ValueError("Gate 0.1 mask_threshold must be finite and non-negative.")

    items = sorted(
        list(volumes), key=lambda item: (item.domain.label, item.record_identity)
    )
    if not items:
        raise ValueError("Gate 0.1 fitting received no training volumes.")
    identities = [item.record_identity for item in items]
    if len(set(identities)) != len(identities):
        raise ValueError("Gate 0.1 fitting received duplicate training record identities.")

    probabilities = torch.linspace(0.0, 1.0, num_quantiles, dtype=torch.float64)
    accumulated: dict[str, list[torch.Tensor]] = {}
    for item in items:
        if item.split != "train":
            raise ValueError(
                f"Gate 0.1 templates may use training records only; "
                f"{item.record_identity!r} is split {item.split!r}."
            )
        if item.cohort != expected_cohort:
            raise ValueError(
                "Gate 0.1 templates may use the retrospective cohort only; "
                f"{item.record_identity!r} is cohort {item.cohort!r}."
            )
        volume = item.volume.detach().cpu().to(torch.float64)
        if volume.ndim < 3:
            raise ValueError(
                f"Training template {item.record_identity!r} is not a full volume."
            )
        if not bool(torch.isfinite(volume).all()):
            raise ValueError(
                f"Training template {item.record_identity!r} contains non-finite values."
            )
        foreground = volume.reshape(-1)
        foreground = foreground[foreground.abs() > mask_threshold]
        if foreground.numel() == 0:
            raise ValueError(
                f"Training template {item.record_identity!r} has no foreground voxels."
            )
        accumulated.setdefault(item.domain.label, []).append(
            _deterministic_quantiles(foreground, probabilities)
        )

    expected_labels = set(all_domain_labels())
    if set(accumulated) != expected_labels:
        raise ValueError(
            "Gate 0.1 fitting requires all 15 domains; "
            f"missing={sorted(expected_labels - set(accumulated))}, "
            f"unexpected={sorted(set(accumulated) - expected_labels)}."
        )

    templates: dict[str, TargetDomainTemplate] = {}
    for label, domain_items in sorted(accumulated.items()):
        quantiles = torch.stack(domain_items).mean(dim=0)
        templates[label] = TargetDomainTemplate(
            quantiles=quantiles,
            volume_count=len(domain_items),
            template_sha256=_quantile_hash(quantiles),
        )

    domain_counts = {
        label: len(domain_items) for label, domain_items in sorted(accumulated.items())
    }
    provenance = {
        "split_fingerprint": split_fingerprint,
        "training_cohort_identity": training_cohort_identity,
        "training_records_sha256": _sha256_text(
            "\n".join(
                f"{item.record_identity}|{item.domain.label}" for item in items
            )
        ),
        "code_commit": code_commit,
        "config": {
            "num_quantiles": num_quantiles,
            "mask_threshold": mask_threshold,
            "low_probability": low_probability,
            "high_probability": high_probability,
            "foreground_rule": "abs(prediction voxel) > mask_threshold",
            **dict(extra_config or {}),
        },
        "balancing": {
            "within_domain": "equal weight per training volume via mean volume quantiles",
            "across_domains": (
                "independent template per domain; evaluation reports each of the 15 "
                "domains and macro-equal directed pairs"
            ),
            "unequal_domain_counts_policy": (
                "retained and disclosed; counts cannot cross-weight independent templates"
            ),
        },
        "domain_volume_counts": domain_counts,
    }
    return PosthocTargetCalibrator(
        probabilities=probabilities,
        templates=templates,
        provenance=provenance,
        mask_threshold=mask_threshold,
        low_probability=low_probability,
        high_probability=high_probability,
    )


_FORBIDDEN_CALIBRATION_KEYS = {
    "target_mask",
    "paired_target_mask",
    "target_histogram",
    "paired_target_histogram",
    "target_quantiles",
    "paired_target_quantiles",
    "target_statistics",
    "target_stats",
    "calibration_target",
}


def reject_target_derived_calibration_fields(payload: Mapping[str, Any]) -> None:
    """Fail closed if a run manifest tries to feed target-derived calibration data."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized in _FORBIDDEN_CALIBRATION_KEYS:
                    raise ValueError(
                        "Gate 0.1 forbids target-derived calibration input "
                        f"{child_path!r}."
                    )
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")


def _deterministic_quantiles(
    values: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    values = values.detach().cpu().to(torch.float64).reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Cannot compute quantiles from non-finite values.")
    limit = 4_000_000
    if values.numel() > limit:
        step = values.numel() // limit + 1
        values = values[::step]
    return torch.quantile(values, probabilities.to(torch.float64))


def _robust_affine_values(
    values: torch.Tensor,
    source_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    probabilities: torch.Tensor,
    low_probability: float,
    high_probability: float,
) -> torch.Tensor:
    low_index = int(torch.argmin((probabilities - low_probability).abs()))
    high_index = int(torch.argmin((probabilities - high_probability).abs()))
    source_low = source_quantiles[low_index]
    source_high = source_quantiles[high_index]
    target_low = target_quantiles[low_index]
    target_high = target_quantiles[high_index]
    span = source_high - source_low
    if float(span.abs()) < 1e-12:
        return values.clone()
    return (values - source_low) * ((target_high - target_low) / span) + target_low


def _interpolate_monotone(
    values: torch.Tensor,
    source_grid: torch.Tensor,
    target_grid: torch.Tensor,
) -> torch.Tensor:
    indices = torch.searchsorted(
        source_grid.contiguous(), values.contiguous()
    ).clamp(1, source_grid.numel() - 1)
    left = source_grid[indices - 1]
    right = source_grid[indices]
    left_target = target_grid[indices - 1]
    right_target = target_grid[indices]
    span = right - left
    nonzero = span.abs() > 1e-12
    weight = torch.zeros_like(values)
    weight[nonzero] = (
        (values[nonzero] - left[nonzero]) / span[nonzero]
    ).clamp(0.0, 1.0)
    mapped = left_target + weight * (right_target - left_target)
    return mapped


def _quantile_hash(quantiles: torch.Tensor) -> str:
    return _sha256_json([float(value) for value in quantiles.to(torch.float64)])


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "FULL_LATENT_BANK_BUILD_COMMIT",
    "GATE0_DIAGNOSTIC_COMMIT",
    "GATE01_CALIBRATION_SEMANTICS",
    "GATE01_CALIBRATOR_CONTRACT_VERSION",
    "PosthocTargetCalibrator",
    "RESPLIT_FINGERPRINT",
    "SB_V2_CHECKPOINT_SHA256",
    "STAGE1_RUN_C_CHECKPOINT_SHA256",
    "TargetDomainTemplate",
    "TrainingTemplateVolume",
    "all_domain_labels",
    "fit_posthoc_target_calibrator",
    "reject_target_derived_calibration_fields",
]
