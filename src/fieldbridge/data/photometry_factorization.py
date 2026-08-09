"""Frozen, training-only photometry factorization for Stage-2 Variant A.

This contract is deliberately independent from Gate 0.1.  Gate 0.1 derives a CDF from
each prediction at evaluation time; this module never does.  It applies only the sealed
domain and contrast quantile grids stored in :class:`FrozenPhotometryArtifact`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Contrast, Domain

PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION = "stage2-photometry-factorization-v1"
PHOTOMETRY_FACTORIZATION_CONFIG_VERSION = "stage2-photometry-variant-a-config-v1"
PHOTOMETRY_FACTORIZATION_SEMANTICS = "fixed-domain-to-contrast-canonical-quantile-map-v1"
PHOTOMETRY_INTERPOLATION_RULE = "monotone-piecewise-linear-v1"
PHOTOMETRY_DUPLICATE_KNOT_RULE = "collapse-equal-source-knots-to-mean-target-v1"
PHOTOMETRY_CLAMPING_RULE = "sealed-endpoint-clamping-v1"
PHOTOMETRY_SUPPORT_POLICY = "source-nonzero-mask-exact-zero-output-v1"
PHOTOMETRY_FIT_ELIGIBILITY = "cohort=R;split=train"
PHOTOMETRY_QUALIFICATION_ELIGIBILITY = "cohort=R;split=validation"
PHOTOMETRY_DEFAULT_QUANTILES = 256
PHOTOMETRY_SOURCE_MODULES = (
    "src/fieldbridge/data/photometry_factorization.py",
    "src/fieldbridge/evaluation/stage2_photometry_baseline.py",
    "src/fieldbridge/evaluation/stage2_photometry_protocol.py",
    "src/fieldbridge/cli.py",
    "src/fieldbridge/data/vae_splits.py",
)
FORBIDDEN_TRAVELLER_IDS = frozenset({"0006", "0007", "0009"})
VARIANT_A_PROSPECTIVE_EXCLUSION_REASON = (
    "prospective-cohort-excluded-before-array-load"
)

_REQUIRED_PROVENANCE_KEYS = {
    "source_split_file_sha256",
    "source_membership_fingerprint",
    "source_recovery_fingerprint",
    "fit_eligibility_rule",
    "accepted_records",
    "accepted_records_sha256",
    "excluded_prospective_records",
    "excluded_prospective_records_sha256",
    "eligibility_proof",
    "domain_volume_counts",
    "weighting",
    "code_commit",
    "code_provenance",
    "resolved_config",
    "resolved_config_sha256",
}


def all_photometry_domain_labels() -> tuple[str, ...]:
    """Return the canonical ordering of all 15 supported domains."""

    return tuple(
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    )


@dataclass(frozen=True, slots=True)
class PhotometryFitVolume:
    """One retrospective training volume and its sealed identities."""

    volume: torch.Tensor
    domain: Domain
    record_identity: str
    subject_identity: str | None
    metadata_prefix: str | None
    source_path_identity: str
    source_file_sha256: str
    split: str = "train"
    cohort: str = "R"


@dataclass(frozen=True, slots=True)
class CanonicalCohortIdentity:
    """Reconciled case namespace used by every Variant-A data boundary."""

    cohort: str
    case_identity: str
    subject_identity: str
    subject_group_identity: str


def classify_variant_a_cohort(
    *,
    case_identity: str,
    metadata_prefix: str | None,
    supplied_cohort: str | None,
    subject_identity: str | None,
    allowed_cohorts: Sequence[str] = ("R", "P"),
) -> CanonicalCohortIdentity:
    """Derive and reconcile the canonical ``R_``/``P_`` identity namespace.

    The case identifier is authoritative only for deriving the namespace; metadata and
    the caller-supplied cohort must independently agree.  No boundary may repair,
    default, or silently reinterpret a missing/conflicting identity.
    """

    identity = str(case_identity).strip()
    if not identity:
        raise ValueError("Variant-A identity requires a nonempty case_id.")
    if identity.startswith("R_"):
        case_cohort = "R"
    elif identity.startswith("P_"):
        case_cohort = "P"
    else:
        raise ValueError("Variant-A case_id must begin with the canonical R_ or P_ prefix.")

    if metadata_prefix is None or not str(metadata_prefix).strip():
        raise ValueError("Variant-A identity requires metadata prefix R or P.")
    metadata_cohort = str(metadata_prefix).strip().upper()
    supplied = "" if supplied_cohort is None else str(supplied_cohort).strip().upper()
    for name, value in (("metadata prefix", metadata_cohort), ("supplied cohort", supplied)):
        if value not in {"R", "P"}:
            raise ValueError(f"Variant-A {name} must be exactly R or P.")
    if len({case_cohort, metadata_cohort, supplied}) != 1:
        raise ValueError(
            "Variant-A cohort identity conflict: "
            f"case_id={case_cohort}, metadata={metadata_cohort}, supplied={supplied}."
        )

    subject = "" if subject_identity is None else str(subject_identity).strip()
    if not subject:
        raise ValueError("Variant-A identity requires a nonempty subject identity.")
    allowed = {str(value).strip().upper() for value in allowed_cohorts}
    if not allowed or not allowed <= {"R", "P"}:
        raise ValueError("Variant-A allowed cohorts must be a nonempty subset of R/P.")
    if case_cohort not in allowed:
        if case_cohort == "P":
            raise ValueError("Variant-A retrospective operation rejects every P record.")
        raise ValueError(f"Variant-A operation rejects cohort {case_cohort}.")
    return CanonicalCohortIdentity(
        cohort=case_cohort,
        case_identity=identity,
        subject_identity=subject,
        subject_group_identity=f"{case_cohort}:{subject}",
    )


@dataclass(frozen=True, slots=True)
class DomainPhotometryTemplate:
    """Equal-volume mean quantiles for one contrast/field domain."""

    quantiles: torch.Tensor
    volume_count: int
    per_volume_weight: float
    template_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantiles": [float(value) for value in self.quantiles],
            "volume_count": int(self.volume_count),
            "per_volume_weight": float(self.per_volume_weight),
            "template_sha256": self.template_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContrastCanonicalTemplate:
    """Exactly equal-field canonical quantiles for one contrast."""

    quantiles: torch.Tensor
    domain_labels: tuple[str, ...]
    per_field_weight: float
    template_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantiles": [float(value) for value in self.quantiles],
            "domain_labels": list(self.domain_labels),
            "per_field_weight": float(self.per_field_weight),
            "template_sha256": self.template_sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceCanonicalizedVolume:
    """Canonical values plus the one source-derived support context.

    Instances are created by :meth:`FrozenPhotometryArtifact.normalize_source`.  The
    artifact identity and source mask travel with the values so target rendering cannot
    replace the support with one derived from a target or prediction.
    """

    values: torch.Tensor
    support_mask: torch.Tensor
    source_domain: Domain
    artifact_sha256: str

    def with_values(self, values: torch.Tensor) -> "SourceCanonicalizedVolume":
        if not isinstance(values, torch.Tensor):
            raise TypeError("Canonical replacement values must be a torch.Tensor.")
        if tuple(values.shape) != tuple(self.values.shape):
            raise ValueError(
                "Canonical replacement shape mismatch: "
                f"{tuple(values.shape)} != {tuple(self.values.shape)}."
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Canonical replacement contains non-finite values.")
        masked = torch.zeros_like(values)
        masked[self.support_mask] = values[self.support_mask]
        return replace(self, values=masked)


@dataclass(frozen=True, slots=True)
class FrozenPhotometryArtifact:
    """Sealed fixed maps ``q_d <-> q_*^contrast`` for all 15 domains."""

    probabilities: torch.Tensor
    domain_templates: Mapping[str, DomainPhotometryTemplate]
    canonical_templates: Mapping[str, ContrastCanonicalTemplate]
    provenance: Mapping[str, Any]
    interpolation_rule: str = PHOTOMETRY_INTERPOLATION_RULE
    duplicate_knot_rule: str = PHOTOMETRY_DUPLICATE_KNOT_RULE
    clamping_rule: str = PHOTOMETRY_CLAMPING_RULE
    support_policy: str = PHOTOMETRY_SUPPORT_POLICY

    def __post_init__(self) -> None:
        probabilities = self.probabilities.detach().cpu().to(torch.float64)
        if probabilities.ndim != 1 or probabilities.numel() < 3:
            raise ValueError("Photometry probabilities must be 1-D with at least 3 values.")
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("Photometry probabilities contain non-finite values.")
        if float(probabilities[0]) != 0.0 or float(probabilities[-1]) != 1.0:
            raise ValueError("Photometry probabilities must include exact endpoints 0 and 1.")
        if not bool((probabilities[1:] > probabilities[:-1]).all()):
            raise ValueError("Photometry probabilities must be strictly increasing.")
        object.__setattr__(self, "probabilities", probabilities)

        if self.interpolation_rule != PHOTOMETRY_INTERPOLATION_RULE:
            raise ValueError("Unsupported photometry interpolation rule.")
        if self.duplicate_knot_rule != PHOTOMETRY_DUPLICATE_KNOT_RULE:
            raise ValueError("Unsupported photometry duplicate-knot rule.")
        if self.clamping_rule != PHOTOMETRY_CLAMPING_RULE:
            raise ValueError("Unsupported photometry clamping rule.")
        if self.support_policy != PHOTOMETRY_SUPPORT_POLICY:
            raise ValueError("Unsupported photometry support policy.")

        missing_provenance = sorted(_REQUIRED_PROVENANCE_KEYS - set(self.provenance))
        if missing_provenance:
            raise ValueError(
                f"Photometry artifact provenance is incomplete: {missing_provenance}."
            )
        _validate_sha256(str(self.provenance["source_split_file_sha256"]), "split file")
        _validate_code_provenance(
            self.provenance["code_provenance"], str(self.provenance["code_commit"])
        )
        if self.provenance["fit_eligibility_rule"] != PHOTOMETRY_FIT_ELIGIBILITY:
            raise ValueError("Photometry artifact has an incompatible fit eligibility rule.")
        resolved_config = self.provenance["resolved_config"]
        if not isinstance(resolved_config, Mapping):
            raise ValueError("Photometry artifact resolved config must be a mapping.")
        if self.provenance["resolved_config_sha256"] != sha256_json(resolved_config):
            raise ValueError("Photometry artifact resolved-config hash mismatch.")
        reject_target_or_prediction_derived_fields(resolved_config)

        expected_labels = set(all_photometry_domain_labels())
        actual_labels = set(self.domain_templates)
        if actual_labels != expected_labels:
            raise ValueError(
                "Photometry factorization requires all 15 domains exactly; "
                f"missing={sorted(expected_labels - actual_labels)}, "
                f"unexpected={sorted(actual_labels - expected_labels)}."
            )
        expected_contrasts = {contrast.value for contrast in CONTRASTS}
        if set(self.canonical_templates) != expected_contrasts:
            raise ValueError("Photometry artifact requires exactly three contrast templates.")

        counts = self.provenance["domain_volume_counts"]
        if not isinstance(counts, Mapping) or set(counts) != expected_labels:
            raise ValueError("Photometry domain counts do not cover all 15 domains.")
        weighting = self.provenance["weighting"]
        if not isinstance(weighting, Mapping) or not math.isclose(
            float(weighting.get("per_field_weight", -1.0)), 0.2, abs_tol=1e-12
        ):
            raise ValueError("Photometry provenance does not seal equal field weights.")
        if weighting.get("within_domain") != (
            "exactly equal weight per eligible volume"
        ) or weighting.get("across_fields_within_contrast") != (
            "exactly 0.2 per field"
        ):
            raise ValueError("Photometry provenance has incompatible weighting semantics.")
        volume_weights = weighting.get("per_volume_weights", {})
        if not isinstance(volume_weights, Mapping) or set(volume_weights) != expected_labels:
            raise ValueError("Photometry provenance does not seal all per-volume weights.")
        for label, template in self.domain_templates.items():
            quantiles = _validated_quantiles(template.quantiles, probabilities, label)
            object.__setattr__(template, "quantiles", quantiles)
            if template.volume_count <= 0 or int(counts[label]) != template.volume_count:
                raise ValueError(f"Photometry domain {label!r} has inconsistent counts.")
            expected_weight = 1.0 / template.volume_count
            if not math.isclose(template.per_volume_weight, expected_weight, abs_tol=1e-12):
                raise ValueError(f"Photometry domain {label!r} is not equally volume-weighted.")
            if not math.isclose(
                float(volume_weights[label]), expected_weight, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Photometry provenance for {label!r} has an invalid volume weight."
                )
            if template.template_sha256 != tensor_sha256(quantiles):
                raise ValueError(f"Photometry domain {label!r} template hash mismatch.")

        for contrast in CONTRASTS:
            label = contrast.value
            canonical = self.canonical_templates[label]
            quantiles = _validated_quantiles(canonical.quantiles, probabilities, label)
            object.__setattr__(canonical, "quantiles", quantiles)
            expected_domains = tuple(
                Domain(field, contrast).label for field in FIELD_STRENGTHS_T
            )
            if canonical.domain_labels != expected_domains:
                raise ValueError(f"Canonical template {label!r} field ordering is incompatible.")
            if not math.isclose(canonical.per_field_weight, 0.2, abs_tol=1e-12):
                raise ValueError(f"Canonical template {label!r} is not equally field-weighted.")
            recomputed = torch.stack(
                [self.domain_templates[item].quantiles for item in expected_domains]
            ).mean(dim=0)
            if not torch.allclose(quantiles, recomputed, rtol=0.0, atol=1e-12):
                raise ValueError(f"Canonical template {label!r} is not the equal-field mean.")
            if canonical.template_sha256 != tensor_sha256(quantiles):
                raise ValueError(f"Canonical template {label!r} hash mismatch.")

        _validate_accepted_records(self.provenance, expected_labels)

    @property
    def artifact_sha256(self) -> str:
        return str(self.to_dict()["artifact_sha256"])

    @property
    def split_fingerprint(self) -> str:
        return str(self.provenance["source_membership_fingerprint"])

    @property
    def recovery_fingerprint(self) -> str:
        return str(self.provenance["source_recovery_fingerprint"])

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
            "semantics": PHOTOMETRY_FACTORIZATION_SEMANTICS,
            "interpolation_rule": self.interpolation_rule,
            "duplicate_knot_rule": self.duplicate_knot_rule,
            "clamping_rule": self.clamping_rule,
            "support_policy": self.support_policy,
            "runtime_statistics": "none; application uses sealed grids only",
            "probabilities": [float(value) for value in self.probabilities],
            "domain_templates": {
                label: template.to_dict()
                for label, template in sorted(self.domain_templates.items())
            },
            "canonical_templates": {
                label: template.to_dict()
                for label, template in sorted(self.canonical_templates.items())
            },
            "provenance": dict(self.provenance),
        }
        payload["artifact_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_split_file_sha256: str | None = None,
        expected_membership_fingerprint: str | None = None,
        expected_recovery_fingerprint: str | None = None,
        expected_artifact_sha256: str | None = None,
    ) -> "FrozenPhotometryArtifact":
        version = str(payload.get("contract_version", ""))
        if version != PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION:
            raise ValueError(
                f"Photometry contract mismatch: {version!r} != "
                f"{PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION!r}."
            )
        if payload.get("semantics") != PHOTOMETRY_FACTORIZATION_SEMANTICS:
            raise ValueError("Photometry artifact has incompatible semantics.")
        domains = {
            str(label): DomainPhotometryTemplate(
                quantiles=torch.tensor(item["quantiles"], dtype=torch.float64),
                volume_count=int(item["volume_count"]),
                per_volume_weight=float(item["per_volume_weight"]),
                template_sha256=str(item["template_sha256"]),
            )
            for label, item in dict(payload.get("domain_templates", {})).items()
        }
        canonicals = {
            str(label): ContrastCanonicalTemplate(
                quantiles=torch.tensor(item["quantiles"], dtype=torch.float64),
                domain_labels=tuple(str(value) for value in item["domain_labels"]),
                per_field_weight=float(item["per_field_weight"]),
                template_sha256=str(item["template_sha256"]),
            )
            for label, item in dict(payload.get("canonical_templates", {})).items()
        }
        artifact = cls(
            probabilities=torch.tensor(payload.get("probabilities", []), dtype=torch.float64),
            domain_templates=domains,
            canonical_templates=canonicals,
            provenance=dict(payload.get("provenance", {})),
            interpolation_rule=str(payload.get("interpolation_rule", "")),
            duplicate_knot_rule=str(payload.get("duplicate_knot_rule", "")),
            clamping_rule=str(payload.get("clamping_rule", "")),
            support_policy=str(payload.get("support_policy", "")),
        )
        stored_hash = str(payload.get("artifact_sha256", ""))
        if stored_hash != artifact.artifact_sha256:
            raise ValueError("Photometry artifact content hash mismatch.")
        expected_values = (
            (
                "split file",
                expected_split_file_sha256,
                artifact.provenance["source_split_file_sha256"],
            ),
            ("membership fingerprint", expected_membership_fingerprint, artifact.split_fingerprint),
            ("recovery fingerprint", expected_recovery_fingerprint, artifact.recovery_fingerprint),
            ("artifact", expected_artifact_sha256, artifact.artifact_sha256),
        )
        for name, expected, actual in expected_values:
            if expected is not None and str(expected) != str(actual):
                raise ValueError(f"Photometry {name} mismatch: {actual} != {expected}.")
        return artifact

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "FrozenPhotometryArtifact":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load photometry artifact {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Photometry artifact root must be a JSON object.")
        return cls.from_dict(payload, **kwargs)

    def save(self, path: str | Path) -> Path:
        return write_json_atomic(path, self.to_dict(), refuse_existing=True)

    def normalize_source(
        self, source: torch.Tensor, source_domain: Domain
    ) -> SourceCanonicalizedVolume:
        source = _validated_runtime_volume(source, "source")
        domain_template = self._domain_template(source_domain)
        canonical_template = self.canonical_templates[source_domain.contrast.value]
        support = source != 0
        mapped = _map_on_support(
            source,
            support,
            domain_template.quantiles,
            canonical_template.quantiles,
        )
        return SourceCanonicalizedVolume(
            values=mapped,
            support_mask=support,
            source_domain=source_domain,
            artifact_sha256=self.artifact_sha256,
        )

    def render_target(
        self, canonical: SourceCanonicalizedVolume, target_domain: Domain
    ) -> torch.Tensor:
        if not isinstance(canonical, SourceCanonicalizedVolume):
            raise TypeError("Target rendering requires a source canonicalization context.")
        if canonical.artifact_sha256 != self.artifact_sha256:
            raise ValueError("Canonical context belongs to a different photometry artifact.")
        if canonical.source_domain.contrast != target_domain.contrast:
            raise ValueError("Variant A supports same-contrast field translation only.")
        _validated_runtime_volume(canonical.values, "canonical values")
        if canonical.support_mask.dtype != torch.bool:
            raise ValueError("Canonical support mask must be boolean.")
        if tuple(canonical.support_mask.shape) != tuple(canonical.values.shape):
            raise ValueError("Canonical support mask shape mismatch.")
        canonical_template = self.canonical_templates[target_domain.contrast.value]
        target_template = self._domain_template(target_domain)
        return _map_on_support(
            canonical.values,
            canonical.support_mask,
            canonical_template.quantiles,
            target_template.quantiles,
        )

    def factorized_identity(
        self, source: torch.Tensor, source_domain: Domain, target_domain: Domain
    ) -> torch.Tensor:
        """Return ``M_s * P_t(N_s(source))`` using no runtime CDF statistics."""

        return self.render_target(self.normalize_source(source, source_domain), target_domain)

    def _domain_template(self, domain: Domain) -> DomainPhotometryTemplate:
        template = self.domain_templates.get(domain.label)
        if template is None:
            raise ValueError(f"Domain {domain.label!r} is not sealed in this artifact.")
        return template


def fit_frozen_photometry(
    volumes: Iterable[PhotometryFitVolume],
    *,
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    code_commit: str,
    code_provenance: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    num_quantiles: int = PHOTOMETRY_DEFAULT_QUANTILES,
    excluded_prospective_records: Sequence[Mapping[str, Any]] = (),
) -> FrozenPhotometryArtifact:
    """Fit fixed domain templates from retrospective training records only."""

    _validate_sha256(source_split_file_sha256, "source split file")
    if not source_membership_fingerprint or not source_recovery_fingerprint:
        raise ValueError("Photometry fitting requires both source split fingerprints.")
    if not code_commit:
        raise ValueError("Photometry fitting requires a code commit.")
    _validate_code_provenance(code_provenance, code_commit)
    if num_quantiles < 3:
        raise ValueError("Photometry fitting requires at least 3 quantiles.")
    reject_target_or_prediction_derived_fields(resolved_config)

    probabilities = torch.linspace(0.0, 1.0, num_quantiles, dtype=torch.float64)
    accumulated: dict[str, list[tuple[str, torch.Tensor]]] = {}
    accepted: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for item in volumes:
        _validate_fit_role(item)
        if item.record_identity in record_ids:
            raise ValueError("Photometry fitting received a duplicate record identity.")
        record_ids.add(item.record_identity)
        _validate_sha256(item.source_file_sha256, "training source file")
        volume = _validated_runtime_volume(item.volume, f"training volume {item.record_identity!r}")
        foreground = volume.detach().cpu().to(torch.float64).reshape(-1)
        foreground = foreground[foreground != 0]
        if foreground.numel() == 0:
            raise ValueError(f"Training volume {item.record_identity!r} has no nonzero support.")
        quantiles = deterministic_quantiles(foreground, probabilities)
        accumulated.setdefault(item.domain.label, []).append((item.record_identity, quantiles))
        accepted.append(
            {
                "record_identity": item.record_identity,
                "record_identity_sha256": sha256_text(item.record_identity),
                "subject_identity": item.subject_identity,
                "metadata_prefix": item.metadata_prefix,
                "subject_group_identity": f"R:{item.subject_identity}",
                "source_path_identity_sha256": sha256_text(item.source_path_identity),
                "source_file_sha256": item.source_file_sha256,
                "canonical_loaded_array_sha256": canonical_tensor_sha256(item.volume),
                "domain": item.domain.label,
                "cohort": item.cohort,
                "split": item.split,
            }
        )

    if not accepted:
        raise ValueError("Photometry fitting received no eligible volumes.")
    expected_labels = set(all_photometry_domain_labels())
    if set(accumulated) != expected_labels:
        raise ValueError(
            "Photometry fitting requires all 15 domains; "
            f"missing={sorted(expected_labels - set(accumulated))}, "
            f"unexpected={sorted(set(accumulated) - expected_labels)}."
        )

    domains: dict[str, DomainPhotometryTemplate] = {}
    for label, values in sorted(accumulated.items()):
        quantiles = torch.stack(
            [item for _, item in sorted(values, key=lambda pair: pair[0])]
        ).mean(dim=0)
        domains[label] = DomainPhotometryTemplate(
            quantiles=quantiles,
            volume_count=len(values),
            per_volume_weight=1.0 / len(values),
            template_sha256=tensor_sha256(quantiles),
        )

    canonicals: dict[str, ContrastCanonicalTemplate] = {}
    for contrast in CONTRASTS:
        labels = tuple(Domain(field, contrast).label for field in FIELD_STRENGTHS_T)
        quantiles = torch.stack([domains[label].quantiles for label in labels]).mean(dim=0)
        canonicals[contrast.value] = ContrastCanonicalTemplate(
            quantiles=quantiles,
            domain_labels=labels,
            per_field_weight=1.0 / len(FIELD_STRENGTHS_T),
            template_sha256=tensor_sha256(quantiles),
        )

    accepted.sort(key=lambda item: (item["domain"], item["record_identity"]))
    excluded = normalize_variant_a_prospective_exclusions(
        excluded_prospective_records, expected_split="train"
    )
    counts = {label: len(values) for label, values in sorted(accumulated.items())}
    config = _json_safe_mapping(resolved_config)
    provenance = {
        "source_split_file_sha256": source_split_file_sha256,
        "source_membership_fingerprint": source_membership_fingerprint,
        "source_recovery_fingerprint": source_recovery_fingerprint,
        "fit_eligibility_rule": PHOTOMETRY_FIT_ELIGIBILITY,
        "accepted_records": accepted,
        "accepted_records_sha256": sha256_json(accepted),
        "excluded_prospective_records": excluded,
        "excluded_prospective_records_sha256": sha256_json(excluded),
        "eligibility_proof": {
            "accepted_count": len(accepted),
            "all_cohort_R": all(item["cohort"] == "R" for item in accepted),
            "all_split_train": all(item["split"] == "train" for item in accepted),
            "prospective_accepted_count": 0,
            "prospective_excluded_count": len(excluded),
            "forbidden_traveller_accepted_count": 0,
        },
        "domain_volume_counts": counts,
        "weighting": {
            "within_domain": "exactly equal weight per eligible volume",
            "across_fields_within_contrast": "exactly 0.2 per field",
            "across_contrasts": "independent canonical template per contrast",
            "per_field_weight": 0.2,
            "per_volume_weights": {
                label: 1.0 / count for label, count in sorted(counts.items())
            },
        },
        "code_commit": code_commit,
        "code_provenance": dict(code_provenance),
        "resolved_config": config,
        "resolved_config_sha256": sha256_json(config),
    }
    return FrozenPhotometryArtifact(
        probabilities=probabilities,
        domain_templates=domains,
        canonical_templates=canonicals,
        provenance=provenance,
    )


def deterministic_quantiles(values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Compute deterministic CPU float64 quantiles without device-dependent sampling."""

    values = values.detach().cpu().to(torch.float64).reshape(-1)
    probabilities = probabilities.detach().cpu().to(torch.float64)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("Cannot compute quantiles from empty or non-finite values.")
    return torch.quantile(values, probabilities)


def interpolate_fixed_grid(
    values: torch.Tensor, source_grid: torch.Tensor, target_grid: torch.Tensor
) -> torch.Tensor:
    """Map through fixed monotone grids with deterministic duplicate handling/clamping."""

    original_device = values.device
    original_dtype = values.dtype
    work = values.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("Photometry interpolation values contain non-finite values.")
    source, target = _collapse_duplicate_knots(source_grid, target_grid)
    if source.numel() == 1:
        mapped = torch.full_like(work, float(target[0]))
    else:
        flat = work.reshape(-1)
        mapped_flat = torch.empty_like(flat)
        below = flat <= source[0]
        above = flat >= source[-1]
        middle = ~(below | above)
        mapped_flat[below] = target[0]
        mapped_flat[above] = target[-1]
        if bool(middle.any()):
            middle_values = flat[middle]
            indices = torch.searchsorted(source.contiguous(), middle_values.contiguous())
            indices = indices.clamp(1, source.numel() - 1)
            left = source[indices - 1]
            right = source[indices]
            weight = (middle_values - left) / (right - left)
            mapped_flat[middle] = target[indices - 1] + weight * (
                target[indices] - target[indices - 1]
            )
        mapped = mapped_flat.reshape(work.shape)
    if not bool(torch.isfinite(mapped).all()):
        raise ValueError("Photometry interpolation produced non-finite values.")
    return mapped.to(device=original_device, dtype=original_dtype)


def capture_photometry_code_provenance(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture clean-checkout and source identities for an external fit."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_head": head,
        "checkout_clean": not bool(status.strip()),
        "module_sha256": {
            relative: sha256_file(root / relative) for relative in PHOTOMETRY_SOURCE_MODULES
        },
    }


def reject_target_or_prediction_derived_fields(payload: Mapping[str, Any]) -> None:
    """Reject configuration attempting runtime target/prediction-derived calibration."""

    forbidden = {
        "evaluation_target",
        "paired_target",
        "target_image",
        "target_mask",
        "target_histogram",
        "target_quantiles",
        "prediction_cdf",
        "prediction_histogram",
        "prediction_quantiles",
        "calibration_target",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized in forbidden:
                    raise ValueError(
                        f"Variant A forbids target- or prediction-derived field {child_path!r}."
                    )
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")


def assert_variant_a_external_path(
    path: str | Path, *, repo_root: str | Path | None = None
) -> Path:
    """Require data-bearing Variant-A inputs/outputs to stay outside the checkout."""

    resolved = Path(path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise ValueError(
        "Variant-A data-bearing inputs and outputs must remain outside the Git repository."
    )


def write_json_atomic(
    path: str | Path, payload: Mapping[str, Any], *, refuse_existing: bool = True
) -> Path:
    """Write deterministic JSON atomically and clean temporary files on failure."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and target.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().to(torch.float64).contiguous().numpy()
    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    canonical = np.ascontiguousarray(values, dtype="<f4")
    header = json.dumps(
        {"dtype": "float32-le", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _map_on_support(
    values: torch.Tensor,
    support: torch.Tensor,
    source_grid: torch.Tensor,
    target_grid: torch.Tensor,
) -> torch.Tensor:
    if support.dtype != torch.bool or tuple(support.shape) != tuple(values.shape):
        raise ValueError("Source support must be boolean and match the source shape.")
    output = torch.zeros_like(values)
    if bool(support.any()):
        output[support] = interpolate_fixed_grid(values[support], source_grid, target_grid)
    if not bool((output[~support] == 0).all()):
        raise AssertionError("Variant-A exact-zero support invariant failed.")
    return output


def _collapse_duplicate_knots(
    source_grid: torch.Tensor, target_grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    source = source_grid.detach().cpu().to(torch.float64).reshape(-1)
    target = target_grid.detach().cpu().to(torch.float64).reshape(-1)
    if source.shape != target.shape or source.numel() < 1:
        raise ValueError("Photometry interpolation grids must be nonempty and shape-matched.")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("Photometry interpolation grids contain non-finite values.")
    if not bool((source[1:] >= source[:-1]).all()):
        raise ValueError("Photometry source grid is not monotonic.")
    if not bool((target[1:] >= target[:-1]).all()):
        raise ValueError("Photometry target grid is not monotonic.")
    unique, inverse, counts = torch.unique_consecutive(
        source, return_inverse=True, return_counts=True
    )
    sums = torch.zeros_like(unique)
    sums.scatter_add_(0, inverse, target)
    collapsed_target = sums / counts.to(torch.float64)
    return unique, collapsed_target


def _validated_quantiles(
    values: torch.Tensor, probabilities: torch.Tensor, label: str
) -> torch.Tensor:
    quantiles = values.detach().cpu().to(torch.float64)
    if quantiles.shape != probabilities.shape:
        raise ValueError(f"Template {label!r} shape does not match probabilities.")
    if not bool(torch.isfinite(quantiles).all()):
        raise ValueError(f"Template {label!r} contains non-finite values.")
    if not bool((quantiles[1:] >= quantiles[:-1]).all()):
        raise ValueError(f"Template {label!r} is not monotonic.")
    if float(quantiles[0]) < 0.0 or float(quantiles[-1]) > 1.0:
        raise ValueError(f"Template {label!r} violates the official [0,1] range.")
    return quantiles


def _validated_runtime_volume(volume: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if volume.ndim < 3:
        raise ValueError(f"{name} must have at least three dimensions.")
    if not volume.dtype.is_floating_point:
        raise ValueError(f"{name} must use a floating-point dtype.")
    if not bool(torch.isfinite(volume).all()):
        raise ValueError(f"{name} contains non-finite values.")
    if float(volume.min()) < 0.0 or float(volume.max()) > 1.0:
        raise ValueError(f"{name} violates the official [0,1] range.")
    return volume


def _validate_fit_role(item: PhotometryFitVolume) -> None:
    if _is_forbidden_traveller(item.record_identity, item.subject_identity):
        raise ValueError(f"Variant A explicitly rejects traveller {item.subject_identity}.")
    identity = classify_variant_a_cohort(
        case_identity=item.record_identity,
        metadata_prefix=item.metadata_prefix,
        supplied_cohort=item.cohort,
        subject_identity=item.subject_identity,
        allowed_cohorts=("R",),
    )
    if _is_forbidden_traveller(identity.case_identity, identity.subject_identity):
        raise ValueError(f"Variant A explicitly rejects traveller {identity.subject_identity}.")
    if item.split != "train":
        raise ValueError("Variant-A fitting accepts split=train only.")
    if not item.record_identity or not item.source_path_identity:
        raise ValueError("Variant-A fitting requires record and source-path identities.")


def validate_qualification_role(
    *,
    record_identity: str,
    subject_identity: str | None,
    metadata_prefix: str | None,
    cohort: str,
    split: str,
) -> CanonicalCohortIdentity:
    """Fail closed on anything other than retrospective validation records."""

    if _is_forbidden_traveller(record_identity, subject_identity):
        raise ValueError(f"Variant A explicitly rejects traveller {subject_identity}.")
    identity = classify_variant_a_cohort(
        case_identity=record_identity,
        metadata_prefix=metadata_prefix,
        supplied_cohort=cohort,
        subject_identity=subject_identity,
        allowed_cohorts=("R",),
    )
    if _is_forbidden_traveller(identity.case_identity, identity.subject_identity):
        raise ValueError(f"Variant A explicitly rejects traveller {identity.subject_identity}.")
    if split != "validation":
        raise ValueError("Variant-A qualification accepts split=validation only.")
    return identity


def _is_forbidden_traveller(record_identity: str, subject_identity: str | None) -> bool:
    normalized = "" if subject_identity is None else str(subject_identity).zfill(4)
    if normalized in FORBIDDEN_TRAVELLER_IDS:
        return True
    upper = record_identity.upper()
    return any(
        f"P_{traveller}" in upper or f"P:{traveller}" in upper
        for traveller in FORBIDDEN_TRAVELLER_IDS
    )


def _validate_accepted_records(provenance: Mapping[str, Any], labels: set[str]) -> None:
    records = provenance["accepted_records"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("Photometry artifact accepted records must be a nonempty sequence.")
    normalized = [dict(item) for item in records if isinstance(item, Mapping)]
    if len(normalized) != len(records):
        raise ValueError("Photometry artifact contains a malformed accepted record.")
    identities: set[str] = set()
    for item in normalized:
        identity = str(item.get("record_identity", ""))
        reconciled = classify_variant_a_cohort(
            case_identity=identity,
            metadata_prefix=item.get("metadata_prefix"),
            supplied_cohort=item.get("cohort"),
            subject_identity=item.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if item.get("split") != "train":
            raise ValueError("Photometry artifact accepted a non-R/train record.")
        if item.get("subject_group_identity") != reconciled.subject_group_identity:
            raise ValueError("Photometry artifact subject-group identity mismatch.")
        if _is_forbidden_traveller(identity, reconciled.subject_identity):
            raise ValueError("Photometry artifact accepted a reserved traveller identity.")
        if not identity or identity in identities:
            raise ValueError("Photometry artifact has a missing or duplicate record identity.")
        identities.add(identity)
        if item.get("record_identity_sha256") != sha256_text(identity):
            raise ValueError("Photometry artifact record-identity hash mismatch.")
        if item.get("domain") not in labels:
            raise ValueError("Photometry artifact accepted an unknown domain.")
        for key in (
            "record_identity_sha256",
            "source_path_identity_sha256",
            "source_file_sha256",
            "canonical_loaded_array_sha256",
        ):
            _validate_sha256(str(item.get(key, "")), key)
    if sha256_json(normalized) != provenance["accepted_records_sha256"]:
        raise ValueError("Photometry accepted-record content hash mismatch.")
    observed_counts = Counter(str(item["domain"]) for item in normalized)
    expected_counts = {
        str(label): int(value)
        for label, value in dict(provenance["domain_volume_counts"]).items()
    }
    if dict(observed_counts) != expected_counts:
        raise ValueError("Photometry accepted-record membership does not match domain counts.")
    excluded = normalize_variant_a_prospective_exclusions(
        provenance["excluded_prospective_records"], expected_split="train"
    )
    if sha256_json(excluded) != provenance["excluded_prospective_records_sha256"]:
        raise ValueError("Photometry excluded-record content hash mismatch.")
    proof = provenance["eligibility_proof"]
    if (
        not isinstance(proof, Mapping)
        or proof.get("all_cohort_R") is not True
        or proof.get("all_split_train") is not True
        or proof.get("accepted_count") != len(normalized)
    ):
        raise ValueError("Photometry artifact eligibility proof is invalid.")
    if (
        proof.get("prospective_accepted_count") != 0
        or proof.get("prospective_excluded_count")
        != len(excluded)
        or proof.get("forbidden_traveller_accepted_count") != 0
    ):
        raise ValueError("Photometry artifact eligibility proof accepted forbidden records.")
    if identities.intersection(item["record_identity"] for item in excluded):
        raise ValueError("Photometry artifact records one identity as accepted and excluded.")


def normalize_variant_a_prospective_exclusions(
    records: Sequence[Mapping[str, Any]], *, expected_split: str
) -> list[dict[str, Any]]:
    """Validate and deterministically order P records excluded before array loading."""

    if expected_split not in {"train", "validation"}:
        raise ValueError("Variant-A exclusion evidence requires train or validation split.")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Variant-A excluded records must be a sequence.")
    required = {
        "record_identity",
        "record_identity_sha256",
        "subject_identity",
        "subject_group_identity",
        "metadata_prefix",
        "cohort",
        "split",
        "source_path_identity_sha256",
        "reason",
    }
    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("Variant-A excluded-record schema is incompatible.")
        item = dict(raw)
        identity = classify_variant_a_cohort(
            case_identity=str(item["record_identity"]),
            metadata_prefix=item["metadata_prefix"],
            supplied_cohort=item["cohort"],
            subject_identity=item["subject_identity"],
            allowed_cohorts=("P",),
        )
        if identity.case_identity in identities:
            raise ValueError("Variant-A exclusion evidence contains a duplicate identity.")
        identities.add(identity.case_identity)
        if item["split"] != expected_split:
            raise ValueError("Variant-A excluded-record split is incompatible.")
        if item["subject_group_identity"] != identity.subject_group_identity:
            raise ValueError("Variant-A excluded-record subject grouping is incompatible.")
        if item["record_identity_sha256"] != sha256_text(identity.case_identity):
            raise ValueError("Variant-A excluded-record identity hash mismatch.")
        _validate_sha256(
            str(item["source_path_identity_sha256"]), "excluded source-path identity"
        )
        if item["reason"] != VARIANT_A_PROSPECTIVE_EXCLUSION_REASON:
            raise ValueError("Variant-A excluded-record reason is incompatible.")
        normalized.append(item)
    normalized.sort(key=lambda item: str(item["record_identity"]))
    return normalized


def _validate_code_provenance(provenance: Any, code_commit: str) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("Photometry code provenance must be a mapping.")
    if provenance.get("checkout_clean") is not True:
        raise ValueError("Photometry fitting requires a clean checkout.")
    if provenance.get("git_head") != code_commit:
        raise ValueError("Photometry fitting commit does not match checkout HEAD.")
    hashes = provenance.get("module_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(PHOTOMETRY_SOURCE_MODULES):
        raise ValueError("Photometry source-module hashes are incomplete.")
    for value in hashes.values():
        _validate_sha256(str(value), "source module")


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Photometry {name} SHA-256 is invalid.")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Resolved photometry config must be a mapping.")
    return decoded


__all__ = [
    "CanonicalCohortIdentity",
    "FORBIDDEN_TRAVELLER_IDS",
    "PHOTOMETRY_CLAMPING_RULE",
    "PHOTOMETRY_DEFAULT_QUANTILES",
    "PHOTOMETRY_DUPLICATE_KNOT_RULE",
    "PHOTOMETRY_FACTORIZATION_CONFIG_VERSION",
    "PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION",
    "PHOTOMETRY_FACTORIZATION_SEMANTICS",
    "PHOTOMETRY_FIT_ELIGIBILITY",
    "PHOTOMETRY_INTERPOLATION_RULE",
    "PHOTOMETRY_QUALIFICATION_ELIGIBILITY",
    "PHOTOMETRY_SOURCE_MODULES",
    "PHOTOMETRY_SUPPORT_POLICY",
    "ContrastCanonicalTemplate",
    "DomainPhotometryTemplate",
    "FrozenPhotometryArtifact",
    "PhotometryFitVolume",
    "SourceCanonicalizedVolume",
    "VARIANT_A_PROSPECTIVE_EXCLUSION_REASON",
    "all_photometry_domain_labels",
    "assert_variant_a_external_path",
    "canonical_tensor_sha256",
    "capture_photometry_code_provenance",
    "classify_variant_a_cohort",
    "deterministic_quantiles",
    "fit_frozen_photometry",
    "interpolate_fixed_grid",
    "normalize_variant_a_prospective_exclusions",
    "reject_target_or_prediction_derived_fields",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "tensor_sha256",
    "validate_qualification_role",
    "write_json_atomic",
]
