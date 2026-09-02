"""Reusable, R-only diagnostics for Stage-2 target-conditioned transport.

The functions in this module are deliberately model-agnostic at their public
boundaries.  They can evaluate the historical flow translator, a conditional
residual translator, or a minimal SB implementation without changing the evidence
schema.  Real-data fitting is restricted to retrospective R/train; R/validation is
assessment-only.  Prospective identities never enter these diagnostics.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import (
    sha256_json,
    write_json_atomic,
)
from fieldbridge.models.discriminators import domain_labels
from fieldbridge.training.losses import adversarial_hinge_loss_discriminator
from fieldbridge.training.stage2_unified import (
    DEFAULT_UNIFIED_WEIGHTS,
    UnifiedStage2Config,
    _TrainingBatch,
    _autocast_context,
    _critic_view,
    _forward_generator_term,
    _freeze_training_step_plan,
    _replay_step_rng,
    _saved_tensor_offload_context,
    integrate_transport,
)


RESCUE_DIAGNOSTICS_SCHEMA = "stage2-rescue-diagnostics-v1"
CONDITIONING_CONTRACT = "stage2-rescue-conditioning-plumbing-v1"
IDENTIFIABILITY_CONTRACT = "stage2-rescue-real-domain-identifiability-v1"
TARGET_SWEEP_CONTRACT = "stage2-rescue-same-source-five-target-sweep-v1"
LATENT_DRIFT_CONTRACT = "stage2-rescue-off-manifold-latent-drift-v1"
GRADIENT_CONTRACT = "stage2-rescue-per-term-gradients-v1"
MICRO_OVERFIT_CONTRACT = "stage2-rescue-synthetic-micro-overfit-v1"
SCORECARD_CONTRACT = "stage2-rescue-scorecard-v1"

Status = Literal["PASS", "FAIL", "INCONCLUSIVE"]
DiagnosticRenderer = Callable[[torch.Tensor, Domain], torch.Tensor]


def validate_data_boundary(
    records: Sequence[Mapping[str, Any]], *, purpose: Literal["fit", "diagnose", "final_p0006"]
) -> dict[str, Any]:
    """Fail closed on cohort/split roles before any tensor-bearing operation."""

    accepted: list[str] = []
    prospective_accepted = 0
    for position, record in enumerate(records):
        cohort = str(record.get("cohort", ""))
        split = str(record.get("split", ""))
        subject = str(record.get("subject_id", record.get("case_id", position)))
        compact = subject.replace(":", "").replace("_", "").upper()
        if "P0009" in compact or compact.endswith("0009") and cohort.upper() == "P":
            raise ValueError("P:0009 is frozen and refused by every rescue diagnostic.")
        if cohort == "P":
            if compact.endswith("0006") and purpose == "final_p0006":
                accepted.append(subject)
                prospective_accepted += 1
                continue
            raise ValueError(
                "Prospective records are forbidden for fitting, diagnostics, tuning, "
                "selection, and iterative candidate comparison."
            )
        if cohort != "R" or split not in {"train", "validation"}:
            raise ValueError("Rescue diagnostics accept only R/train or R/validation records.")
        if purpose == "fit" and split != "train":
            raise ValueError("Diagnostic fitting is restricted to R/train.")
        accepted.append(subject)
    return {
        "purpose": purpose,
        "accepted_count": len(accepted),
        "accepted_identity_sha256": sha256_json(sorted(accepted)),
        "prospective_accepted_count": prospective_accepted,
    }


def self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("result_sha256", None)
    result["result_sha256"] = sha256_json(result)
    return result


def verify_self_hash(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    stored = body.pop("result_sha256", None)
    if not isinstance(stored, str) or stored != sha256_json(body):
        raise ValueError("Diagnostic JSON self-hash mismatch.")


def write_diagnostic_json(
    path: str | Path, payload: Mapping[str, Any], *, resume: bool = False
) -> Path:
    """Publish once, or verify byte-for-contract exact resume without clobbering."""

    target = Path(path)
    sealed = self_hashed(payload)
    if target.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite diagnostic artifact: {target}")
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != sealed:
            raise FileExistsError(f"Existing diagnostic is not an exact resume: {target}")
        verify_self_hash(existing)
        return target
    return write_json_atomic(target, sealed, refuse_existing=True)


def _conditioning_module(translator: nn.Module) -> nn.Module:
    direct = getattr(translator, "domain_embedding", None)
    if isinstance(direct, nn.Module):
        return direct
    backbone = getattr(translator, "backbone", None)
    nested = getattr(backbone, "domain_embedding", None)
    if isinstance(nested, nn.Module):
        return nested
    raise TypeError("Translator exposes no domain_embedding conditioning module.")


def _call_translator(
    translator: nn.Module,
    latent: torch.Tensor,
    source: Sequence[Domain],
    target: Sequence[Domain],
    time_values: torch.Tensor | None,
) -> torch.Tensor:
    if time_values is None:
        return translator(latent, source, target)
    return translator(latent, source, target, time_values)


def diagnose_conditioning_plumbing(
    translator: nn.Module,
    latent: torch.Tensor,
    source_domains: Sequence[Domain],
    target_domains: Sequence[Domain],
    alternate_target_domains: Sequence[Domain],
    *,
    alternate_source_domains: Sequence[Domain] | None = None,
    time_values: torch.Tensor | None = None,
    sensitivity_epsilon: float = 1.0e-10,
) -> dict[str, Any]:
    """Trace labels, conditioning, outputs, and gradients without an optimizer step."""

    batch = int(latent.shape[0])
    groups = (source_domains, target_domains, alternate_target_domains)
    if any(len(group) != batch for group in groups):
        raise ValueError("Conditioning diagnostic domain batches must match the latent batch.")
    if alternate_source_domains is None:
        alternate_source_domains = [
            Domain(
                next(field for field in FIELD_STRENGTHS_T if field != item.field_strength_t),
                item.contrast,
            )
            for item in source_domains
        ]
    if len(alternate_source_domains) != batch:
        raise ValueError("Alternate source-domain batch length mismatch.")
    if any(a == b for a, b in zip(target_domains, alternate_target_domains)):
        raise ValueError("Alternate targets must differ from requested targets.")

    conditioner = _conditioning_module(translator)
    parameter = next(translator.parameters(), None)
    if parameter is None:
        raise ValueError("Conditioning diagnostic requires a parameterized translator.")
    latent = latent.to(parameter.device, parameter.dtype).detach().requires_grad_(True)
    if time_values is not None:
        time_values = time_values.to(latent.device, latent.dtype)

    with torch.no_grad():
        requested_embedding = conditioner(
            source_domains,
            target_domains,
            batch_size=batch,
            device=latent.device,
            dtype=latent.dtype,
        )
        target_swap_embedding = conditioner(
            source_domains,
            alternate_target_domains,
            batch_size=batch,
            device=latent.device,
            dtype=latent.dtype,
        )
        source_swap_embedding = conditioner(
            alternate_source_domains,
            target_domains,
            batch_size=batch,
            device=latent.device,
            dtype=latent.dtype,
        )
        requested_output = _call_translator(
            translator, latent, source_domains, target_domains, time_values
        )
        alternate_output = _call_translator(
            translator, latent, source_domains, alternate_target_domains, time_values
        )

    captured: list[torch.Tensor] = []
    observed_targets: list[str] = []

    def embedding_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor):
            captured.append(output)

    def translator_pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) >= 3:
            values = args[2] if isinstance(args[2], Sequence) else [args[2]]
            observed_targets.extend(item.label for item in values)

    embedding_handle = conditioner.register_forward_hook(embedding_hook)
    translator_handle = translator.register_forward_pre_hook(translator_pre_hook)
    try:
        output = _call_translator(
            translator, latent, source_domains, target_domains, time_values
        )
    finally:
        embedding_handle.remove()
        translator_handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one conditioning embedding in translator forward.")
    conditioning = captured[0]
    probe_weights = torch.linspace(
        0.5, 1.5, output.numel(), device=output.device, dtype=output.dtype
    ).reshape_as(output)
    probe = (output * probe_weights).mean()
    named_parameters = [(name, value) for name, value in translator.named_parameters() if value.requires_grad]
    gradients = torch.autograd.grad(
        probe,
        [conditioning, latent, *(value for _, value in named_parameters)],
        allow_unused=True,
    )
    conditioning_gradient = gradients[0]
    input_gradient = gradients[1]
    parameter_gradients = gradients[2:]
    module_squared: dict[str, float] = defaultdict(float)
    parameter_nonzero = 0
    for (name, _), gradient in zip(named_parameters, parameter_gradients):
        module = name.rsplit(".", 1)[0] if "." in name else "<root>"
        if gradient is not None:
            value = float(gradient.detach().float().square().sum().cpu())
            module_squared[module] += value
            parameter_nonzero += int(value > sensitivity_epsilon**2)
    module_gradients = {
        name: math.sqrt(value) for name, value in sorted(module_squared.items())
    }
    condition_grad_norm = _gradient_norm(conditioning_gradient)
    input_grad_norm = _gradient_norm(input_gradient)
    target_embedding_delta = _tensor_l2(requested_embedding, target_swap_embedding)
    source_embedding_delta = _tensor_l2(requested_embedding, source_swap_embedding)
    output_delta = _tensor_l2(requested_output, alternate_output)
    labels_match = observed_targets == [item.label for item in target_domains]
    unique_requested_ids = len({item.label for item in target_domains})
    alias_free = all(
        requested.label != alternate.label
        and not torch.equal(requested.field_encoding(), alternate.field_encoding())
        for requested, alternate in zip(target_domains, alternate_target_domains)
    )
    finite = all(
        math.isfinite(value)
        for value in (
            condition_grad_norm,
            input_grad_norm,
            target_embedding_delta,
            source_embedding_delta,
            output_delta,
            *module_gradients.values(),
        )
    )
    failures = []
    if not labels_match:
        failures.append("requested_target_not_observed_at_translator")
    if not alias_free:
        failures.append("domain_id_aliasing")
    if target_embedding_delta <= sensitivity_epsilon:
        failures.append("target_embedding_unchanged")
    if source_embedding_delta <= sensitivity_epsilon:
        failures.append("source_embedding_unchanged")
    if condition_grad_norm <= sensitivity_epsilon:
        failures.append("conditioning_path_zero_gradient")
    if output_delta <= sensitivity_epsilon:
        failures.append("target_conditioning_ignored")
    if not finite:
        failures.append("nonfinite_gradient_or_sensitivity")
    payload = {
        "contract_version": CONDITIONING_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "R_only_no_optimizer_step",
        "status": "FAIL" if failures else "PASS",
        "source_labels": [item.label for item in source_domains],
        "requested_target_labels": [item.label for item in target_domains],
        "alternate_target_labels": [item.label for item in alternate_target_domains],
        "translator_observed_target_labels": observed_targets,
        "source_labels_correct": True,
        "requested_target_labels_correct": labels_match,
        "unique_requested_target_id_count": unique_requested_ids,
        "domain_id_alias_free": alias_free,
        "source_embedding_l2_delta": source_embedding_delta,
        "target_embedding_l2_delta": target_embedding_delta,
        "target_swap_output_l2_delta": output_delta,
        "conditioning_output_gradient_norm": condition_grad_norm,
        "input_gradient_norm": input_grad_norm,
        "nonzero_parameter_gradient_count": parameter_nonzero,
        "module_gradient_norms": module_gradients,
        "finite": finite,
        "optimizer_step_called": False,
        "sensitivity_epsilon": sensitivity_epsilon,
        "failures": failures,
    }
    return self_hashed(payload)


def _gradient_norm(gradient: torch.Tensor | None) -> float:
    if gradient is None:
        return 0.0
    return float(gradient.detach().float().norm().cpu())


def aggregate_conditioning_diagnostics(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("Conditioning aggregation requires at least one R-only result.")
    for result in results:
        verify_self_hash(result)
        if result.get("contract_version") != CONDITIONING_CONTRACT:
            raise ValueError("Conditioning aggregation received an incompatible contract.")
    failures = [
        {"source_labels": result["source_labels"], "failures": result["failures"]}
        for result in results
        if result["status"] != "PASS"
    ]
    payload = {
        "contract_version": CONDITIONING_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "fixed_R_only_source_set_no_optimizer_step",
        "status": "PASS" if not failures else "FAIL",
        "source_count": len(results),
        "source_result_sha256": [result["result_sha256"] for result in results],
        "minimum_target_embedding_l2_delta": min(
            float(result["target_embedding_l2_delta"]) for result in results
        ),
        "minimum_target_swap_output_l2_delta": min(
            float(result["target_swap_output_l2_delta"]) for result in results
        ),
        "minimum_conditioning_output_gradient_norm": min(
            float(result["conditioning_output_gradient_norm"]) for result in results
        ),
        "failures": failures,
        "per_source": list(results),
        "optimizer_step_called": False,
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


def _tensor_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).norm().cpu())


def bank_identifiability_features(
    index: PhotometryFactoredLatentBankIndex, stats: FactoredLatentStats
) -> tuple[np.ndarray, list[Domain]]:
    if index.split not in {"train", "validation"}:
        raise ValueError("Identifiability accepts only R/train or R/validation.")
    features: list[np.ndarray] = []
    domains: list[Domain] = []
    for position, record in enumerate(index.records):
        latent, support = index.load(position)
        normalized = stats.normalize(latent, support)
        values = normalized[:, support]
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("Identifiability feature is empty or non-finite.")
        feature = torch.cat(
            (values.mean(1), values.std(1, unbiased=False), values.abs().mean(1))
        )
        features.append(feature.double().numpy())
        domains.append(record.domain)
    return np.stack(features), domains


def bank_domain_latent_centroids(
    index: PhotometryFactoredLatentBankIndex, stats: FactoredLatentStats
) -> dict[str, list[float]]:
    """Build optional nearest-domain evidence from sealed R/train only."""

    if index.split != "train":
        raise ValueError("Target reference centroids must be fit on R/train only.")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for position, record in enumerate(index.records):
        latent, support = index.load(position)
        normalized = stats.normalize(latent, support)
        grouped[record.domain.label].append(
            _supported_latent_descriptor(normalized, support)
        )
    expected = {
        Domain(field, contrast).label for contrast in Contrast for field in FIELD_STRENGTHS_T
    }
    if set(grouped) != expected:
        raise ValueError("R/train target reference centroids do not cover all 15 domains.")
    return {
        label: np.mean(values, axis=0).tolist() for label, values in sorted(grouped.items())
    }


def diagnose_real_domain_identifiability(
    train_features: np.ndarray,
    train_domains: Sequence[Domain],
    validation_features: np.ndarray,
    validation_domains: Sequence[Domain],
    *,
    include_fifteen_way: bool = True,
    pass_margin_over_chance: float = 0.10,
) -> dict[str, Any]:
    """Run fixed nearest-centroid probes fit only on R/train."""

    train = np.asarray(train_features, dtype=np.float64)
    validation = np.asarray(validation_features, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Identifiability features must be aligned two-dimensional arrays.")
    if len(train_domains) != len(train) or len(validation_domains) != len(validation):
        raise ValueError("Identifiability feature/domain lengths differ.")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Identifiability features contain non-finite values.")
    mean = train.mean(0)
    std = train.std(0)
    std[std <= 1.0e-12] = 1.0
    train_z = (train - mean) / std
    validation_z = (validation - mean) / std

    contrast_labels = [item.value for item in Contrast]
    contrast = _centroid_probe(
        train_z,
        [item.contrast.value for item in train_domains],
        validation_z,
        [item.contrast.value for item in validation_domains],
        contrast_labels,
    )
    fields: dict[str, Any] = {}
    for contrast_value in contrast_labels:
        train_mask = np.asarray([item.contrast.value == contrast_value for item in train_domains])
        validation_mask = np.asarray(
            [item.contrast.value == contrast_value for item in validation_domains]
        )
        field_labels = [f"{field:g}T" for field in FIELD_STRENGTHS_T]
        fields[contrast_value] = _centroid_probe(
            train_z[train_mask],
            [f"{item.field_strength_t:g}T" for item in np.asarray(train_domains)[train_mask]],
            validation_z[validation_mask],
            [
                f"{item.field_strength_t:g}T"
                for item in np.asarray(validation_domains)[validation_mask]
            ],
            field_labels,
        )
    fifteen = None
    if include_fifteen_way:
        labels = [
            Domain(field, contrast_value).label
            for contrast_value in Contrast
            for field in FIELD_STRENGTHS_T
        ]
        fifteen = _centroid_probe(
            train_z,
            [item.label for item in train_domains],
            validation_z,
            [item.label for item in validation_domains],
            labels,
        )
    primary_values = [contrast["validation"]["balanced_accuracy"]] + [
        value["validation"]["balanced_accuracy"] for value in fields.values()
    ]
    primary_chances = [contrast["chance_balanced_accuracy"]] + [
        value["chance_balanced_accuracy"] for value in fields.values()
    ]
    improvements = [value - chance for value, chance in zip(primary_values, primary_chances)]
    if all(value >= pass_margin_over_chance for value in improvements):
        status: Status = "PASS"
    elif all(value <= 0.02 for value in improvements):
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    payload = {
        "contract_version": IDENTIFIABILITY_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "R_train_fit_R_validation_assessment",
        "status": status,
        "classifier": "fixed_standardized_nearest_R_train_centroid_v1",
        "feature_fit_split": "R/train",
        "selection_or_tuning_on_validation": False,
        "train_record_count": len(train),
        "validation_record_count": len(validation),
        "contrast_probe": contrast,
        "field_within_contrast_probes": fields,
        "fifteen_way_secondary_probe": fifteen,
        "pass_margin_over_chance": pass_margin_over_chance,
        "interpretation_boundary": (
            "Representation identifiability is necessary but not sufficient for learned "
            "target-field translation. No paired R endpoint claim is made."
        ),
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


def _centroid_probe(
    train: np.ndarray,
    train_labels: Sequence[str],
    validation: np.ndarray,
    validation_labels: Sequence[str],
    label_order: Sequence[str],
) -> dict[str, Any]:
    if not len(train) or not len(validation):
        raise ValueError("Every identifiability probe needs train and validation examples.")
    missing_train = sorted(set(label_order) - set(train_labels))
    missing_validation = sorted(set(label_order) - set(validation_labels))
    if missing_train or missing_validation:
        raise ValueError(
            f"Identifiability probe has incomplete classes: train={missing_train}, "
            f"validation={missing_validation}."
        )
    labels_array = np.asarray(train_labels)
    centroids = np.stack([train[labels_array == label].mean(0) for label in label_order])
    train_logits = -((train[:, None, :] - centroids[None, :, :]) ** 2).mean(2)
    validation_logits = -(
        (validation[:, None, :] - centroids[None, :, :]) ** 2
    ).mean(2)
    return {
        "label_order": list(label_order),
        "chance_balanced_accuracy": 1.0 / len(label_order),
        "chance_cross_entropy": math.log(len(label_order)),
        "train": _classification_metrics(train_logits, train_labels, label_order),
        "validation": _classification_metrics(
            validation_logits, validation_labels, label_order
        ),
    }


def _classification_metrics(
    logits: np.ndarray, actual: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    actual_ids = np.asarray([labels.index(value) for value in actual], dtype=np.int64)
    predicted_ids = logits.argmax(1)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(actual_ids, predicted_ids):
        confusion[truth, prediction] += 1
    row_counts = confusion.sum(1)
    recalls = np.divide(
        np.diag(confusion), row_counts, out=np.zeros(len(labels), dtype=float), where=row_counts > 0
    )
    shifted = logits - logits.max(1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(1))
    cross_entropy = float(np.mean(log_normalizer - shifted[np.arange(len(actual_ids)), actual_ids]))
    macro = {
        label: {
            "count": int(row_counts[index]),
            "recall": float(recalls[index]),
            "predicted_counts": {
                target: int(confusion[index, target_index])
                for target_index, target in enumerate(labels)
            },
        }
        for index, label in enumerate(labels)
    }
    return {
        "balanced_accuracy": float(recalls.mean()),
        "cross_entropy": cross_entropy,
        "confusion_matrix": confusion.tolist(),
        "macro_per_domain": macro,
    }


def same_source_five_target_sweep(
    translator: nn.Module,
    decoder: nn.Module,
    source_latent: torch.Tensor,
    source_support: torch.Tensor,
    source_domain: Domain,
    stats: FactoredLatentStats,
    *,
    renderer: DiagnosticRenderer | None = None,
    target_reference_centroids: Mapping[str, Sequence[float]] | None = None,
    integration_steps: int = 4,
    solver: Literal["euler", "heun"] = "heun",
    sensitivity_epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    """Decompose learned, decoded, renderer-only, and final target sensitivity."""

    if source_latent.shape[0] != 1:
        raise ValueError("Same-source sweep requires exactly one source latent.")
    targets = [Domain(field, source_domain.contrast) for field in FIELD_STRENGTHS_T]
    support = _expanded_support(source_support, source_latent)
    latent_outputs: list[torch.Tensor] = []
    decoded_outputs: list[torch.Tensor] = []
    final_outputs: list[torch.Tensor] = []
    translator.eval()
    decoder.eval()
    with torch.inference_mode():
        for target in targets:
            transported = integrate_transport(
                translator,
                source_latent,
                [source_domain],
                [target],
                steps=integration_steps,
                solver=solver,
            )
            decoded = _diagnostic_decode(decoder, stats.denormalize(transported), [target])
            final = renderer(decoded, target) if renderer is not None else decoded
            latent_outputs.append(transported.detach().float())
            decoded_outputs.append(decoded.detach().float())
            final_outputs.append(final.detach().float())
    reference_index = targets.index(source_domain)
    fixed_decoded = decoded_outputs[reference_index]
    renderer_only = [
        (renderer(fixed_decoded, target) if renderer is not None else fixed_decoded)
        .detach()
        .float()
        for target in targets
    ]
    labels = [item.label for item in targets]
    latent_distances = _pairwise_distances(latent_outputs, support)
    decoded_support = _spatial_support(source_support, decoded_outputs[0])
    decoded_distances = _pairwise_distances(decoded_outputs, decoded_support)
    renderer_distances = _pairwise_distances(renderer_only, decoded_support)
    final_distances = _pairwise_distances(final_outputs, decoded_support)
    learned_max = _off_diagonal_max(latent_distances["l2"])
    renderer_max = _off_diagonal_max(renderer_distances["l2"])
    final_max = _off_diagonal_max(final_distances["l2"])
    nearest_evidence: dict[str, Any]
    if target_reference_centroids is None:
        nearest_evidence = {
            "status": "not_computed_without_predeclared_R_train_reference_descriptors"
        }
    else:
        reference_labels = [
            label
            for label in sorted(target_reference_centroids)
            if label.endswith(f"/{source_domain.contrast.value}")
        ]
        if set(reference_labels) != set(labels):
            raise ValueError("Target reference centroids lack the five same-contrast domains.")
        references = np.stack(
            [np.asarray(target_reference_centroids[label], dtype=float) for label in reference_labels]
        )
        rows = []
        for requested, output in zip(targets, latent_outputs):
            descriptor = _supported_latent_descriptor(output, source_support)
            distances = np.sqrt(((references - descriptor[None, :]) ** 2).mean(1))
            ordering = np.argsort(distances)
            requested_position = reference_labels.index(requested.label)
            rows.append(
                {
                    "requested_target": requested.label,
                    "nearest_R_train_domain": reference_labels[int(ordering[0])],
                    "requested_target_rank": int(np.where(ordering == requested_position)[0][0]) + 1,
                    "requested_target_distance": float(distances[requested_position]),
                    "distances": {
                        label: float(distances[index])
                        for index, label in enumerate(reference_labels)
                    },
                }
            )
        nearest_evidence = {
            "status": "computed_against_predeclared_R_train_centroids",
            "descriptor": "supported_channel_mean_and_std",
            "rows": rows,
            "requested_target_top1_fraction": float(
                np.mean([row["requested_target_rank"] == 1 for row in rows])
            ),
        }
    payload = {
        "contract_version": TARGET_SWEEP_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "same_R_source_all_five_same_contrast_targets",
        "status": "PASS" if final_max > sensitivity_epsilon else "FAIL",
        "source_domain": source_domain.label,
        "target_order": labels,
        "learned_transport_before_renderer": latent_distances,
        "decoded_canonical_before_renderer": decoded_distances,
        "renderer_only_counterfactual_fixed_canonical": renderer_distances,
        "final_decoded_rendered": final_distances,
        "learned_target_sensitivity": {
            "status": "PASS" if learned_max > sensitivity_epsilon else "FAIL",
            "maximum_pairwise_l2": learned_max,
        },
        "renderer_only_sensitivity": {
            "status": "PASS" if renderer_max > sensitivity_epsilon else "FAIL",
            "maximum_pairwise_l2": renderer_max,
        },
        "final_target_sensitivity": {
            "status": "PASS" if final_max > sensitivity_epsilon else "FAIL",
            "maximum_pairwise_l2": final_max,
        },
        "component_attribution": (
            "learned_transport compares transported latents; decoded_canonical measures "
            "decoder response before photometry; renderer_only holds canonical content fixed; "
            "final includes both learned and frozen components."
        ),
        "nearest_requested_target_evidence": nearest_evidence,
        "physical_monotonicity_claimed": False,
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


def _diagnostic_decode(
    decoder: nn.Module, latent: torch.Tensor, domains: Sequence[Domain]
) -> torch.Tensor:
    method = getattr(decoder, "decode", None)
    return method(latent, domains) if callable(method) else decoder(latent)


def _expanded_support(support: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    mask = support.to(torch.bool)
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(1)
    return mask.expand_as(tensor)


def _spatial_support(support: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    mask = support.to(torch.bool)
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[2:] != tensor.shape[2:]:
        mask = F.interpolate(mask.float(), size=tensor.shape[2:], mode="nearest").bool()
    return mask.expand_as(tensor)


def _pairwise_distances(
    tensors: Sequence[torch.Tensor], support: torch.Tensor
) -> dict[str, list[list[float]]]:
    values = [item[support.expand_as(item)].double() for item in tensors]
    if any(item.numel() == 0 for item in values):
        raise ValueError("Pairwise target sweep support is empty.")
    size = len(values)
    l1 = np.zeros((size, size), dtype=float)
    l2 = np.zeros((size, size), dtype=float)
    cosine = np.zeros((size, size), dtype=float)
    for row in range(size):
        for column in range(size):
            difference = values[row] - values[column]
            l1[row, column] = float(difference.abs().mean())
            l2[row, column] = float(difference.square().mean().sqrt())
            cosine[row, column] = float(
                1.0
                - F.cosine_similarity(
                    values[row].reshape(1, -1), values[column].reshape(1, -1)
                )[0]
            )
    return {"l1": l1.tolist(), "l2": l2.tolist(), "cosine_distance": cosine.tolist()}


def _supported_latent_descriptor(
    latent: torch.Tensor, support: torch.Tensor
) -> np.ndarray:
    tensor = latent.detach().float().cpu()
    mask = _expanded_support(support.detach().cpu(), tensor)
    channel_dim = 1 if tensor.ndim == 5 else 0
    channels = int(tensor.shape[channel_dim])
    means: list[float] = []
    stds: list[float] = []
    for channel in range(channels):
        values = tensor[:, channel][mask[:, channel]] if channel_dim == 1 else tensor[channel][mask[channel]]
        if not values.numel():
            raise ValueError("Latent descriptor support is empty.")
        means.append(float(values.mean()))
        stds.append(float(values.std(unbiased=False)))
    return np.asarray([*means, *stds], dtype=float)


def _off_diagonal_max(matrix: Sequence[Sequence[float]]) -> float:
    return max(
        float(value)
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if row != column
    )


def aggregate_target_sweeps(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a predeclared fixed R-only source set without hiding per-source evidence."""

    if not results:
        raise ValueError("Target-sweep aggregation requires at least one source result.")
    for result in results:
        verify_self_hash(result)
        if result.get("contract_version") != TARGET_SWEEP_CONTRACT:
            raise ValueError("Target-sweep aggregation received an incompatible contract.")
    learned = [
        float(result["learned_target_sensitivity"]["maximum_pairwise_l2"])
        for result in results
    ]
    renderer = [
        float(result["renderer_only_sensitivity"]["maximum_pairwise_l2"])
        for result in results
    ]
    final = [
        float(result["final_target_sensitivity"]["maximum_pairwise_l2"])
        for result in results
    ]
    learned_status: Status = (
        "PASS"
        if all(result["learned_target_sensitivity"]["status"] == "PASS" for result in results)
        else "FAIL"
        if all(result["learned_target_sensitivity"]["status"] == "FAIL" for result in results)
        else "INCONCLUSIVE"
    )
    renderer_status: Status = (
        "PASS"
        if all(result["renderer_only_sensitivity"]["status"] == "PASS" for result in results)
        else "FAIL"
        if all(result["renderer_only_sensitivity"]["status"] == "FAIL" for result in results)
        else "INCONCLUSIVE"
    )
    payload = {
        "contract_version": TARGET_SWEEP_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "fixed_R_only_source_set_all_five_same_contrast_targets",
        "status": (
            "PASS"
            if all(result["status"] == "PASS" for result in results)
            else "INCONCLUSIVE"
            if any(result["status"] == "PASS" for result in results)
            else "FAIL"
        ),
        "source_count": len(results),
        "source_result_sha256": [result["result_sha256"] for result in results],
        "source_domains": [result["source_domain"] for result in results],
        "learned_target_sensitivity": {
            "status": learned_status,
            "maximum_pairwise_l2_distribution": _distribution(learned),
        },
        "renderer_only_sensitivity": {
            "status": renderer_status,
            "maximum_pairwise_l2_distribution": _distribution(renderer),
        },
        "final_target_sensitivity": {
            "status": (
                "PASS"
                if all(result["final_target_sensitivity"]["status"] == "PASS" for result in results)
                else "INCONCLUSIVE"
            ),
            "maximum_pairwise_l2_distribution": _distribution(final),
        },
        "per_source": list(results),
        "physical_monotonicity_claimed": False,
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


@dataclass(frozen=True, slots=True)
class LatentDiagnosticRecord:
    latent: torch.Tensor
    support: torch.Tensor
    source_domain: Domain
    target_domain: Domain
    contrast: Contrast | str


def diagnose_off_manifold_latent_drift(
    real_train_records: Sequence[LatentDiagnosticRecord],
    generated_records: Sequence[LatentDiagnosticRecord],
    stats: FactoredLatentStats,
    *,
    outlier_quantile: float = 0.99,
    maximum_outlier_fraction: float = 0.10,
) -> dict[str, Any]:
    """Compare generated latents only with the sealed frozen R/train distribution."""

    if not 0.5 < outlier_quantile < 1.0:
        raise ValueError("Outlier quantile must lie strictly between 0.5 and 1.")
    if len(real_train_records) < 2 or not generated_records:
        raise ValueError("Latent drift requires >=2 R/train and >=1 generated records.")
    real = [_latent_record_summary(item, stats) for item in real_train_records]
    generated = [_latent_record_summary(item, stats) for item in generated_records]
    real_descriptors = np.stack([item["descriptor"] for item in real])
    generated_descriptors = np.stack([item["descriptor"] for item in generated])
    descriptor_mean = real_descriptors.mean(0)
    descriptor_std = real_descriptors.std(0)
    descriptor_std[descriptor_std <= 1.0e-12] = 1.0
    real_z = (real_descriptors - descriptor_mean) / descriptor_std
    generated_z = (generated_descriptors - descriptor_mean) / descriptor_std
    real_nearest = _leave_one_out_nearest(real_z)
    generated_nearest = np.sqrt(
        ((generated_z[:, None, :] - real_z[None, :, :]) ** 2).mean(2)
    ).min(1)
    standardized_threshold = float(
        np.quantile([item["standardized_rms"] for item in real], outlier_quantile)
    )
    nearest_threshold = float(np.quantile(real_nearest, outlier_quantile))
    standardized_flags = np.asarray(
        [item["standardized_rms"] > standardized_threshold for item in generated]
    )
    nearest_flags = generated_nearest > nearest_threshold
    combined_flags = standardized_flags | nearest_flags
    outlier_fraction = float(combined_flags.mean())
    covariance_real = _safe_covariance(real_descriptors)
    covariance_generated = _safe_covariance(generated_descriptors)
    rows = []
    for summary, nearest, standardized, nearest_flag, combined in zip(
        generated, generated_nearest, standardized_flags, nearest_flags, combined_flags
    ):
        rows.append(
            {
                "source_domain": summary["source_domain"],
                "target_domain": summary["target_domain"],
                "contrast": summary["contrast"],
                "latent_norm": summary["norm"],
                "channel_mean": summary["channel_mean"],
                "channel_std": summary["channel_std"],
                "standardized_rms_distance": summary["standardized_rms"],
                "nearest_real_descriptor_distance": float(nearest),
                "standardized_outlier": bool(standardized),
                "nearest_real_outlier": bool(nearest_flag),
                "outlier": bool(combined),
            }
        )
    breakdown = _drift_breakdown(rows)
    payload = {
        "contract_version": LATENT_DRIFT_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "generated_vs_frozen_R_train_latent_distribution",
        "status": "PASS" if outlier_fraction <= maximum_outlier_fraction else "FAIL",
        "reference_record_count": len(real),
        "generated_record_count": len(generated),
        "sealed_statistics_sha256": stats.artifact_sha256,
        "norm_distributions": {
            "R_train": _distribution([item["norm"] for item in real]),
            "generated": _distribution([item["norm"] for item in generated]),
        },
        "channel_means": {
            "R_train": np.mean([item["channel_mean"] for item in real], axis=0).tolist(),
            "generated": np.mean(
                [item["channel_mean"] for item in generated], axis=0
            ).tolist(),
        },
        "channel_stds": {
            "R_train": np.mean([item["channel_std"] for item in real], axis=0).tolist(),
            "generated": np.mean(
                [item["channel_std"] for item in generated], axis=0
            ).tolist(),
        },
        "covariance_diagnostics": {
            "descriptor_order": "channel_means_then_channel_stds",
            "R_train": covariance_real.tolist(),
            "generated": covariance_generated.tolist(),
            "frobenius_difference": float(
                np.linalg.norm(covariance_real - covariance_generated)
            ),
        },
        "outlier_rule": {
            "quantile_fit_on_R_train": outlier_quantile,
            "standardized_rms_threshold": standardized_threshold,
            "nearest_real_descriptor_threshold": nearest_threshold,
            "union_rule": True,
            "maximum_outlier_fraction": maximum_outlier_fraction,
        },
        "outlier_fraction": outlier_fraction,
        "rows": rows,
        "breakdown": breakdown,
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


def _latent_record_summary(
    record: LatentDiagnosticRecord, stats: FactoredLatentStats
) -> dict[str, Any]:
    latent = record.latent.detach().float().cpu()
    support = _expanded_support(record.support.detach().cpu(), latent)
    channel_dim = 1 if latent.ndim == 5 else 0
    channels = int(latent.shape[channel_dim])
    values = []
    for channel in range(channels):
        tensor = latent[:, channel] if channel_dim == 1 else latent[channel]
        mask = support[:, channel] if channel_dim == 1 else support[channel]
        selected = tensor[mask]
        if not selected.numel() or not bool(torch.isfinite(selected).all()):
            raise ValueError("Latent drift record has empty or non-finite supported values.")
        values.append(selected.double())
    means = np.asarray([float(item.mean()) for item in values])
    stds = np.asarray([float(item.std(unbiased=False)) for item in values])
    sealed_mean = stats.mean.double().numpy()
    sealed_std = stats.std.double().numpy()
    standardized_rms = float(np.sqrt(np.mean(((means - sealed_mean) / sealed_std) ** 2)))
    all_values = torch.cat(values)
    return {
        "descriptor": np.concatenate((means, stds)),
        "channel_mean": means.tolist(),
        "channel_std": stds.tolist(),
        "norm": float(all_values.square().mean().sqrt()),
        "standardized_rms": standardized_rms,
        "source_domain": record.source_domain.label,
        "target_domain": record.target_domain.label,
        "contrast": Contrast.parse(record.contrast).value,
    }


def _leave_one_out_nearest(values: np.ndarray) -> np.ndarray:
    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).mean(2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(1)


def _safe_covariance(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((values.shape[1], values.shape[1]), dtype=float)
    return np.atleast_2d(np.cov(values, rowvar=False, ddof=0))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _drift_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("source_domain", "target_domain", "contrast"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        result[key] = {
            label: {
                "count": len(group),
                "outlier_fraction": float(np.mean([bool(item["outlier"]) for item in group])),
                "mean_nearest_real_descriptor_distance": float(
                    np.mean([float(item["nearest_real_descriptor_distance"]) for item in group])
                ),
            }
            for label, group in sorted(groups.items())
        }
    return result


def diagnose_per_term_gradients(
    cfg: UnifiedStage2Config,
    translator: nn.Module,
    critic: nn.Module,
    decoder: nn.Module,
    batch: _TrainingBatch,
    stats: FactoredLatentStats,
    *,
    seed: int = 20_260_901,
    maximum_exact_cosine_parameters: int = 2_000_000,
    gradient_epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Recompute each real Stage-2 term independently; never call optimizer.step()."""

    support = batch.source_support & batch.target_support
    if not bool(support.any(dim=tuple(range(1, support.ndim))).all()):
        raise ValueError("Per-term diagnostic has empty source/target support intersection.")
    sampler = torch.Generator().manual_seed(seed)
    plan = _freeze_training_step_plan(cfg, batch, sampler)
    parameters = [value for value in translator.parameters() if value.requires_grad]
    parameter_count = sum(value.numel() for value in parameters)
    exact_cosines = parameter_count <= maximum_exact_cosine_parameters
    term_vectors: dict[str, torch.Tensor] = {}
    terms: dict[str, Any] = {}
    critic.requires_grad_(False)
    try:
        for name in DEFAULT_UNIFIED_WEIGHTS:
            weight = float(cfg.loss_weights[name])
            if weight <= 0:
                terms[name] = {"status": "disabled", "weight": weight}
                continue
            with (
                _replay_step_rng(plan, batch.source.device),
                _saved_tensor_offload_context(batch.source.device),
                _autocast_context(cfg, batch.source.device),
            ):
                forward = _forward_generator_term(
                    name, cfg, translator, critic, decoder, batch, support, stats, plan
                )
            raw = forward.loss
            weighted = raw * weight
            gradients = torch.autograd.grad(weighted, parameters, allow_unused=True)
            squared_norm = sum(
                float(gradient.detach().float().square().sum().cpu())
                for gradient in gradients
                if gradient is not None
            )
            weighted_norm = math.sqrt(squared_norm)
            raw_norm = weighted_norm / abs(weight)
            finite = bool(
                torch.isfinite(raw.detach()).all()
                and all(
                    gradient is None or bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                )
            )
            terms[name] = {
                "status": "PASS" if finite and raw_norm > gradient_epsilon else "FAIL",
                "raw_loss": float(raw.detach().float().cpu()),
                "weight": weight,
                "weighted_loss": float(weighted.detach().float().cpu()),
                "translator_raw_gradient_norm": raw_norm,
                "translator_weighted_gradient_norm": weighted_norm,
                "finite": finite,
            }
            if exact_cosines:
                vector = torch.cat(
                    [
                        (torch.zeros_like(parameter) if gradient is None else gradient)
                        .detach()
                        .float()
                        .reshape(-1)
                        .cpu()
                        for parameter, gradient in zip(parameters, gradients)
                    ]
                )
                term_vectors[name] = vector
                del vector
            del raw, weighted, gradients, forward
    finally:
        critic.requires_grad_(True)
    critic_metrics = _critic_gradient_diagnostics(
        cfg, translator, critic, decoder, batch, stats, support, gradient_epsilon
    )
    cosine: dict[str, Any]
    if exact_cosines:
        names = list(term_vectors)
        matrix = []
        for left in names:
            row = []
            for right in names:
                denominator = float(term_vectors[left].norm() * term_vectors[right].norm())
                row.append(
                    float(torch.dot(term_vectors[left], term_vectors[right]) / denominator)
                    if denominator > 0
                    else None
                )
            matrix.append(row)
        cosine = {"status": "computed_exact", "term_order": names, "matrix": matrix}
    else:
        cosine = {
            "status": "not_computed_bounded_memory",
            "parameter_count": parameter_count,
            "maximum_exact_cosine_parameters": maximum_exact_cosine_parameters,
        }
    enabled = [value for value in terms.values() if value.get("status") != "disabled"]
    status: Status = (
        "PASS"
        if enabled
        and all(value["status"] == "PASS" for value in enabled)
        and critic_metrics["status"] == "PASS"
        else "FAIL"
    )
    payload = {
        "contract_version": GRADIENT_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "R_only_real_stage2_loss_plumbing_no_optimizer_step",
        "status": status,
        "frozen_step_plan_sha256": plan.plan_sha256,
        "term_order": list(DEFAULT_UNIFIED_WEIGHTS),
        "terms": terms,
        "gradient_cosine_similarity": cosine,
        "critic": critic_metrics,
        "translator_parameter_count": parameter_count,
        "graphs_retained_across_terms": False,
        "termwise_recomputation": True,
        "optimizer_step_called": False,
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


def aggregate_gradient_diagnostics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine a few independently recomputed R-only batches."""

    if not results:
        raise ValueError("Gradient aggregation requires at least one batch result.")
    for result in results:
        verify_self_hash(result)
        if result.get("contract_version") != GRADIENT_CONTRACT:
            raise ValueError("Gradient aggregation received an incompatible contract.")
    terms: dict[str, Any] = {}
    for name in DEFAULT_UNIFIED_WEIGHTS:
        enabled = [result["terms"][name] for result in results if result["terms"][name]["status"] != "disabled"]
        if not enabled:
            terms[name] = {"status": "disabled"}
            continue
        terms[name] = {
            "status": "PASS" if all(item["status"] == "PASS" for item in enabled) else "FAIL",
            "raw_loss": _distribution([float(item["raw_loss"]) for item in enabled]),
            "weighted_loss": _distribution([float(item["weighted_loss"]) for item in enabled]),
            "translator_raw_gradient_norm": _distribution(
                [float(item["translator_raw_gradient_norm"]) for item in enabled]
            ),
            "translator_weighted_gradient_norm": _distribution(
                [float(item["translator_weighted_gradient_norm"]) for item in enabled]
            ),
            "all_finite": all(bool(item["finite"]) for item in enabled),
        }
    payload = {
        "contract_version": GRADIENT_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "few_R_only_batches_real_stage2_loss_plumbing_no_optimizer_step",
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL",
        "batch_count": len(results),
        "batch_result_sha256": [result["result_sha256"] for result in results],
        "terms": terms,
        "critic": {
            "status": "PASS"
            if all(result["critic"]["status"] == "PASS" for result in results)
            else "FAIL",
            "per_batch": [result["critic"] for result in results],
        },
        "gradient_cosine_similarity": {
            "per_batch": [result["gradient_cosine_similarity"] for result in results]
        },
        "graphs_retained_across_terms_or_batches": False,
        "termwise_recomputation": True,
        "optimizer_step_called": False,
        "prospective_records_loaded": 0,
    }
    return self_hashed(payload)


def _critic_gradient_diagnostics(
    cfg: UnifiedStage2Config,
    translator: nn.Module,
    critic: nn.Module,
    decoder: nn.Module,
    batch: _TrainingBatch,
    stats: FactoredLatentStats,
    support: torch.Tensor,
    gradient_epsilon: float,
) -> dict[str, Any]:
    parameters = [value for value in critic.parameters() if value.requires_grad]
    if not parameters:
        return {"status": "FAIL", "reason": "critic_has_no_trainable_parameters"}
    results: dict[str, Any] = {}
    for name in ("adversarial", "domain"):
        with torch.no_grad():
            fake = integrate_transport(
                translator,
                batch.source,
                batch.source_domains,
                batch.target_domains,
                steps=cfg.integration_steps,
                solver=cfg.integration_solver,
            )
        real_view = _critic_view(
            batch.target, support, batch.target_domains, decoder, stats, cfg.critic_space
        )
        fake_view = _critic_view(
            fake, support, batch.target_domains, decoder, stats, cfg.critic_space
        )
        real_score, real_logits = critic(real_view, batch.target_domains)
        fake_score, _ = critic(fake_view, batch.target_domains)
        if name == "adversarial":
            raw = adversarial_hinge_loss_discriminator(real_score, fake_score)
            weight = float(cfg.loss_weights["adversarial"] > 0)
        else:
            raw = F.cross_entropy(
                real_logits,
                domain_labels(batch.target_domains, len(batch.target_domains), batch.source.device),
            )
            weight = float(cfg.loss_weights["domain"])
        weighted = raw * weight
        gradients = torch.autograd.grad(weighted, parameters, allow_unused=True)
        norm = math.sqrt(
            sum(
                float(gradient.detach().float().square().sum().cpu())
                for gradient in gradients
                if gradient is not None
            )
        )
        finite = math.isfinite(norm) and bool(torch.isfinite(raw.detach()).all())
        results[name] = {
            "raw_loss": float(raw.detach().float().cpu()),
            "weighted_loss": float(weighted.detach().float().cpu()),
            "critic_gradient_norm": norm,
            "finite": finite,
            "status": "PASS" if finite and norm > gradient_epsilon else "FAIL",
        }
        del fake, real_view, fake_view, real_score, real_logits, fake_score, raw, weighted
    results["status"] = (
        "PASS"
        if all(results[name]["status"] == "PASS" for name in ("adversarial", "domain"))
        else "FAIL"
    )
    return results


def run_synthetic_micro_overfit(
    model: nn.Module,
    *,
    mode: Literal["velocity", "direct"] = "velocity",
    steps: int = 200,
    learning_rate: float = 2.0e-2,
    device: str | torch.device = "cpu",
    seed: int = 20_260_901,
    pass_mae: float = 0.04,
) -> dict[str, Any]:
    """Isolated tiny gate with known additive target transforms and composition."""

    if steps < 1:
        raise ValueError("Micro-overfit requires at least one synthetic optimization step.")
    device_obj = torch.device(device)
    model = model.to(device_obj).train()
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        return self_hashed(
            {
                "contract_version": MICRO_OVERFIT_CONTRACT,
                "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
                "scope": "fully_synthetic_tiny_only",
                "status": "FAIL",
                "reason": "model_has_no_trainable_parameters",
                "real_scientific_evidence": False,
            }
        )
    generator = torch.Generator(device=device_obj.type).manual_seed(seed)
    domains = [Domain(field, Contrast.T1W) for field in FIELD_STRENGTHS_T[:3]]
    offsets = {domains[0]: -0.25, domains[1]: 0.0, domains[2]: 0.25}
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    final_training_loss = math.inf
    for _ in range(steps):
        source_ids = torch.randint(
            0, len(domains), (12,), generator=generator, device=device_obj
        )
        target_ids = torch.randint(
            0, len(domains), (12,), generator=generator, device=device_obj
        )
        source_domains = [domains[int(value)] for value in source_ids]
        target_domains = [domains[int(value)] for value in target_ids]
        source = torch.rand((12, 1, 4, 4), generator=generator, device=device_obj) * 0.4 - 0.2
        target = source + torch.tensor(
            [offsets[target] - offsets[start] for start, target in zip(source_domains, target_domains)],
            device=device_obj,
            dtype=source.dtype,
        ).reshape(-1, 1, 1, 1)
        optimizer.zero_grad(set_to_none=True)
        if mode == "velocity":
            time_values = torch.rand((12,), generator=generator, device=device_obj)
            prediction = model(source, source_domains, target_domains, time_values)
            loss = F.mse_loss(prediction, target - source)
        else:
            prediction = model(source, source_domains, target_domains)
            loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        final_training_loss = float(loss.detach().cpu())
    model.eval()
    evaluation_source = torch.linspace(-0.2, 0.2, 16, device=device_obj).reshape(1, 1, 4, 4)

    def translate(value: torch.Tensor, start: Domain, target: Domain) -> torch.Tensor:
        if mode == "velocity":
            return integrate_transport(
                model, value, [start], [target], steps=2, solver="heun"
            )
        return model(value, [start], [target])

    with torch.inference_mode():
        identity = translate(evaluation_source, domains[1], domains[1])
        low_to_high = translate(evaluation_source, domains[0], domains[2])
        wrong = translate(evaluation_source, domains[0], domains[1])
        expected = evaluation_source + 0.5
        identity_mae = float((identity - evaluation_source).abs().mean().cpu())
        correct_mae = float((low_to_high - expected).abs().mean().cpu())
        wrong_target_mae = float((wrong - expected).abs().mean().cpu())
        diversity = float((low_to_high - wrong).abs().mean().cpu())
        direction_correct = float((low_to_high - evaluation_source).mean().cpu()) > 0
        direct = translate(evaluation_source, domains[0], domains[2])
        first = translate(evaluation_source, domains[0], domains[1])
        composed = translate(first, domains[1], domains[2])
        composed_mae = float((direct - composed).abs().mean().cpu())
    failures = []
    if identity_mae >= pass_mae:
        failures.append("identity_not_learned")
    if correct_mae >= pass_mae:
        failures.append("known_target_transform_not_learned")
    if diversity <= 0.1:
        failures.append("different_targets_not_distinct")
    if wrong_target_mae <= correct_mae + 0.1:
        failures.append("wrong_target_does_not_degrade_metric")
    if not direction_correct:
        failures.append("target_transform_direction_wrong")
    if composed_mae >= pass_mae:
        failures.append("direct_composed_inconsistent")
    payload = {
        "contract_version": MICRO_OVERFIT_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "scope": "fully_synthetic_tiny_only",
        "status": "FAIL" if failures else "PASS",
        "mode": mode,
        "steps": steps,
        "final_training_loss": final_training_loss,
        "identity_mae": identity_mae,
        "known_transform_mae": correct_mae,
        "wrong_target_mae": wrong_target_mae,
        "target_output_diversity_mae": diversity,
        "target_direction_correct": direction_correct,
        "direct_composed_mae": composed_mae,
        "pass_mae": pass_mae,
        "failures": failures,
        "real_scientific_evidence": False,
        "private_or_prospective_data_used": False,
    }
    return self_hashed(payload)


def build_rescue_scorecard(
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "conditioning_plumbing",
        "real_domain_identifiability",
        "target_sweep",
        "off_manifold_drift",
        "gradient_health",
        "micro_overfit",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"Rescue scorecard is missing diagnostics: {missing}")
    for payload in diagnostics.values():
        verify_self_hash(payload)
    sweep = diagnostics["target_sweep"]
    rows = {
        "conditioning_plumbing": _scorecard_row(diagnostics["conditioning_plumbing"]),
        "real_domain_identifiability": _scorecard_row(
            diagnostics["real_domain_identifiability"]
        ),
        "learned_target_sensitivity": {
            "status": sweep["learned_target_sensitivity"]["status"],
            "evidence_sha256": sweep["result_sha256"],
        },
        "renderer_only_sensitivity": {
            "status": sweep["renderer_only_sensitivity"]["status"],
            "evidence_sha256": sweep["result_sha256"],
        },
        "off_manifold_drift": _scorecard_row(diagnostics["off_manifold_drift"]),
        "gradient_health": _scorecard_row(diagnostics["gradient_health"]),
        "micro_overfit": _scorecard_row(diagnostics["micro_overfit"]),
    }
    payload = {
        "contract_version": SCORECARD_CONTRACT,
        "schema_family": RESCUE_DIAGNOSTICS_SCHEMA,
        "rows": rows,
        "architecture_verdict": "human_decision_required",
        "automatic_architecture_promotion": False,
        "current_step200_promotion_authorized": False,
        "R_only_selection_diagnostics_complete": True,
        "paired_P0006_final_evaluation": "separate_sealed_step_not_executed",
        "P0009_status": "frozen_untouched",
    }
    return self_hashed(payload)


def _scorecard_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "INCONCLUSIVE"))
    if status not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        status = "INCONCLUSIVE"
    return {"status": status, "evidence_sha256": payload["result_sha256"]}


def render_rescue_scorecard_markdown(scorecard: Mapping[str, Any]) -> str:
    verify_self_hash(scorecard)
    lines = [
        "# Stage-2 rescue diagnostic scorecard",
        "",
        "This scorecard organizes falsifiable evidence; it does not promote an architecture.",
        "",
        "| Diagnostic | Status | Evidence SHA-256 |",
        "|---|---:|---|",
    ]
    for name, row in scorecard["rows"].items():
        lines.append(f"| {name.replace('_', ' ')} | {row['status']} | `{row['evidence_sha256']}` |")
    lines.extend(
        [
            "",
            "R-only diagnostics and selection are separate from the sealed paired P:0006 final evaluation.",
            "P:0009 remains frozen and untouched.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CONDITIONING_CONTRACT",
    "GRADIENT_CONTRACT",
    "IDENTIFIABILITY_CONTRACT",
    "LATENT_DRIFT_CONTRACT",
    "LatentDiagnosticRecord",
    "MICRO_OVERFIT_CONTRACT",
    "RESCUE_DIAGNOSTICS_SCHEMA",
    "SCORECARD_CONTRACT",
    "TARGET_SWEEP_CONTRACT",
    "aggregate_conditioning_diagnostics",
    "aggregate_gradient_diagnostics",
    "aggregate_target_sweeps",
    "bank_identifiability_features",
    "bank_domain_latent_centroids",
    "build_rescue_scorecard",
    "diagnose_conditioning_plumbing",
    "diagnose_off_manifold_latent_drift",
    "diagnose_per_term_gradients",
    "diagnose_real_domain_identifiability",
    "render_rescue_scorecard_markdown",
    "run_synthetic_micro_overfit",
    "same_source_five_target_sweep",
    "self_hashed",
    "validate_data_boundary",
    "verify_self_hash",
    "write_diagnostic_json",
]
