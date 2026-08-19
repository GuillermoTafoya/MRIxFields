"""Read-only and sealing preflights for unified retrospective Stage-2.

No function in this module trains a model or constructs a prospective protocol.  Cohort
classification is completed before latent or endpoint arrays are opened.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import (
    FrozenPhotometryArtifact,
    classify_variant_a_cohort,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
)
from fieldbridge.data.vae_splits import (
    load_vae_splits,
    vae_splits_fingerprint,
    vae_splits_recovery_fingerprint_v3,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    OFFICIAL_METRICS,
    VARIANT_A_PAIRED_MANIFEST_VERSION,
)
from fieldbridge.evaluation.stage2_photometry_protocol import (
    PAIRED_EVALUATION_ROLE,
    RAW_IDENTITY_CATASTROPHIC_BOUNDARY,
    seal_paired_evaluation_manifest,
)
from fieldbridge.evaluation.stage2_unified import BASELINE_PREDICTIONS_CONTRACT
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    GATE01_P0006_EVALUATION_PROTOCOL,
    P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
    P0006_EVIDENCE_LIMITATION,
    P0006_IDENTITY_SHA256,
    P0009_CONFIRMATION_STATUS,
)

DOMAIN_SEPARABILITY_CONTRACT = "stage2-factored-latent-domain-separability-v1"
PAIRED_FEASIBILITY_CONTRACT = "stage2-retrospective-paired-feasibility-v2"
MATERIALIZED_VALIDATION_ARRAYS_CONTRACT = "stage2-materialized-r-validation-arrays-v2"
MATERIALIZED_VALIDATION_PRODUCER_CONTRACT = (
    "stage2-r-validation-array-and-stage1-ceiling-producer-v1"
)
BASELINE_SOURCE_CONTRACT = "stage2-existing-gate01-sbv2-baseline-source-v2"
BASELINE_SOURCE_PRODUCER_CONTRACT = "stage2-gate01-sbv2-baseline-export-producer-v1"
LONG_RUN_EVALUATION_READINESS_CONTRACT = "stage2-long-run-evaluation-readiness-v3"


def quantify_factored_domain_separability(
    train_index: PhotometryFactoredLatentBankIndex,
    validation_index: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit deterministic train centroids and score all subject-disjoint validation records."""

    if train_index.split != "train" or validation_index.split != "validation":
        raise ValueError("Domain separability requires R/train and R/validation bank roles.")
    train_subjects = {item.subject_group_id for item in train_index.records}
    validation_subjects = {item.subject_group_id for item in validation_index.records}
    if train_subjects & validation_subjects:
        raise ValueError("Domain-separability train/validation subject groups overlap.")
    labels = [Domain(field, contrast).label for contrast in Contrast for field in FIELD_STRENGTHS_T]
    train_features, train_labels = _bank_features(train_index, stats)
    validation_features, validation_labels = _bank_features(validation_index, stats)
    observed = set(train_labels)
    if observed != set(labels):
        raise ValueError(f"Domain-separability R/train is missing domains: {sorted(set(labels)-observed)}")
    validation_observed = set(validation_labels)
    if validation_observed != set(labels):
        raise ValueError(
            "Domain-separability R/validation is missing domains: "
            f"{sorted(set(labels) - validation_observed)}"
        )
    feature_mean = train_features.mean(0)
    feature_std = train_features.std(0)
    feature_std[feature_std <= 1.0e-12] = 1.0
    standardized_train = (train_features - feature_mean) / feature_std
    standardized_validation = (validation_features - feature_mean) / feature_std
    centroids = np.stack(
        [standardized_train[np.asarray(train_labels) == label].mean(0) for label in labels]
    )
    train_prediction = _nearest_centroid(standardized_train, centroids, labels)
    validation_prediction = _nearest_centroid(standardized_validation, centroids, labels)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(validation_labels, validation_prediction):
        confusion[labels.index(actual), labels.index(predicted)] += 1
    per_domain = {}
    for position, label in enumerate(labels):
        count = int(confusion[position].sum())
        per_domain[label] = {
            "count": count,
            "accuracy": float(confusion[position, position] / count) if count else None,
            "predicted_counts": {
                target: int(confusion[position, target_position])
                for target_position, target in enumerate(labels)
            },
        }
    result: dict[str, Any] = {
        "contract_version": DOMAIN_SEPARABILITY_CONTRACT,
        "scope": "photometry_factored_normalized_latents_R_only",
        "classifier": "deterministic_nearest_train_domain_centroid_v1",
        "feature": "supported_per_channel_mean_and_std",
        "subject_grouped_train_validation": True,
        "train_record_count": len(train_labels),
        "validation_record_count": len(validation_labels),
        "train_inventory_sha256": sha256_json([item.resume_key for item in train_index.records]),
        "validation_inventory_sha256": sha256_json(
            [item.resume_key for item in validation_index.records]
        ),
        "bank_artifact_sha256": train_index.artifact_sha256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "domain_order": labels,
        "train_accuracy": float(np.mean(np.asarray(train_prediction) == np.asarray(train_labels))),
        "validation_accuracy": float(
            np.mean(np.asarray(validation_prediction) == np.asarray(validation_labels))
        ),
        "validation_confusion": confusion.tolist(),
        "per_domain": per_domain,
        "interpretation_boundary": (
            "quantifies residual domain predictability; it does not establish legitimate "
            "StarGAN control or learned disentanglement"
        ),
        "prospective_records_loaded": 0,
    }
    result["result_sha256"] = sha256_json(result)
    if output_path is not None:
        write_json_atomic(output_path, result, refuse_existing=True)
    return result


def audit_retrospective_paired_feasibility(
    split_path: str | Path,
    *,
    output_path: str | Path | None = None,
    hash_source_files: bool = True,
) -> dict[str, Any]:
    """Seal all same-subject/same-contrast directed R/validation edges, if any."""

    source = Path(split_path)
    split_sha = sha256_file(source)
    splits = load_vae_splits(source)
    eligible: list[tuple[Any, Any]] = []
    excluded: list[dict[str, Any]] = []
    # Independent classification and role checks precede every source-file access.
    # Only a positive, reconciled P identity is excluded.  Missing, malformed, or
    # conflicting identities are integrity failures and are never recast as P.
    for record in splits.validation:
        try:
            identity = classify_variant_a_cohort(
                case_identity=record.case_id,
                metadata_prefix=record.metadata.get("prefix"),
                supplied_cohort=record.metadata.get("cohort"),
                subject_identity=record.subject_id,
                allowed_cohorts=("R", "P"),
            )
        except ValueError as exc:
            raise ValueError(
                "Malformed/conflicting validation cohort identity; complete_inventory_no_selection "
                f"cannot be asserted for {record.case_id!r}: {exc}"
            ) from exc
        if identity.cohort == "P":
            excluded.append(
                {
                    "case_id": identity.case_identity,
                    "reason": "independently_and_positively_classified_as_P_before_source_access",
                }
            )
            continue
        eligible.append((record, identity))
    groups: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
    for record, identity in eligible:
        groups[(identity.subject_group_identity, record.domain.contrast.value)].append(
            (record, identity)
        )
    edges: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    for records in groups.values():
        ordered = sorted(records, key=lambda item: (item[0].domain.field_strength_t, item[0].case_id))
        for source_record, source_identity in ordered:
            source_files.setdefault(
                source_record.case_id,
                _source_file_identity(source_record.image_path, hash_source_files),
            )
            for target_record, target_identity in ordered:
                if source_record.domain.field_strength_t == target_record.domain.field_strength_t:
                    continue
                edges.append(
                    {
                        "case_identity": sha256_json(
                            [source_record.case_id, target_record.case_id]
                        ),
                        "subject_group_identity": source_identity.subject_group_identity,
                        "contrast": source_record.domain.contrast.value,
                        "source_record_identity": source_record.case_id,
                        "target_record_identity": target_record.case_id,
                        "source_domain": source_record.domain.to_dict(),
                        "target_domain": target_record.domain.to_dict(),
                    }
                )
                source_files.setdefault(
                    target_record.case_id,
                    _source_file_identity(target_record.image_path, hash_source_files),
                )
    edges.sort(key=lambda item: item["case_identity"])
    result: dict[str, Any] = {
        "contract_version": PAIRED_FEASIBILITY_CONTRACT,
        "scope": "complete_R_validation_inventory",
        "split": {
            "file_sha256": split_sha,
            "membership_fingerprint": vae_splits_fingerprint(splits),
            "recovery_fingerprint_v3": vae_splits_recovery_fingerprint_v3(splits),
            "path_identity_sha256": sha256_text(str(source.resolve())),
        },
        "classification_before_source_file_access": True,
        "prospective_exclusion_rule": "exclude_only_independently_reconciled_positive_P",
        "malformed_or_conflicting_identity_policy": "raise_before_source_file_access",
        "array_payloads_opened": 0,
        "eligible_record_count": len(eligible),
        "eligible_domain_counts": {
            label: sum(record.domain.label == label for record, _ in eligible)
            for label in sorted(
                Domain(field, contrast).label
                for contrast in Contrast
                for field in FIELD_STRENGTHS_T
            )
        },
        "eligible_record_identity_sha256": sha256_json(
            sorted(record.case_id for record, _ in eligible)
        ),
        "excluded_prospective": sorted(excluded, key=lambda item: item["case_id"]),
        "subject_group_count": len({identity.subject_group_identity for _, identity in eligible}),
        "directed_pair_count": len(edges),
        "paired_evaluation_possible": bool(edges),
        "pairs": edges,
        "source_files": dict(sorted(source_files.items())),
        "complete_inventory_no_selection": True,
        "stage1_ceiling_requirement": "one sealed target reconstruction per pair",
        "failure_instruction": (
            None
            if edges
            else "No same-subject, same-contrast, cross-field R/validation endpoints exist; "
            "do not fabricate pairs from unrelated subjects. A P-traveller protocol requires "
            "separate versioned authorization."
        ),
    }
    result["result_sha256"] = sha256_json(result)
    if output_path is not None:
        write_json_atomic(output_path, result, refuse_existing=True)
    return result


def build_retrospective_paired_manifest(
    feasibility_path: str | Path,
    materialized_arrays_path: str | Path,
    artifact: FrozenPhotometryArtifact,
    *,
    output_path: str | Path,
    authorization_reference: str,
) -> dict[str, Any]:
    """Build the complete paired manifest from a feasible plan and sealed .npy arrays."""

    feasibility = _load_self_hashed(feasibility_path, "result_sha256")
    if feasibility.get("contract_version") != PAIRED_FEASIBILITY_CONTRACT:
        raise ValueError("Paired-feasibility contract mismatch.")
    if feasibility.get("paired_evaluation_possible") is not True:
        raise ValueError(str(feasibility.get("failure_instruction")))
    arrays = _load_self_hashed(materialized_arrays_path, "manifest_sha256")
    if arrays.get("contract_version") != MATERIALIZED_VALIDATION_ARRAYS_CONTRACT:
        raise ValueError("Materialized R/validation array contract mismatch.")
    producer = _require_deterministic_producer(
        arrays, MATERIALIZED_VALIDATION_PRODUCER_CONTRACT
    )
    if producer.get("feasibility_result_sha256") != feasibility["result_sha256"]:
        raise ValueError("Materialized arrays were produced from another feasibility plan.")
    entries = arrays.get("records")
    if not isinstance(entries, list):
        raise ValueError("Materialized validation arrays require a records list.")
    by_record = {str(item.get("record_identity")): item for item in entries}
    if set(by_record) != set(feasibility["source_files"]):
        raise ValueError(
            "Materialized arrays must exactly cover the complete feasible record inventory."
        )
    cases = []
    for edge in feasibility["pairs"]:
        source_id = str(edge["source_record_identity"])
        target_id = str(edge["target_record_identity"])
        if source_id not in by_record or target_id not in by_record:
            raise ValueError(
                "Materialized arrays do not cover the complete feasible inventory. "
                "Run the reviewed R/validation materializer and Stage-1 reconstruction export."
            )
        source_spec = _endpoint_spec(by_record[source_id], edge["source_domain"])
        target_spec = _endpoint_spec(by_record[target_id], edge["target_domain"])
        ceiling = by_record[target_id].get("stage1_reconstruction")
        if not isinstance(ceiling, Mapping):
            raise ValueError(f"Missing sealed Stage-1 ceiling for {target_id}.")
        cases.append(
            {
                "case_identity": edge["case_identity"],
                "genuinely_paired": True,
                "source": source_spec,
                "target": target_spec,
                "stage1_reconstruction": dict(ceiling),
            }
        )
    payload = {
        "contract_version": VARIANT_A_PAIRED_MANIFEST_VERSION,
        "data_role": PAIRED_EVALUATION_ROLE,
        "evaluation_identity": sha256_json([item["case_identity"] for item in cases]),
        "metrics": list(OFFICIAL_METRICS),
        "raw_identity_catastrophic_boundary": RAW_IDENTITY_CATASTROPHIC_BOUNDARY,
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "photometry_config_sha256": artifact.provenance["resolved_config_sha256"],
        "split_provenance": {
            "role": "complete retrospective R/validation paired feasibility inventory",
            "file_sha256": feasibility["split"]["file_sha256"],
            "source_membership_fingerprint": feasibility["split"]["membership_fingerprint"],
            "source_recovery_fingerprint": feasibility["split"]["recovery_fingerprint_v3"],
        },
        "provenance": {
            "authorization_reference": authorization_reference,
            "feasibility_result_sha256": feasibility["result_sha256"],
            "materialized_arrays_sha256": arrays["manifest_sha256"],
            "complete_inventory_no_selection": True,
        },
        "cases": cases,
    }
    sealed = seal_paired_evaluation_manifest(payload)
    write_json_atomic(output_path, sealed, refuse_existing=True)
    return sealed


def build_baseline_prediction_manifest(
    paired_manifest_path: str | Path,
    source_artifact_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal complete Gate-0.1/SB-v2 predictions from an existing artifact index."""

    paired = _load_self_hashed(paired_manifest_path, "manifest_sha256")
    source = _load_self_hashed(source_artifact_path, "manifest_sha256")
    if source.get("contract_version") != BASELINE_SOURCE_CONTRACT:
        raise ValueError(
            "Baseline source is unavailable. Export the sealed existing Gate 0.1 calibrated "
            "identity and original SB-v2 predictions with contract "
            f"{BASELINE_SOURCE_CONTRACT}."
        )
    producer = _require_deterministic_producer(
        source, BASELINE_SOURCE_PRODUCER_CONTRACT
    )
    if producer.get("paired_manifest_sha256") != paired["manifest_sha256"]:
        raise ValueError("Baseline predictions were exported for another paired inventory.")
    cases = paired.get("cases")
    source_cases = source.get("cases")
    if not isinstance(cases, list) or not isinstance(source_cases, list):
        raise ValueError("Paired/baseline source manifests require case lists.")
    by_identity = {str(item.get("case_identity")): item for item in source_cases}
    output_cases = []
    for case in cases:
        identity = str(case["case_identity"])
        item = by_identity.get(identity)
        if item is None:
            raise ValueError(
                "Existing Gate 0.1/SB-v2 artifacts do not cover the complete paired inventory."
            )
        classified = classify_variant_a_cohort(
            case_identity=str(item.get("record_identity", "")),
            metadata_prefix=item.get("metadata_prefix"),
            supplied_cohort=item.get("cohort"),
            subject_identity=item.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if item.get("split") != "validation":
            raise ValueError("Existing baseline source is not R/validation.")
        prepared = dict(item)
        prepared["record_identity"] = classified.case_identity
        for method in ("gate01_calibrated_identity", "original_sb_v2"):
            spec = item.get(method)
            if not isinstance(spec, Mapping):
                raise ValueError(f"Baseline source is missing {method} for {identity}.")
            normalized_spec = dict(spec)
            array_path = Path(str(spec.get("path", "")))
            if not array_path.is_absolute():
                array_path = (Path(source_artifact_path).parent / array_path).resolve()
            if not array_path.is_file() or sha256_file(array_path) != spec.get("file_sha256"):
                raise ValueError(f"Existing baseline file identity mismatch for {identity}/{method}.")
            normalized_spec["path"] = str(array_path)
            prepared[method] = normalized_spec
        output_cases.append(prepared)
    if set(by_identity) != {str(item["case_identity"]) for item in cases}:
        raise ValueError("Baseline source contains a selected or extra case inventory.")
    body: dict[str, Any] = {
        "contract_version": BASELINE_PREDICTIONS_CONTRACT,
        "source_artifact_sha256": source["manifest_sha256"],
        "complete_inventory_no_selection": True,
        "cases": output_cases,
    }
    body["manifest_sha256"] = sha256_json(body)
    write_json_atomic(output_path, body, refuse_existing=True)
    return body


def import_retrospective_paired_evaluation_archive(
    feasibility_path: str | Path,
    archive_root: str | Path,
    artifact: FrozenPhotometryArtifact,
    *,
    output_dir: str | Path,
    authorization_reference: str,
) -> dict[str, Any]:
    """Import the two reviewed R-paired producer exports and construct evaluator inputs."""

    root = Path(archive_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"R-paired evaluation archive does not exist: {root}")
    materialized_path = _single_self_hashed_contract(
        root, MATERIALIZED_VALIDATION_ARRAYS_CONTRACT
    )
    baseline_source_path = _single_self_hashed_contract(root, BASELINE_SOURCE_CONTRACT)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paired_path = destination / "stage2_complete_R_validation_paired_manifest.json"
    baselines_path = destination / "stage2_complete_gate01_sbv2_baselines.json"
    readiness_path = destination / "stage2_long_run_evaluation_readiness_v3.json"
    if paired_path.exists():
        paired = _load_self_hashed(paired_path, "manifest_sha256")
        provenance = paired.get("provenance", {})
        feasibility = _load_self_hashed(feasibility_path, "result_sha256")
        materialized = _load_self_hashed(materialized_path, "manifest_sha256")
        if (
            paired.get("photometry_artifact_sha256") != artifact.artifact_sha256
            or provenance.get("feasibility_result_sha256")
            != feasibility["result_sha256"]
            or provenance.get("materialized_arrays_sha256")
            != materialized["manifest_sha256"]
            or provenance.get("authorization_reference") != authorization_reference
        ):
            raise ValueError("Existing R-paired manifest is not an exact import resume.")
    else:
        paired = build_retrospective_paired_manifest(
            feasibility_path,
            materialized_path,
            artifact,
            output_path=paired_path,
            authorization_reference=authorization_reference,
        )
    if baselines_path.exists():
        baselines = _load_self_hashed(baselines_path, "manifest_sha256")
        source = _load_self_hashed(baseline_source_path, "manifest_sha256")
        if (
            baselines.get("source_artifact_sha256") != source["manifest_sha256"]
            or {item.get("case_identity") for item in baselines.get("cases", [])}
            != {item.get("case_identity") for item in paired.get("cases", [])}
        ):
            raise ValueError("Existing R baseline manifest is not an exact import resume.")
    else:
        baselines = build_baseline_prediction_manifest(
            paired_path, baseline_source_path, output_path=baselines_path
        )
    if readiness_path.exists():
        readiness = _load_self_hashed(readiness_path, "readiness_sha256")
        if (
            readiness.get("paired_manifest_sha256") != paired["manifest_sha256"]
            or readiness.get("baseline_predictions_sha256")
            != baselines["manifest_sha256"]
        ):
            raise ValueError("Existing R readiness receipt is not an exact import resume.")
    else:
        readiness = seal_long_run_evaluation_readiness(
            feasibility_path,
            materialized_path,
            paired_path,
            baseline_source_path,
            baselines_path,
            output_path=readiness_path,
        )
    return {
        "contract_version": "stage2-unified-r-paired-evaluation-archive-import-v1",
        "materialized_arrays_path": str(materialized_path),
        "baseline_source_path": str(baseline_source_path),
        "paired_manifest_path": str(paired_path),
        "paired_manifest_sha256": paired["manifest_sha256"],
        "baseline_predictions_path": str(baselines_path),
        "baseline_predictions_sha256": baselines["manifest_sha256"],
        "readiness_path": str(readiness_path),
        "readiness_sha256": readiness["readiness_sha256"],
    }


def seal_long_run_evaluation_readiness(
    feasibility_path: str | Path,
    materialized_arrays_path: str | Path | None = None,
    paired_manifest_path: str | Path | None = None,
    baseline_source_artifact_path: str | Path | None = None,
    baseline_predictions_path: str | Path | None = None,
    *,
    p0006_evaluation_protocol_path: str | Path | None = None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal genuine R pairs or the separately reviewed P:0006 evaluation-only path."""

    feasibility = _load_self_hashed(feasibility_path, "result_sha256")
    if feasibility.get("contract_version") != PAIRED_FEASIBILITY_CONTRACT or (
        feasibility.get("complete_inventory_no_selection") is not True
    ):
        raise ValueError("Long training is blocked: paired-feasibility receipt is invalid.")
    if feasibility.get("paired_evaluation_possible") is not True:
        if p0006_evaluation_protocol_path is None:
            raise ValueError(
                "Long training is blocked: no genuine R/validation pairs and no sealed "
                "P:0006 evaluation-only protocol were supplied."
            )
        protocol = _load_self_hashed(
            p0006_evaluation_protocol_path, "protocol_sha256"
        )
        if (
            protocol.get("contract_version") != GATE01_P0006_EVALUATION_PROTOCOL
            or protocol.get("data_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
            or protocol.get("evidence_interpretation") != P0006_EVIDENCE_LIMITATION
            or protocol.get("population_or_generalization_claims_authorized") is not False
            or protocol.get("traveller_identity_sha256") != P0006_IDENTITY_SHA256
            or protocol.get("acquisition_count") != 15
            or protocol.get("directed_pair_count") != 60
            or protocol.get("wrong_target_reference_count") != 180
            or protocol.get("private_arrays_validated") is not True
            or protocol.get("training_or_model_selection_use") is not False
            or protocol.get("factored_bank", {}).get("P_record_count") != 0
            or protocol.get("frozen_unpaired_validation", {}).get("P_endpoint_count") != 0
            or protocol.get("P0009_confirmation_status") != P0009_CONFIRMATION_STATUS
            or protocol.get("P0009_executed") is not False
        ):
            raise ValueError(
                "Long training is blocked: P:0006 evaluation-only protocol is incomplete."
            )
        result: dict[str, Any] = {
            "contract_version": LONG_RUN_EVALUATION_READINESS_CONTRACT,
            "long_run_authorized_by_evaluation_path": True,
            "evaluation_role": P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
            "evidence_interpretation": P0006_EVIDENCE_LIMITATION,
            "population_or_generalization_claims_authorized": False,
            "prospective_protocol_used": True,
            "prospective_training_or_model_selection_use": False,
            "reviewed_prospective_protocol_available": True,
            "complete_inventory_no_selection": True,
            "directed_pair_count": 60,
            "feasibility_result_sha256": feasibility["result_sha256"],
            "retrospective_pair_feasibility": False,
            "p0006_evaluation_protocol_sha256": protocol["protocol_sha256"],
            "p0006_gate01_result_file_sha256": protocol["gate01_result"][
                "file_sha256"
            ],
            "factored_bank_P_record_count": 0,
            "unpaired_validation_P_endpoint_count": 0,
            "P0009_confirmation_status": P0009_CONFIRMATION_STATUS,
            "P0009_executed": False,
        }
        result["readiness_sha256"] = sha256_json(result)
        write_json_atomic(output_path, result, refuse_existing=True)
        return result
    required_r_paths = (
        materialized_arrays_path,
        paired_manifest_path,
        baseline_source_artifact_path,
        baseline_predictions_path,
    )
    if any(path is None for path in required_r_paths):
        raise ValueError(
            "Long training is blocked: genuine R/validation pairs exist, so the complete "
            "reviewed R-paired arrays, ceilings, and baseline path is required."
        )
    assert materialized_arrays_path is not None
    assert paired_manifest_path is not None
    assert baseline_source_artifact_path is not None
    assert baseline_predictions_path is not None
    arrays = _load_self_hashed(materialized_arrays_path, "manifest_sha256")
    if arrays.get("contract_version") != MATERIALIZED_VALIDATION_ARRAYS_CONTRACT:
        raise ValueError("Long training is blocked: materialized array/ceiling contract mismatch.")
    array_producer = _require_deterministic_producer(
        arrays, MATERIALIZED_VALIDATION_PRODUCER_CONTRACT
    )
    if array_producer.get("feasibility_result_sha256") != feasibility["result_sha256"]:
        raise ValueError("Long training is blocked: array producer used another inventory.")
    paired = _load_self_hashed(paired_manifest_path, "manifest_sha256")
    paired_provenance = paired.get("provenance")
    if not isinstance(paired_provenance, Mapping) or (
        paired_provenance.get("feasibility_result_sha256")
        != feasibility["result_sha256"]
        or paired_provenance.get("materialized_arrays_sha256")
        != arrays["manifest_sha256"]
        or paired_provenance.get("complete_inventory_no_selection") is not True
    ):
        raise ValueError("Long training is blocked: paired manifest provenance is incomplete.")
    baseline_source = _load_self_hashed(
        baseline_source_artifact_path, "manifest_sha256"
    )
    if baseline_source.get("contract_version") != BASELINE_SOURCE_CONTRACT:
        raise ValueError("Long training is blocked: baseline-source contract mismatch.")
    baseline_producer = _require_deterministic_producer(
        baseline_source, BASELINE_SOURCE_PRODUCER_CONTRACT
    )
    if baseline_producer.get("paired_manifest_sha256") != paired["manifest_sha256"]:
        raise ValueError("Long training is blocked: baseline producer used another inventory.")
    baselines = _load_self_hashed(baseline_predictions_path, "manifest_sha256")
    if (
        baselines.get("contract_version") != BASELINE_PREDICTIONS_CONTRACT
        or baselines.get("source_artifact_sha256")
        != baseline_source["manifest_sha256"]
        or baselines.get("complete_inventory_no_selection") is not True
    ):
        raise ValueError("Long training is blocked: final baseline manifest is incomplete.")
    case_ids = {
        str(case.get("case_identity")) for case in paired.get("cases", [])
    }
    baseline_case_ids = {
        str(case.get("case_identity")) for case in baselines.get("cases", [])
    }
    if not case_ids or case_ids != baseline_case_ids or len(case_ids) != int(
        feasibility["directed_pair_count"]
    ):
        raise ValueError("Long training is blocked: evaluation case inventories disagree.")
    result: dict[str, Any] = {
        "contract_version": LONG_RUN_EVALUATION_READINESS_CONTRACT,
        "long_run_authorized_by_evaluation_path": True,
        "evaluation_role": "complete_genuine_paired_R_validation",
        "prospective_protocol_used": False,
        "reviewed_prospective_protocol_available": False,
        "complete_inventory_no_selection": True,
        "directed_pair_count": len(case_ids),
        "feasibility_result_sha256": feasibility["result_sha256"],
        "materialized_arrays_sha256": arrays["manifest_sha256"],
        "materialized_array_producer": dict(array_producer),
        "paired_manifest_sha256": paired["manifest_sha256"],
        "baseline_source_sha256": baseline_source["manifest_sha256"],
        "baseline_source_producer": dict(baseline_producer),
        "baseline_predictions_sha256": baselines["manifest_sha256"],
        "case_inventory_sha256": sha256_json(sorted(case_ids)),
    }
    result["readiness_sha256"] = sha256_json(result)
    write_json_atomic(output_path, result, refuse_existing=True)
    return result


def _bank_features(
    index: PhotometryFactoredLatentBankIndex, stats: FactoredLatentStats
) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    # All identities were already classified by the strict index constructor.
    for position, record in enumerate(index.records):
        latent, support = index.load(position)
        normalized = stats.normalize(latent, support)
        values = normalized[:, support]
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("Domain-separability record has empty/nonfinite supported values.")
        feature = torch.cat((values.mean(1), values.std(1, unbiased=False)))
        features.append(feature.double().cpu().numpy())
        labels.append(record.domain.label)
    return np.stack(features), labels


def _single_self_hashed_contract(root: Path, contract: str) -> Path:
    matches: list[Path] = []
    for path in root.rglob("*.json"):
        try:
            payload = _load_self_hashed(path, "manifest_sha256")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("contract_version") == contract:
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(
            f"R-paired archive requires exactly one {contract!r} export; found {len(matches)}."
        )
    return matches[0]


def _nearest_centroid(features: np.ndarray, centroids: np.ndarray, labels: Sequence[str]) -> list[str]:
    distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
    return [labels[int(value)] for value in distances.argmin(1)]


def _source_file_identity(path: str | Path, hash_file: bool) -> dict[str, Any]:
    source = Path(path)
    result = {"path_identity_sha256": sha256_text(str(source))}
    if hash_file:
        if not source.is_file():
            raise ValueError(f"R/validation source file is missing: {source}")
        result["file_sha256"] = sha256_file(source)
    else:
        result["file_sha256"] = None
    return result


def _endpoint_spec(entry: Mapping[str, Any], domain: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "record_identity",
        "subject_identity",
        "array_path",
        "file_sha256",
        "loaded_array_sha256",
        "content_identity",
        "shape",
        "dtype",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"Materialized endpoint is missing {missing}.")
    return {
        "case_id": entry["record_identity"],
        "subject_id": entry["subject_identity"],
        "metadata_prefix": "R",
        "cohort": "R",
        "domain": dict(domain),
        **{key: entry[key] for key in required[2:]},
    }


def _require_deterministic_producer(
    artifact: Mapping[str, Any], expected_contract: str
) -> dict[str, Any]:
    producer = artifact.get("producer")
    if not isinstance(producer, Mapping):
        raise ValueError(
            f"Artifact requires deterministic producer provenance {expected_contract}."
        )
    required_hashes = ("source_code_provenance_sha256", "resolved_config_sha256")
    if (
        producer.get("contract_version") != expected_contract
        or producer.get("deterministic") is not True
        or producer.get("complete_inventory_no_selection") is not True
        or producer.get("full_volume_arithmetic") is not True
        or any(
            not isinstance(producer.get(key), str) or len(str(producer.get(key))) != 64
            for key in required_hashes
        )
    ):
        raise ValueError(f"Artifact producer provenance does not satisfy {expected_contract}.")
    return dict(producer)


def _load_self_hashed(path: str | Path, hash_key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Sealed artifact root must be an object.")
    normalized = dict(payload)
    stored = normalized.pop(hash_key, None)
    if stored != sha256_json(normalized):
        raise ValueError(f"Sealed artifact {hash_key} mismatch.")
    normalized[hash_key] = stored
    return normalized


__all__ = [
    "BASELINE_SOURCE_CONTRACT",
    "BASELINE_SOURCE_PRODUCER_CONTRACT",
    "DOMAIN_SEPARABILITY_CONTRACT",
    "LONG_RUN_EVALUATION_READINESS_CONTRACT",
    "MATERIALIZED_VALIDATION_ARRAYS_CONTRACT",
    "MATERIALIZED_VALIDATION_PRODUCER_CONTRACT",
    "PAIRED_FEASIBILITY_CONTRACT",
    "audit_retrospective_paired_feasibility",
    "build_baseline_prediction_manifest",
    "import_retrospective_paired_evaluation_archive",
    "build_retrospective_paired_manifest",
    "quantify_factored_domain_separability",
    "seal_long_run_evaluation_readiness",
]
