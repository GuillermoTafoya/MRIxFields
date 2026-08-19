"""Retrospective R/validation evaluation for the complete unified Stage-2 model."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fieldbridge.data.domains import Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.latent_bank import encode_latent
from fieldbridge.data.photometry_factored_latent_bank import (
    propagate_encoder_local_valid_core_support,
)
from fieldbridge.data.photometry_factorization import (
    FrozenPhotometryArtifact,
    canonical_tensor_sha256,
    classify_variant_a_cohort,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    OfficialQualificationMetrics,
    PairedEvaluationCase,
)
from fieldbridge.evaluation.stage2_photometry_protocol import (
    RAW_IDENTITY_CATASTROPHIC_BOUNDARY,
    load_paired_evaluation_manifest,
)
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
    P0006_EVIDENCE_LIMITATION,
    P0009_CONFIRMATION_STATUS,
    load_gate01_p0006_evaluation_protocol,
)
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.stage2_unified import (
    anatomy_preservation_components,
    graph_consistency_loss,
    integrate_transport,
)

UNIFIED_EVALUATION_CONTRACT = "stage2-unified-selected-best-evaluation-v4"
BASELINE_PREDICTIONS_CONTRACT = "stage2-unified-retrospective-baseline-predictions-v1"


@torch.inference_mode()
def evaluate_stage2_unified(
    *,
    translator: BaseTranslator,
    encoder: torch.nn.Module | None,
    decoder: torch.nn.Module,
    artifact: FrozenPhotometryArtifact,
    bank: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    paired_manifest_path: str | Path | None,
    baseline_predictions_path: str | Path | None,
    p0006_evaluation_protocol_path: str | Path | None = None,
    output_dir: str | Path,
    sb_only_translator: BaseTranslator | None = None,
    ablation_translators: Mapping[str, BaseTranslator] | None = None,
    device: str | torch.device = "cuda",
    integration_steps: int = 20,
    solver: str = "heun",
    resume: bool = False,
) -> dict[str, Any]:
    """Score genuine-R pairs or P:0006 development-validation evidence."""

    if bank.split != "validation":
        raise ValueError("Unified evaluation is restricted to the complete R/validation bank.")
    if p0006_evaluation_protocol_path is not None:
        if paired_manifest_path is not None or baseline_predictions_path is not None:
            raise ValueError("Choose either genuine R pairs or P:0006 evaluation, never both.")
        if encoder is None:
            raise ValueError("P:0006 evaluation requires the frozen full-volume VAE encoder.")
        manifest, cases, baselines = load_gate01_p0006_evaluation_protocol(
            p0006_evaluation_protocol_path
        )
        evaluation_role = P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
        evaluation_identity_sha256 = str(manifest["protocol_sha256"])
        baseline_identity = {
            "contract_version": manifest["contract_version"],
            "protocol_sha256": manifest["protocol_sha256"],
            "case_count": len(cases),
            "source": "sealed_Gate01Private_8012a3f_graph",
        }
    else:
        if paired_manifest_path is None or baseline_predictions_path is None:
            raise ValueError("Genuine R evaluation requires paired and baseline manifests.")
        _preflight_paired_retrospective_manifest(paired_manifest_path)
        manifest, cases = load_paired_evaluation_manifest(
            paired_manifest_path, artifact=artifact
        )
        baselines, baseline_identity = _load_baseline_predictions(
            baseline_predictions_path, cases
        )
        evaluation_role = "complete_genuine_paired_R_validation"
        evaluation_identity_sha256 = str(manifest["manifest_sha256"])
    if not cases:
        raise ValueError("Unified evaluation requires at least one sealed pair.")
    device_obj = torch.device(device)
    translator = translator.to(device_obj).eval()
    decoder = decoder.to(device_obj).eval()
    if encoder is not None:
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("Unified evaluation requires a frozen VAE encoder.")
        encoder = encoder.to(device_obj).eval()
    if sb_only_translator is not None:
        sb_only_translator = sb_only_translator.to(device_obj).eval()
    ablation_translators = dict(ablation_translators or {})
    for name, model in ablation_translators.items():
        if not name or name in {"full", "sb_only"}:
            raise ValueError("Ablation translator names must be nonempty and unambiguous.")
        ablation_translators[name] = model.to(device_obj).eval()
    root = Path(output_dir)
    result_path = root / "result.json"
    existing_result: dict[str, Any] | None = None
    if result_path.exists():
        if not resume:
            raise FileExistsError("Unified evaluation result exists; pass resume to verify it.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stored = result.pop("result_sha256", None)
        if stored != sha256_json(result):
            raise ValueError("Unified evaluation result hash mismatch.")
        result["result_sha256"] = stored
        existing_result = result
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("Unified evaluation output is nonempty; pass resume to verify it.")
    root.mkdir(parents=True, exist_ok=True)
    montage_dir = root / "montages"
    montage_dir.mkdir(exist_ok=True)
    shard_dir = root / "case_shards"
    shard_dir.mkdir(exist_ok=True)
    run_contract: dict[str, Any] = {
        "contract_version": "stage2-unified-selected-best-evaluation-run-v4",
        "evaluation_role": evaluation_role,
        "evidence_interpretation": (
            P0006_EVIDENCE_LIMITATION
            if evaluation_role == P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
            else "complete genuine paired R/validation development evidence"
        ),
        "evaluation_protocol_sha256": evaluation_identity_sha256,
        "bank_artifact_sha256": bank.artifact_sha256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "baseline_predictions": baseline_identity,
        "full_translator_state_sha256": _module_state_sha256(translator),
        "sb_only_translator_state_sha256": (
            _module_state_sha256(sb_only_translator) if sb_only_translator is not None else None
        ),
        "ablation_translator_state_sha256": {
            name: _module_state_sha256(model)
            for name, model in sorted(ablation_translators.items())
        },
        "frozen_decoder_state_sha256": _module_state_sha256(decoder),
        "frozen_encoder_state_sha256": (
            _module_state_sha256(encoder) if encoder is not None else None
        ),
        "integration": {"steps": integration_steps, "solver": solver},
        "case_identities": sorted(case.case_identity for case in cases),
    }
    run_contract["run_contract_sha256"] = sha256_json(run_contract)
    contract_path = root / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise ValueError("Unified evaluation exact-resume contract mismatch.")
    else:
        write_json_atomic(contract_path, run_contract, refuse_existing=True)
    if existing_result is not None:
        if existing_result.get("run_contract_sha256") != run_contract["run_contract_sha256"]:
            raise ValueError("Unified result belongs to a different run contract.")
        return existing_result
    metrics = OfficialQualificationMetrics(device=str(device_obj))
    by_case = {record.case_id: index for index, record in enumerate(bank.records)}
    rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: item.case_identity):
        shard_path = shard_dir / f"{sha256_text(case.case_identity)}.json"
        if shard_path.exists():
            if not resume:
                raise FileExistsError("Unified evaluation case shard exists without resume.")
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            stored_hash = shard.pop("shard_sha256", None)
            if stored_hash != sha256_json(shard) or shard.get(
                "run_contract_sha256"
            ) != run_contract["run_contract_sha256"]:
                raise ValueError("Unified evaluation case shard exact-resume mismatch.")
            row = dict(shard["case"])
            montage_path = root / str(row["montage"]["path"])
            if not montage_path.is_file() or sha256_file(montage_path) != row["montage"][
                "file_sha256"
            ]:
                raise ValueError("Unified evaluation montage is missing or changed.")
            rows.append(row)
            continue
        canonical_context = artifact.normalize_source(case.source, case.source_domain)
        canonical_support = canonical_context.support_mask.detach().to(torch.bool)
        while canonical_support.ndim > 3 and canonical_support.shape[0] == 1:
            canonical_support = canonical_support[0]
        if canonical_support.ndim != 3:
            raise ValueError("Unified evaluation source support is not a 3-D full volume.")
        image_support_batch = canonical_support[None, None].to(device_obj)
        source_id = str(case.source_provenance["case_id"])
        if evaluation_role == P0006_DEVELOPMENT_VALIDATION_DATA_ROLE:
            assert encoder is not None
            canonical = canonical_context.values
            if canonical.ndim == 3:
                canonical = canonical[None, None]
            elif canonical.ndim == 4:
                canonical = canonical[None]
            if canonical.ndim != 5:
                raise ValueError("P:0006 canonical source is not a full 3-D VAE batch.")
            latent, path_used = encode_latent(
                encoder,
                canonical.to(device_obj),
                case.source_domain,
                strategy="full",
                block_size=(1, 1, 1),
                halo=(0, 0, 0),
                precision="fp32",
            )
            if path_used != "full":
                raise ValueError("P:0006 evaluation requires actual full VAE encoding.")
            support = propagate_encoder_local_valid_core_support(
                canonical_support,
                encoder,
                expected_rule_sha256=str(
                    bank.manifest["operational_support_rule"]["rule_sha256"]
                ),
            )
        else:
            if source_id not in by_case:
                raise ValueError(
                    f"Paired source {source_id!r} is absent from the validation bank."
                )
            latent, support = bank.load(by_case[source_id])
            latent = latent[None].to(device_obj)
            path_used = "frozen_factored_bank_full_encode"
        support_batch = support[None, None].to(device_obj)
        z = stats.normalize(latent.to(device_obj), support_batch)
        generated_z = integrate_transport(
            translator,
            z,
            [case.source_domain],
            [case.target_domain],
            steps=integration_steps,
            solver=solver,  # type: ignore[arg-type]
        )
        decoded_canonical = _decode(
            decoder, stats.denormalize(generated_z), [case.target_domain]
        )[0].cpu()
        full_prediction = artifact.render_target(
            canonical_context.with_values(decoded_canonical), case.target_domain
        )
        methods: dict[str, torch.Tensor] = {
            "raw_identity": case.source,
            "gate01_calibrated_identity": baselines[case.case_identity][
                "gate01_calibrated_identity"
            ],
            "original_sb_v2": baselines[case.case_identity]["original_sb_v2"],
            "full_unified_model": full_prediction,
            "stage1_reconstruction_ceiling": _require_ceiling(case),
        }
        if sb_only_translator is not None:
            sb_z = integrate_transport(
                sb_only_translator,
                z,
                [case.source_domain],
                [case.target_domain],
                steps=integration_steps,
                solver=solver,  # type: ignore[arg-type]
            )
            sb_canonical = _decode(
                decoder, stats.denormalize(sb_z), [case.target_domain]
            )[0].cpu()
            methods["photometry_factored_sb_only"] = artifact.render_target(
                canonical_context.with_values(sb_canonical), case.target_domain
            )
        for name, ablation_model in sorted(ablation_translators.items()):
            ablation_z = integrate_transport(
                ablation_model,
                z,
                [case.source_domain],
                [case.target_domain],
                steps=integration_steps,
                solver=solver,  # type: ignore[arg-type]
            )
            ablation_canonical = _decode(
                decoder, stats.denormalize(ablation_z), [case.target_domain]
            )[0].cpu()
            methods[f"ablation_{name}"] = artifact.render_target(
                canonical_context.with_values(ablation_canonical), case.target_domain
            )
        method_metrics = {
            name: dict(metrics(prediction, case.target))
            for name, prediction in methods.items()
        }
        identical_support = (case.source != 0) & (case.target != 0)
        if not bool(identical_support.any()):
            raise ValueError("Paired case has empty identical source/target support.")
        identical_support_metrics = {
            name: dict(
                metrics(
                    prediction.masked_fill(~identical_support, 0.0),
                    case.target.masked_fill(~identical_support, 0.0),
                )
            )
            for name, prediction in methods.items()
        }
        anatomy = anatomy_preservation_components(
            canonical_context.values[None].to(device_obj),
            decoded_canonical[None].to(device_obj),
            image_support_batch,
        )
        graph_paths = []
        for intermediate in _all_intermediates(case.source_domain, case.target_domain):
            graph, direct, composed = graph_consistency_loss(
                translator,
                z,
                [case.source_domain],
                [intermediate],
                [case.target_domain],
                support_batch,
                steps=integration_steps,
                solver=solver,  # type: ignore[arg-type]
            )
            graph_paths.append(
                {
                    "path": (
                        f"{case.source_domain.label}->{intermediate.label}->"
                        f"{case.target_domain.label}"
                    ),
                    "intermediate_domain": intermediate.label,
                    "direct_vs_composed_l1": float(graph.cpu()),
                    "direct_vs_composed_mse": float(F.mse_loss(direct, composed).cpu()),
                }
            )
        requested = method_metrics["full_unified_model"]["nrmse"]
        wrong = _wrong_target_controls(
            translator,
            decoder,
            artifact,
            stats,
            z,
            canonical_context,
            case,
            integration_steps,
            solver,
            device_obj,
        )
        support_image = canonical_context.support_mask
        outside = ~support_image
        raw_decoder_background = (
            float(decoded_canonical[outside].abs().mean()) if bool(outside.any()) else 0.0
        )
        rendered_background = (
            float(full_prediction[outside].abs().mean()) if bool(outside.any()) else 0.0
        )
        montage_path = montage_dir / f"{sha256_text(case.case_identity)}.png"
        _render_montage(montage_path, case, methods)
        row = {
            "case_identity": case.case_identity,
            "subject_group_identity": case.subject_group_identity,
            "contrast": case.source_domain.contrast.value,
            "source_domain": case.source_domain.label,
            "target_domain": case.target_domain.label,
            "directed_field_pair": (
                f"{case.source_domain.field_strength_t:g}T->"
                f"{case.target_domain.field_strength_t:g}T"
            ),
            "raw_identity_stratum": (
                "catastrophic"
                if method_metrics["raw_identity"]["nrmse"] > RAW_IDENTITY_CATASTROPHIC_BOUNDARY
                else "ordinary"
            ),
            "methods": method_metrics,
            "identical_support_methods": identical_support_metrics,
            "identical_support_voxel_count": int(identical_support.sum()),
            "requested_vs_wrong_target": {
                "requested_nrmse": requested,
                "wrong_targets": wrong,
                "requested_better_than_every_wrong_target": all(
                    requested < item["requested_render_nrmse"] for item in wrong
                ),
                "mechanistic_rendering": "all conditions rendered through requested target map",
                "condition_native_rendering_role": "separately_labelled_diagnostic_only",
            },
            "graph_consistency": {
                "all_valid_intermediate_fields": True,
                "paths": graph_paths,
                "mean_direct_vs_composed_l1": float(
                    np.mean([item["direct_vs_composed_l1"] for item in graph_paths])
                ),
                "max_direct_vs_composed_l1": float(
                    np.max([item["direct_vs_composed_l1"] for item in graph_paths])
                ),
            },
            "anatomy_preservation": {
                key: float(value.cpu()) for key, value in anatomy.items()
            },
            "raw_pre_mask_decoder_background_leakage_mae": raw_decoder_background,
            "rendered_post_mask_background_leakage_mae": rendered_background,
            "source_provenance": dict(case.source_provenance),
            "target_provenance": dict(case.target_provenance),
            "source_encoding_path": path_used,
            "montage": {
                "path": montage_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(montage_path),
            },
        }
        rows.append(row)
        shard = {
            "contract_version": "stage2-unified-retrospective-case-shard-v2",
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "case": row,
        }
        shard["shard_sha256"] = sha256_json(shard)
        write_json_atomic(shard_path, shard, refuse_existing=True)
        print(
            json.dumps(
                {
                    "event": "unified_evaluation_case",
                    "case": case.case_identity,
                    "index": len(rows),
                    "total": len(cases),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result: dict[str, Any] = {
        "contract_version": UNIFIED_EVALUATION_CONTRACT,
        "scope": evaluation_role,
        "evidence_interpretation": (
            P0006_EVIDENCE_LIMITATION
            if evaluation_role == P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
            else "complete genuine paired R/validation development evidence"
        ),
        "population_or_generalization_claims_authorized": False,
        "P0009_confirmation_status": P0009_CONFIRMATION_STATUS,
        "P0009_executed": False,
        "prospective_records_loaded": (
            len(rows) if evaluation_role == P0006_DEVELOPMENT_VALIDATION_DATA_ROLE else 0
        ),
        "training_or_model_selection_prospective_records": 0,
        "evaluation_protocol_sha256": evaluation_identity_sha256,
        "bank_artifact_sha256": bank.artifact_sha256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "baseline_predictions": baseline_identity,
        "run_contract_sha256": run_contract["run_contract_sha256"],
        "methods": sorted(rows[0]["methods"]),
        "trained_ablation_methods_evaluated": sorted(
            f"ablation_{name}" for name in ablation_translators
        ),
        "case_count": len(rows),
        "cases": rows,
        "reductions": _reductions(rows),
        "montage_semantics": {
            "slice": "deterministic-central-axis-2",
            "columns": [
                "source",
                "target",
                "raw_identity",
                "gate01_calibrated_identity",
                "original_sb_v2",
                *( ["photometry_factored_sb_only"] if sb_only_translator is not None else [] ),
                "full_unified_model",
                "stage1_reconstruction_ceiling",
                "absolute_full_error",
            ],
        },
        "descriptor_coupling_used": False,
        "learned_disentanglement_claim": "none",
    }
    result["result_sha256"] = sha256_json(result)
    write_json_atomic(result_path, result, refuse_existing=True)
    _write_markdown(root / "report.md", result)
    return result


def _load_baseline_predictions(
    path: str | Path, cases: Sequence[PairedEvaluationCase]
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    stored = payload.pop("manifest_sha256", None)
    if stored != sha256_json(payload):
        raise ValueError("Baseline prediction manifest hash mismatch.")
    if payload.get("contract_version") != BASELINE_PREDICTIONS_CONTRACT:
        raise ValueError("Baseline prediction manifest contract mismatch.")
    expected = {case.case_identity: case for case in cases}
    entries = payload.get("cases")
    if not isinstance(entries, list) or {str(item.get("case_identity")) for item in entries} != set(expected):
        raise ValueError("Baseline predictions must cover the complete paired inventory exactly.")
    # Classify all case identities and validate every path/hash before opening arrays.
    prepared: list[tuple[str, str, Path, str]] = []
    for entry in entries:
        identity = str(entry["case_identity"])
        cohort_identity = classify_variant_a_cohort(
            case_identity=str(entry.get("record_identity", identity)),
            metadata_prefix=entry.get("metadata_prefix", "R"),
            supplied_cohort=entry.get("cohort", "R"),
            subject_identity=entry.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if entry.get("split") != "validation":
            raise ValueError("Baseline prediction entry is not R/validation.")
        if cohort_identity.subject_group_identity != expected[identity].subject_group_identity:
            raise ValueError("Baseline prediction subject group differs from the paired case.")
        for method in ("gate01_calibrated_identity", "original_sb_v2"):
            spec = entry.get(method)
            if not isinstance(spec, Mapping):
                raise ValueError(f"Missing baseline {method} for {identity}.")
            array_path = (source.parent / str(spec["path"])).resolve()
            if not array_path.is_file() or sha256_file(array_path) != spec.get("file_sha256"):
                raise ValueError(f"Baseline file identity mismatch for {identity}/{method}.")
            prepared.append((identity, method, array_path, str(spec["tensor_sha256"])))
    result: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for identity, method, array_path, expected_tensor_hash in prepared:
        array = np.load(array_path, allow_pickle=False)
        tensor = torch.from_numpy(np.asarray(array)).to(torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tuple(tensor.shape) != tuple(expected[identity].target.shape):
            raise ValueError("Baseline tensor shape mismatch.")
        if canonical_tensor_sha256(tensor) != expected_tensor_hash:
            raise ValueError("Baseline tensor content hash mismatch.")
        result[identity][method] = tensor
    return dict(result), {
        "contract_version": BASELINE_PREDICTIONS_CONTRACT,
        "file_sha256": sha256_file(source),
        "manifest_sha256": stored,
        "case_count": len(entries),
        "classification_before_array_load": True,
    }


def _preflight_paired_retrospective_manifest(path: str | Path) -> None:
    """Classify the complete paired inventory before its loader opens any endpoint array."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Paired retrospective manifest must be an object.")
    stored_hash = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stored_hash != sha256_json(body):
        raise ValueError("Paired retrospective manifest hash mismatch during preflight.")
    split = payload.get("split_provenance")
    if not isinstance(split, Mapping) or "validation" not in str(split.get("role", "")).lower():
        raise ValueError("Unified paired evaluation requires a sealed retrospective validation role.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Paired retrospective validation inventory is empty.")
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Malformed paired retrospective case.")
        for role in ("source", "target"):
            endpoint = case.get(role)
            if not isinstance(endpoint, Mapping):
                raise ValueError(f"Paired retrospective {role} endpoint is missing.")
            classify_variant_a_cohort(
                case_identity=str(endpoint.get("case_id", "")),
                metadata_prefix=endpoint.get("metadata_prefix"),
                supplied_cohort=endpoint.get("cohort"),
                subject_identity=endpoint.get("subject_id"),
                allowed_cohorts=("R",),
            )


def _wrong_target_controls(
    translator: BaseTranslator,
    decoder: torch.nn.Module,
    artifact: FrozenPhotometryArtifact,
    stats: FactoredLatentStats,
    z: torch.Tensor,
    canonical_context: Any,
    case: PairedEvaluationCase,
    steps: int,
    solver: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in FIELD_STRENGTHS_T:
        if field == case.target_domain.field_strength_t:
            continue
        wrong_domain = Domain(field, case.target_domain.contrast)
        wrong_z = integrate_transport(
            translator,
            z,
            [case.source_domain],
            [wrong_domain],
            steps=steps,
            solver=solver,  # type: ignore[arg-type]
        )
        wrong_canonical = _decode(
            decoder, stats.denormalize(wrong_z), [wrong_domain]
        )[0].cpu()
        # Mechanistic condition control: rendering is held fixed at the requested
        # target, so only the translator condition changes.
        requested_render_prediction = artifact.render_target(
            canonical_context.with_values(wrong_canonical), case.target_domain
        )
        native_render_prediction = artifact.render_target(
            canonical_context.with_values(wrong_canonical), wrong_domain
        )
        target = case.target.detach().cpu().double().numpy()
        denominator = float(np.sqrt(np.mean(target**2)))
        requested_prediction = requested_render_prediction.detach().cpu().double().numpy()
        native_prediction = native_render_prediction.detach().cpu().double().numpy()
        rows.append(
            {
                "conditioned_domain": wrong_domain.label,
                "requested_render_domain": case.target_domain.label,
                "requested_render_nrmse": float(
                    np.sqrt(np.mean((requested_prediction - target) ** 2))
                    / max(denominator, 1e-12)
                ),
                "condition_native_render_domain": wrong_domain.label,
                "condition_native_render_nrmse": float(
                    np.sqrt(np.mean((native_prediction - target) ** 2))
                    / max(denominator, 1e-12)
                ),
            }
        )
    return rows


def _all_intermediates(source: Domain, target: Domain) -> list[Domain]:
    return [
        Domain(value, source.contrast)
        for value in FIELD_STRENGTHS_T
        if value not in {source.field_strength_t, target.field_strength_t}
    ]


def _decode(decoder: torch.nn.Module, latent: torch.Tensor, domains: Sequence[Domain]) -> torch.Tensor:
    if hasattr(decoder, "decode"):
        return decoder.decode(latent, domains)  # type: ignore[attr-defined,no-any-return]
    return decoder(latent)  # type: ignore[no-any-return]


def _require_ceiling(case: PairedEvaluationCase) -> torch.Tensor:
    if case.stage1_reconstruction is None:
        raise ValueError("Every full-model evaluation case requires its Stage-1 ceiling.")
    return case.stage1_reconstruction


def _reductions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "overall": lambda row: "all",
        "per_source_domain": lambda row: str(row["source_domain"]),
        "per_target_domain": lambda row: str(row["target_domain"]),
        "per_contrast": lambda row: str(row["contrast"]),
        "per_directed_field_pair": lambda row: str(row["directed_field_pair"]),
        "ordinary_vs_catastrophic": lambda row: str(row["raw_identity_stratum"]),
    }
    output: dict[str, Any] = {}
    for name, grouper in dimensions.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[grouper(row)].append(row)
        output[name] = {}
        for key, items in sorted(grouped.items()):
            methods = items[0]["methods"]
            output[name][key] = {
                "count": len(items),
                "methods": {
                    method: {
                        metric: float(np.mean([item["methods"][method][metric] for item in items]))
                        for metric in values
                    }
                    for method, values in methods.items()
                },
                "identical_support_methods": {
                    method: {
                        metric: float(
                            np.mean(
                                [item["identical_support_methods"][method][metric] for item in items]
                            )
                        )
                        for metric in values
                    }
                    for method, values in items[0]["identical_support_methods"].items()
                },
                "requested_better_than_every_wrong_target_fraction": float(
                    np.mean(
                        [
                            item["requested_vs_wrong_target"][
                                "requested_better_than_every_wrong_target"
                            ]
                            for item in items
                        ]
                    )
                ),
                "graph_mean_direct_vs_composed_l1": float(
                    np.mean(
                        [
                            item["graph_consistency"]["mean_direct_vs_composed_l1"]
                            for item in items
                        ]
                    )
                ),
                "anatomy_preservation": {
                    metric: float(
                        np.mean([item["anatomy_preservation"][metric] for item in items])
                    )
                    for metric in ("low_mid", "edge", "gradient", "total")
                },
                "raw_pre_mask_decoder_background_leakage_mae": float(
                    np.mean(
                        [
                            item["raw_pre_mask_decoder_background_leakage_mae"]
                            for item in items
                        ]
                    )
                ),
            }
    return output


def _render_montage(
    path: Path, case: PairedEvaluationCase, methods: Mapping[str, torch.Tensor]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes = [
        ("source", case.source),
        ("target", case.target),
        ("raw", methods["raw_identity"]),
        ("calibrated", methods["gate01_calibrated_identity"]),
        ("SB-v2", methods["original_sb_v2"]),
        *(
            [("factored SB", methods["photometry_factored_sb_only"])]
            if "photometry_factored_sb_only" in methods
            else []
        ),
        ("full", methods["full_unified_model"]),
        ("ceiling", methods["stage1_reconstruction_ceiling"]),
        ("|full-target|", (methods["full_unified_model"] - case.target).abs()),
    ]
    index = int(case.target.shape[-1] // 2)
    fig, axes = plt.subplots(1, len(volumes), figsize=(2.5 * len(volumes), 2.8))
    for axis, (title, volume) in zip(axes, volumes):
        axis.imshow(volume.squeeze(0)[..., index].detach().cpu(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        fig.savefig(temp_path, format="png", dpi=110)
        plt.close(fig)
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    overall = result["reductions"]["overall"]["all"]
    lines = [
        "# Unified retrospective Stage-2 evaluation",
        "",
        f"Cases: {result['case_count']} (R/validation only; P loaded: 0)",
        "",
        "| Method | nRMSE | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
    ]
    for method, values in sorted(overall["methods"].items()):
        lines.append(
            f"| {method} | {values['nrmse']:.6f} | {values['ssim']:.6f} | {values['lpips']:.6f} |"
        )
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        temp_path.write_bytes(encoded)
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _module_state_sha256(module: torch.nn.Module | None) -> str | None:
    if module is None:
        return None
    from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256

    return sha256_json(
        {
            name: storage_tensor_sha256(value.detach().cpu().contiguous())
            for name, value in sorted(module.state_dict().items())
        }
    )


__all__ = [
    "BASELINE_PREDICTIONS_CONTRACT",
    "UNIFIED_EVALUATION_CONTRACT",
    "evaluate_stage2_unified",
]
