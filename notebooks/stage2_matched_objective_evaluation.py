"""Fixed-update matched acquired-target evaluation on the frozen P:0006 direction list."""
from __future__ import annotations

import json
import math
import os
from itertools import groupby
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

import stage2_zero_transport_operator as op
from stage2_decoder_context_inputs import (
    ARRAY_KEYS, SENTINEL_CASE_ID, SOURCE_KEYS, Direction, as_plan_row, calibrated_source_reconstruction,
    load_qualified_codec, load_source, load_target, read_p0006_metadata,
)
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import sha256_json
from fieldbridge.evaluation.decoder_context import DECODER_CONTEXT_CONTRACT, SOURCE_NATIVE_EXTERIOR
from fieldbridge.evaluation.paired_decoder_context import prepare_source, score_direction
from fieldbridge.evaluation.stage2_step200_inference_audit import _apply_shape_bound_calibrator
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import P0006_IDENTITY_SHA256

OBJECTIVES = ("forward_drift", "current_velocity")
CONTROLS = ("zero", "photometry_only", "stage1_source", "acquired_source")
EXPERIMENT = "p0006-matched-bridge-objectives-step200-v1"


def frozen_directions(cohort, receipts, seeds):
    """Bind the sealed case order to acquisition identities, wrong requests, and solver seeds."""
    rows = cohort.rows
    if any(len(items) != 60 for items in (cohort.manifest.case_specs, rows, receipts, seeds)):
        raise ValueError("Expected the frozen 60-direction manifest, receipts and seed plan.")
    expected = {(Domain(source, c), Domain(target, c)) for c in CONTRASTS
                for source in FIELD_STRENGTHS_T for target in FIELD_STRENGTHS_T if source != target}
    directions = {}
    acquisitions = {}
    for ordinal, (receipt, seed) in enumerate(zip(receipts, seeds, strict=True), 1):
        case = receipt["case_identity"]
        row = rows[case]
        if (row["traveller_identity_sha256"] != P0006_IDENTITY_SHA256 or seed["index"] != ordinal
                or seed["source_image_sha256"] != receipt["source_image_sha256"]
                or any(row[k] != receipt[k] or row[k] != seed[k] for k in ("source_domain", "target_domain"))):
            raise ValueError(f"Participant or direction-specific seed mismatch for {case}.")
        source, target = Domain.from_dict(row["source_domain"]), Domain.from_dict(row["target_domain"])
        pair = (source, target)
        if pair not in expected or pair in directions:
            raise ValueError(f"Unexpected or duplicate field direction {pair}.")
        hashes = {k: receipt[h] for k, h in ARRAY_KEYS.items()}
        if any(row[k]["sha256"] != hashes[k] for k in ARRAY_KEYS):
            raise ValueError(f"Archived acquisition or frozen reconstruction mismatch for {case}.")
        source_refs = tuple((row[k]["path"], hashes[k]) for k in SOURCE_KEYS)
        if acquisitions.setdefault(source, source_refs) != source_refs:
            raise ValueError(f"Inconsistent source identity across targets for {source.label}.")
        # This is the rotation from the frozen 60-direction notebook, for each source/contrast.
        alternatives = [f for f in FIELD_STRENGTHS_T if f != source.field_strength_t]
        wrong = Domain(alternatives[(alternatives.index(target.field_strength_t) + 1) % len(alternatives)],
                       target.contrast)
        directions[pair] = Direction(case, source, target, wrong, seed["seed"], ordinal,
                                     hashes, receipt["calibrated_identity_sha256"])
    if set(directions) != expected or len(acquisitions) != 15:
        raise ValueError("Incomplete frozen study grid.")
    for spec in directions.values():
        ref = rows[spec.case_id]["target"]
        if (ref["path"], ref["sha256"]) != acquisitions[spec.target_domain][0]:
            raise ValueError(f"The measured target for {spec.case_id} differs from its sealed acquisition.")
    pilot = directions[Domain(3, "T1w"), Domain(7, "T1w")]
    if (pilot.case_id, pilot.wrong_domain, pilot.ordinal, pilot.seed) != (
            SENTINEL_CASE_ID, Domain(0.1, "T1w"), 12, 1091568062):
        raise ValueError("Original pilot identity differs from the frozen study.")
    return sorted(directions.values(), key=lambda d: (
        d.source_domain.contrast_index, d.source_domain.field_strength_t, d.target_domain.field_strength_t))


def matched_scores(analyses):
    """Check the common controls/inputs and extract matched, calibrated whole-volume nRMSE."""
    forward, current = (analyses[key] for key in OBJECTIVES)
    for table in ("raw_scores", "calibrated_scores"):
        for control in CONTROLS:
            if forward[table][control] != current[table][control]:
                raise ValueError(f"{table}/{control} changed between the matched fits.")
    for key, value in forward["trace"].items():
        if key != "requests" and current["trace"][key] != value:
            raise ValueError(f"Matched checkpoints differ in source, mask, seed or target trace: {key}.")
    if forward["trace"]["requests"]["zero"] != current["trace"]["requests"]["zero"]:
        raise ValueError("Matched zero transports differ.")
    for request in ("correct", "wrong"):
        for key in ("velocity_evaluations", "velocity_times", "velocity_labels", "neural_request",
                    "decoder_domain", "renderer_domain"):
            if forward["trace"]["requests"][request][key] != current["trace"]["requests"][request][key]:
                raise ValueError(f"Matched solver/label routing differs in {request}/{key}.")
    scores = {objective: {method: analyses[objective]["calibrated_scores"][method]["full_nrmse"]
                         for method in ("correct", "wrong", *CONTROLS)} for objective in OBJECTIVES}
    if not all(math.isfinite(value) and value >= 0 for methods in scores.values() for value in methods.values()):
        raise ValueError("Expected finite calibrated acquired-target scores.")
    return scores


def summarize(rows):
    """Pair objectives by direction before reducing. The original pilot has its own stratum."""
    if len(rows) != 60 or SENTINEL_CASE_ID not in rows:
        raise ValueError("A final summary requires the pilot and all 59 development directions.")
    by_direction = []
    for case_id, row in sorted(rows.items(), key=lambda item: item[1]["plan"]["ordinal"]):
        scores = matched_scores(row["analyses"])
        by_direction.append({"plan": row["plan"], "calibrated_full_nrmse": scores,
                             "correct_current_minus_forward": scores["current_velocity"]["correct"]
                                                              - scores["forward_drift"]["correct"]})
    pilot = next(r for r in by_direction if r["plan"]["case_id"] == SENTINEL_CASE_ID)
    development = [r for r in by_direction if r["plan"]["case_id"] != SENTINEL_CASE_ID]
    if len(development) != 59:
        raise ValueError("Misidentified pilot exclusion.")

    def reduce_group(cases):
        diffs = [r["correct_current_minus_forward"] for r in cases]
        means = {o: {method: fmean(r["calibrated_full_nrmse"][o][method] for r in cases)
                     for method in ("correct", "wrong", *CONTROLS)} for o in OBJECTIVES}
        return {"count": len(cases), "mean_paired_current_minus_forward": fmean(diffs),
                "directions_current_lower": sum(d < 0 for d in diffs),
                "directions_tied": sum(d == 0 for d in diffs), "mean_calibrated_full_nrmse": means}

    return {"metric": "Calibrated whole-volume Task 3 nRMSE versus the acquired target",
            "difference": "current_velocity minus forward_drift for the correct request, paired within direction",
            "participant": "P:0006", "update": 200, "development_59": reduce_group(development),
            "original_pilot": {**pilot, **reduce_group([pilot])}, "by_direction": by_direction}


def _write_or_verify(path, data):
    if not path.exists():
        try:
            op.write_result(path, data)
            return
        except FileExistsError:
            pass
    if json.loads(path.read_bytes()) != data:
        raise ValueError(f"Existing evaluation file has a different identity or contents: {path.name}")


def _read_case(path, plan, contract_hash):
    if not path.exists():
        return None
    record = json.loads(path.read_bytes())
    digest = record.pop("receipt_sha256")
    if (sha256_json(record) != digest or record["evaluation_sha256"] != contract_hash
            or record["plan"] != plan):
        raise ValueError(f"Case receipt digest, plan or evaluation identity differs: {path.name}")
    return record


def _evaluate_group(specs, *, cohort, models, encoder, decoder, artifact, stats, rule, device, save, mark):
    source = load_source(cohort, specs[0])
    prepared = prepare_source(encoder=encoder, artifact=artifact, stats=stats, support_rule_sha256=rule,
                              source=source.source_image, source_domain=specs[0].source_domain, device=device,
                              mark=mark)
    for spec in specs:
        target = load_target(cohort, spec, tuple(source.source_image.shape))
        baseline = calibrated_source_reconstruction(cohort, spec, source)

        def calibrate(image):
            return _apply_shape_bound_calibrator(cohort.calibrator, image, spec.target_domain, source.support_mask,
                                                 expected_shape=tuple(source.source_image.shape),
                                                 role="matched calibration")

        analyses = {}
        for objective in OBJECTIVES:
            mark(f"{spec.source_domain.label} -> {spec.target_domain.label}   {objective}")
            analyses[objective] = score_direction(
                prepared=prepared, decoder=decoder, translator=models[objective], artifact=artifact, stats=stats,
                target=target, target_domain=spec.target_domain, wrong_domain=spec.wrong_domain,
                raw_stage1_source=source.raw_identity, calibrated_stage1_source=baseline,
                calibrate=calibrate, case_seed=spec.seed, device=device, mark=mark,
            )
        matched_scores(analyses)
        save(spec, analyses)


def evaluate_step200(*, run_dir, stage_dir, commit, fits, device):
    """Resume compatible case receipts and compare the two checkpoints at the fixed update."""
    from stage2_matched_objective_training import load_step200_translator

    device = torch.device(device)
    cohort, receipts, seeds = read_p0006_metadata(op.DRIVE)
    specs = frozen_directions(cohort, receipts, seeds)
    if set(fits) != set(OBJECTIVES):
        raise ValueError("Both matched training receipts are required.")
    checkpoints = {}
    for objective in OBJECTIVES:
        fit = fits[objective]
        if (fit["bridge_objective"] != objective or fit["step"] != 200 or fit["implementation_commit"] != commit):
            raise ValueError(f"Incorrect fit objective, implementation, or checkpoint step for {objective}.")
        checkpoints[objective] = {k: fit[k] for k in ("checkpoint_sha256", "translator_state_sha256",
                                                       "run_fingerprint", "validation_plan_sha256")}
    contract = {"version": EXPERIMENT, "implementation_commit": commit, "step": 200, "checkpoints": checkpoints,
                "context_contract": DECODER_CONTEXT_CONTRACT, "context_policy": SOURCE_NATIVE_EXTERIOR,
                "cohort": cohort.provenance, "inventory": [as_plan_row(s) for s in specs],
                "solver": {"method": "heun", "steps": 4, "precision": "bf16"},
                "backend": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "numpy": np.__version__,
                            "gpu": torch.cuda.get_device_name(device), "cudnn": torch.backends.cudnn.version(),
                            "cpu_threads": torch.get_num_threads(), "workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                            "deterministic": torch.are_deterministic_algorithms_enabled(),
                            "cudnn_benchmark": torch.backends.cudnn.benchmark}}
    digest = sha256_json(contract)
    directory = Path(run_dir) / "evaluation"
    cases_dir = directory / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    _write_or_verify(directory / "contract.json", contract)
    rows = {}
    pending = []
    for spec in specs:
        row = _read_case(cases_dir / (spec.case_id + ".json"), as_plan_row(spec), digest)
        if row is None:
            pending.append(spec)
        else:
            rows[spec.case_id] = row

    def mark(message):
        print(message, flush=True)

    def save(spec, analyses):
        row = {"plan": as_plan_row(spec), "analyses": analyses, "evaluation_sha256": digest}
        _write_or_verify(cases_dir / (spec.case_id + ".json"), {**row, "receipt_sha256": sha256_json(row)})
        rows[spec.case_id] = row
        mark(f"Saved matched direction {len(rows)}/60.")

    if pending:
        stage = Path(stage_dir)
        encoder, decoder, artifact, stats, rule, model_info = load_qualified_codec(stage, stage / "bank", device, mark)
        # The qualification uses a synthetic input on this GPU and the same staged VAE used for evaluation.
        _write_or_verify(directory / "codec.json", model_info)
        models = {objective: load_step200_translator(fits[objective], device) for objective in OBJECTIVES}
        for _, group in groupby(pending, key=lambda d: d.source_domain):
            _evaluate_group(list(group), cohort=cohort, models=models, encoder=encoder, decoder=decoder,
                            artifact=artifact, stats=stats, rule=rule, device=device, mark=mark, save=save)
    summary = {"status": "complete", "implementation_commit": commit, "evaluation_sha256": digest,
               "checkpoints": checkpoints, **summarize(rows)}
    _write_or_verify(directory / "summary.json", summary)
    return summary
