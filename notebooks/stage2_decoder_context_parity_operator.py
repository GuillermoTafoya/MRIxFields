"""Replay the archived P:0006 3T -> 7T T1w pilot with the packaged L4 evaluator."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import stage2_zero_transport_operator as op
from stage2_decoder_context_inputs import SENTINEL_CASE_ID, as_plan_row, load_pilot_inputs, stage_and_load_models
from fieldbridge.evaluation.decoder_context import DECODER_CONTEXT_CONTRACT, SOURCE_NATIVE_EXTERIOR
from fieldbridge.evaluation.paired_decoder_context import prepare_source, score_direction
from fieldbridge.evaluation.stage2_step200_inference_audit import _apply_shape_bound_calibrator

EXPERIMENT = "stage2_decoder_context_parity"
FROZEN_COMMIT = "37063b561d9e9dee4469c23567e5c05afbc5bfa5"
PILOT_DIR = Path("step200_p0006_paired_matched_calibration/20261121T045754Z-92d41266")
QUALIFICATION = Path("low_memory_reference_R0398/20261121T021240Z-7801ec2e/qualification.json")
REFERENCE_HASHES = {
    PILOT_DIR / "receipt.json": "07072c6041e84c5118fab3bcf1200347a2f14183bf5e6e4f00fad482bbc43493",
    PILOT_DIR / "summary.json": "fce039abb9e8651a73ebfea3043907fa6e71302e3fd481f04433f5c794fd4f89",
    QUALIFICATION: "df26ec301faa8a03b86b092ffed0ce0cbde50daf2d6661e12179ead41c1db659",
}
METHODS = ("acquired_source", "stage1_source", "photometry_only", "zero", "correct", "wrong")
METRICS = ("full_nrmse", "shared_foreground_rmse", "shared_foreground_nrmse")
TOLERANCE = 1e-10


def read_reference(root):
    for path, expected in REFERENCE_HASHES.items():
        if op.sha256_file(root / path) != expected:
            raise ValueError(f"Archived reference digest changed: {path}")
    pilot = json.loads((root / PILOT_DIR / "receipt.json").read_bytes())
    if (pilot["implementation_commit"] != FROZEN_COMMIT or pilot["case_id"] != SENTINEL_CASE_ID
            or pilot["status"] != "completed" or pilot["analysis"]["replay"]["status"] != "passed"):
        raise ValueError("Expected the completed, replay-qualified frozen pilot.")
    return pilot


def compare_pilot(observed, expected):
    """Use the measured notebook's exact trace/image checks and absolute scalar tolerance."""
    trace_matches = observed["trace"] == expected["trace"]
    failures = [] if trace_matches else ["trace"]
    differences = {}
    image_matches = {}
    for table in ("raw_scores", "calibrated_scores"):
        if set(observed[table]) != set(METHODS) or set(expected[table]) != set(METHODS):
            raise ValueError(f"Expected all six pilot methods in {table}.")
        for method in METHODS:
            name = f"{table}/{method}"
            actual, reference = observed[table][method], expected[table][method]
            image_matches[name] = actual["rendered_prediction_sha256"] == reference["rendered_prediction_sha256"]
            if not image_matches[name]:
                failures.append(name + "/rendered_prediction_sha256")
            for metric in METRICS:
                key = name + "/" + metric
                difference = actual[metric] - reference[metric]
                if not math.isfinite(difference):
                    raise ValueError(f"Nonfinite score difference: {key}")
                differences[key] = difference
                if abs(difference) > TOLERANCE:
                    failures.append(key)
    return {"status": "failed" if failures else "passed", "trace_matches": trace_matches,
            "image_hash_matches": image_matches, "score_minus_reference": differences,
            "absolute_tolerance": TOLERANCE, "failures": failures}


def evaluate_pilot(stage_dir, pilot, device):
    cohort, spec, source, target, baseline = load_pilot_inputs(op.DRIVE)
    plan = as_plan_row(spec)
    reference = pilot["paired_input_provenance"]
    if (spec.case_id != SENTINEL_CASE_ID or plan["source_domain"] != pilot["source_domain"]
            or plan["target_domain"] != pilot["target_domain"]
            or plan["wrong_domain"] != pilot["wrong_neural_domain"] or spec.seed != pilot["inference_seed"]
            or plan["array_sha256"] != reference["array_sha256"]
            or plan["calibrated_identity_sha256"] != reference["calibrated_identity_sha256"]):
        raise ValueError("Case, labels, seed or input identities differ from the archived pilot.")

    def mark(message):
        print(message, flush=True)

    encoder, decoder, translator, artifact, stats, rule, provenance = stage_and_load_models(stage_dir, device, mark)
    prepared = prepare_source(encoder=encoder, artifact=artifact, stats=stats, support_rule_sha256=rule,
                              source=source.source_image, source_domain=spec.source_domain, device=device, mark=mark)

    def calibrate(prediction):
        return _apply_shape_bound_calibrator(cohort.calibrator, prediction, spec.target_domain,
                                             source.support_mask, expected_shape=tuple(source.source_image.shape),
                                             role="matched calibration")

    analysis = score_direction(prepared=prepared, decoder=decoder, translator=translator, artifact=artifact,
                               stats=stats, target=target, target_domain=spec.target_domain,
                               wrong_domain=spec.wrong_domain, raw_stage1_source=source.raw_identity,
                               calibrated_stage1_source=baseline, calibrate=calibrate,
                               case_seed=spec.seed, device=device, mark=mark)
    return {"plan": plan, "input_provenance": cohort.provenance, "model_provenance": provenance,
            "analysis": analysis, "parity": compare_pilot(analysis, pilot["analysis"])}


def main():
    commit = op.verified_commit(Path(__file__).resolve().parents[1])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set the frozen CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python.")
    if (not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0)
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise RuntimeError("The archived parity run requires a Colab NVIDIA L4 with native BF16.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(op.SEED)
    pilot = read_reference(op.OUTPUT_ROOT)
    out = op.new_run_directory(op.OUTPUT_ROOT / EXPERIMENT / commit)
    print(f"Writing the one-direction parity result to {out}", flush=True)
    report = {"experiment": EXPERIMENT, "participant": "P:0006", "case_id": SENTINEL_CASE_ID,
              "implementation_commit": commit, "decoder_context_policy": SOURCE_NATIVE_EXTERIOR,
              "decoder_context_contract": DECODER_CONTEXT_CONTRACT,
              "reference_commit": FROZEN_COMMIT, "reference_files": {str(p): h for p, h in REFERENCE_HASHES.items()},
              "seed": op.SEED, "backend": {"gpu": torch.cuda.get_device_name(device),
                                          "capability": list(torch.cuda.get_device_capability(device)),
                                          "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                          "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
                                          "cpu_threads": torch.get_num_threads(), "workspace": ":4096:8"}}
    try:
        with TemporaryDirectory(prefix="fieldbridge-pilot-", dir="/content") as scratch:
            result = evaluate_pilot(Path(scratch), pilot, device)
    except Exception as error:
        op.write_result(out / "summary.json", {**report, "status": "error", "error": str(error)})
        raise
    status = result["parity"]["status"]
    op.write_result(out / "summary.json", {**report, **result, "status": status})
    print(f"{status.upper()} — {out / 'summary.json'}", flush=True)
    if status != "passed":
        raise RuntimeError(f"Archived pilot parity failed: {result['parity']['failures']}")


if __name__ == "__main__":
    main()
