"""Synthetic controls for the fixed-update P:0006 acquired-target study operator."""
import json
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain


@pytest.fixture
def operator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks"))
    return import_module("stage2_matched_objective_evaluation")


def synthetic_grid(op):
    pairs = [(Domain(s, c), Domain(t, c))
             for c in CONTRASTS for s in FIELD_STRENGTHS_T for t in FIELD_STRENGTHS_T if s != t]
    pilot = Domain(3, "T1w"), Domain(7, "T1w")
    pairs.remove(pilot)
    pairs.insert(11, pilot)
    rows, receipts, seeds = {}, [], []
    for i, (src, dst) in enumerate(pairs, 1):
        case = op.SENTINEL_CASE_ID if i == 12 else f"synthetic-case-{i}"
        role = {"source_domain": src.to_dict(), "target_domain": dst.to_dict()}
        row = {**role, "case_id": case, "traveller_identity_sha256": op.P0006_IDENTITY_SHA256}
        for key in op.ARRAY_KEYS:
            identity = (dst if key == "target" else src).label + ("/source_image" if key == "target" else "/" + key)
            row[key] = {"path": identity, "sha256": identity}
        rows[case] = row
        receipts.append({**role, "case_identity": case, "calibrated_identity_sha256": case,
                         **{v: row[k]["sha256"] for k, v in op.ARRAY_KEYS.items()}})
        seeds.append({**role, "index": i, "seed": 1091568062 if i == 12 else i,
                      "source_image_sha256": row["source_image"]["sha256"]})
    cohort = SimpleNamespace(rows=rows, manifest=SimpleNamespace(case_specs=list(rows.values())),
                             provenance={"participant": "P:0006"}, calibrator="calibrator")
    return cohort, receipts, seeds


def test_grid_rejects_different_participant_or_target(operator):
    cohort, receipts, seeds = synthetic_grid(operator)
    assert len(operator.frozen_directions(cohort, receipts, seeds)) == 60
    pilot = cohort.rows[operator.SENTINEL_CASE_ID]
    pilot["traveller_identity_sha256"] = "different-synthetic-participant"
    with pytest.raises(ValueError, match="Participant"):
        operator.frozen_directions(cohort, receipts, seeds)
    pilot["traveller_identity_sha256"] = operator.P0006_IDENTITY_SHA256
    pilot["target"]["path"] = "different-synthetic-acquisition"
    with pytest.raises(ValueError, match="measured target"):
        operator.frozen_directions(cohort, receipts, seeds)


def test_matched_study_scores_both_checkpoints_and_resumes(operator, tmp_path, monkeypatch):
    op = operator
    grid = synthetic_grid(op)
    monkeypatch.setattr(op, "read_p0006_metadata", lambda _: grid)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(op, "load_qualified_codec", lambda *_: ("encoder", "decoder", "photo", "stats", "rule", {}))
    training = SimpleNamespace(load_step200_translator=lambda fit, device: fit["bridge_objective"])
    monkeypatch.setitem(sys.modules, "stage2_matched_objective_training", training)
    image = torch.ones(1, 2, 3, 4)
    mask = image.bool()
    monkeypatch.setattr(op, "load_source", lambda *_: SimpleNamespace(
        source_image=image, support_mask=mask, raw_identity=image))
    encodes = []
    def prepare(*, source_domain, **kwargs):
        encodes.append(source_domain)
        return source_domain
    monkeypatch.setattr(op, "prepare_source", prepare)
    monkeypatch.setattr(op, "load_target", lambda _, spec, shape: spec)
    monkeypatch.setattr(op, "calibrated_source_reconstruction", lambda *_: image)
    monkeypatch.setattr(op, "_apply_shape_bound_calibrator",
                        lambda calibrator, x, domain, mask, **kwargs: (calibrator, x, domain))
    calls = []
    def score(*, prepared, translator, target, wrong_domain, target_domain, case_seed, calibrate, **kwargs):
        assert (target.source_domain, target.target_domain, target.wrong_domain, target.seed) == (
            prepared, target_domain, wrong_domain, case_seed)
        assert calibrate(image)[2] == target.target_domain
        calls.append((target.case_id, translator))
        # Opposite effects for the original pilot and the other directions expose accidental pooling.
        delta = 0.2 if target.case_id == op.SENTINEL_CASE_ID else -0.01
        value = 0.4 + (delta if translator == "current_velocity" else 0)
        scores = {k: {"full_nrmse": value if k == "correct" else 0.6, "rendered_prediction_sha256": k}
                  for k in ("correct", "wrong", *op.CONTROLS)}
        requests = {k: {"velocity_evaluations": 8, "velocity_times": [], "velocity_labels": [],
                        "neural_request": d.label, "decoder_domain": target_domain.label,
                        "renderer_domain": target_domain.label} for k, d in (
                            ("correct", target_domain), ("wrong", wrong_domain))}
        requests["zero"] = {}
        return {"raw_scores": scores, "calibrated_scores": scores,
                "trace": {"case_seed": case_seed, "requests": requests}}
    monkeypatch.setattr(op, "score_direction", score)
    fits = {o: {"bridge_objective": o, "implementation_commit": "synthetic", "step": 200,
                "checkpoint_sha256": o, "translator_state_sha256": o,
                "validation_plan_sha256": "plan", "run_fingerprint": o} for o in op.OBJECTIVES}
    kwargs = dict(run_dir=tmp_path, stage_dir=tmp_path, commit="synthetic", fits=fits, device="cpu")
    report = op.evaluate_step200(**kwargs)
    assert len(encodes) == len(set(encodes)) == 15
    assert len(calls) == len(set(calls)) == 120
    assert report["development_59"]["count"] == 59
    assert report["development_59"]["mean_paired_current_minus_forward"] == pytest.approx(-0.01)
    assert report["original_pilot"]["mean_paired_current_minus_forward"] == pytest.approx(0.2)
    assert set(report["development_59"]["mean_calibrated_full_nrmse"]["current_velocity"]) == {
        "correct", "wrong", *op.CONTROLS}
    assert op.evaluate_step200(**kwargs) == report
    assert len(calls) == 120
    with pytest.raises(ValueError, match="different identity"):
        op.evaluate_step200(**{**kwargs, "fits": {**fits, "forward_drift":
                             {**fits["forward_drift"], "checkpoint_sha256": "changed"}}})
    directory = tmp_path / "evaluation"
    pilot_path = directory / "cases" / (op.SENTINEL_CASE_ID + ".json")
    pilot = json.loads(pilot_path.read_bytes())
    changed = deepcopy(pilot["analyses"])
    changed["current_velocity"]["calibrated_scores"]["zero"]["rendered_prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="zero changed"):
        op.matched_scores(changed)
    pilot["plan"]["seed"] = 0
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt digest"):
        op.evaluate_step200(**kwargs)
