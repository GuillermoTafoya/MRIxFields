"""The one-direction Colab operator uses the frozen pilot's strict comparison."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest


def test_archived_pilot_comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks"))
    operator = import_module("stage2_decoder_context_parity_operator")
    scores = {name: {**dict.fromkeys(operator.METRICS, 0.25), "rendered_prediction_sha256": "0" * 64}
              for name in operator.METHODS}
    reference = {"trace": {"integration_count": 2}, "raw_scores": scores, "calibrated_scores": scores}
    observed = {key: deepcopy(value) for key, value in reference.items()}
    observed["calibrated_scores"]["zero"]["full_nrmse"] += 1e-12
    result = operator.compare_pilot(observed, reference)
    assert result["status"] == "passed"
    assert len(result["image_hash_matches"]) == 12
    assert len(result["score_minus_reference"]) == 36
    observed["trace"]["integration_count"] = 1
    observed["raw_scores"]["zero"]["rendered_prediction_sha256"] = "1" * 64
    observed["calibrated_scores"]["wrong"]["shared_foreground_rmse"] -= 2e-8
    result = operator.compare_pilot(observed, reference)
    assert result["status"] == "failed"
    assert set(result["failures"]) == {
        "trace", "raw_scores/zero/rendered_prediction_sha256", "calibrated_scores/wrong/shared_foreground_rmse",
    }
    assert result["score_minus_reference"]["calibrated_scores/wrong/shared_foreground_rmse"] < 0
    del observed["calibrated_scores"]["correct"]
    with pytest.raises(ValueError, match="six pilot methods"):
        operator.compare_pilot(observed, reference)
