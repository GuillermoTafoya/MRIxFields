"""The matched-fit operator selects terminal states and verifies shared training draws."""
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest

from fieldbridge.training.checkpoints import resolve_git_commit
from fieldbridge.training.stage2_unified import build_unified_validation_plan
from test_stage2_unified import SyntheticFactoredIndex, TinyDecoder, TinyTranslator, _stats


def test_v7_matched_fits_publish_recover_and_authenticate_actual_updates(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks"))
    operator = import_module("stage2_matched_objective_training")
    train = SyntheticFactoredIndex("train")
    validation = SyntheticFactoredIndex("validation")
    steps = 2
    monkeypatch.setattr(operator, "STEPS", steps)
    monkeypatch.setattr(operator, "CHECKPOINT", "stage2_unified_full_step000000002.pt")
    monkeypatch.setattr(operator, "_translator", lambda spec: TinyTranslator())
    original_arm_config = operator._arm_config

    def cpu_config(recipe, objective, attempt):
        cfg = original_arm_config(recipe, objective, attempt)
        return replace(cfg, device="cpu", precision="fp32", critic_channels=(4,), anatomy_pool_scales=(1,),
                       anatomy_support_erosion=0, decoder_activation_checkpoint_mode="disabled")

    monkeypatch.setattr(operator, "_arm_config", cpu_config)
    recipe = {"model": {"name": "tiny"}, "training": {"batch_size": 1, "checkpoint": {}, "pilot": {}}}
    plan = build_unified_validation_plan(validation, validation_seed=20260818)
    monkeypatch.setattr(operator, "STEP200_VALIDATION_PLAN_SHA256", plan["validation_plan_sha256"])
    common = {"inputs": {}, "implementation_commit": resolve_git_commit(), "backend": {"device": "cpu"}}
    decoder, stats = TinyDecoder(), _stats()

    def fit(objective):
        return operator._fit_one(tmp_path / objective, objective, recipe, train, validation, stats, decoder, common)

    fits = {objective: fit(objective) for objective in operator.OBJECTIVES}
    operator.verify_match(fits)
    assert fits["current_velocity"]["run_fingerprint"] != fits["forward_drift"]["run_fingerprint"]
    assert fits["current_velocity"]["translator_state_sha256"] != fits["forward_drift"]["translator_state_sha256"]
    for receipt in fits.values():
        restored = operator.load_step200_translator(receipt, "cpu")
        assert not restored.training
        assert receipt["step"] == steps

    def forbid_another_fit(*args, **kwargs):
        raise AssertionError("Tried to re-fit an already published terminal checkpoint.")

    monkeypatch.setattr(operator, "run_stage2_unified_train", forbid_another_fit)
    # Exercise the reconnect window between atomic checkpoint publication and fit.json.
    checkpoint_dir = Path(fits["forward_drift"]["checkpoint"]).parent
    (checkpoint_dir.parent / "fit.json").unlink()
    assert fit("forward_drift") == fits["forward_drift"]
    assert fit("current_velocity") == fits["current_velocity"]
    mismatched = {**fits, "current_velocity": {**fits["current_velocity"], "training_step_plans_sha256": "f" * 64}}
    with pytest.raises(ValueError, match="draw sequence"):
        operator.verify_match(mismatched)
