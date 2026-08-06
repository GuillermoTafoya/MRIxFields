from __future__ import annotations

import json

import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.evaluation.stage2_gate01 import (
    Gate01Case,
    evaluate_gate01,
    fixed_montage_specifications,
    frozen_artifact_provenance,
    render_gate01_markdown,
    write_gate01_outputs,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    RESPLIT_FINGERPRINT,
    TrainingTemplateVolume,
    fit_posthoc_target_calibrator,
)


def _calibrator():
    records = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            volume = torch.linspace(0.0, 0.8, 64).reshape(4, 4, 4)
            volume = volume * (0.5 + 0.03 * field_index + 0.02 * contrast_index)
            records.append(
                TrainingTemplateVolume(
                    volume=volume,
                    domain=Domain(field, contrast),
                    record_identity=f"template-{contrast.value}-{field:g}",
                )
            )
    return fit_posthoc_target_calibrator(
        records,
        split_fingerprint=RESPLIT_FINGERPRINT,
        training_cohort_identity="synthetic-retrospective-training",
        code_commit="fit-commit",
        num_quantiles=9,
    )


def _metric_fn(prediction, target, metrics, device):
    del device
    error = float((prediction.to(torch.float64) - target).abs().mean())
    values = {
        "nrmse": error,
        "ssim": 1.0 - min(error, 1.0),
        "lpips": error * 0.5,
    }
    return {metric: values[metric] for metric in metrics}


def _volume(value: float) -> torch.Tensor:
    result = torch.full((4, 4, 4), value, dtype=torch.float32)
    result[0, 0, 0] = 0.0
    return result


def _case(
    contrast,
    source: float,
    target: float,
    *,
    wrong: bool = False,
) -> Gate01Case:
    target_value = 0.25 + 0.02 * FIELD_STRENGTHS_T.index(target)
    identity_value = 2.0 if source == 7.0 else target_value + 0.25
    sb_value = target_value + (0.08 if source < target else 0.12)
    wrong_predictions = {"1.5T": _volume(target_value + 0.5)} if wrong else {}
    if target == 1.5:
        wrong_predictions = {"3T": _volume(target_value + 0.5)} if wrong else {}
    return Gate01Case(
        case_id=f"synthetic-{contrast.value}-{source:g}-{target:g}",
        source_domain=Domain(source, contrast),
        target_domain=Domain(target, contrast),
        target=_volume(target_value),
        raw_identity=_volume(identity_value),
        raw_sb_v2=_volume(sb_value),
        stage1_reconstruction_ceiling=_volume(target_value + 0.01),
        wrong_target_sb_v2=wrong_predictions,
    )


def _all_cases() -> list[Gate01Case]:
    cases = []
    for contrast in CONTRASTS:
        for source in FIELD_STRENGTHS_T:
            for target in FIELD_STRENGTHS_T:
                if source == target:
                    continue
                cases.append(
                    _case(
                        contrast,
                        source,
                        target,
                        wrong=(contrast == CONTRASTS[0] and source == 0.1 and target == 7.0),
                    )
                )
    return cases


def _evaluate(*, include_robust_affine: bool = True):
    return evaluate_gate01(
        _all_cases(),
        calibrator=_calibrator(),
        artifact_provenance=frozen_artifact_provenance(),
        code_commit="evaluation-commit",
        evidence_scope={
            "role": "synthetic development evidence",
            "traveller": "synthetic-traveller",
            "private_data_run": False,
        },
        input_manifest_sha256="a" * 64,
        metrics=("nrmse", "ssim", "lpips"),
        device="cpu",
        include_robust_affine=include_robust_affine,
        metric_fn=_metric_fn,
    )


def test_gate01_reduces_aggregate_contrast_and_all_directed_pairs() -> None:
    result = _evaluate()
    assert result["num_pairs"] == 60
    assert result["overall"]["num_pairs"] == 60
    assert set(result["by_contrast"]) == {contrast.value for contrast in CONTRASTS}
    assert all(block["num_pairs"] == 20 for block in result["by_contrast"].values())

    raw_pair_values = [
        pair["methods"]["raw_identity"]["nrmse"] for pair in result["pairs"]
    ]
    assert result["overall"]["methods"]["raw_identity"]["nrmse"] == pytest.approx(
        sum(raw_pair_values) / len(raw_pair_values)
    )

    for contrast in CONTRASTS:
        directed = result["directed_pair_results"][contrast.value]
        assert len(directed) == 20
        assert all(row["num_pairs"] == 1 for row in directed)
        matrix = result["directed_pair_matrices"][contrast.value]
        assert matrix["axes"]["rows"] == "source_field_t"
        assert matrix["raw_identity"]["nrmse"]["0.1T"]["0.1T"] is None
        assert matrix["raw_identity"]["nrmse"]["0.1T"]["7T"] is not None


def test_catastrophic_stratum_is_frozen_from_raw_identity_only() -> None:
    result = _evaluate(include_robust_affine=False)
    expected = sum(
        pair["methods"]["raw_identity"]["nrmse"] > 1.0
        for pair in result["pairs"]
    )
    assert result["strata"]["assignment_source"] == "raw_identity only"
    assert result["strata"]["catastrophic_identity"]["num_pairs"] == expected
    assert all(
        (pair["stratum"] == "catastrophic_identity")
        == (pair["methods"]["raw_identity"]["nrmse"] > 1.0)
        for pair in result["pairs"]
    )


def test_per_pair_deltas_wins_and_wrong_target_diagnostics_are_labeled() -> None:
    result = _evaluate()
    for pair in result["pairs"]:
        for metric in ("nrmse", "ssim", "lpips"):
            comparison = pair["central_comparison"][metric]
            expected_delta = (
                pair["methods"]["calibrated_sb_v2"][metric]
                - pair["methods"]["calibrated_identity"][metric]
            )
            assert comparison[
                "calibrated_sb_minus_calibrated_identity"
            ] == pytest.approx(expected_delta)
            assert comparison["winner"] in {"sb_v2", "identity", "tie"}

    for metric, block in result["central_paired_deltas_and_wins"]["overall"][
        "metrics"
    ].items():
        del metric
        assert sum(block["wins"].values()) == 60

    diagnostic = result["requested_vs_wrong_target_diagnostic"]
    assert diagnostic["role"].startswith("diagnostic-only")
    assert diagnostic["available_pairs"] == 1
    assert diagnostic["not_rerun"] is True


def test_official_and_diagnostic_roles_are_unambiguous() -> None:
    result = _evaluate()
    official = result["metric_roles"]["official"]
    assert official["names"] == ["nrmse", "ssim", "lpips"]
    assert official["formulas_modified"] is False
    assert "mrixfields2026-task3" in official["contract"]
    assert (
        result["method_roles"]["diagnostic_robust_affine_sb_v2"]
        == "diagnostic-only robust-affine calibration"
    )
    assert (
        result["scientific_status"]["promotion_decision"]
        == "unset_pending_private_data_run"
    )
    assert result["scientific_status"]["population_or_challenge_claim"] is False


def test_montage_spec_is_fixed_and_does_not_claim_anatomical_plane() -> None:
    spec = fixed_montage_specifications()
    assert spec["selection_frozen_before_private_run"] is True
    assert len(spec["directed_pairs_per_contrast"]) == 4
    assert spec["relative_slice_positions"] == [0.35, 0.5, 0.65]
    assert "no anatomical plane name" in spec["tensor_axis_convention"]


def test_case_shape_and_nonfinite_values_fail_closed() -> None:
    kwargs = {
        "case_id": "bad",
        "source_domain": Domain(0.1, "T1w"),
        "target_domain": Domain(1.5, "T1w"),
        "target": torch.ones(4, 4, 4),
        "raw_identity": torch.ones(4, 4, 4),
        "raw_sb_v2": torch.ones(4, 4, 4),
        "stage1_reconstruction_ceiling": torch.ones(4, 4, 4),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        Gate01Case(**{**kwargs, "raw_sb_v2": torch.ones(3, 4, 4)})

    nonfinite = torch.ones(4, 4, 4)
    nonfinite[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        Gate01Case(**{**kwargs, "raw_identity": nonfinite})


def test_markdown_and_atomic_outputs_distinguish_development_status(tmp_path) -> None:
    result = _evaluate(include_robust_affine=False)
    markdown = render_gate01_markdown(result)
    assert "development evidence only" in markdown
    assert "unset pending a private-data run" in markdown
    assert "paired evaluation target" in markdown

    written = write_gate01_outputs(
        result,
        json_path=tmp_path / "result.json",
        markdown_path=tmp_path / "report.md",
        contract_path=tmp_path / "contract.json",
    )
    assert set(written) == {"json", "markdown", "contract"}
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    contract = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    assert payload["contract"] == contract
    assert contract["target_independence_guarantee"][
        "paired_target_for_calibration"
    ] is False
