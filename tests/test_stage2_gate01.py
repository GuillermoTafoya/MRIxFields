from __future__ import annotations

import json
import hashlib
import gc
import weakref
from dataclasses import replace

import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.evaluation.stage2_gate01 import (
    Gate01Case,
    evaluate_gate01,
    fixed_montage_specifications,
    frozen_artifact_provenance,
    gate01_selection_fingerprint,
    render_gate01_markdown,
    write_gate01_outputs,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    RESPLIT_FINGERPRINT,
    TrainingTemplateVolume,
    fit_posthoc_target_calibrator,
)
from fieldbridge.evaluation.stage2_gate01_montage import (
    Gate01MontageCollector,
    render_gate01_montages,
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
        support_mask=_volume(1.0).bool(),
        traveller_identity_sha256=hashlib.sha256(
            b"synthetic-traveller"
        ).hexdigest(),
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


def _evaluate(
    *,
    include_robust_affine: bool = True,
    execution_mode: str = "scientific",
    cases=None,
    metrics=("nrmse", "ssim", "lpips"),
    checkout_clean: bool = True,
):
    cases = list(cases if cases is not None else _all_cases())
    traveller_hash = cases[0].traveller_identity_sha256
    selection_fingerprint = gate01_selection_fingerprint(
        {
            "case_identity_sha256": case.case_identity_sha256,
            "traveller_identity_sha256": case.traveller_identity_sha256,
            "contrast": case.target_domain.contrast,
            "source_field_t": case.source_domain.field_strength_t,
            "target_field_t": case.target_domain.field_strength_t,
        }
        for case in cases
    )
    return evaluate_gate01(
        iter(cases),
        calibrator=_calibrator(),
        artifact_provenance=frozen_artifact_provenance(),
        code_commit="evaluation-commit",
        evidence_scope={
            "role": "synthetic development evidence",
            "traveller_identity_sha256": traveller_hash,
            "private_data_run": False,
        },
        input_manifest_sha256="a" * 64,
        execution_mode=execution_mode,
        selection_fingerprint_sha256=selection_fingerprint,
        code_provenance={
            "git_head": "evaluation-commit",
            "checkout_clean": checkout_clean,
            "module_sha256": {
                "src/fieldbridge/evaluation/stage2_gate01.py": "1" * 64,
                "src/fieldbridge/evaluation/stage2_gate01_calibration.py": "2" * 64,
                "src/fieldbridge/evaluation/mrixfields2026_official.py": "3" * 64,
            },
        },
        metrics=metrics,
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
    assert result["scientific_status"]["eligible_for_scientific_conclusions"] is True
    runtime = result["contract"]["official_runtime_provenance"]
    assert runtime["python"]
    assert runtime["numpy"]
    assert runtime["torch"]
    assert runtime["lpips_device"] == "cpu"


def test_scientific_mode_rejects_incomplete_duplicate_mixed_and_dirty_inputs() -> None:
    cases = _all_cases()
    with pytest.raises(ValueError, match="exactly 60"):
        _evaluate(cases=cases[:-1])

    duplicate_id = cases + [cases[0]]
    with pytest.raises(ValueError, match="duplicate case IDs"):
        _evaluate(cases=duplicate_id)

    duplicate_direction = cases + [
        replace(cases[0], case_id="synthetic-duplicate-direction")
    ]
    with pytest.raises(ValueError, match="duplicate contrast/directed-field pair"):
        _evaluate(cases=duplicate_direction)

    mixed = list(cases)
    mixed[-1] = replace(mixed[-1], traveller_identity_sha256="f" * 64)
    with pytest.raises(ValueError, match="mixes multiple travellers"):
        _evaluate(cases=mixed)

    with pytest.raises(ValueError, match="requires nRMSE, SSIM, and LPIPS"):
        _evaluate(metrics=("nrmse",))

    with pytest.raises(ValueError, match="clean checkout"):
        _evaluate(checkout_clean=False)


def test_raw_pre_mask_background_leakage_is_reported_and_calibration_is_zero() -> None:
    cases = _all_cases()
    cases[0].raw_sb_v2[0, 0, 0] = 1e-7
    observed = {}

    def observer(case, predictions):
        if case.case_identity_sha256 == cases[0].case_identity_sha256:
            observed.update(predictions)

    # Reproduce the helper call with an observer to inspect the calibrated tensor.
    traveller_hash = cases[0].traveller_identity_sha256
    fingerprint = gate01_selection_fingerprint(
        {
            "case_identity_sha256": case.case_identity_sha256,
            "traveller_identity_sha256": case.traveller_identity_sha256,
            "contrast": case.target_domain.contrast,
            "source_field_t": case.source_domain.field_strength_t,
            "target_field_t": case.target_domain.field_strength_t,
        }
        for case in cases
    )
    result = evaluate_gate01(
        iter(cases),
        calibrator=_calibrator(),
        artifact_provenance=frozen_artifact_provenance(),
        code_commit="evaluation-commit",
        evidence_scope={
            "role": "synthetic",
            "traveller_identity_sha256": traveller_hash,
            "private_data_run": False,
        },
        input_manifest_sha256="a" * 64,
        execution_mode="scientific",
        selection_fingerprint_sha256=fingerprint,
        code_provenance={
            "git_head": "evaluation-commit",
            "checkout_clean": True,
            "module_sha256": {
                "src/fieldbridge/evaluation/stage2_gate01.py": "1" * 64,
                "src/fieldbridge/evaluation/stage2_gate01_calibration.py": "2" * 64,
                "src/fieldbridge/evaluation/mrixfields2026_official.py": "3" * 64,
            },
        },
        metrics=("nrmse", "ssim", "lpips"),
        device="cpu",
        metric_fn=_metric_fn,
        case_observer=observer,
    )
    row = next(
        row
        for row in result["pairs"]
        if row["case_identity_sha256"] == cases[0].case_identity_sha256
    )
    assert row["raw_pre_mask_background_leakage"]["raw_sb_v2"][
        "nonzero_voxel_count"
    ] == 1
    assert result["raw_pre_mask_background_leakage"]["overall"]["methods"][
        "raw_sb_v2"
    ]["nonzero_voxel_count"] == 1
    assert observed["calibrated_sb_v2"][0, 0, 0].item() == 0.0


def test_evaluator_streams_cases_with_bounded_full_volume_liveness() -> None:
    traveller_hash = hashlib.sha256(b"synthetic-traveller").hexdigest()
    descriptors = []
    for contrast in CONTRASTS:
        for source in FIELD_STRENGTHS_T:
            for target in FIELD_STRENGTHS_T:
                if source == target:
                    continue
                case_id = f"synthetic-{contrast.value}-{source:g}-{target:g}"
                descriptors.append(
                    {
                        "case_identity_sha256": hashlib.sha256(
                            case_id.encode("utf-8")
                        ).hexdigest(),
                        "traveller_identity_sha256": traveller_hash,
                        "contrast": contrast.value,
                        "source_field_t": source,
                        "target_field_t": target,
                    }
                )
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    maximum_alive = 0

    def stream():
        nonlocal maximum_alive
        for contrast in CONTRASTS:
            for source in FIELD_STRENGTHS_T:
                for target in FIELD_STRENGTHS_T:
                    if source == target:
                        continue
                    case = _case(contrast, source, target)
                    references.extend(
                        weakref.ref(value)
                        for value in (
                            case.target,
                            case.raw_identity,
                            case.raw_sb_v2,
                            case.stage1_reconstruction_ceiling,
                            case.support_mask,
                        )
                    )
                    gc.collect()
                    maximum_alive = max(
                        maximum_alive,
                        sum(reference() is not None for reference in references),
                    )
                    yield case

    result = evaluate_gate01(
        stream(),
        calibrator=_calibrator(),
        artifact_provenance=frozen_artifact_provenance(),
        code_commit="evaluation-commit",
        evidence_scope={
            "role": "synthetic",
            "traveller_identity_sha256": traveller_hash,
            "private_data_run": False,
        },
        input_manifest_sha256="a" * 64,
        execution_mode="scientific",
        selection_fingerprint_sha256=gate01_selection_fingerprint(descriptors),
        code_provenance={
            "git_head": "evaluation-commit",
            "checkout_clean": True,
            "module_sha256": {
                "src/fieldbridge/evaluation/stage2_gate01.py": "1" * 64,
                "src/fieldbridge/evaluation/stage2_gate01_calibration.py": "2" * 64,
                "src/fieldbridge/evaluation/mrixfields2026_official.py": "3" * 64,
            },
        },
        metrics=("nrmse", "ssim", "lpips"),
        device="cpu",
        metric_fn=_metric_fn,
    )
    gc.collect()
    assert result["num_pairs"] == 60
    assert maximum_alive <= 10
    assert sum(reference() is not None for reference in references) == 0


def test_montage_spec_is_fixed_and_does_not_claim_anatomical_plane() -> None:
    spec = fixed_montage_specifications()
    assert spec["selection_frozen_before_private_run"] is True
    assert len(spec["directed_pairs_per_contrast"]) == 4
    assert spec["relative_slice_positions"] == [0.35, 0.5, 0.65]
    assert "no anatomical plane name" in spec["tensor_axis_convention"]


def test_frozen_montage_renderer_is_deterministic_and_hash_linked(tmp_path) -> None:
    case = _case(CONTRASTS[0], 0.1, 7.0)
    predictions = {
        "raw_identity": case.raw_identity,
        "calibrated_identity": case.raw_identity * 0.9,
        "raw_sb_v2": case.raw_sb_v2,
        "calibrated_sb_v2": case.raw_sb_v2 * 0.9,
        "stage1_reconstruction_ceiling": case.stage1_reconstruction_ceiling,
    }
    first = Gate01MontageCollector(fixed_montage_specifications())
    second = Gate01MontageCollector(fixed_montage_specifications())
    first.observe(case, predictions)
    second.observe(case, predictions)
    first_manifest = render_gate01_montages(
        first, tmp_path / "first", require_complete=False
    )
    second_manifest = render_gate01_montages(
        second, tmp_path / "second", require_complete=False
    )

    assert first_manifest["entries"][0]["png_sha256"] == second_manifest["entries"][0][
        "png_sha256"
    ]
    assert first_manifest["manifest_sha256"] == second_manifest["manifest_sha256"]
    png = (tmp_path / "first" / first_manifest["entries"][0]["png"]).read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "first" / "montage_manifest.json").is_file()


def test_case_shape_and_nonfinite_values_fail_closed() -> None:
    kwargs = {
        "case_id": "bad",
        "source_domain": Domain(0.1, "T1w"),
        "target_domain": Domain(1.5, "T1w"),
        "target": torch.ones(4, 4, 4),
        "raw_identity": torch.ones(4, 4, 4),
        "raw_sb_v2": torch.ones(4, 4, 4),
        "stage1_reconstruction_ceiling": torch.ones(4, 4, 4),
        "support_mask": torch.ones(4, 4, 4, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        Gate01Case(**{**kwargs, "raw_sb_v2": torch.ones(3, 4, 4)})

    nonfinite = torch.ones(4, 4, 4)
    nonfinite[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        Gate01Case(**{**kwargs, "raw_identity": nonfinite})


def test_markdown_and_atomic_outputs_distinguish_development_status(tmp_path) -> None:
    result = _evaluate(
        include_robust_affine=False, execution_mode="development-incomplete"
    )
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
