from __future__ import annotations

import json

import pytest

import fieldbridge.evaluation.mrixfields2026_official as official
from fieldbridge.cli import build_parser, main
from fieldbridge.evaluation import board_score
from fieldbridge.evaluation.board_score import (
    BoardUnit,
    aggregate_task3_board,
    board_units_from_payload,
    evaluate_task3_board_directory,
    format_board_table,
    rank_submissions,
    submissions_from_payload,
)


def _units() -> list[BoardUnit]:
    return [
        BoardUnit.from_cases(
            "T1W",
            "0.1T_to_7T",
            [
                {"nrmse": 0.10, "ssim": 0.90, "lpips": 0.05},
                {"nrmse": 0.20, "ssim": 0.80, "lpips": 0.07},
            ],
        ),
        BoardUnit.from_cases(
            "T1W",
            "1.5T_to_3T",
            [{"nrmse": 0.30, "ssim": 0.70, "lpips": 0.09}],
        ),
        BoardUnit.from_cases(
            "T2W",
            "0.1T_to_7T",
            [{"nrmse": 0.40, "ssim": 0.60, "lpips": 0.11}],
        ),
    ]


def test_aggregate_pools_cases_within_contrast_and_balances_contrasts() -> None:
    aggregate = aggregate_task3_board(_units())

    assert list(aggregate.per_contrast) == ["T1W", "T2W"]
    t1w = aggregate.per_contrast["T1W"]
    assert t1w.case_count == 3
    assert t1w.subtask_count == 2
    assert t1w.means == pytest.approx({"nrmse": 0.20, "ssim": 0.80, "lpips": 0.07})
    t2w = aggregate.per_contrast["T2W"]
    assert t2w.means == pytest.approx({"nrmse": 0.40, "ssim": 0.60, "lpips": 0.11})

    # balanced overall = mean of the two per-contrast means.
    assert aggregate.overall.means == pytest.approx(
        {"nrmse": 0.30, "ssim": 0.70, "lpips": 0.09}
    )
    assert aggregate.overall.case_count == 4
    assert aggregate.overall.subtask_count == 3


def test_pooled_weighting_differs_from_balanced_under_unequal_case_counts() -> None:
    pooled = aggregate_task3_board(_units(), contrast_weighting="pooled")
    # pooled overall = mean over all four cases equally.
    assert pooled.overall.means == pytest.approx(
        {"nrmse": 0.25, "ssim": 0.75, "lpips": 0.08}
    )


def test_per_subtask_breakdown_reports_unit_level_means() -> None:
    aggregate = aggregate_task3_board(_units())
    first = aggregate.per_subtask[0]
    assert first["target_contrast"] == "T1W"
    assert first["subtask"] == "0.1T_to_7T"
    assert first["case_count"] == 2
    assert first["means"] == pytest.approx(
        {"nrmse": 0.15, "ssim": 0.85, "lpips": 0.06}
    )


def test_empty_units_rejected() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        aggregate_task3_board([])


def test_from_cases_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="no cases"):
        BoardUnit.from_cases("T1W", "x", [])


def test_rank_sum_orders_by_official_metric_directions() -> None:
    ranking = rank_submissions(
        {
            "fm": {"nrmse": 0.12, "ssim": 0.93, "lpips": 0.07},
            "sb": {"nrmse": 0.10, "ssim": 0.92, "lpips": 0.06},
            "baseline": {"nrmse": 0.30, "ssim": 0.88, "lpips": 0.09},
        }
    )

    assert ranking.per_metric_rank["nrmse"] == {"sb": 1, "fm": 2, "baseline": 3}
    assert ranking.per_metric_rank["ssim"] == {"fm": 1, "sb": 2, "baseline": 3}
    assert ranking.per_metric_rank["lpips"] == {"sb": 1, "fm": 2, "baseline": 3}
    assert ranking.rank_sum == {"sb": 4, "fm": 5, "baseline": 9}
    assert ranking.order == ("sb", "fm", "baseline")


def test_rank_sum_uses_average_ranks_for_ties() -> None:
    ranking = rank_submissions(
        {
            "a": {"nrmse": 0.10, "ssim": 0.90, "lpips": 0.05},
            "b": {"nrmse": 0.10, "ssim": 0.80, "lpips": 0.05},
        }
    )
    # nrmse and lpips tie → average rank 1.5 each; ssim splits 1/2.
    assert ranking.per_metric_rank["nrmse"] == {"a": 1.5, "b": 1.5}
    assert ranking.per_metric_rank["lpips"] == {"a": 1.5, "b": 1.5}
    assert ranking.per_metric_rank["ssim"] == {"a": 1, "b": 2}
    assert ranking.rank_sum == {"a": 4.0, "b": 5.0}
    assert ranking.order == ("a", "b")


def test_rank_sum_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="missing metric 'lpips'"):
        rank_submissions({"a": {"nrmse": 0.1, "ssim": 0.9}})


def test_payload_parsers_round_trip() -> None:
    units = board_units_from_payload(
        {
            "units": [
                {
                    "target_contrast": "T1W",
                    "subtask": "0.1T_to_7T",
                    "cases": [{"nrmse": 0.1, "ssim": 0.9, "lpips": 0.05}],
                }
            ]
        }
    )
    assert units[0].target_contrast == "T1W"
    subs = submissions_from_payload(
        {"submissions": {"ours": {"nrmse": 0.1, "ssim": 0.9, "lpips": 0.05}}}
    )
    assert subs["ours"]["ssim"] == pytest.approx(0.9)
    # flat mapping without the wrapper key is also accepted.
    flat = submissions_from_payload({"ours": {"nrmse": 0.1, "ssim": 0.9, "lpips": 0.05}})
    assert flat["ours"]["nrmse"] == pytest.approx(0.1)


def test_format_board_table_lists_contrasts_and_mean_row() -> None:
    table = format_board_table(aggregate_task3_board(_units()))
    assert "T1W" in table
    assert "T2W" in table
    assert "MEAN (balanced)" in table
    assert "NRMSE" in table


def test_evaluate_board_directory_walks_tree_and_aggregates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pred = tmp_path / "pred"
    target = tmp_path / "target"
    layout = {
        ("T1W", "0.1T_to_7T"): [("0001", {"nrmse": 0.10, "ssim": 0.90, "lpips": 0.05})],
        ("T1W", "1.5T_to_3T"): [("0002", {"nrmse": 0.30, "ssim": 0.70, "lpips": 0.09})],
        ("T2W", "0.1T_to_7T"): [("0003", {"nrmse": 0.40, "ssim": 0.60, "lpips": 0.11})],
    }
    for (contrast, subtask), cases in layout.items():
        for subject, _metrics in cases:
            for root in (pred, target):
                leaf = root / contrast / subtask
                leaf.mkdir(parents=True, exist_ok=True)
                (leaf / f"P_{contrast}_7T_{subject}.nii.gz").touch()

    def fake_directory(prediction_dir, target_dir, *, metrics, device):
        del target_dir, metrics, device
        contrast = prediction_dir.parent.name
        subtask = prediction_dir.name
        cases = layout[(contrast, subtask)]
        return {
            "OFFICIAL_TASK3_METRIC_CONTRACT": official.OFFICIAL_TASK3_METRIC_CONTRACT,
            "cases": [
                {"subject": subject, "metrics": dict(m)} for subject, m in cases
            ],
        }

    monkeypatch.setattr(
        official, "evaluate_official_task3_directory", fake_directory
    )

    aggregate, provenance = evaluate_task3_board_directory(
        pred, target, device="cpu"
    )
    assert list(aggregate.per_contrast) == ["T1W", "T2W"]
    assert aggregate.overall.means == pytest.approx(
        {"nrmse": 0.30, "ssim": 0.70, "lpips": 0.09}
    )
    assert (
        provenance["OFFICIAL_TASK3_METRIC_CONTRACT"]
        == official.OFFICIAL_TASK3_METRIC_CONTRACT
    )
    assert len(provenance["units"]) == 3


def test_board_command_is_exposed_in_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["mrixfields2026-task3-board", "--help"])
    assert exc_info.value.code == 0


def test_board_cli_ranks_summaries_end_to_end(tmp_path, capsys) -> None:
    summaries = tmp_path / "summaries.json"
    summaries.write_text(
        json.dumps(
            {
                "submissions": {
                    "fm": {"nrmse": 0.12, "ssim": 0.93, "lpips": 0.07},
                    "sb": {"nrmse": 0.10, "ssim": 0.92, "lpips": 0.06},
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "ranking.json"
    code = main(
        [
            "mrixfields2026-task3-board",
            "--summaries-json",
            str(summaries),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ranking"]["order"] == ["sb", "fm"]
    printed = capsys.readouterr().out
    assert "rank_sum" in printed


def test_board_cli_aggregates_board_json(tmp_path, capsys) -> None:
    board_json = tmp_path / "board.json"
    board_json.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "target_contrast": "T1W",
                        "subtask": "0.1T_to_7T",
                        "cases": [{"nrmse": 0.1, "ssim": 0.9, "lpips": 0.05}],
                    },
                    {
                        "target_contrast": "T2W",
                        "subtask": "0.1T_to_7T",
                        "cases": [{"nrmse": 0.3, "ssim": 0.7, "lpips": 0.09}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    code = main(
        ["mrixfields2026-task3-board", "--board-json", str(board_json), "--rank-as", "ours"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "MEAN (balanced)" in printed
    assert '"ours"' in printed


def test_board_cli_requires_target_dir_with_pred_dir(tmp_path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    with pytest.raises(ValueError, match="requires --target-dir"):
        main(["mrixfields2026-task3-board", "--pred-dir", str(pred)])
