"""Board-shaped aggregation for the official MRIxFields2026 Task-3 metrics.

The per-file metric values come from ``mrixfields2026_official`` (source-pinned to the
published ``Evaluation/evaluate.py``). This module only combines those values the way
the Synapse leaderboard presents them and never redefines a metric:

- group evaluated units by *target contrast* (T1W / T2W / T2FLAIR);
- report a per-contrast mean of each metric plus a contrast-balanced overall mean;
- rank a set of submissions by the official rule — rank-sum over (nRMSE, SSIM, LPIPS),
  lowest total wins.

The leaderboard's exact subtask-to-row combination is not published (see
``docs/MRIXFIELDS2026_TASK3_METRICS.md``). The two aggregation choices made here are
explicit and configurable rather than guessed silently:

- within a target contrast, per-case values are pooled (each case weighs equally),
  matching a flat "60 pairs/contrast" average;
- the overall ``mean`` weighs each *contrast* equally ("balanced"), because the board
  reports three symmetric per-contrast sub-rankings and the project bar is stated as a
  balanced-across-contrasts target. ``pooled`` (each case equal, contrasts unweighted)
  is available for callers that want it. When every contrast carries the same case
  count the two coincide.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fieldbridge.official.mrixfields2026 import OFFICIAL_MODALITIES, normalize_modality

TASK3_METRICS: tuple[str, ...] = ("nrmse", "ssim", "lpips")

# Optimization direction per metric: lower nRMSE/LPIPS is better, higher SSIM is better.
# Used both for ranking and for orienting the rank-sum. These are metric definitions,
# not tunable knobs.
METRIC_LOWER_IS_BETTER: dict[str, bool] = {
    "nrmse": True,
    "ssim": False,
    "lpips": True,
}

ContrastWeighting = Literal["balanced", "pooled"]


@dataclass(frozen=True, slots=True)
class BoardUnit:
    """One evaluated Task-3 unit: cases sharing a target contrast and subtask label.

    ``cases`` holds one dict per matched subject with the raw official metric values
    (``nrmse``/``ssim``/``lpips``, any subset). ``target_contrast`` is normalized to an
    official modality label; ``subtask`` is a free label (e.g. a source→target field
    pair) used only for the per-subtask breakdown and traceability.
    """

    target_contrast: str
    subtask: str
    cases: tuple[Mapping[str, float], ...]

    @classmethod
    def from_cases(
        cls, target_contrast: str, subtask: str, cases: Iterable[Mapping[str, float]]
    ) -> "BoardUnit":
        contrast = normalize_modality(target_contrast)
        materialized = tuple(dict(case) for case in cases)
        if not materialized:
            raise ValueError(
                f"Board unit {contrast!r}/{subtask!r} has no cases to aggregate."
            )
        return cls(target_contrast=contrast, subtask=str(subtask), cases=materialized)


@dataclass(frozen=True, slots=True)
class MetricGroup:
    """Per-metric mean over a set of pooled cases, with the contributing case count."""

    means: dict[str, float]
    case_count: int
    subtask_count: int = 1


@dataclass(frozen=True, slots=True)
class BoardAggregate:
    """Board-shaped result: per-contrast means, per-subtask means, and the overall mean."""

    metrics: tuple[str, ...]
    per_subtask: tuple[dict[str, Any], ...]
    per_contrast: dict[str, MetricGroup]
    overall: MetricGroup
    contrast_weighting: ContrastWeighting

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": list(self.metrics),
            "contrast_weighting": self.contrast_weighting,
            "per_subtask": [dict(entry) for entry in self.per_subtask],
            "per_contrast": {
                contrast: {
                    "means": dict(group.means),
                    "case_count": group.case_count,
                    "subtask_count": group.subtask_count,
                }
                for contrast, group in self.per_contrast.items()
            },
            "overall": {
                "means": dict(self.overall.means),
                "case_count": self.overall.case_count,
                "subtask_count": self.overall.subtask_count,
            },
        }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _pool_case_means(
    cases: Sequence[Mapping[str, float]], metrics: Sequence[str]
) -> dict[str, float]:
    means: dict[str, float] = {}
    for metric in metrics:
        present = [float(case[metric]) for case in cases if metric in case]
        if present:
            means[metric] = _mean(present)
    return means


def aggregate_task3_board(
    units: Iterable[BoardUnit],
    *,
    metrics: Sequence[str] = TASK3_METRICS,
    contrast_weighting: ContrastWeighting = "balanced",
) -> BoardAggregate:
    """Combine evaluated units into the leaderboard-shaped per-contrast/overall means.

    - per subtask: mean over that unit's cases;
    - per contrast: pooled mean over every case of that contrast (each case equal);
    - overall: ``balanced`` weighs each contrast equally, ``pooled`` weighs each case
      equally across all contrasts.
    """

    metric_names = tuple(metrics)
    unsupported = sorted(set(metric_names) - set(TASK3_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported Task-3 board metrics: {unsupported}.")

    unit_list = list(units)
    if not unit_list:
        raise ValueError("aggregate_task3_board requires at least one unit.")

    per_subtask: list[dict[str, Any]] = []
    by_contrast: dict[str, list[BoardUnit]] = {}
    for unit in unit_list:
        per_subtask.append(
            {
                "target_contrast": unit.target_contrast,
                "subtask": unit.subtask,
                "case_count": len(unit.cases),
                "means": _pool_case_means(unit.cases, metric_names),
            }
        )
        by_contrast.setdefault(unit.target_contrast, []).append(unit)

    per_contrast: dict[str, MetricGroup] = {}
    for contrast in _ordered_contrasts(by_contrast):
        contrast_units = by_contrast[contrast]
        pooled_cases = [case for unit in contrast_units for case in unit.cases]
        per_contrast[contrast] = MetricGroup(
            means=_pool_case_means(pooled_cases, metric_names),
            case_count=len(pooled_cases),
            subtask_count=len(contrast_units),
        )

    overall = _overall_group(
        unit_list, per_contrast, metric_names, contrast_weighting
    )
    return BoardAggregate(
        metrics=metric_names,
        per_subtask=tuple(per_subtask),
        per_contrast=per_contrast,
        overall=overall,
        contrast_weighting=contrast_weighting,
    )


def _ordered_contrasts(by_contrast: Mapping[str, Any]) -> list[str]:
    known = [c for c in OFFICIAL_MODALITIES if c in by_contrast]
    extra = sorted(c for c in by_contrast if c not in OFFICIAL_MODALITIES)
    return known + extra


def _overall_group(
    units: Sequence[BoardUnit],
    per_contrast: Mapping[str, MetricGroup],
    metrics: Sequence[str],
    contrast_weighting: ContrastWeighting,
) -> MetricGroup:
    total_cases = sum(len(unit.cases) for unit in units)
    total_subtasks = len(units)
    if contrast_weighting == "pooled":
        pooled_cases = [case for unit in units for case in unit.cases]
        return MetricGroup(
            means=_pool_case_means(pooled_cases, metrics),
            case_count=total_cases,
            subtask_count=total_subtasks,
        )
    if contrast_weighting != "balanced":
        raise ValueError(
            f"Unknown contrast_weighting {contrast_weighting!r}; "
            "expected 'balanced' or 'pooled'."
        )
    means: dict[str, float] = {}
    for metric in metrics:
        contrast_values = [
            group.means[metric]
            for group in per_contrast.values()
            if metric in group.means
        ]
        if contrast_values:
            means[metric] = _mean(contrast_values)
    return MetricGroup(
        means=means, case_count=total_cases, subtask_count=total_subtasks
    )


@dataclass(frozen=True, slots=True)
class SubmissionRanking:
    """Rank-sum ordering of submissions under the official Task-3 rule."""

    metrics: tuple[str, ...]
    per_metric_rank: dict[str, dict[str, float]]
    rank_sum: dict[str, float]
    order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": list(self.metrics),
            "per_metric_rank": {
                metric: dict(ranks) for metric, ranks in self.per_metric_rank.items()
            },
            "rank_sum": dict(self.rank_sum),
            "order": list(self.order),
        }


def _average_ranks(
    named_values: Sequence[tuple[str, float]], *, lower_is_better: bool
) -> dict[str, float]:
    """Rank names by value with average (fractional) ranks for ties; best rank is 1."""

    ordered = sorted(
        named_values, key=lambda item: item[1], reverse=not lower_is_better
    )
    ranks: dict[str, float] = {}
    index = 0
    position = 1
    while index < len(ordered):
        tie_end = index
        while (
            tie_end + 1 < len(ordered)
            and ordered[tie_end + 1][1] == ordered[index][1]
        ):
            tie_end += 1
        span = tie_end - index + 1
        average_rank = position + (span - 1) / 2.0
        for offset in range(span):
            ranks[ordered[index + offset][0]] = average_rank
        position += span
        index = tie_end + 1
    return ranks


def rank_submissions(
    submissions: Mapping[str, Mapping[str, float]],
    *,
    metrics: Sequence[str] = TASK3_METRICS,
) -> SubmissionRanking:
    """Rank submissions by rank-sum over the metrics; lowest total wins (ties broken by name).

    ``submissions`` maps a submission name to its aggregate metric means (e.g. the
    ``overall`` means of a ``BoardAggregate``). Each metric must be present for every
    submission so the per-metric ranking is well defined.
    """

    metric_names = tuple(metrics)
    unsupported = sorted(set(metric_names) - set(TASK3_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported Task-3 board metrics: {unsupported}.")
    if not submissions:
        raise ValueError("rank_submissions requires at least one submission.")

    names = list(submissions)
    per_metric_rank: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        named_values: list[tuple[str, float]] = []
        for name in names:
            if metric not in submissions[name]:
                raise ValueError(
                    f"Submission {name!r} is missing metric {metric!r}."
                )
            named_values.append((name, float(submissions[name][metric])))
        per_metric_rank[metric] = _average_ranks(
            named_values, lower_is_better=METRIC_LOWER_IS_BETTER[metric]
        )

    rank_sum = {
        name: float(sum(per_metric_rank[metric][name] for metric in metric_names))
        for name in names
    }
    order = tuple(sorted(names, key=lambda name: (rank_sum[name], name)))
    return SubmissionRanking(
        metrics=metric_names,
        per_metric_rank=per_metric_rank,
        rank_sum=rank_sum,
        order=order,
    )


def format_board_table(aggregate: BoardAggregate) -> str:
    """Render the per-contrast + overall means as an aligned text table."""

    metrics = aggregate.metrics
    header = ["target", *[m.upper() for m in metrics], "n_cases", "n_subtasks"]
    rows: list[list[str]] = []
    for contrast, group in aggregate.per_contrast.items():
        rows.append(_board_row(contrast, group, metrics))
    rows.append(
        _board_row(
            f"MEAN ({aggregate.contrast_weighting})", aggregate.overall, metrics
        )
    )
    return _render_table(header, rows)


def _board_row(label: str, group: MetricGroup, metrics: Sequence[str]) -> list[str]:
    cells = [label]
    for metric in metrics:
        value = group.means.get(metric)
        cells.append("—" if value is None else f"{value:.4f}")
    cells.append(str(group.case_count))
    cells.append(str(group.subtask_count))
    return cells


def format_rank_sum_table(ranking: SubmissionRanking) -> str:
    """Render the rank-sum standings (winner first) as an aligned text table."""

    metrics = ranking.metrics
    header = [
        "rank",
        "submission",
        *[f"{m.upper()}_rank" for m in metrics],
        "rank_sum",
    ]
    rows: list[list[str]] = []
    for position, name in enumerate(ranking.order, start=1):
        row = [str(position), name]
        for metric in metrics:
            row.append(_format_rank(ranking.per_metric_rank[metric][name]))
        row.append(_format_rank(ranking.rank_sum[name]))
        rows.append(row)
    return _render_table(header, rows)


def _format_rank(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _render_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    columns = len(header)
    widths = [len(str(header[i])) for i in range(columns)]
    for row in rows:
        for i in range(columns):
            widths[i] = max(widths[i], len(str(row[i])))
    lines = [_render_row(header, widths), _render_divider(widths)]
    lines.extend(_render_row(row, widths) for row in rows)
    return "\n".join(lines)


def _render_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))


def _render_divider(widths: Sequence[int]) -> str:
    return "  ".join("-" * width for width in widths)


def board_units_from_payload(payload: Mapping[str, Any]) -> list[BoardUnit]:
    """Parse a ``{"units": [{target_contrast, subtask, cases:[...]}, ...]}`` mapping."""

    raw_units = payload.get("units")
    if not isinstance(raw_units, Sequence) or not raw_units:
        raise ValueError("Board payload must contain a non-empty 'units' list.")
    units: list[BoardUnit] = []
    for index, entry in enumerate(raw_units):
        if not isinstance(entry, Mapping):
            raise ValueError(f"units[{index}] must be a mapping.")
        try:
            contrast = entry["target_contrast"]
            subtask = entry["subtask"]
            cases = entry["cases"]
        except KeyError as exc:
            raise ValueError(
                f"units[{index}] missing required key {exc.args[0]!r}."
            ) from exc
        units.append(BoardUnit.from_cases(contrast, subtask, cases))
    return units


def submissions_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Parse ``{"submissions": {name: {nrmse, ssim, lpips}}}`` or a flat name→metrics map."""

    raw = payload.get("submissions", payload)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "Submissions payload must be a non-empty mapping of name → metrics."
        )
    submissions: dict[str, dict[str, float]] = {}
    for name, metrics in raw.items():
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Submission {name!r} must map metric names to values.")
        submissions[str(name)] = {
            str(metric): float(value) for metric, value in metrics.items()
        }
    return submissions


def evaluate_task3_board_directory(
    prediction_root: str | Path,
    target_root: str | Path,
    *,
    metrics: Sequence[str] = TASK3_METRICS,
    contrast_weighting: ContrastWeighting = "balanced",
    device: str = "cuda",
) -> tuple[BoardAggregate, dict[str, Any]]:
    """Run the official evaluator over a ``<contrast>/<subtask>/`` prediction tree.

    Each leaf directory holds ``*.nii.gz`` predictions (unique subject IDs) matched
    against the same-named leaf under ``target_root`` by the published subject-ID rule.
    A contrast directory that holds NIfTIs directly is treated as a single subtask.
    Returns the board aggregate plus the raw per-unit official payloads (which carry the
    pinned metric contract and runtime provenance).
    """

    from fieldbridge.evaluation.mrixfields2026_official import (
        evaluate_official_task3_directory,
    )

    prediction_base = _existing_directory(prediction_root, "prediction")
    target_base = _existing_directory(target_root, "target")
    leaves = _discover_board_leaves(prediction_base)
    if not leaves:
        raise ValueError(
            f"No <contrast>/<subtask> prediction leaves found under {prediction_base}."
        )

    units: list[BoardUnit] = []
    raw_payloads: list[dict[str, Any]] = []
    for contrast, subtask, pred_leaf in leaves:
        target_leaf = target_base / pred_leaf.relative_to(prediction_base)
        if not target_leaf.is_dir():
            raise FileNotFoundError(
                f"Target leaf missing for {contrast}/{subtask}: {target_leaf}"
            )
        payload = evaluate_official_task3_directory(
            pred_leaf, target_leaf, metrics=metrics, device=device
        )
        cases = [case["metrics"] for case in payload["cases"]]
        units.append(BoardUnit.from_cases(contrast, subtask, cases))
        raw_payloads.append(
            {"target_contrast": contrast, "subtask": subtask, "payload": payload}
        )

    aggregate = aggregate_task3_board(
        units, metrics=metrics, contrast_weighting=contrast_weighting
    )
    provenance = {
        "OFFICIAL_TASK3_METRIC_CONTRACT": raw_payloads[0]["payload"][
            "OFFICIAL_TASK3_METRIC_CONTRACT"
        ],
        "units": raw_payloads,
    }
    return aggregate, provenance


def _existing_directory(path: str | Path, role: str) -> Path:
    resolved = Path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Board {role} directory not found: {resolved}")
    return resolved


def _discover_board_leaves(prediction_base: Path) -> list[tuple[str, str, Path]]:
    leaves: list[tuple[str, str, Path]] = []
    for contrast_dir in sorted(p for p in prediction_base.iterdir() if p.is_dir()):
        contrast = normalize_modality(contrast_dir.name)
        subtask_dirs = sorted(p for p in contrast_dir.iterdir() if p.is_dir())
        if subtask_dirs:
            for subtask_dir in subtask_dirs:
                leaves.append((contrast, subtask_dir.name, subtask_dir))
        elif any(contrast_dir.glob("*.nii.gz")):
            leaves.append((contrast, contrast_dir.name, contrast_dir))
    return leaves


__all__ = [
    "TASK3_METRICS",
    "METRIC_LOWER_IS_BETTER",
    "BoardUnit",
    "MetricGroup",
    "BoardAggregate",
    "SubmissionRanking",
    "aggregate_task3_board",
    "rank_submissions",
    "format_board_table",
    "format_rank_sum_table",
    "board_units_from_payload",
    "submissions_from_payload",
    "evaluate_task3_board_directory",
]
