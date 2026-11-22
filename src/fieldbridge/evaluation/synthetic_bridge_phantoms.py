"""Synthetic multi-field volumes and held-out subject and population distances."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import torch
from scipy.optimize import linear_sum_assignment

SOURCE_FIELD = 3.0
TARGET_FIELDS = (1.5, 7.0)
TISSUE_INTENSITY = {1.5: 0.38, 3.0: 0.68, 7.0: 0.93}
DEPTH = 4


@dataclass(frozen=True)
class SubjectBatch:
    ids: tuple[str, ...]
    source: torch.Tensor
    targets: dict[float, torch.Tensor]
    tissue: torch.Tensor


def make_cohort(*, n_train: int, n_test: int, side: int, seed: int) -> tuple[SubjectBatch, SubjectBatch]:
    """Draw distinct anatomies once, split whole subjects, and render all fields on the same grid."""
    if n_train < 2 or n_test < 2 or side < 8:
        raise ValueError("Need two or more subjects in each split and at least 8 voxels per side.")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    axis = torch.linspace(-1, 1, side, device="cpu", dtype=torch.float32)
    z, y, x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, DEPTH, device="cpu", dtype=torch.float32), axis, axis, indexing="ij"
    )

    def uniform(lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand((), generator=rng, dtype=torch.float32, device="cpu").item()

    labels = []
    seen = set()
    for _ in range(n_train + n_test):
        # The shell, interior and inclusion move with the individual subject.
        for _attempt in range(100):
            cx, cy, cz = uniform(-0.23, 0.23), uniform(-0.23, 0.23), uniform(-0.10, 0.10)
            rx, ry, rz = uniform(0.68, 0.91), uniform(0.66, 0.87), uniform(1.05, 1.35)
            angle = uniform(-0.6, 0.6)
            u = (x - cx) * cos(angle) + (y - cy) * sin(angle)
            v = (y - cy) * cos(angle) - (x - cx) * sin(angle)
            radius = (u / rx).square() + (v / ry).square() + ((z - cz) / rz).square()
            body = radius <= 1.0
            interior = radius <= uniform(0.46, 0.66)
            inclusion = (
                ((u - uniform(-0.31, 0.31)) / uniform(0.25, 0.39)).square()
                + ((v - uniform(-0.28, 0.28)) / uniform(0.28, 0.43)).square()
                + ((z - cz - uniform(-0.15, 0.15)) / uniform(0.82, 1.08)).square()
            ) <= 1.0
            inclusion &= body
            label = body.to(torch.uint8) + interior.to(torch.uint8)
            label[inclusion] = 3
            key = label.numpy().tobytes()
            if key not in seen and label.unique().numel() == 4:
                seen.add(key)
                labels.append(label)
                break
        else:
            raise ValueError("Could not draw a distinct phantom with all four labels at this resolution.")

    anatomy = torch.stack(labels).unsqueeze(1)
    order = torch.randperm(len(labels), generator=rng, device="cpu")
    levels = torch.tensor([0.0, 0.23, 0.48, TISSUE_INTENSITY[SOURCE_FIELD]], dtype=torch.float32, device="cpu")

    def render(indices: torch.Tensor) -> SubjectBatch:
        sublabels = anatomy.index_select(0, indices)
        tissue = sublabels == 3
        source = levels[sublabels.long()]
        targets = {field: torch.where(tissue, TISSUE_INTENSITY[field], source) for field in TARGET_FIELDS}
        ids = tuple(f"synthetic-{i:04d}" for i in indices.tolist())
        return SubjectBatch(ids, source, targets, tissue)

    return render(order[:n_train]), render(order[n_train:])


def measure(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float | list[float]]:
    """Score paired subjects, the unordered empirical volume populations, and their pooled intensities.

    The empirical 2-Wasserstein distance uses full-volume voxel RMSE as its ground metric.
    With equal subject masses, minimizing total squared RMSE is a linear assignment problem.
    The pooled 1-Wasserstein distance in intensity units also ignores spatial arrangement.
    """
    if predictions.shape != targets.shape or predictions.ndim != 5 or predictions.numel() == 0:
        raise ValueError("Expected matching nonempty [subjects, channels, depth, height, width] cohorts.")
    pred = predictions.detach().cpu().double().flatten(1)
    ref = targets.detach().cpu().double().flatten(1)
    if not (torch.isfinite(pred).all() and torch.isfinite(ref).all()):
        raise ValueError("Metrics require finite images.")
    # One row at a time bounds temporary memory. Direct subtraction preserves exact zeros.
    costs = torch.stack([(ref - row).square().mean(dim=1) for row in pred])
    rows, cols = linear_sum_assignment(costs.numpy())
    paired = costs.diag().sqrt()
    intensity_w1 = (pred.flatten().sort().values - ref.flatten().sort().values).abs().mean()
    return {
        "subject_rmse": paired.tolist(),
        "mean_subject_rmse": paired.mean().item(),
        "population_image_w2": costs[rows, cols].mean().sqrt().item(),
        "population_intensity_w1": intensity_w1.item(),
    }
