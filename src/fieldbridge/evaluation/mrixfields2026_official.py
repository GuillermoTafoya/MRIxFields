"""Source-pinned MRIxFields2026 Task-3 evaluation adapter.

The numerical behavior in this module mirrors ``Evaluation/evaluate.py`` at the pinned
upstream state below. Keep this adapter separate from differentiable training proxies
and frozen project audit contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_REPOSITORY = "MRIxFields/MRIxFields2026"
UPSTREAM_COMMIT = "5d55309253951d9dfb7847856f4f46893a44d63b"
UPSTREAM_EVALUATE_BLOB = "4e21a48b097ef274f9fceeef536f9790eb451385"
UPSTREAM_README_BLOB = "7d3a73a38990450c58b9d05a89b82acb0b73e638"
OFFICIAL_TASK3_METRIC_CONTRACT = (
    "mrixfields2026-task3-evaluate.py@"
    f"{UPSTREAM_COMMIT}#blob-{UPSTREAM_EVALUATE_BLOB}"
)


def load_official_nifti(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one NIfTI exactly as the published evaluator does.

    The image is reoriented with ``nib.as_closest_canonical`` before voxel data is
    requested as float32. ``nibabel`` is imported lazily so core synthetic tests do not
    require the optional official-evaluation dependencies.
    """

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Official MRIxFields2026 file evaluation requires nibabel. "
            "Install the 'official-evaluation' optional dependency group."
        ) from exc

    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)
    return image.get_fdata(dtype=np.float32), image.affine


def official_task3_nrmse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Published full-volume L2 error divided by the target L2 norm.

    Task 3 calls this without a mask. The optional mask is retained solely because the
    published function exposes it.
    """

    pred = prediction.astype(np.float64)
    tgt = target.astype(np.float64)
    if mask is not None:
        pred, tgt = pred[mask > 0], tgt[mask > 0]
    norm = np.linalg.norm(tgt)
    return (
        float(np.linalg.norm(pred - tgt) / norm)
        if norm > 1e-10
        else 0.0
    )


def official_task3_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    slice_axis: int = 2,
) -> float:
    """Published Task-3 SSIM: slice-wise scikit-image SSIM.

    The target's full-volume range is used for every slice. Constant target slices are
    omitted, and an all-constant target returns one.
    """

    structural_similarity = _load_structural_similarity()
    pred = prediction.astype(np.float64)
    tgt = target.astype(np.float64)
    data_range = tgt.max() - tgt.min()
    if data_range < 1e-10:
        return 1.0
    if pred.ndim == 2:
        return float(
            structural_similarity(pred, tgt, data_range=data_range)
        )

    values: list[float] = []
    for index in range(pred.shape[slice_axis]):
        selection = [slice(None)] * pred.ndim
        selection[slice_axis] = index
        key = tuple(selection)
        pred_slice, target_slice = pred[key], tgt[key]
        if target_slice.max() - target_slice.min() < 1e-10:
            continue
        values.append(
            float(
                structural_similarity(
                    pred_slice,
                    target_slice,
                    data_range=data_range,
                )
            )
        )
    return float(np.mean(values)) if values else 1.0


def official_task3_lpips(
    prediction: np.ndarray,
    target: np.ndarray,
    slice_axis: int = 2,
    device: str = "cuda",
) -> float:
    """Published Task-3 slice-wise AlexNet LPIPS implementation."""

    import torch

    try:
        import lpips as lpips_module
    except ImportError as exc:
        raise ImportError(
            "Official MRIxFields2026 LPIPS evaluation requires lpips. "
            "Install the 'official-evaluation' optional dependency group."
        ) from exc

    selected_device = device
    if not torch.cuda.is_available() and selected_device == "cuda":
        selected_device = "cpu"
    network = lpips_module.LPIPS(net="alex").to(selected_device)
    network.eval()
    pred_normalized = prediction.astype(np.float64) * 2.0 - 1.0
    target_normalized = target.astype(np.float64) * 2.0 - 1.0

    def evaluate_2d(
        pred_slice: np.ndarray, target_slice: np.ndarray
    ) -> float:
        pred_tensor = (
            torch.from_numpy(pred_slice)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .to(selected_device)
        )
        target_tensor = (
            torch.from_numpy(target_slice)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .to(selected_device)
        )
        with torch.no_grad():
            return float(network(pred_tensor, target_tensor).item())

    if prediction.ndim == 2:
        return evaluate_2d(pred_normalized, target_normalized)

    values: list[float] = []
    for index in range(prediction.shape[slice_axis]):
        selection = [slice(None)] * prediction.ndim
        selection[slice_axis] = index
        key = tuple(selection)
        if np.abs(target_normalized[key]).max() < 1e-10:
            continue
        values.append(
            evaluate_2d(pred_normalized[key], target_normalized[key])
        )
    return float(np.mean(values)) if values else 0.0


def evaluate_official_task3_pair(
    prediction_path: str | Path,
    target_path: str | Path,
    *,
    metrics: Iterable[str] = ("nrmse", "ssim", "lpips"),
    device: str = "cuda",
) -> dict[str, float]:
    """Load and evaluate one prediction-target NIfTI pair under the pinned contract."""

    prediction, _ = load_official_nifti(prediction_path)
    target, _ = load_official_nifti(target_path)
    requested = tuple(metrics)
    unsupported = sorted(set(requested) - {"nrmse", "ssim", "lpips"})
    if unsupported:
        raise ValueError(
            f"Unsupported official Task-3 metrics: {unsupported}."
        )

    results: dict[str, float] = {}
    if "nrmse" in requested:
        results["nrmse"] = official_task3_nrmse(prediction, target)
    if "ssim" in requested:
        results["ssim"] = official_task3_ssim(prediction, target)
    if "lpips" in requested:
        results["lpips"] = official_task3_lpips(
            prediction, target, device=device
        )
    return results


def _load_structural_similarity() -> Any:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "Official MRIxFields2026 SSIM evaluation requires scikit-image. "
            "Install the 'official-evaluation' optional dependency group."
        ) from exc
    return structural_similarity
