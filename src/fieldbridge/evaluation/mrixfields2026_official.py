"""Source-pinned MRIxFields2026 Task-3 evaluation adapter.

The numerical behavior in this module mirrors ``Evaluation/evaluate.py`` at the pinned
upstream state below. Keep this adapter separate from differentiable training proxies
and frozen project audit contracts.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Iterable
from importlib import metadata
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

    _validate_official_inputs(prediction, target, mask=mask)
    pred = prediction.astype(np.float64)
    tgt = target.astype(np.float64)
    if mask is not None:
        pred, tgt = pred[mask > 0], tgt[mask > 0]
    norm = np.linalg.norm(tgt)
    result = (
        float(np.linalg.norm(pred - tgt) / norm)
        if norm > 1e-10
        else 0.0
    )
    return _finite_metric_result("nrmse", result)


def official_task3_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    slice_axis: int = 2,
) -> float:
    """Published Task-3 SSIM: slice-wise scikit-image SSIM.

    The target's full-volume range is used for every slice. Constant target slices are
    omitted, and an all-constant target returns one.
    """

    _validate_official_inputs(prediction, target)
    structural_similarity = _load_structural_similarity()
    pred = prediction.astype(np.float64)
    tgt = target.astype(np.float64)
    data_range = tgt.max() - tgt.min()
    if data_range < 1e-10:
        return 1.0
    if pred.ndim == 2:
        return _finite_metric_result(
            "ssim",
            float(structural_similarity(pred, tgt, data_range=data_range)),
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
    result = float(np.mean(values)) if values else 1.0
    return _finite_metric_result("ssim", result)


def official_task3_lpips(
    prediction: np.ndarray,
    target: np.ndarray,
    slice_axis: int = 2,
    device: str = "cuda",
) -> float:
    """Published Task-3 slice-wise AlexNet LPIPS implementation."""

    import torch

    _validate_official_inputs(prediction, target)
    try:
        import lpips as lpips_module
    except ImportError as exc:
        raise ImportError(
            "Official MRIxFields2026 LPIPS evaluation requires lpips. "
            "Install the 'official-evaluation' optional dependency group."
        ) from exc

    selected_device = resolve_official_lpips_device(device)
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
        return _finite_metric_result(
            "lpips", evaluate_2d(pred_normalized, target_normalized)
        )

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
    result = float(np.mean(values)) if values else 0.0
    return _finite_metric_result("lpips", result)


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
    _validate_official_inputs(prediction, target)
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
    for name, value in results.items():
        _finite_metric_result(name, value)
    return results


def evaluate_official_task3_directory(
    prediction_dir: str | Path,
    target_dir: str | Path,
    *,
    metrics: Iterable[str] = ("nrmse", "ssim", "lpips"),
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate and aggregate one published-style Task-3 directory.

    Pairs are matched by the published subject-ID extraction rule. Each directory must
    contain at most one NIfTI per subject ID; challenge callers should evaluate one
    target field/contrast unit at a time. Per-metric summary values use NumPy's
    population mean and standard deviation exactly as the published evaluator does.
    """

    requested = tuple(metrics)
    unsupported = sorted(set(requested) - {"nrmse", "ssim", "lpips"})
    if unsupported:
        raise ValueError(
            f"Unsupported official Task-3 metrics: {unsupported}."
        )
    if not requested:
        raise ValueError("At least one official Task-3 metric is required.")

    prediction_root = _validated_directory(prediction_dir, "prediction")
    target_root = _validated_directory(target_dir, "target")
    pairs = match_official_task3_pairs(prediction_root, target_root)
    if not pairs:
        raise ValueError(
            "No matching prediction-target NIfTI pairs were found."
        )

    cases: list[dict[str, Any]] = []
    for prediction_path, target_path in pairs:
        subject = _extract_subject_id(prediction_path.name)
        case_metrics = evaluate_official_task3_pair(
            prediction_path,
            target_path,
            metrics=requested,
            device=device,
        )
        for name, value in case_metrics.items():
            _finite_metric_result(name, value)
        cases.append(
            {
                "subject": subject,
                "prediction": prediction_path.relative_to(
                    prediction_root
                ).as_posix(),
                "target": target_path.relative_to(target_root).as_posix(),
                "metrics": case_metrics,
            }
        )

    summary: dict[str, float] = {}
    for name in requested:
        values = np.asarray(
            [case["metrics"][name] for case in cases],
            dtype=np.float64,
        )
        summary[f"{name}_mean"] = _finite_metric_result(
            f"{name}_mean", float(np.mean(values))
        )
        summary[f"{name}_std"] = _finite_metric_result(
            f"{name}_std", float(np.std(values))
        )

    return {
        "OFFICIAL_TASK3_METRIC_CONTRACT": OFFICIAL_TASK3_METRIC_CONTRACT,
        "metric_contract": OFFICIAL_TASK3_METRIC_CONTRACT,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "evaluate_blob": UPSTREAM_EVALUATE_BLOB,
            "readme_blob": UPSTREAM_README_BLOB,
        },
        "runtime": official_task3_runtime_provenance(
            metrics=requested, device=device
        ),
        "aggregation": {
            "per_case": "published full-volume metric functions",
            "summary": "numpy mean and population std (ddof=0)",
        },
        "case_count": len(cases),
        "metrics": list(requested),
        "summary": summary,
        "cases": cases,
    }


def match_official_task3_pairs(
    prediction_dir: str | Path, target_dir: str | Path
) -> list[tuple[Path, Path]]:
    """Match ``*.nii.gz`` files by the published subject-ID prefix rule."""

    prediction_root = Path(prediction_dir)
    target_root = Path(target_dir)
    predictions = sorted(prediction_root.rglob("*.nii.gz"))
    targets = sorted(target_root.rglob("*.nii.gz"))
    prediction_lookup = _unique_subject_lookup(
        predictions, directory_role="prediction"
    )
    target_lookup = _unique_subject_lookup(
        targets, directory_role="target"
    )
    return [
        (prediction_lookup[subject], target_lookup[subject])
        for subject in sorted(prediction_lookup)
        if subject in target_lookup
    ]


def official_task3_runtime_provenance(
    *,
    metrics: Iterable[str],
    device: str,
) -> dict[str, Any]:
    """Return dependency versions and the resolved LPIPS device."""

    import torch

    requested = tuple(metrics)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nibabel": _package_version("nibabel"),
        "scikit-image": _package_version("scikit-image"),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "lpips": _package_version("lpips"),
        "lpips_device": (
            resolve_official_lpips_device(device)
            if "lpips" in requested
            else None
        ),
    }


def resolve_official_lpips_device(device: str) -> str:
    """Apply the published CUDA-to-CPU fallback rule."""

    import torch

    if not torch.cuda.is_available() and device == "cuda":
        return "cpu"
    return device


def _load_structural_similarity() -> Any:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "Official MRIxFields2026 SSIM evaluation requires scikit-image. "
            "Install the 'official-evaluation' optional dependency group."
        ) from exc
    return structural_similarity


def _validate_official_inputs(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Official Task-3 prediction/target shape mismatch: "
            f"{prediction.shape} != {target.shape}; NumPy broadcasting is forbidden."
        )
    if not np.isfinite(prediction).all():
        raise ValueError(
            "Official Task-3 prediction contains non-finite values."
        )
    if not np.isfinite(target).all():
        raise ValueError("Official Task-3 target contains non-finite values.")
    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError(
                "Official Task-3 mask shape mismatch: "
                f"{mask.shape} != {target.shape}."
            )
        if not np.isfinite(mask).all():
            raise ValueError(
                "Official Task-3 mask contains non-finite values."
            )


def _finite_metric_result(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"Official Task-3 metric {name!r} produced a non-finite result."
        )
    return result


def _validated_directory(path: str | Path, role: str) -> Path:
    resolved = Path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Official Task-3 {role} directory not found: {resolved}"
        )
    return resolved


def _extract_subject_id(filename: str) -> str:
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[-1]
    return parts[0]


def _unique_subject_lookup(
    paths: Iterable[Path], *, directory_role: str
) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in paths:
        subject = _extract_subject_id(path.name)
        if subject in lookup:
            raise ValueError(
                f"Official Task-3 {directory_role} directory contains duplicate "
                f"subject ID {subject!r}: {lookup[subject].name!r} and {path.name!r}."
            )
        lookup[subject] = path
    return lookup


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"
