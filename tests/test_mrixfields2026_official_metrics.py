from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

import fieldbridge.evaluation.mrixfields2026_official as official


def test_official_source_contract_is_pinned_to_verified_blobs() -> None:
    assert (
        official.UPSTREAM_COMMIT
        == "5d55309253951d9dfb7847856f4f46893a44d63b"
    )
    assert (
        official.UPSTREAM_EVALUATE_BLOB
        == "4e21a48b097ef274f9fceeef536f9790eb451385"
    )
    assert (
        official.UPSTREAM_README_BLOB
        == "7d3a73a38990450c58b9d05a89b82acb0b73e638"
    )


def test_official_nrmse_matches_published_l2_ratio_and_zero_norm_rule() -> None:
    prediction = np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    target = np.asarray([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    expected = np.linalg.norm(
        prediction.astype(np.float64) - target.astype(np.float64)
    ) / np.linalg.norm(target.astype(np.float64))

    assert official.official_task3_nrmse(prediction, target) == pytest.approx(
        expected
    )
    assert (
        official.official_task3_nrmse(
            np.ones((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
        == 0.0
    )


def test_official_metrics_reject_shape_mismatch_before_broadcasting() -> None:
    prediction = np.zeros((8, 8, 2), dtype=np.float32)
    target = np.zeros((8, 8, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch.*broadcasting"):
        official.official_task3_nrmse(prediction, target)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_official_metrics_reject_nonfinite_input(invalid: float) -> None:
    prediction = np.zeros((8, 8, 2), dtype=np.float32)
    prediction[0, 0, 0] = invalid
    target = np.zeros_like(prediction)

    with pytest.raises(ValueError, match="prediction contains non-finite"):
        official.official_task3_nrmse(prediction, target)


def test_official_ssim_uses_global_target_range_and_skips_constant_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def fake_structural_similarity(
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        data_range: float,
    ) -> float:
        calls.append((prediction, target, float(data_range)))
        return float(target.mean())

    monkeypatch.setattr(
        official,
        "_load_structural_similarity",
        lambda: fake_structural_similarity,
    )
    target = np.zeros((8, 8, 3), dtype=np.float32)
    target[..., 1] = np.linspace(0.0, 0.5, 64).reshape(8, 8)
    target[..., 2] = np.linspace(0.25, 1.0, 64).reshape(8, 8)
    prediction = target + 0.1

    result = official.official_task3_ssim(prediction, target)

    assert len(calls) == 2
    assert all(call[2] == pytest.approx(1.0) for call in calls)
    assert result == pytest.approx(
        np.mean([target[..., 1].mean(), target[..., 2].mean()])
    )


def test_official_ssim_constant_target_returns_one_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_be_called(*args: object, **kwargs: object) -> float:
        del args, kwargs
        raise AssertionError("structural_similarity should not be called")

    monkeypatch.setattr(
        official,
        "_load_structural_similarity",
        lambda: should_not_be_called,
    )
    target = np.zeros((8, 8, 2), dtype=np.float32)
    prediction = np.ones_like(target)

    assert official.official_task3_ssim(prediction, target) == 1.0


def test_official_ssim_matches_real_scikit_image_on_deterministic_volume() -> None:
    skimage_metrics = pytest.importorskip("skimage.metrics")
    target = np.zeros((8, 8, 3), dtype=np.float32)
    target[..., 1] = np.linspace(0.0, 0.75, 64).reshape(8, 8)
    target[..., 2] = np.linspace(0.25, 1.0, 64).reshape(8, 8)
    prediction = np.clip(target * 0.85 + 0.05, 0.0, 1.0)

    pred64 = prediction.astype(np.float64)
    target64 = target.astype(np.float64)
    data_range = target64.max() - target64.min()
    expected_slices = []
    for index in range(pred64.shape[2]):
        pred_slice = pred64[..., index]
        target_slice = target64[..., index]
        if target_slice.max() - target_slice.min() < 1e-10:
            continue
        expected_slices.append(
            skimage_metrics.structural_similarity(
                pred_slice,
                target_slice,
                data_range=data_range,
            )
        )
    expected = float(np.mean(expected_slices))

    assert official.official_task3_ssim(
        prediction, target
    ) == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_official_nifti_loader_canonicalizes_before_float32_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    affine = np.eye(4)

    class FakeCanonicalImage:
        def __init__(self) -> None:
            self.affine = affine

        def get_fdata(self, *, dtype: object) -> np.ndarray:
            events.append(("get_fdata", dtype))
            return np.ones((2, 3, 4), dtype=np.float32)

    source_image = object()
    canonical_image = FakeCanonicalImage()
    fake_nibabel = types.ModuleType("nibabel")

    def fake_load(path: str) -> object:
        events.append(("load", path))
        return source_image

    def fake_canonicalize(image: object) -> FakeCanonicalImage:
        events.append(("canonical", image))
        return canonical_image

    fake_nibabel.load = fake_load  # type: ignore[attr-defined]
    fake_nibabel.as_closest_canonical = fake_canonicalize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nibabel", fake_nibabel)

    values, loaded_affine = official.load_official_nifti("synthetic.nii.gz")

    assert events == [
        ("load", "synthetic.nii.gz"),
        ("canonical", source_image),
        ("get_fdata", np.float32),
    ]
    assert values.dtype == np.float32
    assert loaded_affine is affine


def test_official_lpips_uses_alexnet_signed_slices_and_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations: dict[str, object] = {"calls": []}

    class FakeLPIPS(torch.nn.Module):
        def __init__(self, *, net: str) -> None:
            super().__init__()
            observations["net"] = net

        def to(self, device: str) -> FakeLPIPS:
            observations["device"] = device
            return self

        def eval(self) -> FakeLPIPS:
            observations["eval"] = True
            return self

        def forward(
            self, prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            calls = observations["calls"]
            assert isinstance(calls, list)
            calls.append((prediction.clone(), target.clone()))
            return torch.abs(prediction - target).mean()

    fake_lpips = types.ModuleType("lpips")
    fake_lpips.LPIPS = FakeLPIPS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lpips", fake_lpips)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    target = np.zeros((2, 2, 2), dtype=np.float32)
    target[..., 0] = 0.5
    target[..., 1] = 0.25
    prediction = target.copy()
    prediction[..., 1] = 0.75

    score = official.official_task3_lpips(prediction, target)

    assert observations["net"] == "alex"
    assert observations["device"] == "cpu"
    assert observations["eval"] is True
    calls = observations["calls"]
    assert isinstance(calls, list) and len(calls) == 1
    pred_tensor, target_tensor = calls[0]
    assert pred_tensor.shape == (1, 3, 2, 2)
    assert torch.all(pred_tensor == 0.5)
    assert torch.all(target_tensor == -0.5)
    assert score == pytest.approx(1.0)


def test_official_pair_rejects_nonfinite_metric_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = iter(
        [
            (np.zeros((8, 8, 2), dtype=np.float32), np.eye(4)),
            (np.ones((8, 8, 2), dtype=np.float32), np.eye(4)),
        ]
    )
    monkeypatch.setattr(official, "load_official_nifti", lambda path: next(arrays))
    monkeypatch.setattr(
        official, "official_task3_nrmse", lambda prediction, target: float("nan")
    )

    with pytest.raises(ValueError, match="non-finite result"):
        official.evaluate_official_task3_pair(
            "prediction.nii.gz",
            "target.nii.gz",
            metrics=("nrmse",),
        )


def test_official_directory_aggregates_published_mean_and_population_std(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    for subject in ("0001", "0002"):
        _touch_nifti(prediction_dir / f"P_T1W_7T_{subject}.nii.gz")
        _touch_nifti(target_dir / f"P_T1W_7T_{subject}.nii.gz")

    values = {
        "0001": {"nrmse": 1.0, "ssim": 0.5, "lpips": 0.1},
        "0002": {"nrmse": 3.0, "ssim": 0.7, "lpips": 0.3},
    }

    def fake_pair(prediction_path, target_path, *, metrics, device):
        del target_path, metrics, device
        subject = official._extract_subject_id(prediction_path.name)
        return dict(values[subject])

    monkeypatch.setattr(
        official, "evaluate_official_task3_pair", fake_pair
    )
    monkeypatch.setattr(
        official,
        "official_task3_runtime_provenance",
        lambda **kwargs: {"lpips_device": "cpu"},
    )

    payload = official.evaluate_official_task3_directory(
        prediction_dir, target_dir
    )

    assert payload["case_count"] == 2
    assert payload["summary"] == pytest.approx(
        {
            "nrmse_mean": 2.0,
            "nrmse_std": 1.0,
            "ssim_mean": 0.6,
            "ssim_std": 0.1,
            "lpips_mean": 0.2,
            "lpips_std": 0.1,
        }
    )
    assert [case["subject"] for case in payload["cases"]] == [
        "0001",
        "0002",
    ]


def test_official_directory_output_records_contract_and_runtime_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    _touch_nifti(prediction_dir / "P_T2W_5T_0001.nii.gz")
    _touch_nifti(target_dir / "P_T2W_5T_0001.nii.gz")
    monkeypatch.setattr(
        official,
        "evaluate_official_task3_pair",
        lambda *args, **kwargs: {
            "nrmse": 0.1,
            "ssim": 0.9,
            "lpips": 0.2,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    payload = official.evaluate_official_task3_directory(
        prediction_dir, target_dir, device="cuda"
    )

    assert (
        payload["OFFICIAL_TASK3_METRIC_CONTRACT"]
        == official.OFFICIAL_TASK3_METRIC_CONTRACT
    )
    assert payload["metric_contract"] == official.OFFICIAL_TASK3_METRIC_CONTRACT
    assert payload["upstream"] == {
        "repository": official.UPSTREAM_REPOSITORY,
        "commit": official.UPSTREAM_COMMIT,
        "evaluate_blob": official.UPSTREAM_EVALUATE_BLOB,
        "readme_blob": official.UPSTREAM_README_BLOB,
    }
    assert set(payload["runtime"]) == {
        "python",
        "numpy",
        "nibabel",
        "scikit-image",
        "torch",
        "torchvision",
        "lpips",
        "lpips_device",
    }
    assert all(
        isinstance(payload["runtime"][name], str)
        for name in (
            "python",
            "numpy",
            "nibabel",
            "scikit-image",
            "torch",
            "torchvision",
            "lpips",
        )
    )
    assert payload["runtime"]["lpips_device"] == "cpu"


def _touch_nifti(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
