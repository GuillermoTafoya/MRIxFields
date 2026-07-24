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
