import importlib.util
import json
from pathlib import Path

import pytest
import torch

from fieldbridge.data.paired_loso import (
    RealPairedSliceDataset,
    build_loso_folds,
    fit_train_only_affine_calibrations,
    prepare_preprocessed_tensor_cache,
)
from fieldbridge.data.preprocessing import SlicePreprocessingSpec
from fieldbridge.models.factory import build_translator
from fieldbridge.training.paired_loso import validate_one_batch_training_compatibility
from fieldbridge.training.pseudo_pair_epochs import PseudoPairEpochConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "notebooks/prospective_paired_loso_residual_runner.py"
FIELDS = (0.1, 1.5, 3.0, 5.0, 7.0)


def _spec() -> SlicePreprocessingSpec:
    return SlicePreprocessingSpec(
        slice_start=0,
        slice_end=3,
        slices_per_volume=None,
        normalization="official_01",
        model_range="minus_one_one",
        resize_mode="fit_pad",
        output_height=6,
        output_width=8,
        slice_axis="z",
    )


def _volumes(cases=("0006", "0007", "0009")):
    volumes = {}
    for case_offset, case_id in enumerate(cases):
        generator = torch.Generator().manual_seed(31 + case_offset)
        source = torch.rand((1, 4, 5, 3), generator=generator) * 0.8 + 0.1
        volumes[case_id] = {
            field: (source + field / 100.0).clamp(0.0, 1.0)
            for field in FIELDS
        }
    return volumes


def _input_fingerprints(cases, *, marker="a"):
    return {
        case_id: {
            field: f"{case_offset + field_offset + 1:064x}"
            for field_offset, field in enumerate(FIELDS)
        }
        for case_offset, case_id in enumerate(cases)
    }


def _prepare(tmp_path, volumes, cases, **overrides):
    arguments = {
        "cache_root": tmp_path,
        "case_ids": cases,
        "preprocessing": _spec(),
        "slice_indices": (0, 1, 2),
        "manifest_fingerprint": "a" * 64,
        "input_fingerprints": _input_fingerprints(cases),
        "code_fingerprint": "b" * 40,
        "config_fingerprint": "c" * 64,
    }
    arguments.update(overrides)
    return prepare_preprocessed_tensor_cache(volumes, **arguments)


def _model_config():
    return {
        "name": "conditional_residual_unet_field_translator",
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": [4],
        "latent_channels": 8,
        "cond_dim": 8,
        "spatial_dims": 2,
        "upsample_mode": "interpolate",
        "skip_mode": "gated",
        "pad_to_multiple": True,
        "model_range": "minus_one_one",
    }


def test_cache_is_bit_exact_to_direct_dataset_and_affine(tmp_path: Path) -> None:
    cases = ("0006", "0007")
    volumes = _volumes(cases)
    cache = _prepare(tmp_path, volumes, cases)
    direct = RealPairedSliceDataset(
        volumes,
        case_ids=cases,
        preprocessing=_spec(),
        slice_indices=(0, 1, 2),
    )
    cached = RealPairedSliceDataset(
        volumes,
        case_ids=cases,
        preprocessing=_spec(),
        slice_indices=(0, 1, 2),
        preprocessed_cache=cache,
    )
    assert direct.samples == cached.samples
    for index in range(len(direct)):
        direct_sample = direct[index]
        cached_sample = cached[index]
        assert torch.equal(direct_sample.x_low, cached_sample.x_low)
        assert torch.equal(direct_sample.x_high, cached_sample.x_high)
        assert torch.equal(direct_sample.mask, cached_sample.mask)
        assert direct_sample.geometry == cached_sample.geometry
        assert direct_sample.slice_index == cached_sample.slice_index

    direct_affine = fit_train_only_affine_calibrations(
        volumes,
        train_case_ids=cases,
        preprocessing=_spec(),
        slice_indices=(0, 1, 2),
    )
    cached_affine = fit_train_only_affine_calibrations(
        volumes,
        train_case_ids=cases,
        preprocessing=_spec(),
        slice_indices=(0, 1, 2),
        preprocessed_cache=cache,
    )
    assert cached_affine == direct_affine


def test_cache_reuse_invalidation_and_corruption_are_fail_closed(tmp_path: Path) -> None:
    cases = ("0006",)
    volumes = _volumes(cases)
    first = _prepare(tmp_path, volumes, cases)
    second = _prepare(tmp_path, volumes, cases)
    assert second.fingerprint == first.fingerprint
    assert second.stats.cache_hit is True

    changed_inputs = _input_fingerprints(cases)
    changed_inputs["0006"][7.0] = "f" * 64
    input_invalidated = _prepare(
        tmp_path,
        volumes,
        cases,
        input_fingerprints=changed_inputs,
    )
    config_invalidated = _prepare(
        tmp_path,
        volumes,
        cases,
        config_fingerprint="d" * 64,
    )
    assert len(
        {first.fingerprint, input_invalidated.fingerprint, config_invalidated.fingerprint}
    ) == 3

    cache_dir = tmp_path / "paired_loso_preprocessed_v1" / first.fingerprint
    tensor_path = next(cache_dir.glob("*.pt"))
    with tensor_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        _prepare(tmp_path, volumes, cases)
    building = cache_dir.with_name(cache_dir.name + ".building")
    building.mkdir()
    with pytest.raises(ValueError, match="incomplete preprocessing cache"):
        _prepare(tmp_path, volumes, cases)


def test_dry_run_uses_one_cached_real_batch_and_never_fits_affine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    cases = ("0006", "0007", "0009")
    volumes = _volumes(cases)
    cache = _prepare(tmp_path, volumes, cases)
    config = _model_config()
    synthetic_model = build_translator(
        config["name"],
        **{key: value for key, value in config.items() if key != "name"},
    )
    seen_batch_ids = []

    def validate(config, *, model, batch, device):
        seen_batch_ids.append(id(batch))
        return {
            "cuda": True,
            "amp_enabled": True,
            "batch_size": int(batch.x_low.shape[0]),
            "forward": True,
            "loss": True,
            "backward": True,
            "optimizer": "AdamW",
            "optimizer_step": True,
        }

    monkeypatch.setattr(runner, "validate_one_batch_training_compatibility", validate)
    monkeypatch.setattr(
        runner,
        "fit_train_only_affine_calibrations",
        lambda *args, **kwargs: pytest.fail("dry-run must not fit affine calibrations"),
    )
    result = runner._run_training_dry_run(
        fold=build_loso_folds(cases)[0],
        case_slot="case_01",
        volumes=volumes,
        preprocessed_cache=cache,
        preprocessing=_spec(),
        train_indices=(0, 1, 2),
        train_cfg=PseudoPairEpochConfig(batch_size=4, seed=13, amp=True),
        model_config=config,
        synthetic_state={"model": synthetic_model.state_dict()},
        num_workers=0,
        device=torch.device("cpu"),
    )
    assert result["affine_fit_executed"] is False
    assert result["real_batch"] is True
    assert len(seen_batch_ids) == 2 and len(set(seen_batch_ids)) == 1
    assert set(result["arms"]) == {
        "identity_initialization",
        "synthetic_initialization",
    }


def test_actual_dry_run_step_requires_cuda() -> None:
    dataset = RealPairedSliceDataset(
        _volumes(("0006",)),
        case_ids=("0006",),
        preprocessing=_spec(),
        slice_indices=(0, 1, 2),
    )
    from fieldbridge.data.pseudo_pairs import collate_pseudo_pair_slices

    batch = collate_pseudo_pair_slices([dataset[0], dataset[1]])
    model_config = _model_config()
    model = build_translator(
        model_config["name"],
        **{key: value for key, value in model_config.items() if key != "name"},
    )
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        validate_one_batch_training_compatibility(
            PseudoPairEpochConfig(amp=True),
            model=model,
            batch=batch,
            device=torch.device("cpu"),
        )


def test_sanitized_telemetry_reports_cache_batch_training_and_amp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    telemetry = runner.NvidiaSmiTelemetry(
        device=torch.device("cpu"),
        amp_enabled=False,
    )
    telemetry.start()
    cache = _prepare(tmp_path / "cache", _volumes(("0006",)), ("0006",))
    telemetry.set_cache_preparation(cache.stats)
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "train": {
                    "batch_preparation_samples_per_second": 1200.0,
                    "training_steps_per_second": 8.5,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry.add_training_history(history_path)
    telemetry.stop()
    payload = telemetry.summary()
    assert payload["cuda_device"] == "unavailable"
    assert payload["amp_enabled"] is False
    assert payload["cache_preparation"]["tensors"] == 15
    assert payload["batch_preparation_samples_per_second"]["mean"] == 1200.0
    assert payload["training_steps_per_second"]["mean"] == 8.5
    assert str(tmp_path) not in json.dumps(payload)


def test_dry_run_preflight_validation_is_read_only(tmp_path: Path) -> None:
    runner = _load_runner()
    alignment_dir = tmp_path / "alignment_preflight"
    alignment_dir.mkdir()
    fingerprint = "a" * 64
    preflight = {"ok": True, "alignment_contract_sha256": fingerprint}
    preflight_path = tmp_path / "preflight_sanitized.json"
    contract_path = alignment_dir / "contract_sanitized.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    contract_path.write_text(
        json.dumps({"alignment_contract_sha256": fingerprint}),
        encoding="utf-8",
    )
    for case_offset in range(1, 4):
        for field in (1.5, 3.0, 5.0, 7.0):
            for slice_index in (72, 103, 135, 166, 197, 228, 260, 291):
                field_label = str(field).replace(".", "p")
                panel_path = alignment_dir / (
                    f"case_{case_offset:02d}_{field_label}T_{slice_index:03d}.png"
                )
                panel_path.write_bytes(b"panel")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    runner._validate_preflight_artifacts(
        output_dir=tmp_path,
        expected_preflight=preflight,
        alignment_fingerprint=fingerprint,
    )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def _load_runner():
    spec = importlib.util.spec_from_file_location("paired_loso_cache_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
