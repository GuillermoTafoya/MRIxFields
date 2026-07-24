from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.datasets import ALL_DOMAINS
from fieldbridge.data.vae_splits import VaeSplits
from fieldbridge.evaluation import stage1_full_volume_audit as audit_module
from fieldbridge.evaluation.metrics import stage1_full_volume_ssim3d_v1
from fieldbridge.evaluation.stage1_full_volume_audit import (
    AuditRuntime,
    aggregate_domain_balanced,
    audit_stage1_checkpoint,
    compute_full_volume_metrics,
    freeze_stage1_audit_selection,
    prepare_audit_root,
)
from fieldbridge.evaluation.stage1_report import sliding_window_reconstruct


class _IdentityEncoder(torch.nn.Module):
    latent_channels = 1

    def encode_dist(self, value: torch.Tensor, domain: object):
        return value, torch.zeros_like(value)


class _IdentityDecoder(torch.nn.Module):
    def decode(self, value: torch.Tensor, domain: object):
        return value


def test_frozen_audit_ssim3d_v1_has_pre_v3_regression_value() -> None:
    target = torch.linspace(
        0.0, 1.0, 8 * 8 * 8, dtype=torch.float32
    ).reshape(1, 1, 8, 8, 8)
    reconstruction = target * 0.8 + 0.1

    value = stage1_full_volume_ssim3d_v1(reconstruction, target)

    assert audit_module.stage1_full_volume_ssim3d_v1 is (
        stage1_full_volume_ssim3d_v1
    )
    assert float(value) == pytest.approx(
        0.9868708848953247, rel=0.0, abs=1e-7
    )


def _records(per_domain: int = 4) -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    for domain_index, domain in enumerate(ALL_DOMAINS):
        for index in range(per_domain):
            identity = f"private-record-d{domain_index:02d}-{index:02d}"
            records.append(
                VolumeRecord(
                    case_id=identity,
                    image_path=f"C:/private/data/{identity}.nii.gz",
                    domain=domain,
                    subject_id=f"private-subject-{domain_index:02d}-{index:02d}",
                )
            )
    return records


def _splits(records: list[VolumeRecord] | None = None) -> VaeSplits:
    return VaeSplits(
        train=(),
        validation=(),
        test=tuple(records or _records()),
        seed=13,
        fractions=(0.0, 0.0, 1.0),
    )


def _freeze(tmp_path: Path, splits: VaeSplits | None = None, *, seed: int = 13):
    return freeze_stage1_audit_selection(
        splits or _splits(),
        private_path=tmp_path / "selection_private.json",
        sanitized_path=tmp_path / "selection_sanitized.json",
        seed=seed,
    )


def test_selection_is_exactly_four_by_fifteen_and_sanitized(tmp_path: Path) -> None:
    payload = _freeze(tmp_path)
    assert len(payload["selected"]) == 60
    assert len({item["domain"] for item in payload["selected"]}) == 15
    assert all(
        sum(item["domain"] == domain.label for item in payload["selected"]) == 4
        for domain in ALL_DOMAINS
    )
    sanitized = (tmp_path / "selection_sanitized.json").read_text(encoding="utf-8")
    assert "private-record" not in sanitized
    assert "C:/private" not in sanitized
    assert "case_slot" in sanitized


def test_selection_is_stable_under_input_shuffle(tmp_path: Path) -> None:
    records = _records(per_domain=6)
    shuffled = list(records)
    random.Random(991).shuffle(shuffled)
    first = freeze_stage1_audit_selection(
        _splits(records),
        private_path=tmp_path / "first.json",
        sanitized_path=tmp_path / "first_sanitized.json",
    )
    second = freeze_stage1_audit_selection(
        _splits(shuffled),
        private_path=tmp_path / "second.json",
        sanitized_path=tmp_path / "second_sanitized.json",
    )
    assert first["selection_fingerprint"] == second["selection_fingerprint"]
    assert [item["record_id"] for item in first["selected"]] == [
        item["record_id"] for item in second["selected"]
    ]


@pytest.mark.parametrize("mode", ["missing", "insufficient"])
def test_selection_fails_closed_on_domain_coverage(tmp_path: Path, mode: str) -> None:
    records = _records()
    first_domain = ALL_DOMAINS[0]
    matching = [record for record in records if record.domain == first_domain]
    remove = matching if mode == "missing" else matching[-1:]
    records = [record for record in records if record not in remove]
    with pytest.raises(ValueError, match="eligible test records"):
        _freeze(tmp_path, _splits(records))


def test_selection_rejects_subject_leakage(tmp_path: Path) -> None:
    test_records = _records()
    leaked = test_records[0]
    train = VolumeRecord(
        case_id="different-record",
        image_path="C:/private/train.nii.gz",
        domain=leaked.domain,
        subject_id=leaked.subject_id,
    )
    splits = VaeSplits(
        train=(train,), validation=(), test=tuple(test_records), seed=13, fractions=(0.1, 0.0, 0.9)
    )
    with pytest.raises(Exception, match="leakage"):
        _freeze(tmp_path, splits)


def test_existing_selection_rejects_seed_or_record_fingerprint_change(tmp_path: Path) -> None:
    _freeze(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        _freeze(tmp_path, seed=99)
    changed_split_seed = VaeSplits(
        train=(),
        validation=(),
        test=tuple(_records()),
        seed=99,
        fractions=(0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="source_split_seed"):
        _freeze(tmp_path, changed_split_seed)
    changed = _records()
    record = changed[0]
    changed[0] = VolumeRecord(
        case_id=record.case_id,
        image_path="C:/private/moved.nii.gz",
        domain=record.domain,
        subject_id=record.subject_id,
    )
    with pytest.raises(ValueError, match="record_fingerprint"):
        _freeze(tmp_path, _splits(changed))


def _supported_target() -> torch.Tensor:
    target = torch.zeros(1, 1, 9, 10, 11)
    target[..., 2:7, 2:8, 2:9] = torch.linspace(0.1, 0.9, 5 * 6 * 7).reshape(1, 1, 5, 6, 7)
    return target


def test_identity_metrics_are_exact_where_mathematically_expected() -> None:
    target = _supported_target()
    metrics = compute_full_volume_metrics(target=target, raw_reconstruction=target)
    for key in (
        "foreground_mae",
        "foreground_nrmse",
        "gradient_mae",
        "background_leakage",
        "signed_foreground_bias",
        "prediction_minus_source_residual_magnitude",
        "high_intensity_tail_mae",
        "high_intensity_tail_signed_bias",
        "foreground_histogram_wasserstein_cdf",
        "raw_fraction_below_zero",
        "raw_fraction_above_one",
    ):
        assert metrics[key] == pytest.approx(0.0, abs=1e-8)
    assert metrics["ssim3d"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["correlation"] == pytest.approx(1.0)
    for quantile in ("q01", "q05", "q50", "q95", "q99"):
        assert metrics[f"quantile_{quantile}_signed_error"] == pytest.approx(0.0)
        assert metrics[f"quantile_{quantile}_absolute_error"] == pytest.approx(0.0)


def test_constant_shifted_metrics_have_explicit_contract() -> None:
    target = torch.full((1, 1, 8, 8, 8), 0.4)
    prediction = torch.full_like(target, 0.5)
    metrics = compute_full_volume_metrics(target=target, raw_reconstruction=prediction)
    assert metrics["foreground_mae"] == pytest.approx(0.1)
    assert metrics["foreground_nrmse"] == pytest.approx(0.1)
    assert metrics["signed_foreground_bias"] == pytest.approx(0.1)
    assert metrics["gradient_mae"] == pytest.approx(0.0)
    assert metrics["correlation"] == 0.0
    assert metrics["correlation_status"] == "constant_undefined_reported_zero"
    assert metrics["background_leakage"] is None
    assert metrics["background_leakage_status"] == "not_available_no_background"
    assert metrics["foreground_histogram_wasserstein_cdf"] == pytest.approx(26 / 256)


def test_quantile_q99_tail_histogram_and_raw_range_contracts() -> None:
    target = _supported_target()
    shifted = target.clone()
    shifted[target > 0] += 0.05
    metrics = compute_full_volume_metrics(target=target, raw_reconstruction=shifted)
    assert metrics["quantile_q50_signed_error"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["high_intensity_tail_mae"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["high_intensity_tail_signed_bias"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["foreground_histogram_wasserstein_cdf"] > 0.0

    raw = target.clone()
    raw[target == 0] = -0.2
    raw[target > 0] = 1.2
    ranged = compute_full_volume_metrics(target=target, raw_reconstruction=raw)
    assert ranged["raw_reconstruction_min"] == pytest.approx(-0.2)
    assert ranged["raw_reconstruction_max"] == pytest.approx(1.2)
    assert ranged["raw_fraction_below_zero"] > 0
    assert ranged["raw_fraction_above_one"] > 0


@pytest.mark.parametrize("size", [3, 4])
def test_linear_quantiles_use_torch_at_or_below_threshold(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_module, "TORCH_QUANTILE_MAX_ELEMENTS", 4)
    monkeypatch.setattr(
        audit_module.np,
        "quantile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("NumPy fallback used")),
    )
    values = torch.linspace(0.1, 0.9, size, dtype=torch.float32)
    quantiles = (0.05, 0.5, 0.95)

    actual = audit_module._linear_quantiles(values, quantiles)

    expected = torch.quantile(values, torch.tensor(quantiles, dtype=torch.float32))
    assert torch.equal(actual, expected)


def test_linear_quantiles_use_numpy_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_module, "TORCH_QUANTILE_MAX_ELEMENTS", 4)
    original_torch_quantile = torch.quantile
    monkeypatch.setattr(
        audit_module.torch,
        "quantile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch path used")),
    )
    values = torch.tensor([0.7, 0.1, 0.9, 0.3, 0.5], dtype=torch.float64)
    quantiles = (0.05, 0.5, 0.95)

    actual = audit_module._linear_quantiles(values, quantiles)

    reference_values = values.float()
    expected = original_torch_quantile(
        reference_values, torch.tensor(quantiles, dtype=torch.float32)
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        expected,
        rtol=torch.finfo(torch.float32).eps,
        atol=0.0,
    )


def test_numpy_linear_quantiles_agree_with_torch_on_reference_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_module, "TORCH_QUANTILE_MAX_ELEMENTS", 1)
    values = torch.tensor([0.1, 0.7, 0.3, 0.9, 0.2, 0.4, 0.8], dtype=torch.float32)
    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99)

    actual = audit_module._linear_quantiles(values, quantiles)
    expected = torch.quantile(values, torch.tensor(quantiles, dtype=torch.float32))

    torch.testing.assert_close(
        actual,
        expected,
        rtol=torch.finfo(torch.float32).eps,
        atol=0.0,
    )


def test_numpy_linear_quantiles_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_module, "TORCH_QUANTILE_MAX_ELEMENTS", 1)
    values = torch.linspace(0.0, 1.0, 257, dtype=torch.float32).flip(0)
    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99)

    first = audit_module._linear_quantiles(values, quantiles)
    second = audit_module._linear_quantiles(values, quantiles)

    assert torch.equal(first, second)


def test_empty_foreground_fails_closed() -> None:
    target = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="foreground mask is empty"):
        compute_full_volume_metrics(target=target, raw_reconstruction=target)


def test_equal_domain_macro_is_not_micro_weighted() -> None:
    rows = [
        {"domain": "a", "metrics": {"foreground_nrmse": 0.0}},
        {"domain": "a", "metrics": {"foreground_nrmse": 0.0}},
        {"domain": "a", "metrics": {"foreground_nrmse": 0.0}},
        {"domain": "b", "metrics": {"foreground_nrmse": 10.0}},
    ]
    _, macro = aggregate_domain_balanced(rows, expected_domains=("a", "b"))
    assert macro["macro_metrics"]["foreground_nrmse"] == 5.0
    assert macro["micro_metrics_secondary"]["foreground_nrmse"] == 2.5


def test_non_divisible_sliding_window_identity_roundtrip() -> None:
    torch.manual_seed(4)
    target = torch.rand(1, 1, 9, 10, 11)
    reconstruction = sliding_window_reconstruct(
        _IdentityEncoder(),
        _IdentityDecoder(),
        target,
        patch_size=(4, 5, 6),
        domain=None,
        overlap=0.5,
        clamp_output=False,
    )
    assert torch.allclose(reconstruction, target, atol=1e-6)


def test_recovery_is_deterministic_and_rejects_incompatible_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _freeze(tmp_path)
    runtime = AuditRuntime(patch_size=(5, 6, 7), overlap=0.5)
    root = prepare_audit_root(
        tmp_path / "audit",
        selection=selection,
        audit_commit="audit-commit",
        config_sha256="c" * 64,
        runtime=runtime,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(audit_module, "render_audit_panel", lambda *args, path, **kwargs: path.write_bytes(b"panel"))

    def loader(record: VolumeRecord) -> torch.Tensor:
        return _supported_target()[0]

    checkpoint_dir = tmp_path / "audit" / "checkpoints" / "checkpoint-01"
    kwargs = dict(
        encoder=_IdentityEncoder(),
        decoder=_IdentityDecoder(),
        volume_loader=loader,
        selection=selection,
        out_dir=checkpoint_dir,
        checkpoint_slot="checkpoint-01",
        checkpoint_label="identity",
        checkpoint_sha256="a" * 64,
        checkpoint_metadata={"training_commit": "training"},
        root_contract=root,
        runtime=runtime,
        device=torch.device("cpu"),
        progress_path=tmp_path / "audit" / "run_progress_sanitized.json",
    )
    first = audit_stage1_checkpoint(**kwargs)
    serialized = (checkpoint_dir / "per_volume_metrics.jsonl").read_bytes()

    kwargs["volume_loader"] = lambda record: (_ for _ in ()).throw(AssertionError("recomputed"))
    second = audit_stage1_checkpoint(**kwargs, resume=True)
    assert first["macro_metrics"] == second["macro_metrics"]
    assert serialized == (checkpoint_dir / "per_volume_metrics.jsonl").read_bytes()

    state_path = checkpoint_dir / ".audit_state" / "domain-01-case-01.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["checkpoint_contract_sha256"] = "bad"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible fingerprint"):
        audit_stage1_checkpoint(**kwargs, resume=True)


def test_interrupted_recovery_reuses_18_volumes_and_resumes_at_domain_05_case_03(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _freeze(tmp_path)
    runtime = AuditRuntime(patch_size=(5, 6, 7), overlap=0.5)
    root = prepare_audit_root(
        tmp_path / "audit",
        selection=selection,
        audit_commit="audit-commit",
        config_sha256="c" * 64,
        runtime=runtime,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        audit_module,
        "render_audit_panel",
        lambda *args, path, **kwargs: path.write_bytes(b"panel"),
    )
    checkpoint_dir = tmp_path / "audit" / "checkpoints" / "checkpoint-01"
    loaded_case_slots: list[str] = []
    record_to_slot = {
        str(item["record_id"]): str(item["case_slot"]) for item in selection["selected"]
    }

    def interrupted_loader(record: VolumeRecord) -> torch.Tensor:
        case_slot = record_to_slot[str(record.case_id)]
        loaded_case_slots.append(case_slot)
        if case_slot == "domain-05-case-03":
            raise RuntimeError("synthetic interruption")
        return _supported_target()[0]

    kwargs = dict(
        encoder=_IdentityEncoder(),
        decoder=_IdentityDecoder(),
        volume_loader=interrupted_loader,
        selection=selection,
        out_dir=checkpoint_dir,
        checkpoint_slot="checkpoint-01",
        checkpoint_label="identity",
        checkpoint_sha256="a" * 64,
        checkpoint_metadata={"training_commit": "training"},
        root_contract=root,
        runtime=runtime,
        device=torch.device("cpu"),
        progress_path=tmp_path / "audit" / "run_progress_sanitized.json",
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        audit_stage1_checkpoint(**kwargs)

    completed = sorted((checkpoint_dir / ".audit_state").glob("*.json"))
    assert len(completed) == 18
    assert loaded_case_slots[-1] == "domain-05-case-03"

    resumed_case_slots: list[str] = []

    def resumed_loader(record: VolumeRecord) -> torch.Tensor:
        resumed_case_slots.append(record_to_slot[str(record.case_id)])
        return _supported_target()[0]

    kwargs["volume_loader"] = resumed_loader
    result = audit_stage1_checkpoint(**kwargs, resume=True)

    assert resumed_case_slots[0] == "domain-05-case-03"
    assert len(resumed_case_slots) == 42
    assert result["volume_count"] == 60
    for path in (tmp_path / "audit").rglob("*.json*"):
        text = path.read_text(encoding="utf-8")
        assert "C:/private" not in text
        assert "private-record" not in text
        assert "private-subject" not in text


def test_non_resume_refuses_existing_audit_and_outputs_are_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _freeze(tmp_path)
    runtime = AuditRuntime(patch_size=(5, 6, 7))
    root = prepare_audit_root(
        tmp_path / "audit",
        selection=selection,
        audit_commit="audit-commit",
        config_sha256="c" * 64,
        runtime=runtime,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(audit_module, "render_audit_panel", lambda *args, path, **kwargs: path.write_bytes(b"panel"))
    checkpoint_dir = tmp_path / "audit" / "checkpoints" / "checkpoint-01"
    kwargs = dict(
        encoder=_IdentityEncoder(),
        decoder=_IdentityDecoder(),
        volume_loader=lambda record: _supported_target()[0],
        selection=selection,
        out_dir=checkpoint_dir,
        checkpoint_slot="checkpoint-01",
        checkpoint_label="identity",
        checkpoint_sha256="a" * 64,
        checkpoint_metadata={"training_commit": "training"},
        root_contract=root,
        runtime=runtime,
        device=torch.device("cpu"),
        progress_path=tmp_path / "audit" / "run_progress_sanitized.json",
    )
    audit_stage1_checkpoint(**kwargs)
    with pytest.raises(FileExistsError):
        audit_stage1_checkpoint(**kwargs)
    for path in (tmp_path / "audit").rglob("*.json*"):
        text = path.read_text(encoding="utf-8")
        assert "C:/private" not in text
        assert "private-record" not in text
        assert "private-subject" not in text
