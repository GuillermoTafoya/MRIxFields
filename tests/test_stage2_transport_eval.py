from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.latent_bank_dataset import LatentStats
from fieldbridge.evaluation.stage2_transport_eval import (
    DecodeSpec,
    TransportSamplerConfig,
    evaluate_transport_travellers,
    sample_transport,
)

C, X, FACTOR = 4, 4, 2


class _ConstantField(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, z, source_domain, target_domain, t):  # noqa: ANN001
        return self.velocity.expand_as(z)


class _StubDecoder(nn.Module):
    downsample_factor = FACTOR
    latent_channels = C

    def decode(self, latent: torch.Tensor, domain) -> torch.Tensor:  # noqa: ANN001
        image = latent.mean(dim=1, keepdim=True)
        return F.interpolate(image, scale_factor=self.downsample_factor, mode="nearest").clamp(0.0, 1.0)


def _stub_metric(prediction, target, *, metrics, device):  # noqa: ANN001
    err = float(torch.sqrt(torch.mean((prediction - target) ** 2)))
    return {name: err for name in metrics}


def test_sample_transport_recovers_constant_field() -> None:
    z0 = torch.randn(1, C, X, X, X)
    velocity = torch.randn(1, C, X, X, X)
    field = _ConstantField(velocity)
    dom = Domain(0.1, Contrast.T1W)
    dom_t = Domain(3.0, Contrast.T1W)
    for solver in ("euler", "heun"):
        z1 = sample_transport(field, z0, dom, dom_t, TransportSamplerConfig(solver=solver, n_steps=8))
        assert torch.allclose(z1, z0 + velocity, atol=1e-4)  # integral of a constant field over [0,1]


def _make_traveller_bank(tmp_path, subject="0006", contrast=Contrast.T1W, fields=(0.1, 3.0)):
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    records, manifest_records = [], []
    for field in fields:
        case_id = f"P_{contrast.name}_{field}_{subject}"
        torch.save({"case_id": case_id, "latent": torch.rand(C, X, X, X, dtype=torch.float16)},
                   split_dir / f"{case_id}.pt")
        manifest_records.append({"case_id": case_id, "path": f"train/{case_id}.pt"})
        records.append(VolumeRecord(
            case_id=case_id, image_path=f"/nonexistent/{case_id}.nii.gz",
            domain=Domain(field, contrast), subject_id=subject, split="Training_prospective",
            metadata={"prefix": "P"},
        ))
    (tmp_path / "latent_stats.json").write_text(
        json.dumps({"per_channel_mean": [0.0] * C, "per_channel_std": [1.0] * C}), encoding="utf-8"
    )
    manifest = {"records": manifest_records}
    return records, manifest


def test_evaluate_transport_travellers_zero_field_matches_identity(tmp_path) -> None:
    records, manifest = _make_traveller_bank(tmp_path)
    stats = LatentStats.from_json(tmp_path / "latent_stats.json")
    fixed_target = torch.rand(1, 1, X * FACTOR, X * FACTOR, X * FACTOR)

    result = evaluate_transport_travellers(
        translator=_ConstantField(torch.zeros(1, C, X, X, X)),  # zero velocity -> transport == source decode
        decoder=_StubDecoder(),
        records=records,
        bank_manifest=manifest,
        bank_dir=tmp_path,
        stats=stats,
        sampler=TransportSamplerConfig(solver="heun", n_steps=4),
        decode=DecodeSpec(precision="float32"),
        device=torch.device("cpu"),
        metrics=("ssim", "nrmse"),
        metric_fn=_stub_metric,
        volume_loader=lambda record: fixed_target.clone(),
        log=False,
    )

    assert result["num_pairs"] == 2  # (0.1->3.0) and (3.0->0.1)
    assert result["subjects"] == ["0006"]
    for row in result["pairs"]:
        # zero-velocity transport decodes the source latent, exactly like the identity baseline.
        assert row["transport"] == pytest.approx(row["identity"])
        for method in ("transport", "identity", "ceiling"):
            assert all(torch.isfinite(torch.tensor(v)) for v in row[method].values())
    assert set(result["overall"]) == {"transport", "identity", "ceiling"}


def test_evaluate_transport_raises_without_travellers(tmp_path) -> None:
    records, manifest = _make_traveller_bank(tmp_path)
    stats = LatentStats.from_json(tmp_path / "latent_stats.json")
    with pytest.raises(ValueError, match="No traveller"):
        evaluate_transport_travellers(
            translator=_ConstantField(torch.zeros(1, C, X, X, X)),
            decoder=_StubDecoder(), records=records, bank_manifest=manifest, bank_dir=tmp_path,
            stats=stats, sampler=TransportSamplerConfig(), decode=DecodeSpec(precision="float32"),
            device=torch.device("cpu"), metrics=("nrmse",), subjects=["9999"],
            metric_fn=_stub_metric, volume_loader=lambda record: torch.zeros(1, 1, 8, 8, 8), log=False,
        )


def test_eval_and_resplit_commands_exposed_in_help() -> None:
    from fieldbridge.cli import build_parser

    for command in ("eval-stage2-transport", "resplit-travellers"):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args([command, "--help"])
        assert exc_info.value.code == 0
