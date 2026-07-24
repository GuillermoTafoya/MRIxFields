from __future__ import annotations

import json

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.latent_bank_dataset import LatentBankIndex, LatentStats
from fieldbridge.models.translators.flow_transport import FlowMatchingLatentTranslator
from fieldbridge.training.stage2_transport import (
    Stage2TransportConfig,
    _bridge_sample,
    _ot_assignment,
    run_stage2_transport_train,
)

C, X = 4, 8


def _make_bank(tmp_path, per_split=6):
    domains = [Domain(0.1, Contrast.T1W), Domain(3.0, Contrast.T2W), Domain(7.0, Contrast.T2_FLAIR)]
    records = []
    for split in ("train", "validation"):
        split_dir = tmp_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            domain = domains[i % len(domains)]
            latent = torch.randn(C, X, X, X, dtype=torch.float16)
            case_id = f"{split}_{i}"
            path = split_dir / f"{case_id}.pt"
            torch.save(
                {"case_id": case_id, "subject_id": f"s{i}", "split": split,
                 "domain": domain.to_dict(), "latent": latent}, path
            )
            records.append(
                {"case_id": case_id, "subject_id": f"s{i}", "split": split,
                 "domain": domain.to_dict(), "latent_shape": [C, X, X, X],
                 "source_shape": [1, X * 4, X * 4, X * 4], "path": f"{split}/{case_id}.pt"}
            )
    (tmp_path / "latent_bank_manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (tmp_path / "latent_stats.json").write_text(
        json.dumps({"per_channel_mean": [0.0] * C, "per_channel_std": [1.0] * C, "computed_over": "train"}),
        encoding="utf-8",
    )
    return tmp_path


def _model():
    return FlowMatchingLatentTranslator(
        latent_channels=C, hidden_channels=(8, 16), bottleneck_channels=16,
        cond_dim=16, time_embed_dim=16, spatial_dims=3,
    )


def test_ot_assignment_recovers_optimal_permutation() -> None:
    # z1 is a shuffled copy of z0; the OT plan should undo the shuffle.
    z0 = torch.randn(4, C, X, X, X)
    perm = torch.tensor([2, 0, 3, 1])
    z1 = z0[perm]
    recovered = _ot_assignment(z0, z1)
    # applying the recovered column order to z1 should reproduce z0.
    assert torch.allclose(z1[recovered], z0, atol=1e-5)


def test_bridge_ot_cfm_endpoints_and_target() -> None:
    z0 = torch.randn(3, C, X, X, X)
    z1 = torch.randn(3, C, X, X, X)
    cfg = Stage2TransportConfig(bridge="ot_cfm")
    gen = torch.Generator().manual_seed(0)
    z_at0, target0 = _bridge_sample(z0, z1, torch.zeros(3), cfg, gen)
    z_at1, _ = _bridge_sample(z0, z1, torch.ones(3), cfg, gen)
    assert torch.allclose(z_at0, z0, atol=1e-6)
    assert torch.allclose(z_at1, z1, atol=1e-6)
    assert torch.allclose(target0, z1 - z0, atol=1e-6)


def test_bridge_schrodinger_target_matches_drift_formula() -> None:
    z0 = torch.randn(3, C, X, X, X)
    z1 = torch.randn(3, C, X, X, X)
    cfg = Stage2TransportConfig(bridge="schrodinger", sigma=0.2, time_eps=1e-3)
    t = torch.full((3,), 0.4)
    gen = torch.Generator().manual_seed(1)
    z_t, target = _bridge_sample(z0, z1, t, cfg, gen)
    t_b = t.reshape(-1, 1, 1, 1, 1)
    expected = (z1 - z_t) / (1.0 - t_b)
    assert torch.allclose(target, expected, atol=1e-5)


def test_transport_training_runs_and_checkpoints(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    train_index = LatentBankIndex(bank, "train")
    val_index = LatentBankIndex(bank, "validation")
    stats = LatentStats.from_json(bank / "latent_stats.json")
    ckpt_dir = tmp_path / "ckpt"
    cfg = Stage2TransportConfig(
        steps=12, batch_size=4, precision="fp32", coupling="ot", bridge="ot_cfm",
        checkpoint_dir=ckpt_dir, checkpoint_at_end=True, val_every_steps=6, val_batches=2,
        variant="fm_test",
    )
    result = run_stage2_transport_train(
        cfg, translator=_model(), train_index=train_index, stats=stats, val_index=val_index
    )
    assert result.steps == 12
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert result.best_val is not None
    assert (ckpt_dir / "transport_fm_test_last.pt").exists()
    assert (ckpt_dir / "transport_fm_test_best.pt").exists()


def test_transport_training_resumes(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    train_index = LatentBankIndex(bank, "train")
    stats = LatentStats.from_json(bank / "latent_stats.json")
    ckpt_dir = tmp_path / "ckpt"
    base = dict(precision="fp32", coupling="independent", bridge="ot_cfm",
                checkpoint_dir=ckpt_dir, variant="fm_test")
    run_stage2_transport_train(
        Stage2TransportConfig(steps=6, batch_size=4, **base),
        translator=_model(), train_index=train_index, stats=stats,
    )
    resumed = run_stage2_transport_train(
        Stage2TransportConfig(
            steps=4, batch_size=4, resume_from=ckpt_dir / "transport_fm_test_last.pt", **base
        ),
        translator=_model(), train_index=train_index, stats=stats,
    )
    assert resumed.steps == 4


def test_cycle_weight_is_guarded(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    train_index = LatentBankIndex(bank, "train")
    stats = LatentStats.from_json(bank / "latent_stats.json")
    cfg = Stage2TransportConfig(
        steps=1, batch_size=2, precision="fp32",
        loss_weights={"flow": 1.0, "transport_cost": 0.0, "identity": 0.0, "cycle": 1.0},
    )
    with pytest.raises(NotImplementedError, match="cycle loss"):
        run_stage2_transport_train(cfg, translator=_model(), train_index=train_index, stats=stats)


def test_train_transport_command_is_exposed_in_help() -> None:
    from fieldbridge.cli import build_parser

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["train-stage2-transport", "--help"])
    assert exc_info.value.code == 0
