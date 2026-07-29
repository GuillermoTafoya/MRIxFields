"""Global nearest-neighbour coupling, zero-init velocity, and the decode-path contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.latent_bank_dataset import LatentBankIndex, LatentStats
from fieldbridge.evaluation.stage2_transport_eval import DecodeSpec
from fieldbridge.models.translators.flow_transport import FlowMatchingLatentTranslator
from fieldbridge.training.stage2_transport import (
    Stage2TransportConfig,
    _FieldPools,
    _sample_constrained_pair,
    build_latent_descriptors,
)

CONTRASTS = ("T1w", "T2w")
FIELDS = (0.1, 7.0)


def _make_bank(root: Path, *, per_cell: int = 6, latent_shape=(2, 4, 4, 4)) -> Path:
    """A bank whose latents carry a per-record 'anatomy' signature the NN cost can find."""

    bank = root / "bank"
    (bank / "train").mkdir(parents=True)
    records = []
    generator = torch.Generator().manual_seed(7)
    for contrast in CONTRASTS:
        for subject in range(per_cell):
            anatomy = torch.randn(latent_shape, generator=generator)
            for field in FIELDS:
                case_id = f"P_{contrast}_{subject}_{field}"
                # Same subject => same anatomy, plus a field-specific offset. The globally
                # nearest target of a source is therefore its own subject at the other field.
                latent = anatomy + 0.01 * field
                path = bank / "train" / f"{case_id}.pt"
                torch.save({"case_id": case_id, "subject_id": str(subject), "split": "train",
                            "domain": {"field_strength_t": field, "contrast": contrast},
                            "latent": latent}, path)
                records.append({"case_id": case_id, "subject_id": str(subject), "split": "train",
                                "domain": {"field_strength_t": field, "contrast": contrast},
                                "path": f"train/{case_id}.pt"})
    (bank / "latent_bank_manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    return bank


def _stats(channels: int = 2) -> LatentStats:
    return LatentStats(mean=torch.zeros(channels), std=torch.ones(channels))


def test_descriptors_are_cached_and_the_cache_is_invalidated_by_pool_size(tmp_path: Path) -> None:
    index = LatentBankIndex(_make_bank(tmp_path), "train")
    cache = tmp_path / "desc.pt"

    first = build_latent_descriptors(index, _stats(), pool_size=2, cache_path=cache)
    assert cache.exists()
    assert torch.equal(build_latent_descriptors(index, _stats(), pool_size=2, cache_path=cache), first)

    other = build_latent_descriptors(index, _stats(), pool_size=4, cache_path=cache)
    assert other.shape[1] != first.shape[1]


def test_nn_coupling_pairs_a_source_with_its_own_subject(tmp_path: Path) -> None:
    """The whole point: the target must be the anatomically closest volume, not a random one."""

    bank = _make_bank(tmp_path)
    index = LatentBankIndex(bank, "train")
    pools = _FieldPools.from_index(index).with_nn_tables(
        build_latent_descriptors(index, _stats(), pool_size=2), candidates=1
    )
    subject_of = {i: record.subject_id for i, record in enumerate(index.records)}

    table = pools.nn_tables[(Contrast.parse("T1w"), 0.1, 7.0)]
    sources = pools.by_contrast_field[Contrast.parse("T1w")][0.1]
    for row, source_position in enumerate(sources):
        assert subject_of[int(table[row, 0])] == subject_of[source_position]


def test_nn_batch_is_paired_elementwise_and_stays_in_the_transition(tmp_path: Path) -> None:
    bank = _make_bank(tmp_path)
    index = LatentBankIndex(bank, "train")
    pools = _FieldPools.from_index(index).with_nn_tables(
        build_latent_descriptors(index, _stats(), pool_size=2), candidates=1
    )
    cfg = Stage2TransportConfig(coupling="nn", batch_size=4, same_contrast=True, ot_pool_size=2)

    z0, dom_s, z1, dom_t = _sample_constrained_pair(
        index, pools, _stats(), cfg, torch.device("cpu"), torch.Generator().manual_seed(0)
    )

    assert z0.shape == z1.shape == (4, 2, 4, 4, 4)
    assert {d.contrast for d in dom_s} == {d.contrast for d in dom_t}
    assert len({d.field_strength_t for d in dom_s}) == 1
    assert len({d.field_strength_t for d in dom_t}) == 1
    assert dom_s[0].field_strength_t != dom_t[0].field_strength_t
    # Paired on the same subject => the displacement is the field offset only, not anatomy.
    assert torch.allclose(z1 - z0, (z1 - z0)[0].expand_as(z1), atol=1e-5)


def test_nn_candidates_greater_than_one_does_not_always_return_the_same_target(tmp_path: Path) -> None:
    bank = _make_bank(tmp_path, per_cell=8)
    index = LatentBankIndex(bank, "train")
    pools = _FieldPools.from_index(index).with_nn_tables(
        build_latent_descriptors(index, _stats(), pool_size=2), candidates=3
    )
    cfg = Stage2TransportConfig(coupling="nn", batch_size=8, same_contrast=True, ot_pool_size=2)

    seen = set()
    for seed in range(6):
        _, _, z1, _ = _sample_constrained_pair(
            index, pools, _stats(), cfg, torch.device("cpu"), torch.Generator().manual_seed(seed)
        )
        seen.add(float(z1.sum()))
    assert len(seen) > 1


def test_nn_without_tables_fails_closed(tmp_path: Path) -> None:
    index = LatentBankIndex(_make_bank(tmp_path), "train")
    pools = _FieldPools.from_index(index)
    cfg = Stage2TransportConfig(coupling="nn", batch_size=2, same_contrast=True, ot_pool_size=2)

    with pytest.raises(ValueError, match="needs the precomputed neighbour tables"):
        _sample_constrained_pair(
            index, pools, _stats(), cfg, torch.device("cpu"), torch.Generator().manual_seed(0)
        )


def test_nn_requires_same_contrast() -> None:
    from fieldbridge.training.stage2_transport import _validate_config

    with pytest.raises(ValueError, match="requires same_contrast"):
        _validate_config(Stage2TransportConfig(coupling="nn", same_contrast=False))


def test_zero_init_makes_the_untrained_transport_the_identity() -> None:
    translator = FlowMatchingLatentTranslator(
        latent_channels=2, hidden_channels=(8,), bottleneck_channels=16,
        cond_dim=8, time_embed_dim=8, spatial_dims=3, zero_init_output=True,
    ).eval()
    z = torch.randn(1, 2, 8, 8, 8)
    domain_s, domain_t = Domain(field_strength_t=0.1, contrast="T1w"), Domain(field_strength_t=7.0, contrast="T1w")

    with torch.no_grad():
        velocity = translator(z, [domain_s], [domain_t], torch.tensor([0.3]))

    assert torch.count_nonzero(velocity) == 0


def test_zero_init_can_be_turned_off() -> None:
    translator = FlowMatchingLatentTranslator(
        latent_channels=2, hidden_channels=(8,), bottleneck_channels=16,
        cond_dim=8, time_embed_dim=8, spatial_dims=3, zero_init_output=False,
    ).eval()

    assert torch.count_nonzero(translator.output_projection.weight) > 0


def test_decode_spec_does_not_inherit_the_encode_halo() -> None:
    """The run-C gate regression: a `--halo 16` meant for an unused tiled *encode* fallback
    was inherited by the decode, halving the decoder's receptive field."""

    manifest = {"config": {"block_size": [128, 128, 128], "halo": [16, 16, 16],
                           "precision": "bfloat16", "strategy": "full"}}

    spec = DecodeSpec.from_bank_manifest(manifest)

    assert spec.halo == (64, 64, 64)
    assert spec.precision == "bfloat16"  # precision IS a shared choice and is still taken
    assert spec.strategy == "auto"


def test_training_runs_end_to_end_with_nn_coupling_and_builds_its_descriptor_cache(
    tmp_path: Path,
) -> None:
    """Covers the wiring unit tests bypass: descriptor build + cache + table attach + steps."""

    from fieldbridge.training.stage2_transport import run_stage2_transport_train

    bank = _make_bank(tmp_path, per_cell=6)
    index = LatentBankIndex(bank, "train")
    translator = FlowMatchingLatentTranslator(
        latent_channels=2, hidden_channels=(8,), bottleneck_channels=16,
        cond_dim=8, time_embed_dim=8, spatial_dims=3, zero_init_output=True,
    )
    cfg = Stage2TransportConfig(
        steps=6, batch_size=4, coupling="nn", same_contrast=True, ot_pool_size=2,
        nn_candidates=3, device="cpu", precision="fp32", checkpoint_dir=None,
        descriptor_cache=tmp_path / "cache",
        loss_weights={"flow": 1.0, "transport_cost": 0.1, "identity": 0.1, "cycle": 0.0},
    )

    result = run_stage2_transport_train(cfg, translator=translator, train_index=index, stats=_stats())

    assert result.steps == 6
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
    assert (tmp_path / "cache" / "descriptors_train_pool2.pt").is_file()


def test_transport_cost_and_identity_terms_are_reported_when_enabled(tmp_path: Path) -> None:
    """The ladder terms must actually enter the loss, not just sit in the config."""

    from fieldbridge.training.stage2_transport import _FieldPools, _transport_loss

    bank = _make_bank(tmp_path)
    index = LatentBankIndex(bank, "train")
    pools = _FieldPools.from_index(index).with_nn_tables(
        build_latent_descriptors(index, _stats(), pool_size=2), candidates=2
    )
    translator = FlowMatchingLatentTranslator(
        latent_channels=2, hidden_channels=(8,), bottleneck_channels=16,
        cond_dim=8, time_embed_dim=8, spatial_dims=3,
    )
    cfg = Stage2TransportConfig(
        batch_size=2, coupling="nn", same_contrast=True, ot_pool_size=2,
        loss_weights={"flow": 1.0, "transport_cost": 0.5, "identity": 0.5, "cycle": 0.0},
    )

    total, terms = _transport_loss(
        translator, index, pools, _stats(), cfg, torch.device("cpu"),
        torch.Generator().manual_seed(0),
    )

    assert {"flow", "transport_cost", "identity"} <= set(terms)
    assert torch.isfinite(total)


def test_decode_spec_overrides_win() -> None:
    spec = DecodeSpec.from_bank_manifest({"config": {}}, strategy="tiled", halo=(32, 32, 32))

    assert spec.strategy == "tiled"
    assert spec.halo == (32, 32, 32)
