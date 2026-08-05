"""Explicit paired supervision from the travellers, and the cohort trap it must not fall into.

The official data description gives retrospective IDs disjoint per field strength and
prospective IDs where the same number IS the same volunteer at every field. So same-subject
cross-field pairs may only ever be built from the prospective cohort; doing it on the bare
subject_id would fabricate "pairs" out of two unrelated retrospective people.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fieldbridge.data.domains import Contrast
from fieldbridge.data.latent_bank_dataset import LatentBankIndex, LatentStats
from fieldbridge.training.stage2_transport import (
    Stage2TransportConfig,
    _FieldPools,
    _sample_constrained_pair,
    _validate_config,
)

FIELDS = (0.1, 7.0)


def _bank(root: Path, *, travellers: int, retrospective: int) -> Path:
    bank = root / "bank"
    (bank / "train").mkdir(parents=True, exist_ok=True)
    records = []
    generator = torch.Generator().manual_seed(3)

    def write(case_id: str, subject: str, field: float, latent: torch.Tensor) -> None:
        path = bank / "train" / f"{case_id}.pt"
        torch.save({"case_id": case_id, "subject_id": subject, "split": "train",
                    "domain": {"field_strength_t": field, "contrast": "T1w"},
                    "latent": latent}, path)
        records.append({"case_id": case_id, "subject_id": subject, "split": "train",
                        "domain": {"field_strength_t": field, "contrast": "T1w"},
                        "path": f"train/{case_id}.pt"})

    for subject in range(travellers):
        anatomy = torch.randn(2, 4, 4, 4, generator=generator)
        for field in FIELDS:
            write(f"P_T1W_{field}T_{subject:04d}", f"{subject:04d}", field, anatomy + 0.01 * field)

    # Retrospective volunteers reuse the SAME numeric ids at BOTH fields — different people.
    for subject in range(retrospective):
        for field in FIELDS:
            write(f"R_T1W_{field}T_{subject:04d}", f"{subject:04d}", field,
                  torch.randn(2, 4, 4, 4, generator=generator))

    (bank / "latent_bank_manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    return bank


def _stats() -> LatentStats:
    return LatentStats(mean=torch.zeros(2), std=torch.ones(2))


def test_paired_pool_counts_only_prospective_subjects() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        index = LatentBankIndex(_bank(Path(tmp), travellers=2, retrospective=5), "train")
        pools = _FieldPools.from_index(index)

        # 2 travellers x 2 ordered field pairs = 4. The 5 retrospective numbers contribute 0.
        assert pools.paired_pair_count() == 4
        assert set(pools.paired_by_transition) == {
            (Contrast.parse("T1w"), 0.1, 7.0), (Contrast.parse("T1w"), 7.0, 0.1)
        }


def test_paired_pool_is_empty_when_no_traveller_is_in_the_split(tmp_path: Path) -> None:
    index = LatentBankIndex(_bank(tmp_path, travellers=0, retrospective=6), "train")

    assert _FieldPools.from_index(index).paired_pair_count() == 0


def test_paired_fraction_one_always_draws_true_same_subject_pairs(tmp_path: Path) -> None:
    index = LatentBankIndex(_bank(tmp_path, travellers=3, retrospective=4), "train")
    pools = _FieldPools.from_index(index)
    cfg = Stage2TransportConfig(
        coupling="independent", batch_size=4, same_contrast=True, paired_fraction=1.0
    )
    subject_of = {i: (str(r.case_id)[:1], r.subject_id) for i, r in enumerate(index.records)}
    positions = {
        (str(r.case_id)[:1], r.subject_id, float(r.domain.field_strength_t)): i
        for i, r in enumerate(index.records)
    }

    for seed in range(5):
        z0, dom_s, z1, dom_t, already_paired = _sample_constrained_pair(
            index, pools, _stats(), cfg, torch.device("cpu"), torch.Generator().manual_seed(seed)
        )
        assert already_paired
        # Every element must be the SAME prospective subject at the two fields.
        for i in range(4):
            fs, ft = dom_s[i].field_strength_t, dom_t[i].field_strength_t
            assert fs != ft
            matched = [
                subject
                for cohort, subject, field in positions
                if cohort == "P" and field == fs
                and torch.allclose(
                    z0[i], _stats().normalize(index.load_latent(positions[("P", subject, fs)])),
                    atol=1e-5,
                )
                and torch.allclose(
                    z1[i], _stats().normalize(index.load_latent(positions[("P", subject, ft)])),
                    atol=1e-5,
                )
            ]
            assert matched, "paired draw returned a cross-subject pair"


def test_paired_fraction_zero_never_uses_the_paired_pool(tmp_path: Path) -> None:
    index = LatentBankIndex(_bank(tmp_path, travellers=3, retrospective=4), "train")
    pools = _FieldPools.from_index(index)
    cfg = Stage2TransportConfig(
        coupling="independent", batch_size=4, same_contrast=True, paired_fraction=0.0
    )

    for seed in range(5):
        *_, already_paired = _sample_constrained_pair(
            index, pools, _stats(), cfg, torch.device("cpu"), torch.Generator().manual_seed(seed)
        )
        assert not already_paired


def test_unpaired_batches_still_get_their_ot_assignment(tmp_path: Path) -> None:
    """With 0 < paired_fraction < 1 the OT permutation must stay per-batch, not be disabled."""

    index = LatentBankIndex(_bank(tmp_path, travellers=1, retrospective=6), "train")
    pools = _FieldPools.from_index(index)
    cfg = Stage2TransportConfig(
        coupling="ot", batch_size=4, same_contrast=True, ot_pool_size=2, paired_fraction=0.5
    )

    flags = {
        _sample_constrained_pair(
            index, pools, _stats(), cfg, torch.device("cpu"), torch.Generator().manual_seed(seed)
        )[-1]
        for seed in range(12)
    }
    assert flags == {True, False}


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rejects_an_out_of_range_fraction(value: float) -> None:
    with pytest.raises(ValueError, match="paired_fraction must be in"):
        _validate_config(Stage2TransportConfig(paired_fraction=value))


def test_paired_fraction_requires_same_contrast() -> None:
    with pytest.raises(ValueError, match="needs same_contrast"):
        _validate_config(Stage2TransportConfig(paired_fraction=0.5, same_contrast=False, coupling="ot"))
