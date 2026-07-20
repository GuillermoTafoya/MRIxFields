"""Tests for the Etapa 1 VAE subject-level split (data/vae_splits.py)."""

from __future__ import annotations

import pytest

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.vae_splits import (
    build_vae_splits,
    audit_vae_splits,
    load_vae_splits,
    save_vae_splits,
    summarize_vae_splits,
    vae_splits_fingerprint,
)
from fieldbridge.data.volume_splits import VolumeSplitError

_FIELDS = [0.1, 1.5, 3.0, 5.0, 7.0]
_CONTRASTS = ["T1W", "T2W", "T2FLAIR"]


def _pool(retro_per_domain: int = 12, travellers: int = 5) -> list[VolumeRecord]:
    """Retrospective (unique subject per domain) + prospective travellers (one subject in
    all 15 domains), like the official Training pool."""

    records: list[VolumeRecord] = []
    for field in _FIELDS:
        for contrast in _CONTRASTS:
            for s in range(retro_per_domain):
                # Collision-free per (field, contrast, s) — real retrospective IDs are unique.
                sid = f"R{int(field * 10):03d}{contrast}{s:02d}"
                records.append(
                    VolumeRecord(
                        case_id=f"R_{contrast}_{field}_{sid}",
                        image_path=f"/d/R/{contrast}/{field}/{sid}.nii.gz",
                        domain={"field_strength_t": field, "contrast": contrast},
                        subject_id=sid,
                        metadata={"prefix": "R"},
                    )
                )
    for t in range(travellers):
        sid = f"{t:04d}"
        for field in _FIELDS:
            for contrast in _CONTRASTS:
                records.append(
                    VolumeRecord(
                        case_id=f"P_{contrast}_{field}_{sid}",
                        image_path=f"/d/P/{contrast}/{field}/{sid}.nii.gz",
                        domain={"field_strength_t": field, "contrast": contrast},
                        subject_id=sid,
                        metadata={"prefix": "P"},
                    )
                )
    return records


def test_split_is_leakage_free_and_covers_all_records() -> None:
    splits = build_vae_splits(_pool(), seed=13)
    assert audit_vae_splits(splits).ok
    total = len(splits.train) + len(splits.validation) + len(splits.test)
    assert total == len(_pool())


def test_travellers_never_straddle_splits() -> None:
    """Every volume of a prospective traveller (same subject_id across 15 domains) must
    land in exactly one split, or the held-out sets leak the subject."""

    splits = build_vae_splits(_pool(), seed=13)
    location: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        for record in splits.records_for(name):
            if record.metadata.get("prefix") == "P":
                location.setdefault(record.subject_id, set()).add(name)
    assert location, "expected prospective travellers in the pool"
    assert all(len(v) == 1 for v in location.values())


def test_split_is_deterministic_for_a_seed() -> None:
    a = build_vae_splits(_pool(), seed=13)
    b = build_vae_splits(_pool(), seed=13)
    assert vae_splits_fingerprint(a) == vae_splits_fingerprint(b)


def test_different_seed_changes_the_split() -> None:
    a = build_vae_splits(_pool(), seed=13)
    b = build_vae_splits(_pool(), seed=99)
    assert vae_splits_fingerprint(a) != vae_splits_fingerprint(b)


def test_validation_and_test_cover_every_domain() -> None:
    """With enough subjects per domain, stratification puts each of the 15 domains in
    validation (0.1T in particular — the field whose held-out score matters most)."""

    splits = build_vae_splits(_pool(retro_per_domain=20), seed=13)
    val_domains = summarize_vae_splits(splits)["splits"]["validation"]["per_domain"]
    assert "0.1T/T1w" in val_domains
    assert len(val_domains) == len(_FIELDS) * len(_CONTRASTS)


def test_roundtrip_save_load_preserves_split() -> None:
    import tempfile
    from pathlib import Path

    splits = build_vae_splits(_pool(), seed=7)
    path = Path(tempfile.mkdtemp()) / "split.json"
    save_vae_splits(splits, path)
    reloaded = load_vae_splits(path)
    assert vae_splits_fingerprint(reloaded) == vae_splits_fingerprint(splits)
    assert reloaded.seed == splits.seed
    assert reloaded.fractions == splits.fractions


def test_empty_records_rejected() -> None:
    with pytest.raises(VolumeSplitError):
        build_vae_splits([], seed=13)


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(VolumeSplitError, match="sum to 1"):
        build_vae_splits(_pool(), train_frac=0.8, val_frac=0.1, test_frac=0.2)


def test_non_official_records_fall_back_to_case_id_without_leakage() -> None:
    """A manifest without prefix/subject_id metadata still splits (each record its own
    subject) rather than crashing."""

    records = [
        VolumeRecord(
            case_id=f"c{i}",
            image_path=f"/d/c{i}.nii.gz",
            domain={"field_strength_t": 3.0, "contrast": "T1W"},
        )
        for i in range(30)
    ]
    splits = build_vae_splits(records, seed=13)
    assert audit_vae_splits(splits).ok
    assert len(splits.train) + len(splits.validation) + len(splits.test) == 30
