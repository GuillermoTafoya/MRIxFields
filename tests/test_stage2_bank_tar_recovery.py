from __future__ import annotations

import io
import json
import os
import tarfile
from collections import namedtuple
from pathlib import Path

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_file


BANK_ARTIFACT_SHA256 = "8" * 64


def _drive_root(tmp_path: Path) -> Path:
    root = tmp_path / "MRIxFields2026"
    if os.name == "nt":
        return Path(chr(92) * 2 + "?" + chr(92) + str(root.resolve()))
    return root


def _synthetic_bank(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    (root / p0006.STAGE2_BANK_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "contract_version": "synthetic-bank-manifest-v1",
                "artifact_sha256": BANK_ARTIFACT_SHA256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "train").mkdir()
    (root / "train" / "record-0001.json").write_text(
        json.dumps({"cohort": "R", "split": "train"}, sort_keys=True),
        encoding="utf-8",
    )
    (root / "train" / "record-0001.bin").write_bytes(b"synthetic-bank-bytes")
    return p0006.stage2_bank_tree_identity(root)


def _reviewed_tar(path: Path, bank: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as bundle:
        bundle.add(bank, arcname="photometry_factored_latent_bank_v2")
    return sha256_file(path)


def _restore(
    archive: Path,
    destination: Path,
    identity: dict[str, object],
    **overrides: object,
) -> p0006.VerifiedStage2BankRestore:
    expected = {
        "expected_archive_sha256": sha256_file(archive),
        "expected_tree_sha256": identity["tree_sha256"],
        "expected_bank_artifact_sha256": BANK_ARTIFACT_SHA256,
        "expected_file_count": identity["file_count"],
        "expected_total_bytes": identity["total_bytes"],
    }
    expected.update(overrides)
    return p0006.restore_verified_stage2_bank_tar(
        archive,
        destination,
        **expected,
    )


def test_real_persisted_topology_restores_reviewed_tar_and_ignores_empty_dir(
    tmp_path: Path,
) -> None:
    drive_root = _drive_root(tmp_path)
    output_root = drive_root / "UnifiedStage2_1ca2b4a_01"
    stage2_root = output_root / "stage2_unified_v7"
    training_namespace = (
        stage2_root / "bank_8081ce89a0ea" / "implementation_82633d66e5ea"
    )
    selection_receipt = (
        training_namespace
        / "unified_full_objective_pilot_200"
        / "scientific_attempts"
        / "attempt-0001"
        / "checkpoints"
        / p0006.STAGE2_RECOVERY_SELECTION_RECEIPT_NAME
    )
    selection_receipt.parent.mkdir(parents=True)
    selection_receipt.write_text("{}", encoding="utf-8")
    pair_feasibility = output_root / "stage2_retrospective_pair_feasibility_v2.json"
    pair_feasibility.write_text("{}", encoding="utf-8")
    empty_unreceipted = output_root / "photometry_factored_latent_bank_v2"
    empty_unreceipted.mkdir()

    source_bank = tmp_path / "source" / "photometry_factored_latent_bank_v2"
    identity = _synthetic_bank(source_bank)
    archive = output_root / "photometry_factored_latent_bank_v2.tar"
    archive_sha = _reviewed_tar(archive, source_bank)
    layout = p0006.resolve_stage2_recovery_drive_layout(drive_root)
    local_bank = drive_root / "local-scratch" / "photometry_factored_latent_bank_v2"
    restored = _restore(layout.bank_archive, local_bank, identity)

    assert layout.output_root == output_root.resolve()
    assert layout.stage2_v7_root == stage2_root.resolve()
    assert layout.training_namespace == training_namespace.resolve()
    assert layout.bank_archive == archive.resolve()
    assert layout.pair_feasibility == pair_feasibility.resolve()
    assert layout.selection_receipt == selection_receipt.resolve()
    assert layout.ignored_empty_unreceipted_bank_directory is True
    assert list(empty_unreceipted.iterdir()) == []
    assert restored.archive_file_sha256 == archive_sha
    assert restored.bank_root == local_bank.resolve()
    assert restored.tree_sha256 == identity["tree_sha256"]
    assert restored.file_count == identity["file_count"]
    assert restored.total_bytes == identity["total_bytes"]
    assert restored.bank_artifact_sha256 == BANK_ARTIFACT_SHA256
    assert restored.restored_from_tar is True


def test_drive_layout_rejects_nonempty_unreceipted_bank_directory(
    tmp_path: Path,
) -> None:
    drive_root = _drive_root(tmp_path)
    output_root = drive_root / p0006.STAGE2_RECOVERY_OUTPUT_ROOT_NAME
    selection_receipt = (
        output_root
        / p0006.STAGE2_RECOVERY_V7_ROOT_NAME
        / p0006.STAGE2_RECOVERY_BANK_NAMESPACE_NAME
        / p0006.STAGE2_RECOVERY_TRAINING_NAMESPACE_NAME
        / "unified_full_objective_pilot_200"
        / "scientific_attempts"
        / "attempt-0001"
        / "checkpoints"
        / p0006.STAGE2_RECOVERY_SELECTION_RECEIPT_NAME
    )
    selection_receipt.parent.mkdir(parents=True)
    selection_receipt.write_text("{}", encoding="utf-8")
    (output_root / p0006.STAGE2_RECOVERY_BANK_TAR_NAME).write_bytes(b"synthetic")
    (output_root / p0006.STAGE2_RECOVERY_PAIR_FEASIBILITY_NAME).write_text(
        "{}", encoding="utf-8"
    )
    unreceipted = output_root / p0006.STAGE2_RECOVERY_UNRECEIPTED_BANK_DIR_NAME
    unreceipted.mkdir()
    (unreceipted / "unsealed.bin").write_bytes(b"must-not-be-discovered")

    with pytest.raises(ValueError, match="not empty"):
        p0006.resolve_stage2_recovery_drive_layout(drive_root)


def test_drive_layout_does_not_guess_old_root_locations(tmp_path: Path) -> None:
    drive_root = tmp_path / "MRIxFields2026"
    (drive_root / "stage2_unified_v7").mkdir(parents=True)
    (drive_root / p0006.STAGE2_RECOVERY_BANK_TAR_NAME).write_bytes(b"synthetic")

    with pytest.raises(FileNotFoundError, match="output root"):
        p0006.resolve_stage2_recovery_drive_layout(drive_root)


def test_exact_local_restore_is_reused_without_clobber(tmp_path: Path) -> None:
    source_bank = tmp_path / "source-bank"
    identity = _synthetic_bank(source_bank)
    archive = tmp_path / "output" / "photometry_factored_latent_bank_v2.tar"
    _reviewed_tar(archive, source_bank)
    destination = tmp_path / "scratch" / "bank"
    first = _restore(archive, destination, identity)
    before = p0006.stage2_bank_tree_identity(destination)
    second = _restore(archive, destination, identity)
    assert first.restored_from_tar is True
    assert second.restored_from_tar is False
    assert p0006.stage2_bank_tree_identity(destination) == before
    assert list(destination.parent.glob("*.partial")) == []


def test_reviewed_tar_file_sha_is_verified_before_extraction(tmp_path: Path) -> None:
    source_bank = tmp_path / "source-bank"
    identity = _synthetic_bank(source_bank)
    archive = tmp_path / "photometry_factored_latent_bank_v2.tar"
    original_sha = _reviewed_tar(archive, source_bank)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tar SHA-256 mismatch"):
        p0006.restore_verified_stage2_bank_tar(
            archive,
            tmp_path / "scratch" / "bank",
            expected_archive_sha256=original_sha,
            expected_tree_sha256=str(identity["tree_sha256"]),
            expected_bank_artifact_sha256=BANK_ARTIFACT_SHA256,
            expected_file_count=int(identity["file_count"]),
            expected_total_bytes=int(identity["total_bytes"]),
        )
    assert not (tmp_path / "scratch" / "bank").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_reviewed_tar_rejects_escape_and_link_members(
    tmp_path: Path, kind: str
) -> None:
    archive = tmp_path / "photometry_factored_latent_bank_v2.tar"
    with tarfile.open(archive, mode="w") as bundle:
        manifest = json.dumps({"artifact_sha256": BANK_ARTIFACT_SHA256}).encode()
        manifest_info = tarfile.TarInfo(
            "photometry_factored_latent_bank_v2/"
            + p0006.STAGE2_BANK_MANIFEST_FILENAME
        )
        manifest_info.size = len(manifest)
        bundle.addfile(manifest_info, io.BytesIO(manifest))
        if kind == "traversal":
            malicious = tarfile.TarInfo("../outside.txt")
            payload = b"escape"
            malicious.size = len(payload)
            bundle.addfile(malicious, io.BytesIO(payload))
        else:
            malicious = tarfile.TarInfo(
                "photometry_factored_latent_bank_v2/link"
            )
            malicious.type = tarfile.SYMTYPE
            malicious.linkname = "../../outside.txt"
            bundle.addfile(malicious)
    expected_message = "path traversal" if kind == "traversal" else "link or special"
    with pytest.raises(ValueError, match=expected_message):
        p0006.restore_verified_stage2_bank_tar(
            archive,
            tmp_path / "scratch" / "bank",
            expected_archive_sha256=sha256_file(archive),
            expected_tree_sha256="1" * 64,
            expected_bank_artifact_sha256=BANK_ARTIFACT_SHA256,
            expected_file_count=1,
            expected_total_bytes=1,
        )
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_tree_sha256": "1" * 64}, "tree count, byte total, or SHA"),
        ({"expected_file_count": 999}, "tree count, byte total, or SHA"),
        ({"expected_total_bytes": 999}, "tree count, byte total, or SHA"),
        ({"expected_bank_artifact_sha256": "2" * 64}, "artifact SHA-256"),
    ],
)
def test_reviewed_tar_fails_closed_on_wrong_sealed_identity(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    source_bank = tmp_path / "source-bank"
    identity = _synthetic_bank(source_bank)
    archive = tmp_path / "photometry_factored_latent_bank_v2.tar"
    _reviewed_tar(archive, source_bank)
    with pytest.raises(ValueError, match=message):
        _restore(archive, tmp_path / "scratch" / "bank", identity, **override)


def test_local_copy_hash_progress_extract_progress_and_identity_are_exact(
    tmp_path: Path,
) -> None:
    test_root = _drive_root(tmp_path)
    source_bank = test_root / "source-bank"
    identity = _synthetic_bank(source_bank)
    drive_archive = test_root / "drive" / "photometry_factored_latent_bank_v2.tar"
    archive_sha = _reviewed_tar(drive_archive, source_bank)
    scratch = test_root / "scratch"
    local_archive = scratch / "photometry_factored_latent_bank_v2.tar"
    local_bank = scratch / "photometry_factored_latent_bank_v2"

    capacity = p0006.preflight_stage2_local_disk_capacity(
        drive_archive,
        scratch,
        local_archive_path=local_archive,
        local_bank_root=local_bank,
        expected_extracted_bytes=int(identity["total_bytes"]),
        reserve_bytes=0,
    )
    assert capacity.archive_bytes == drive_archive.stat().st_size
    assert capacity.additional_archive_bytes == capacity.archive_bytes
    assert capacity.additional_extracted_bytes == identity["total_bytes"]

    copy_events: list[dict[str, object]] = []
    verified = p0006.copy_verified_stage2_bank_tar_to_local(
        drive_archive,
        local_archive,
        expected_archive_sha256=archive_sha,
        progress_callback=copy_events.append,
        progress_interval_bytes=1,
    )
    assert verified.archive_file_sha256 == archive_sha
    assert verified.archive_bytes == drive_archive.stat().st_size
    assert verified.copied_from_source is True
    assert sha256_file(local_archive) == archive_sha
    assert [event["status"] for event in copy_events] == [
        "start",
        "periodic",
        "end",
    ]

    restore_events: list[dict[str, object]] = []
    restored = p0006.restore_verified_stage2_bank_tar(
        verified,
        local_bank,
        expected_archive_sha256=archive_sha,
        expected_tree_sha256=str(identity["tree_sha256"]),
        expected_bank_artifact_sha256=BANK_ARTIFACT_SHA256,
        expected_file_count=int(identity["file_count"]),
        expected_total_bytes=int(identity["total_bytes"]),
        progress_callback=restore_events.append,
        progress_interval_bytes=1,
        tree_progress_interval_files=1,
    )
    assert restored.tree_sha256 == identity["tree_sha256"]
    assert restored.bank_artifact_sha256 == BANK_ARTIFACT_SHA256
    assert p0006.stage2_bank_tree_identity(local_bank) == identity
    for stage in ("bank_archive_extraction", "bank_tree_verification"):
        stage_events = [event for event in restore_events if event["stage"] == stage]
        assert stage_events[0]["status"] == "start"
        assert any(event["status"] == "periodic" for event in stage_events)
        assert stage_events[-1]["status"] == "end"
    for event in [*copy_events, *restore_events]:
        assert not any(
            token in key
            for key in event
            for token in ("path", "sha", "identity", "subject")
        )

    resume_events: list[dict[str, object]] = []
    resumed = p0006.copy_verified_stage2_bank_tar_to_local(
        drive_archive,
        local_archive,
        expected_archive_sha256=archive_sha,
        progress_callback=resume_events.append,
        progress_interval_bytes=1,
    )
    assert resumed.copied_from_source is False
    assert resumed.archive_file_sha256 == verified.archive_file_sha256
    assert resume_events[0]["mode"] == "verify_existing"
    assert resume_events[-1]["status"] == "end"


def test_local_archive_copy_preserves_attempts_and_rejects_tampered_resume(
    tmp_path: Path,
) -> None:
    source_bank = tmp_path / "source-bank"
    _synthetic_bank(source_bank)
    drive_archive = tmp_path / "drive" / "photometry_factored_latent_bank_v2.tar"
    archive_sha = _reviewed_tar(drive_archive, source_bank)
    local_archive = tmp_path / "scratch" / "photometry_factored_latent_bank_v2.tar"

    with pytest.raises(ValueError, match="during local copy"):
        p0006.copy_verified_stage2_bank_tar_to_local(
            drive_archive,
            local_archive,
            expected_archive_sha256="0" * 64,
        )
    preserved = (
        local_archive.parent
        / f"{local_archive.name}.copy-attempt-0001.partial"
    )
    assert preserved.is_file()
    verified = p0006.copy_verified_stage2_bank_tar_to_local(
        drive_archive,
        local_archive,
        expected_archive_sha256=archive_sha,
    )
    assert verified.copied_from_source is True
    assert preserved.is_file()
    assert local_archive.is_file()

    tampered = bytearray(local_archive.read_bytes())
    tampered[0] ^= 1
    local_archive.write_bytes(tampered)
    with pytest.raises(ValueError, match="local Stage-2 bank tar SHA-256 mismatch"):
        p0006.copy_verified_stage2_bank_tar_to_local(
            drive_archive,
            local_archive,
            expected_archive_sha256=archive_sha,
        )


def test_local_disk_capacity_preflight_fails_before_large_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_bank = tmp_path / "source-bank"
    identity = _synthetic_bank(source_bank)
    drive_archive = tmp_path / "drive" / "photometry_factored_latent_bank_v2.tar"
    _reviewed_tar(drive_archive, source_bank)
    disk_usage = namedtuple("disk_usage", "total used free")
    monkeypatch.setattr(
        p0006.shutil,
        "disk_usage",
        lambda _path: disk_usage(total=1, used=1, free=0),
    )
    with pytest.raises(OSError, match="Insufficient local disk"):
        p0006.preflight_stage2_local_disk_capacity(
            drive_archive,
            tmp_path / "scratch",
            local_archive_path=tmp_path / "scratch" / "bank.tar",
            local_bank_root=tmp_path / "scratch" / "bank",
            expected_extracted_bytes=int(identity["total_bytes"]),
            reserve_bytes=0,
        )
