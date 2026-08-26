from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.evaluation.stage2_gate01 import GATE01_CONTRACT_VERSION


def _authentic_result() -> dict[str, object]:
    result: dict[str, object] = {
        "contract_version": GATE01_CONTRACT_VERSION,
        "evidence_scope": {"synthetic": True},
        "scientific_status": {"synthetic": True},
        "central_question": "synthetic-only",
        "metric_roles": {},
        "method_roles": {},
        "num_pairs": 0,
        "methods": [],
        "overall": {},
        "strata": {},
        "raw_pre_mask_background_leakage": {},
        "by_contrast": {},
        "directed_pair_results": {},
        "directed_pair_matrices": {},
        "central_paired_deltas_and_wins": {},
        "requested_vs_wrong_target_diagnostic": {},
        "montage_specifications": {},
        "pairs": [],
        "contract": {},
        "montage_rendering": {},
    }
    assert len(result) == 20
    assert "result_sha256" not in result
    return result


def _snapshot_bytes(result: dict[str, object] | None = None) -> bytes:
    payload = _authentic_result() if result is None else result
    return b"\xef\xbb\xbf" + json.dumps(payload, sort_keys=True).encode("utf-8")


def _entry(snapshot: bytes) -> p0006.VerifiedGate01InventoryEntry:
    return p0006.VerifiedGate01InventoryEntry(
        relative_path="gate01-results.json",
        sha256=hashlib.sha256(snapshot).hexdigest(),
        size_bytes=len(snapshot),
        resolution_rule="synthetic_exact_path",
        stored_path_label_sha256="0" * 64,
    )


def _load(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    snapshot: bytes,
    *,
    entry: p0006.VerifiedGate01InventoryEntry | None = None,
    expected_sha256: str | None = None,
):
    reviewed_entry = _entry(snapshot) if entry is None else entry
    expected = reviewed_entry.sha256 if expected_sha256 is None else expected_sha256
    monkeypatch.setattr(
        p0006,
        "REVIEWED_GATE01_RESULT_FILE_SHA256",
        reviewed_entry.sha256,
    )
    return p0006._load_verified_gate01_result_snapshot(
        path,
        reviewed_entry,
        expected_gate01_result_sha256=expected,
    )


def test_authentic_twenty_field_result_uses_one_hashed_and_parsed_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot_bytes()
    path = tmp_path / "gate01-results.json"
    path.write_bytes(snapshot)
    original_read_bytes = Path.read_bytes
    original_write_bytes = Path.write_bytes
    calls = 0

    def read_once_then_mutate(candidate: Path) -> bytes:
        nonlocal calls
        captured = original_read_bytes(candidate)
        if candidate == path:
            calls += 1
            changed = captured.replace(b"synthetic-only", b"mutated-value!", 1)
            original_write_bytes(candidate, changed)
        return captured

    monkeypatch.setattr(Path, "read_bytes", read_once_then_mutate)
    result, file_sha256 = _load(monkeypatch, path, snapshot)
    assert calls == 1
    assert result == _authentic_result()
    assert len(result) == 20
    assert file_sha256 == hashlib.sha256(snapshot).hexdigest()
    assert path.read_bytes() != snapshot


def test_mutated_result_bytes_fail_inventory_sha_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot_bytes()
    path = tmp_path / "gate01-results.json"
    path.write_bytes(snapshot.replace(b"synthetic-only", b"mutated-value!", 1))
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        _load(monkeypatch, path, snapshot)


def test_inventory_and_expected_result_sha_disagreement_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot_bytes()
    path = tmp_path / "gate01-results.json"
    path.write_bytes(snapshot)
    with pytest.raises(ValueError, match="exact reviewed result path"):
        _load(monkeypatch, path, snapshot, expected_sha256="0" * 64)


def test_expected_result_sha_must_equal_pinned_authentic_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _snapshot_bytes()
    path = tmp_path / "gate01-results.json"
    path.write_bytes(snapshot)
    entry = _entry(snapshot)
    monkeypatch.setattr(
        p0006,
        "REVIEWED_GATE01_RESULT_FILE_SHA256",
        "f" * 64,
    )
    with pytest.raises(ValueError, match="pinned authentic"):
        p0006._load_verified_gate01_result_snapshot(
            path,
            entry,
            expected_gate01_result_sha256=entry.sha256,
        )


def test_wrong_result_contract_version_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _authentic_result()
    result["contract_version"] = "stage2-gate01-equal-photometry-v1"
    snapshot = _snapshot_bytes(result)
    path = tmp_path / "gate01-results.json"
    path.write_bytes(snapshot)
    with pytest.raises(ValueError, match="incompatible contract"):
        _load(monkeypatch, path, snapshot)


def test_duplicate_result_json_keys_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = (
        b'{"contract_version":"stage2-gate01-equal-photometry-v2",'
        b'"contract_version":"stage2-gate01-equal-photometry-v2"}'
    )
    path = tmp_path / "gate01-results.json"
    path.write_bytes(snapshot)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load(monkeypatch, path, snapshot)


def test_importer_source_contains_no_fabricated_result_self_hash() -> None:
    source = Path(p0006.__file__).read_text(encoding="utf-8")
    for forbidden in (
        'result_body.pop("result_sha256"',
        '"result_sha256": stored_result_hash',
        "_Gate01MetadataPreflight.stored_result_hash",
    ):
        assert forbidden not in source
    assert "verified_result_file_sha256" in source
    assert '"internal_self_hash_defined": False' in source


def test_private_runbook_documents_strict_module_comparison_object() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "STAGE2_GATE01_PRIVATE_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    for field in (
        "changed_modules",
        "evaluation_git_commit",
        "evaluation_module_sha256",
        "previous_evaluation_git_commit",
        "previous_evaluation_module_sha256",
    ):
        assert f'"{field}"' in runbook
    assert "Both maps must contain exactly the same 31 keys" in runbook
    assert "previous commit and\nmap are provenance only" in runbook
    assert "An old module list cannot\nsubstitute" in runbook
    assert "[pscustomobject]@{ module = $_; sha256" not in runbook
