from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_file


LEGACY_REQUIRED = {
    "gate01-private-manifest.json",
    "gate01-private-build-state.json",
    "gate01-results.json",
    "gate01-result-contract.json",
    "gate01-protocol-lock.json",
    "gate01-protocol-spec.json",
    "gate01-target-calibrator.json",
    "gate01-producer-spec.json",
    "gate01-prospective-selection.json",
    "frozen-scientific-resplit.json",
    "colab-operational-source-split.json",
    "gate01-reviewed-module-sha256-8012a3f.json",
}
MODERN_REQUIRED = (
    LEGACY_REQUIRED
    - {
        "colab-operational-source-split.json",
        "gate01-reviewed-module-sha256-8012a3f.json",
    }
    | {
        "original-split-v3.json",
        "producer-state.json",
        "private-build-plan.json",
        "gate01-report.md",
    }
)


def _write_dependencies(root: Path, names: set[str]) -> None:
    root.mkdir()
    for index, name in enumerate(sorted(names)):
        path = root / name
        if path.suffix == ".json":
            path.write_text(
                json.dumps({"name": name, "synthetic_index": index}), encoding="utf-8"
            )
        else:
            path.write_text(f"synthetic:{name}\\n", encoding="utf-8")


def _legacy_rows(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": f"/content/historical/Gate01Private_8012a3f/{name}",
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in sorted(LEGACY_REQUIRED)
    ]


def _write_legacy_inventory(root: Path, rows: list[dict[str, object]]) -> Path:
    archive = root / "archive"
    archive.mkdir(exist_ok=True)
    path = archive / "sha256-inventory.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def _legacy_bundle(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    root = tmp_path / "Gate01Private_8012a3f"
    _write_dependencies(root, LEGACY_REQUIRED)
    rows = _legacy_rows(root)
    _write_legacy_inventory(root, rows)
    return root, rows


def test_modern_flat_csv_layout_still_passes(tmp_path: Path) -> None:
    root = tmp_path / "modern"
    _write_dependencies(root, MODERN_REQUIRED)
    inventory = root / "sha256-inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Algorithm", "Hash", "Path"))
        writer.writeheader()
        for name in sorted(MODERN_REQUIRED):
            writer.writerow(
                {
                    "Algorithm": "SHA256",
                    "Hash": sha256_file(root / name).upper(),
                    "Path": f"D:/reviewed/archive/{name}",
                }
            )
    layout = p0006.resolve_gate01_p0006_archive_layout(root)
    assert layout.layout_contract == p0006.GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1
    assert layout.inventory_format == "csv"
    assert len(layout.entries) == len(MODERN_REQUIRED)


def test_reviewed_legacy_parent_root_nested_json_inventory_passes(
    tmp_path: Path,
) -> None:
    root, _ = _legacy_bundle(tmp_path)
    layout = p0006.resolve_gate01_p0006_archive_layout(root)
    result_sha = sha256_file(root / "gate01-results.json")
    assert layout.layout_contract == (
        p0006.GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1
    )
    assert layout.inventory_format == "reviewed_legacy_json"
    assert layout.file_with_sha256(result_sha) == (root / "gate01-results.json").resolve()
    assert layout.inventory_file_sha256 == sha256_file(
        root / "archive" / "sha256-inventory.json"
    )
    assert len(layout.entries) == len(LEGACY_REQUIRED)


def test_legacy_absolute_paths_are_labels_and_cannot_redirect_access(
    tmp_path: Path,
) -> None:
    root, rows = _legacy_bundle(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_result = outside / "gate01-results.json"
    outside_result.write_bytes(b"untrusted-outside-bytes")
    result_row = next(row for row in rows if str(row["path"]).endswith("gate01-results.json"))
    result_row["path"] = str(outside_result.resolve())
    _write_legacy_inventory(root, rows)
    layout = p0006.resolve_gate01_p0006_archive_layout(root)
    assert layout.file_with_sha256(str(result_row["sha256"])) == (
        root / "gate01-results.json"
    ).resolve()


@pytest.mark.parametrize("tamper", ["bytes", "size"])
def test_every_legacy_inventory_row_is_size_and_hash_verified(
    tmp_path: Path, tamper: str
) -> None:
    root, rows = _legacy_bundle(tmp_path)
    target = root / "gate01-reviewed-module-sha256-8012a3f.json"
    if tamper == "bytes":
        target.write_bytes(target.read_bytes() + b"tamper")
    else:
        row = next(row for row in rows if str(row["path"]).endswith(target.name))
        row["size_bytes"] = int(row["size_bytes"]) + 1
        _write_legacy_inventory(root, rows)
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_legacy_malformed_digest_fails(tmp_path: Path) -> None:
    root, rows = _legacy_bundle(tmp_path)
    rows[0]["sha256"] = str(rows[0]["sha256"]).upper()
    _write_legacy_inventory(root, rows)
    with pytest.raises(ValueError, match="malformed SHA-256"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize("size", [True, "1", -1, 1.0])
def test_legacy_malformed_or_noninteger_size_fails(
    tmp_path: Path, size: object
) -> None:
    root, rows = _legacy_bundle(tmp_path)
    rows[0]["size_bytes"] = size
    _write_legacy_inventory(root, rows)
    with pytest.raises(ValueError, match="invalid size_bytes"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_empty_or_duplicate_legacy_inventory_fails(tmp_path: Path) -> None:
    root, rows = _legacy_bundle(tmp_path)
    _write_legacy_inventory(root, [])
    with pytest.raises(ValueError, match="empty"):
        p0006.resolve_gate01_p0006_archive_layout(root)
    duplicate = dict(rows[0])
    duplicate["path"] = f"/another/historical/root/{Path(str(rows[0]['path'])).name}"
    _write_legacy_inventory(root, [*rows, duplicate])
    with pytest.raises(ValueError, match="duplicate basenames"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_legacy_inventory_rejects_exact_duplicate_entry(tmp_path: Path) -> None:
    root, rows = _legacy_bundle(tmp_path)
    _write_legacy_inventory(root, [*rows, dict(rows[0])])
    with pytest.raises(ValueError, match="duplicate entry"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize(
    "stored_path",
    ("/content/historical/", ".", "../gate01-results.json", " padded.json "),
)
def test_legacy_inventory_rejects_malformed_stored_paths(
    tmp_path: Path, stored_path: str
) -> None:
    root, rows = _legacy_bundle(tmp_path)
    rows[0]["path"] = stored_path
    _write_legacy_inventory(root, rows)
    with pytest.raises(ValueError, match="malformed|traversal|empty basename"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_missing_inventoried_file_or_required_dependency_fails(tmp_path: Path) -> None:
    root, rows = _legacy_bundle(tmp_path)
    missing = root / "gate01-reviewed-module-sha256-8012a3f.json"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        p0006.resolve_gate01_p0006_archive_layout(root)

    root2 = tmp_path / "missing-dependency"
    _write_dependencies(root2, LEGACY_REQUIRED)
    rows2 = _legacy_rows(root2)
    rows2 = [row for row in rows2 if not str(row["path"]).endswith("gate01-producer-spec.json")]
    _write_legacy_inventory(root2, rows2)
    with pytest.raises(ValueError, match="required scientific dependencies"):
        p0006.resolve_gate01_p0006_archive_layout(root2)


def test_dot_traversal_is_rejected_but_absolute_label_is_not_opened(
    tmp_path: Path,
) -> None:
    root, rows = _legacy_bundle(tmp_path)
    rows[0]["path"] = f"/historical/../{Path(str(rows[0]['path'])).name}"
    _write_legacy_inventory(root, rows)
    with pytest.raises(ValueError, match="dot traversal"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_symlink_escape_fails_where_supported(tmp_path: Path) -> None:
    root, rows = _legacy_bundle(tmp_path)
    name = "gate01-reviewed-module-sha256-8012a3f.json"
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (root / name).unlink()
    try:
        os.symlink(target, root / name)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    row = next(row for row in rows if str(row["path"]).endswith(name))
    row["sha256"] = sha256_file(target)
    row["size_bytes"] = target.stat().st_size
    _write_legacy_inventory(root, rows)
    with pytest.raises(ValueError, match="symlink"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_simultaneous_modern_and_legacy_layouts_fail_closed(tmp_path: Path) -> None:
    root, _ = _legacy_bundle(tmp_path)
    (root / "sha256-inventory.csv").write_text("Algorithm,Hash,Path\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_incorrect_expected_result_sha_fails_before_contract_loading(
    tmp_path: Path,
) -> None:
    root, _ = _legacy_bundle(tmp_path)
    with pytest.raises(ValueError, match="exactly one file"):
        p0006.preflight_gate01_p0006_archive(
            root, expected_gate01_result_sha256="0" * 64
        )
