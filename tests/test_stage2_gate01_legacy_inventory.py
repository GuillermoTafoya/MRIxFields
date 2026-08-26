from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json, sha256_text
from fieldbridge.data.vae_splits import VaeSplits, save_vae_splits
from fieldbridge.evaluation.stage2_gate01_protocol import GATE01_SCIENTIFIC_MODULES


LEGACY_RELATIVE_PATHS = (
    "frozen-scientific-resplit.json",
    "archive/split_v3.json",
    "gate01-target-calibrator.json",
    "gate01-prospective-selection.json",
    "gate01-protocol-spec.json",
    "gate01-protocol-lock.json",
    "gate01-producer-spec.json",
    "producer-state/producer-state.json",
    "producer-output/private-build-plan.json",
    "gate01-private-manifest.json",
    "gate01-private-build-state.json",
    "gate01-results.json",
    "gate01-report.md",
    "gate01-result-contract.json",
)
MODERN_REQUIRED = {
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
    "original-split-v3.json",
    "producer-state.json",
    "private-build-plan.json",
    "gate01-report.md",
}
PINNED_SPLIT_SHA256 = (
    "f6a19d7a31c4c3bb73edd92088ea078192e88ee4b276309bad81c548ab7f94d5"
)
PINNED_SPLIT_SIZE = 798444
PINNED_OPERATIONAL_SHA256 = (
    "972e497e2d29755e928414a4aa51f906951674ec0a950b0e9ac73881fffd0c54"
)
PINNED_OPERATIONAL_SIZE = 799986
PINNED_MODULE_SHA256 = (
    "ea5f40b580cbba26766ee60ce243d466ab93d32b1856125c067eace9a7d1ed36"
)
PINNED_MODULE_SIZE = 7634
CURRENT_EVALUATION_COMMIT = "c" * 40
PREVIOUS_EVALUATION_COMMIT = "b" * 40
REVIEWED_CHANGED_MODULE = "src/fieldbridge/models/translators/flow_transport.py"


def _pad_json_file(path: Path, size_bytes: int) -> None:
    payload = path.read_bytes()
    assert len(payload) <= size_bytes
    path.write_bytes(payload + b" " * (size_bytes - len(payload)))


def _module_comparison_evidence() -> dict[str, object]:
    current = {
        module: sha256_text(module) for module in GATE01_SCIENTIFIC_MODULES
    }
    previous = dict(current)
    previous[REVIEWED_CHANGED_MODULE] = sha256_text(
        "previous:" + REVIEWED_CHANGED_MODULE
    )
    return {
        "changed_modules": [REVIEWED_CHANGED_MODULE],
        "evaluation_git_commit": CURRENT_EVALUATION_COMMIT,
        "evaluation_module_sha256": current,
        "previous_evaluation_git_commit": PREVIOUS_EVALUATION_COMMIT,
        "previous_evaluation_module_sha256": previous,
    }


def _write_normal_dependency(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(
            json.dumps({"relative_path": path.as_posix(), "synthetic_index": index}),
            encoding="utf-8",
        )
    else:
        path.write_text(f"synthetic:{index}\n", encoding="utf-8")


def _install_pinned_hashes(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> dict[str, str]:
    pinned = {
        "archive/split_v3.json": PINNED_SPLIT_SHA256,
        "colab-operational-source-split.json": PINNED_OPERATIONAL_SHA256,
        "gate01-reviewed-module-sha256-8012a3f.json": PINNED_MODULE_SHA256,
    }
    real_sha256_file = sha256_file

    def synthetic_pinned_sha256(path: str | Path) -> str:
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(root.resolve()).as_posix()
        except ValueError:
            return real_sha256_file(candidate)
        return pinned.get(relative, real_sha256_file(candidate))

    monkeypatch.setattr(p0006, "sha256_file", synthetic_pinned_sha256)
    return pinned


def _legacy_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, list[dict[str, object]], dict[str, str]]:
    root = tmp_path / "Gate01Private_8012a3f"
    root.mkdir()
    for index, relative_path in enumerate(LEGACY_RELATIVE_PATHS):
        path = root.joinpath(*relative_path.split("/"))
        if relative_path == "archive/split_v3.json":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"reviewed_split": True}), encoding="utf-8")
            _pad_json_file(path, PINNED_SPLIT_SIZE)
        else:
            _write_normal_dependency(path, index)

    # Real observed duplicate basenames and deliberately wrong identities.
    decoys = (
        "archive/frozen-scientific-resplit.json",
        "archive/gate01-results.json",
        "archive/producer-state.json",
        "archive/producer-state/producer-state.json",
        "archive/private-build-plan.json",
        "archive/producer-output/private-build-plan.json",
    )
    for index, relative_path in enumerate(decoys, start=100):
        _write_normal_dependency(root.joinpath(*relative_path.split("/")), index)

    operational = root / "colab-operational-source-split.json"
    save_vae_splits(
        VaeSplits(
            train=(),
            validation=(),
            test=(),
            seed=8012,
            fractions=(1.0, 0.0, 0.0),
            metadata={"fixture": "synthetic-only"},
        ),
        operational,
    )
    _pad_json_file(operational, PINNED_OPERATIONAL_SIZE)
    modules = root / "gate01-reviewed-module-sha256-8012a3f.json"
    modules.write_text(
        json.dumps(_module_comparison_evidence()),
        encoding="utf-8",
    )
    _pad_json_file(modules, PINNED_MODULE_SIZE)

    pinned = _install_pinned_hashes(monkeypatch, root)
    rows: list[dict[str, object]] = []
    for relative_path in LEGACY_RELATIVE_PATHS:
        candidate = root.joinpath(*relative_path.split("/"))
        if relative_path == "archive/split_v3.json":
            stored_path = "/content/drive/MyDrive/MRIxFields2026/split_v3.json"
        else:
            stored_path = (
                "/content/historical/Gate01Private_8012a3f/" + relative_path
            )
        rows.append(
            {
                "path": stored_path,
                "sha256": p0006.sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        )
    _write_inventory(root, rows)
    return root, rows, pinned


def _write_inventory(root: Path, rows: list[dict[str, object]]) -> Path:
    inventory = root / "archive" / "sha256-inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return inventory


def _write_modern_dependencies(root: Path) -> None:
    root.mkdir()
    for index, name in enumerate(sorted(MODERN_REQUIRED)):
        _write_normal_dependency(root / name, index)


def test_modern_flat_csv_layout_still_passes(tmp_path: Path) -> None:
    root = tmp_path / "modern"
    _write_modern_dependencies(root)
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
    assert {entry.relative_path for entry in layout.entries} == MODERN_REQUIRED
    assert layout.supplemental_dependencies == ()


def test_exact_reviewed_legacy_topology_resolves_all_14_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    layout = p0006.resolve_gate01_p0006_archive_layout(root)
    assert layout.layout_contract == (
        p0006.GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2
    )
    assert [entry.relative_path for entry in layout.entries] == sorted(
        LEGACY_RELATIVE_PATHS
    )
    assert len(layout.entries) == 14
    assert layout.path_for("producer-state/producer-state.json") == (
        root / "producer-state/producer-state.json"
    ).resolve()
    assert layout.path_for("producer-output/private-build-plan.json") == (
        root / "producer-output/private-build-plan.json"
    ).resolve()
    split = next(
        entry
        for entry in layout.entries
        if entry.relative_path == "archive/split_v3.json"
    )
    assert split.resolution_rule == "reviewed_pinned_external_split_relocation"
    assert split.sha256 == PINNED_SPLIT_SHA256
    assert len(layout.supplemental_dependencies) == 2
    assert not {
        dependency.relative_path for dependency in layout.supplemental_dependencies
    } & {entry.relative_path for entry in layout.entries}
    assert layout.normalized_inventory_sha256 == sha256_json(
        [entry.to_dict() for entry in layout.entries]
    )


def test_archive_decoys_cannot_replace_reviewed_exact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    (root / "gate01-results.json").unlink()
    assert (root / "archive/gate01-results.json").is_file()
    with pytest.raises(FileNotFoundError, match="gate01-results.json"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_all_inventory_rows_are_size_and_hash_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, pinned = _legacy_bundle(monkeypatch, tmp_path)
    real_sha = sha256_file
    verified: list[str] = []

    def recording_sha(path: str | Path) -> str:
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(root.resolve()).as_posix()
        except ValueError:
            return real_sha(candidate)
        if relative in LEGACY_RELATIVE_PATHS:
            verified.append(relative)
        return pinned.get(relative, real_sha(candidate))

    monkeypatch.setattr(p0006, "sha256_file", recording_sha)
    p0006.resolve_gate01_p0006_archive_layout(root)
    assert set(verified) == set(LEGACY_RELATIVE_PATHS)


@pytest.mark.parametrize("tamper", ["bytes", "size", "digest"])
def test_inventoried_row_tamper_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    root, rows, _ = _legacy_bundle(monkeypatch, tmp_path)
    target = root / "gate01-report.md"
    row = next(row for row in rows if str(row["path"]).endswith("gate01-report.md"))
    if tamper == "bytes":
        target.write_bytes(b"changed-but-same")
        row["size_bytes"] = target.stat().st_size
        _write_inventory(root, rows)
    elif tamper == "size":
        row["size_bytes"] = int(row["size_bytes"]) + 1
        _write_inventory(root, rows)
    else:
        row["sha256"] = "0" * 64
        _write_inventory(root, rows)
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize("size", [True, "1", -1, 1.0])
def test_legacy_malformed_digest_or_size_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: object
) -> None:
    root, rows, _ = _legacy_bundle(monkeypatch, tmp_path)
    rows[0]["size_bytes"] = size
    _write_inventory(root, rows)
    with pytest.raises(ValueError, match="invalid size_bytes"):
        p0006.resolve_gate01_p0006_archive_layout(root)
    rows[0]["size_bytes"] = (root / LEGACY_RELATIVE_PATHS[0]).stat().st_size
    rows[0]["sha256"] = str(rows[0]["sha256"]).upper()
    _write_inventory(root, rows)
    with pytest.raises(ValueError, match="malformed SHA-256"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize(
    "stored_path",
    (
        "/content/historical/no-reviewed-marker.json",
        "/content/Gate01Private_8012a3f/../gate01-results.json",
        "/content/Gate01Private_8012a3f/Gate01Private_8012a3f/gate01-results.json",
        "/content/Gate01Private_8012a3f\\gate01-results.json",
        "/content/Gate01Private_8012a3f/gate01-results.json\x00",
        "/content/Gate01Private_8012a3f/C:/gate01-results.json",
    ),
)
def test_unanchored_repeated_traversal_or_ambiguous_paths_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_path: str
) -> None:
    root, rows, _ = _legacy_bundle(monkeypatch, tmp_path)
    rows[11]["path"] = stored_path
    _write_inventory(root, rows)
    with pytest.raises(ValueError, match="marker|traversal|malformed|drive-qualified"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize("change", ["path", "hash", "size", "destination"])
def test_split_relocation_is_exactly_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    root, rows, _ = _legacy_bundle(monkeypatch, tmp_path)
    row = rows[1]
    if change == "path":
        row["path"] = "/content/drive/MyDrive/MRIxFields2026/other/split_v3.json"
    elif change == "hash":
        row["sha256"] = "0" * 64
    elif change == "size":
        row["size_bytes"] = PINNED_SPLIT_SIZE + 1
    else:
        (root / "archive/split_v3.json").rename(root / "split_v3.json")
    _write_inventory(root, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_empty_duplicate_and_casefold_collision_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, rows, _ = _legacy_bundle(monkeypatch, tmp_path)
    _write_inventory(root, [])
    with pytest.raises(ValueError, match="exactly 14"):
        p0006.resolve_gate01_p0006_archive_layout(root)
    _write_inventory(root, [*rows[:-1], dict(rows[0])])
    with pytest.raises(ValueError, match="duplicate entry"):
        p0006.resolve_gate01_p0006_archive_layout(root)
    case_variant = "Gate01-Results.json"
    if os.name != "nt":
        (root / case_variant).write_bytes((root / "gate01-results.json").read_bytes())
    rows[10]["path"] = (
        "/content/historical/Gate01Private_8012a3f/" + case_variant
    )
    rows[10]["sha256"] = rows[11]["sha256"]
    rows[10]["size_bytes"] = rows[11]["size_bytes"]
    _write_inventory(root, rows)
    with pytest.raises(ValueError, match="case-colliding"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize("supplemental", list(p0006._GATE01_LEGACY_SUPPLEMENTAL_SPECS))
def test_missing_or_changed_supplemental_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, supplemental: str
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    target = root / supplemental
    target.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_supplemental_hash_and_size_changes_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    target = root / "colab-operational-source-split.json"
    pinned_sha = p0006.sha256_file

    def changed_hash(path: str | Path) -> str:
        if Path(path).resolve() == target.resolve():
            return "0" * 64
        return pinned_sha(path)

    monkeypatch.setattr(p0006, "sha256_file", changed_hash)
    with pytest.raises(ValueError, match="supplemental dependency hash mismatch"):
        p0006.resolve_gate01_p0006_archive_layout(root)
    monkeypatch.setattr(p0006, "sha256_file", pinned_sha)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="supplemental dependency size mismatch"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_malformed_module_supplemental_fails_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    target = root / "gate01-reviewed-module-sha256-8012a3f.json"
    target.write_text(json.dumps([{"module": "wrong", "sha256": "0" * 64}]))
    _pad_json_file(target, PINNED_MODULE_SIZE)
    with pytest.raises(ValueError, match="must be a JSON object"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_special_inventoried_file_fails_where_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    target = root / "gate01-report.md"
    target.unlink()
    try:
        os.mkfifo(target)
    except OSError as exc:
        pytest.skip(f"FIFO creation unavailable: {exc}")
    with pytest.raises(ValueError, match="not a regular file"):
        p0006.resolve_gate01_p0006_archive_layout(root)


@pytest.mark.parametrize("kind", ["file", "parent"])
def test_symlink_file_or_parent_escape_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    root, rows, _ = _legacy_bundle(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "file":
        target = root / "gate01-report.md"
        outside_target = outside / target.name
        outside_target.write_bytes(target.read_bytes())
        target.unlink()
        link = target
        destination = outside_target
    else:
        target = root / "producer-state/producer-state.json"
        outside_target = outside / target.name
        outside_target.write_bytes(target.read_bytes())
        target.unlink()
        (root / "producer-state").rmdir()
        link = root / "producer-state"
        destination = outside
    try:
        os.symlink(destination, link, target_is_directory=(kind == "parent"))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    _write_inventory(root, rows)
    with pytest.raises(ValueError, match="symlink"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_simultaneous_modern_and_legacy_layouts_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    (root / "sha256-inventory.csv").write_text(
        "Algorithm,Hash,Path\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        p0006.resolve_gate01_p0006_archive_layout(root)


def test_incorrect_expected_result_sha_fails_before_contract_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = _legacy_bundle(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exact reviewed result path"):
        p0006.preflight_gate01_p0006_archive(
            root, expected_gate01_result_sha256="0" * 64
        )
