from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fieldbridge.evaluation.stage2_unified_gate01_p0006 as p0006
from fieldbridge.data.photometry_factorization import sha256_text
from fieldbridge.evaluation.stage2_gate01_protocol import GATE01_SCIENTIFIC_MODULES


CURRENT_COMMIT = "c" * 40
PREVIOUS_COMMIT = "b" * 40
FLOW_MODULE = "src/fieldbridge/models/translators/flow_transport.py"


def _maps() -> tuple[dict[str, str], dict[str, str]]:
    current = {
        module: sha256_text(module) for module in GATE01_SCIENTIFIC_MODULES
    }
    previous = dict(current)
    previous[FLOW_MODULE] = sha256_text("previous:" + FLOW_MODULE)
    return current, previous


def _payload() -> dict[str, object]:
    current, previous = _maps()
    return {
        "changed_modules": [FLOW_MODULE],
        "evaluation_git_commit": CURRENT_COMMIT,
        "evaluation_module_sha256": current,
        "previous_evaluation_git_commit": PREVIOUS_COMMIT,
        "previous_evaluation_module_sha256": previous,
    }


def _write_payload(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(tmp_path: Path, payload: object | None = None):
    selected = _payload() if payload is None else payload
    return p0006._load_reviewed_module_comparison_evidence(
        _write_payload(tmp_path / "reviewed.json", selected)
    )


def _verify_linkage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, object],
    *,
    lock_commit: str = CURRENT_COMMIT,
    lock_modules: dict[str, str] | None = None,
) -> None:
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    reviewed = _write_payload(tmp_path / "reviewed.json", payload)

    class _Layout:
        def supplemental_path_for(self, relative_path: str) -> Path:
            if relative_path == "colab-operational-source-split.json":
                return split
            if relative_path == "gate01-reviewed-module-sha256-8012a3f.json":
                return reviewed
            raise AssertionError(relative_path)

    monkeypatch.setattr(p0006, "load_vae_splits", lambda path: object())
    monkeypatch.setattr(p0006, "vae_splits_fingerprint", lambda split: "bank")
    lock = SimpleNamespace(
        bank_source_split_fingerprint="bank",
        evaluation_git_commit=lock_commit,
        evaluation_module_sha256=(
            _maps()[0] if lock_modules is None else lock_modules
        ),
    )
    p0006._verify_supplemental_linkage(_Layout(), lock)


def test_exact_five_field_comparison_evidence_parses_and_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _load(tmp_path)
    current, previous = _maps()
    assert evidence.evaluation_git_commit == CURRENT_COMMIT
    assert evidence.current_module_hashes == current
    assert evidence.previous_evaluation_git_commit == PREVIOUS_COMMIT
    assert evidence.previous_module_hashes == previous
    assert evidence.changed_modules == (FLOW_MODULE,)
    assert len(evidence.evaluation_module_sha256) == 31
    assert current != previous
    _verify_linkage(monkeypatch, tmp_path, _payload())


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_top_level_key_set_is_exact(tmp_path: Path, mutation: str) -> None:
    payload = _payload()
    if mutation == "missing":
        del payload["changed_modules"]
    else:
        payload["unexpected"] = True
    with pytest.raises(ValueError, match="incorrect top-level key set"):
        _load(tmp_path, payload)


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_json_keys_are_rejected_at_every_level(
    tmp_path: Path, nested: bool
) -> None:
    payload = _payload()
    text = json.dumps(payload)
    if nested:
        current = payload["evaluation_module_sha256"]
        assert isinstance(current, dict)
        module = next(iter(current))
        needle = json.dumps(module) + ": " + json.dumps(current[module])
    else:
        needle = '"evaluation_git_commit": ' + json.dumps(CURRENT_COMMIT)
    text = text.replace(needle, needle + ", " + needle, 1)
    path = tmp_path / "reviewed.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        p0006._load_reviewed_module_comparison_evidence(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_git_commit", "C" * 40),
        ("evaluation_git_commit", "c" * 39),
        ("previous_evaluation_git_commit", "z" * 40),
        ("previous_evaluation_git_commit", CURRENT_COMMIT),
    ],
)
def test_commit_identities_are_strict_and_distinct(
    tmp_path: Path, field: str, value: str
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match="Git commit identity|must be distinct"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    "map_name",
    ["evaluation_module_sha256", "previous_evaluation_module_sha256"],
)
@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_both_maps_require_exact_scientific_module_keys(
    tmp_path: Path, map_name: str, mutation: str
) -> None:
    payload = _payload()
    module_map = payload[map_name]
    assert isinstance(module_map, dict)
    if mutation == "missing":
        module_map.pop(next(iter(module_map)))
    else:
        module_map["src/fieldbridge/models/unexpected.py"] = "0" * 64
    with pytest.raises(ValueError, match="missing or unexpected scientific modules"):
        _load(tmp_path, payload)


def test_malformed_module_digest_fails(tmp_path: Path) -> None:
    payload = _payload()
    current = payload["evaluation_module_sha256"]
    assert isinstance(current, dict)
    current[next(iter(current))] = "A" * 64
    with pytest.raises(ValueError, match="malformed SHA-256"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    "changed_modules",
    [[], [FLOW_MODULE, FLOW_MODULE], ["../flow_transport.py"]],
)
def test_changed_modules_must_be_canonical_unique_and_match_computed_diff(
    tmp_path: Path, changed_modules: list[str]
) -> None:
    payload = _payload()
    payload["changed_modules"] = changed_modules
    with pytest.raises(ValueError, match="changed_modules|canonical module path"):
        _load(tmp_path, payload)


def test_unexpected_computed_change_set_fails(tmp_path: Path) -> None:
    payload = _payload()
    current = payload["evaluation_module_sha256"]
    previous = payload["previous_evaluation_module_sha256"]
    assert isinstance(current, dict) and isinstance(previous, dict)
    second = next(module for module in current if module != FLOW_MODULE)
    previous[second] = sha256_text("previous:" + second)
    payload["changed_modules"] = sorted([FLOW_MODULE, second])
    with pytest.raises(ValueError, match="unexpected reviewed change set"):
        _load(tmp_path, payload)


def test_arbitrary_list_cannot_substitute_for_comparison_evidence(
    tmp_path: Path,
) -> None:
    current, _ = _maps()
    legacy_list = [
        {"module": module, "sha256": digest} for module, digest in current.items()
    ]
    with pytest.raises(ValueError, match="must be a JSON object"):
        _load(tmp_path, legacy_list)


def test_incorrect_current_map_fails_protocol_lock_linkage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _payload()
    current = payload["evaluation_module_sha256"]
    previous = payload["previous_evaluation_module_sha256"]
    assert isinstance(current, dict) and isinstance(previous, dict)
    unchanged = next(module for module in current if module != FLOW_MODULE)
    current[unchanged] = sha256_text("other-current:" + unchanged)
    previous[unchanged] = current[unchanged]
    with pytest.raises(ValueError, match="current module identities"):
        _verify_linkage(monkeypatch, tmp_path, payload)


def test_incorrect_current_commit_fails_protocol_lock_linkage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _payload()
    payload["evaluation_git_commit"] = "d" * 40
    with pytest.raises(ValueError, match="current evaluation commit"):
        _verify_linkage(monkeypatch, tmp_path, payload)


def test_previous_map_and_commit_cannot_authorize_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _payload()
    previous = payload["previous_evaluation_module_sha256"]
    assert isinstance(previous, dict)
    with pytest.raises(ValueError, match="current evaluation commit"):
        _verify_linkage(
            monkeypatch,
            tmp_path,
            payload,
            lock_commit=PREVIOUS_COMMIT,
            lock_modules=previous,
        )
