from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fieldbridge.data import photometry_factorization as photometry
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SOURCE_MODULES,
    VARIANT_A_PROSPECTIVE_EXCLUSION_REASON,
    PhotometryFitVolume,
    fit_frozen_photometry,
    sha256_file,
    sha256_json,
    sha256_text,
)
from fieldbridge.evaluation import stage2_step200_inference_audit as audit


@dataclass(frozen=True)
class SyntheticReviewedArtifact:
    path: Path
    file_sha256: str
    artifact_sha256: str
    protected_module_sha256: str


def _tiny_volume(offset: int) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        dtype=torch.float32,
    ).reshape(2, 2, 2) + float(offset % 3) * 0.001


def _fit_volume(
    *,
    ordinal: int,
    domain: Domain,
    subject_identity: str,
    cohort: str = "R",
    metadata_prefix: str = "R",
    record_identity: str | None = None,
) -> PhotometryFitVolume:
    identity = record_identity or f"{cohort}_SYNTHETIC_{ordinal:04d}"
    return PhotometryFitVolume(
        volume=_tiny_volume(ordinal),
        domain=domain,
        record_identity=identity,
        subject_identity=subject_identity,
        metadata_prefix=metadata_prefix,
        source_path_identity=f"synthetic/{identity}.nii.gz",
        source_file_sha256=sha256_text(f"synthetic-file-{identity}"),
        split="train",
        cohort=cohort,
    )


def _excluded_record(ordinal: int) -> dict[str, object]:
    identity = f"P_SYNTHETIC_{ordinal:04d}"
    subject = f"{2_000 + ordinal:04d}"
    return {
        "record_identity": identity,
        "record_identity_sha256": sha256_text(identity),
        "subject_identity": subject,
        "subject_group_identity": f"P:{subject}",
        "metadata_prefix": "P",
        "cohort": "P",
        "split": "train",
        "source_path_identity_sha256": sha256_text(
            f"synthetic/{identity}.nii.gz"
        ),
        "reason": VARIANT_A_PROSPECTIVE_EXCLUSION_REASON,
    }


@pytest.fixture(scope="module")
def reviewed_artifact(tmp_path_factory: pytest.TempPathFactory) -> SyntheticReviewedArtifact:
    root = tmp_path_factory.mktemp("reviewed-photometry")
    path = root / audit.REVIEWED_PHOTOMETRY_BASENAME
    protected_module_sha = sha256_file(Path(photometry.__file__))
    module_hashes = {
        name: (
            protected_module_sha
            if name == "src/fieldbridge/data/photometry_factorization.py"
            else hashlib.sha256(name.encode("utf-8")).hexdigest()
        )
        for name in PHOTOMETRY_SOURCE_MODULES
    }
    volumes: list[PhotometryFitVolume] = []
    ordinal = 0
    for contrast in CONTRASTS:
        for field in FIELD_STRENGTHS_T:
            domain = Domain(field, contrast)
            for _ in range(104):
                if ordinal < 3:
                    subject = "0006"
                elif ordinal < 6:
                    subject = "0009"
                else:
                    subject = f"{10_000 + ordinal:05d}"
                volumes.append(
                    _fit_volume(
                        ordinal=ordinal,
                        domain=domain,
                        subject_identity=subject,
                    )
                )
                ordinal += 1
    assert len(volumes) == 1_560
    previous_module_sha = audit.PROTECTED_PHOTOMETRY_MODULE_SHA256
    audit.PROTECTED_PHOTOMETRY_MODULE_SHA256 = protected_module_sha
    try:
        with audit._reviewed_photometry_namespace_scope():
            artifact = fit_frozen_photometry(
                volumes,
                source_split_file_sha256="a" * 64,
                source_membership_fingerprint="synthetic-membership",
                source_recovery_fingerprint="synthetic-recovery",
                code_commit=audit.REVIEWED_PHOTOMETRY_PRODUCTION_COMMIT,
                code_provenance={
                    "git_head": audit.REVIEWED_PHOTOMETRY_PRODUCTION_COMMIT,
                    "checkout_clean": True,
                    "module_sha256": module_hashes,
                },
                resolved_config={"contract": "synthetic-reviewed-photometry"},
                num_quantiles=3,
                excluded_prospective_records=tuple(
                    _excluded_record(index) for index in range(30)
                ),
            )
    finally:
        audit.PROTECTED_PHOTOMETRY_MODULE_SHA256 = previous_module_sha
    artifact.save(path)
    return SyntheticReviewedArtifact(
        path=path,
        file_sha256=sha256_file(path),
        artifact_sha256=artifact.artifact_sha256,
        protected_module_sha256=protected_module_sha,
    )


def _configure_synthetic_pins(
    monkeypatch: pytest.MonkeyPatch,
    fixture: SyntheticReviewedArtifact,
) -> None:
    monkeypatch.setattr(audit, "REVIEWED_PHOTOMETRY_FILE_SHA256", fixture.file_sha256)
    monkeypatch.setattr(
        audit, "REVIEWED_PHOTOMETRY_ARTIFACT_SHA256", fixture.artifact_sha256
    )
    monkeypatch.setattr(
        audit, "PROTECTED_PHOTOMETRY_MODULE_SHA256", fixture.protected_module_sha256
    )


@pytest.mark.parametrize("subject", ["0006", "0007", "0009"])
def test_reviewed_scope_accepts_canonical_retrospective_numeric_collisions(
    monkeypatch: pytest.MonkeyPatch, subject: str
) -> None:
    local_sha = sha256_file(Path(photometry.__file__))
    monkeypatch.setattr(audit, "PROTECTED_PHOTOMETRY_MODULE_SHA256", local_sha)
    item = _fit_volume(
        ordinal=1,
        domain=Domain(0.1, "T1w"),
        subject_identity=subject,
    )
    with pytest.raises(ValueError, match="explicitly rejects traveller"):
        photometry._validate_fit_role(item)
    with audit._reviewed_photometry_namespace_scope():
        photometry._validate_fit_role(item)
    assert photometry._is_forbidden_traveller is audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER
    assert audit.photometry_namespace_compatibility_active() is False


@pytest.mark.parametrize("subject", ["0006", "0007", "0009"])
def test_reviewed_scope_still_rejects_prospective_and_embedded_tokens(
    monkeypatch: pytest.MonkeyPatch, subject: str
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    prospective = _fit_volume(
        ordinal=1,
        domain=Domain(0.1, "T1w"),
        subject_identity=subject,
        cohort="P",
        metadata_prefix="P",
        record_identity=f"P_{subject}_SYNTHETIC",
    )
    embedded = _fit_volume(
        ordinal=2,
        domain=Domain(0.1, "T1w"),
        subject_identity="1234",
        record_identity=f"R_SYNTHETIC_P_{subject}_TOKEN",
    )
    with audit._reviewed_photometry_namespace_scope():
        with pytest.raises(ValueError, match="explicitly rejects traveller"):
            photometry._validate_fit_role(prospective)
        with pytest.raises(ValueError, match="explicitly rejects traveller"):
            photometry._validate_fit_role(embedded)


def test_reviewed_scope_rejects_conflicting_or_missing_cohort_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    with audit._reviewed_photometry_namespace_scope():
        for item in (
            _fit_volume(
                ordinal=1,
                domain=Domain(0.1, "T1w"),
                subject_identity="0006",
                metadata_prefix="P",
            ),
            _fit_volume(
                ordinal=2,
                domain=Domain(0.1, "T1w"),
                subject_identity="0006",
                cohort="P",
            ),
            _fit_volume(
                ordinal=3,
                domain=Domain(0.1, "T1w"),
                subject_identity="",
            ),
        ):
            with pytest.raises(ValueError):
                photometry._validate_fit_role(item)


def test_exact_synthetic_reviewed_artifact_preflight_and_base_rejection(
    monkeypatch: pytest.MonkeyPatch,
    reviewed_artifact: SyntheticReviewedArtifact,
) -> None:
    _configure_synthetic_pins(monkeypatch, reviewed_artifact)
    with pytest.raises(ValueError, match="reserved traveller"):
        photometry.FrozenPhotometryArtifact.load(reviewed_artifact.path)
    preflight = audit.preflight_reviewed_photometry_namespace_artifact(
        reviewed_artifact.path
    )
    assert preflight.accepted_record_count == 1_560
    assert preflight.prospective_accepted_count == 0
    assert preflight.prospective_excluded_count == 30
    assert preflight.retrospective_numeric_collision_count == 6
    assert preflight.collision_group_counts == (
        audit.REVIEWED_PHOTOMETRY_COLLISION_GROUP_COUNTS
    )
    assert photometry._is_forbidden_traveller is audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER
    assert audit.photometry_namespace_compatibility_active() is False


def test_scope_restores_after_failure_and_rejects_nested_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with audit._reviewed_photometry_namespace_scope():
            raise RuntimeError("synthetic failure")
    assert photometry._is_forbidden_traveller is audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER
    with audit._reviewed_photometry_namespace_scope():
        with pytest.raises(RuntimeError, match="Nested or concurrent"):
            with audit._reviewed_photometry_namespace_scope():
                pass
    assert photometry._is_forbidden_traveller is audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER


def test_concurrent_compatibility_scope_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    failures: list[BaseException] = []

    def enter_concurrently() -> None:
        try:
            with audit._reviewed_photometry_namespace_scope():
                pass
        except BaseException as error:
            failures.append(error)

    with audit._reviewed_photometry_namespace_scope():
        thread = threading.Thread(target=enter_concurrently)
        thread.start()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert "Nested or concurrent" in str(failures[0])
    assert photometry._is_forbidden_traveller is audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER


def test_scope_restores_when_existing_artifact_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    with pytest.raises(ValueError, match="synthetic artifact failure"):
        with audit._reviewed_photometry_namespace_scope():
            raise ValueError("synthetic artifact failure")
    assert photometry._is_forbidden_traveller is audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER
    assert audit.photometry_namespace_compatibility_active() is False


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        (
            "HISTORICAL_PHOTOMETRY_OPERATOR_OVERLAY_SHA256",
            "0" * 64,
            "operator-overlay",
        ),
        (
            "REVIEWED_NAMESPACE_PREDICATE_SOURCE_SHA256",
            "1" * 64,
            "namespace-predicate",
        ),
    ],
)
def test_overlay_and_predicate_provenance_substitution_fails(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: str,
    match: str,
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    monkeypatch.setattr(audit, attribute, value)
    with pytest.raises(ValueError, match=match):
        with audit._reviewed_photometry_namespace_scope():
            pass
    assert audit.photometry_namespace_compatibility_active() is False


def test_base_or_compatibility_function_substitution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "PROTECTED_PHOTOMETRY_MODULE_SHA256",
        sha256_file(Path(photometry.__file__)),
    )
    monkeypatch.setattr(
        photometry, "_is_forbidden_traveller", lambda *_args: False
    )
    with pytest.raises(RuntimeError, match="helper was substituted"):
        with audit._reviewed_photometry_namespace_scope():
            pass

    monkeypatch.setattr(
        photometry,
        "_is_forbidden_traveller",
        audit._BASE_PHOTOMETRY_FORBIDDEN_TRAVELLER,
    )
    monkeypatch.setattr(
        audit, "_namespace_aware_forbidden_traveller", lambda *_args: False
    )
    with pytest.raises(RuntimeError, match="predicate was substituted"):
        with audit._reviewed_photometry_namespace_scope():
            pass


def test_substituted_active_module_location_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substitute = tmp_path / "photometry_factorization.py"
    substitute.write_text("# substituted module role", encoding="utf-8")
    monkeypatch.setattr(photometry, "__file__", str(substitute))
    with pytest.raises(ValueError, match="outside the detached implementation checkout"):
        with audit._reviewed_photometry_namespace_scope():
            pass


def test_wrong_file_or_internal_hash_fails_before_compatibility_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reviewed_artifact: SyntheticReviewedArtifact,
) -> None:
    _configure_synthetic_pins(monkeypatch, reviewed_artifact)
    wrong_raw = tmp_path / audit.REVIEWED_PHOTOMETRY_BASENAME
    wrong_raw.write_bytes(reviewed_artifact.path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw-file SHA-256"):
        audit.preflight_reviewed_photometry_namespace_artifact(wrong_raw)
    assert audit.photometry_namespace_compatibility_active() is False

    payload = json.loads(reviewed_artifact.path.read_text(encoding="utf-8"))
    payload["runtime_statistics"] = "changed"
    wrong_internal = tmp_path / "internal" / audit.REVIEWED_PHOTOMETRY_BASENAME
    wrong_internal.parent.mkdir()
    wrong_internal.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        audit, "REVIEWED_PHOTOMETRY_FILE_SHA256", sha256_file(wrong_internal)
    )
    with pytest.raises(ValueError, match="internal SHA-256"):
        audit.preflight_reviewed_photometry_namespace_artifact(wrong_internal)
    assert audit.photometry_namespace_compatibility_active() is False


def test_production_or_module_provenance_change_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reviewed_artifact: SyntheticReviewedArtifact,
) -> None:
    original = json.loads(reviewed_artifact.path.read_text(encoding="utf-8"))
    for label, mutate, match in (
        (
            "commit",
            lambda payload: payload["provenance"].__setitem__(
                "code_commit", "b" * 40
            ),
            "production commit",
        ),
        (
            "module",
            lambda payload: payload["provenance"]["code_provenance"][
                "module_sha256"
            ].__setitem__(
                "src/fieldbridge/data/photometry_factorization.py", "c" * 64
            ),
            "module hash",
        ),
    ):
        payload = copy.deepcopy(original)
        mutate(payload)
        if label == "commit":
            payload["provenance"]["code_provenance"]["git_head"] = "b" * 40
        body = dict(payload)
        body.pop("artifact_sha256")
        payload["artifact_sha256"] = sha256_json(body)
        path = tmp_path / label / audit.REVIEWED_PHOTOMETRY_BASENAME
        path.parent.mkdir()
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        _configure_synthetic_pins(monkeypatch, reviewed_artifact)
        monkeypatch.setattr(audit, "REVIEWED_PHOTOMETRY_FILE_SHA256", sha256_file(path))
        monkeypatch.setattr(
            audit, "REVIEWED_PHOTOMETRY_ARTIFACT_SHA256", payload["artifact_sha256"]
        )
        with pytest.raises(ValueError, match=match):
            audit.preflight_reviewed_photometry_namespace_artifact(path)


def test_membership_and_prospective_mutations_fail(
    monkeypatch: pytest.MonkeyPatch,
    reviewed_artifact: SyntheticReviewedArtifact,
) -> None:
    _configure_synthetic_pins(monkeypatch, reviewed_artifact)
    preflight = audit.preflight_reviewed_photometry_namespace_artifact(
        reviewed_artifact.path
    )
    provenance = copy.deepcopy(dict(preflight.artifact.provenance))
    provenance["accepted_records"] = provenance["accepted_records"][:-1]
    with pytest.raises(ValueError, match="accepted-record count"):
        audit._validate_reviewed_photometry_membership(
            SimpleNamespace(provenance=provenance)
        )

    provenance = copy.deepcopy(dict(preflight.artifact.provenance))
    provenance["accepted_records_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="accepted-record content hash"):
        audit._validate_reviewed_photometry_membership(
            SimpleNamespace(provenance=provenance)
        )

    provenance = copy.deepcopy(dict(preflight.artifact.provenance))
    provenance["excluded_prospective_records_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="excluded-record content hash"):
        audit._validate_reviewed_photometry_membership(
            SimpleNamespace(provenance=provenance)
        )

    provenance = copy.deepcopy(dict(preflight.artifact.provenance))
    provenance["accepted_records"][0]["record_identity"] = "P_0006_SUBSTITUTED"
    provenance["accepted_records"][0]["metadata_prefix"] = "P"
    provenance["accepted_records"][0]["cohort"] = "P"
    provenance["accepted_records"][0]["subject_group_identity"] = "P:0006"
    provenance["accepted_records_sha256"] = sha256_json(
        provenance["accepted_records"]
    )
    with pytest.raises(ValueError, match="rejects every P|prospective"):
        audit._validate_reviewed_photometry_membership(
            SimpleNamespace(provenance=provenance)
        )


def test_bank_cross_check_is_field_specific(
    monkeypatch: pytest.MonkeyPatch,
    reviewed_artifact: SyntheticReviewedArtifact,
) -> None:
    _configure_synthetic_pins(monkeypatch, reviewed_artifact)
    preflight = audit.preflight_reviewed_photometry_namespace_artifact(
        reviewed_artifact.path
    )
    manifest = {
        "photometry": {
            "artifact_file_sha256": reviewed_artifact.file_sha256,
            "artifact_sha256": reviewed_artifact.artifact_sha256,
        }
    }
    monkeypatch.setattr(
        audit,
        "PhotometryFactoredLatentBankIndex",
        lambda *_args: SimpleNamespace(manifest=manifest),
    )
    verified = audit.verify_reviewed_photometry_bank_provenance(
        preflight, bank_dir=reviewed_artifact.path.parent
    )
    assert verified.preflight.artifact is preflight.artifact

    manifest["photometry"]["artifact_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file SHA-256"):
        audit.verify_reviewed_photometry_bank_provenance(
            preflight, bank_dir=reviewed_artifact.path.parent
        )
    manifest["photometry"]["artifact_file_sha256"] = reviewed_artifact.file_sha256
    manifest["photometry"]["artifact_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="internal SHA-256"):
        audit.verify_reviewed_photometry_bank_provenance(
            preflight, bank_dir=reviewed_artifact.path.parent
        )


def test_runtime_rehash_fails_before_checkpoint_or_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reviewed_artifact: SyntheticReviewedArtifact,
) -> None:
    _configure_synthetic_pins(monkeypatch, reviewed_artifact)
    preflight = audit.preflight_reviewed_photometry_namespace_artifact(
        reviewed_artifact.path
    )
    copied = tmp_path / audit.REVIEWED_PHOTOMETRY_BASENAME
    copied.write_bytes(reviewed_artifact.path.read_bytes())
    copied_preflight = replace(preflight, path=copied)
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    verified = audit.ReviewedPhotometryBankProvenance(
        preflight=copied_preflight,
        bank_dir=bank_dir,
        bank_manifest_artifact_file_sha256=reviewed_artifact.file_sha256,
        bank_manifest_artifact_sha256=reviewed_artifact.artifact_sha256,
    )
    copied.write_bytes(copied.read_bytes() + b" ")
    reached: list[str] = []
    monkeypatch.setattr(
        audit,
        "load_checkpoint",
        lambda *_args, **_kwargs: reached.append("checkpoint"),
    )
    monkeypatch.setattr(
        audit,
        "build_translator",
        lambda *_args, **_kwargs: reached.append("model"),
    )
    monkeypatch.setattr(
        audit,
        "preflight_frozen_stage1_run_c_config",
        lambda _path: SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="raw-file SHA-256"):
        audit.load_unified_step200_inference_runtime(
            checkpoint_path=tmp_path / "checkpoint.pt",
            resolved_config_path=tmp_path / "resolved.json",
            vae_config_path=tmp_path / "stage1-run-c.yaml",
            vae_checkpoint_path=tmp_path / "vae.pt",
            photometry_artifact_path=copied,
            bank_dir=bank_dir,
            verified_photometry_provenance=verified,
        )
    assert reached == []


def test_operator_preflights_photometry_before_bank_and_model_work() -> None:
    source = Path(
        "notebooks/stage2_step200_inference_audit_operator.py"
    ).read_text(encoding="utf-8")
    preflight = source.index(
        "photometry_preflight = preflight_reviewed_photometry_namespace_artifact("
    )
    capacity = source.index("capacity = preflight_stage2_local_disk_capacity(")
    copy_bank = source.index("local_archive = copy_verified_stage2_bank_tar_to_local(")
    restore = source.index("bank_restore = restore_verified_stage2_bank_tar(")
    checkpoint = source.index("completed = verify_completed_stage2_pilot_evidence(")
    runtime = source.index("runtime = load_unified_step200_inference_runtime(")
    assert preflight < capacity < copy_bank < restore < checkpoint < runtime
    assert "verified_photometry_provenance=verified_photometry_provenance" in source
    assert "FrozenPhotometryArtifact.load" not in source
