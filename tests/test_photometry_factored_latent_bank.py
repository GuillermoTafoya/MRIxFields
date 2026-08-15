from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.config import load_yaml_config
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.latent_bank import LATENT_BANK_CONTRACT_VERSION
from fieldbridge.data.latent_bank_dataset import LatentBankIndex
from fieldbridge.data.photometry_factored_latent_bank import (
    FACTORED_LATENT_BANK_MANIFEST,
    FACTORED_LATENT_STATS_FILE,
    PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
    PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
    STRUCTURAL_DESCRIPTOR_MANIFEST,
    PhotometryFactoredLatentBankConfig,
    audit_photometry_factored_latent_bank,
    build_photometry_factored_latent_bank,
    downsample_source_support,
    pack_support_mask,
    structural_descriptor,
    unpack_support_mask,
)
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SOURCE_MODULES,
    FrozenPhotometryArtifact,
    PhotometryFitVolume,
    fit_frozen_photometry,
    sha256_file,
    sha256_json,
    sha256_text,
)
from fieldbridge.data.stage2_canonical_volume import (
    CANONICAL_VOLUME_SOURCE_MODULES,
    CanonicalVolumeBuildConfig,
    audit_canonical_volume_artifact,
    build_canonical_volume_artifact,
    load_canonical_volume_manifest,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    VARIANT_A_QUALIFICATION_CONTRACT_VERSION,
)

_CONFIG_PATH = Path("configs/experiment/stage2_canonical_artifacts_v1.yaml")
_FAKE_SPLIT_SHA = sha256_text("synthetic canonical split file")
_FAKE_MEMBERSHIP = sha256_text("synthetic membership")
_FAKE_RECOVERY = sha256_text("synthetic recovery")
_VAE_CONFIG_SHA = sha256_text("synthetic frozen VAE config")
_VAE_CHECKPOINT_SHA = sha256_text("synthetic frozen VAE checkpoint")
_PHOTOMETRY_FILE_SHA = sha256_text("synthetic photometry artifact file")
_QUALIFICATION_FILE_SHA = sha256_text("synthetic qualification file")


class _FrozenPosteriorMeanEncoder(nn.Module):
    downsample_factor = 2
    latent_channels = 2

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode_dist(
        self, volume: torch.Tensor, domain: Domain
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        pooled = F.avg_pool3d(volume, kernel_size=2, stride=2)
        field_scaled = pooled * float(domain.field_strength_t / 7.0)
        mean = torch.cat((pooled, field_scaled), dim=1)
        return mean, torch.zeros_like(mean)


@dataclass
class _Fixture:
    records_by_split: dict[str, tuple[VolumeRecord, ...]]
    volumes: dict[str, torch.Tensor]
    artifact: FrozenPhotometryArtifact
    qualification: dict
    config: dict
    code_provenance: dict
    p_path: str


def _code_provenance() -> dict:
    return {
        "git_head": "synthetic-commit",
        "checkout_clean": True,
        "module_sha256": {
            path: sha256_text(path) for path in CANONICAL_VOLUME_SOURCE_MODULES
        },
    }


def _photometry_code_provenance() -> dict:
    return {
        "git_head": "synthetic-photometry-commit",
        "checkout_clean": True,
        "module_sha256": {path: sha256_text(path) for path in PHOTOMETRY_SOURCE_MODULES},
    }


def _volume(domain: Domain, split_index: int) -> torch.Tensor:
    values = torch.linspace(0.1, 0.9, 8 * 8 * 8, dtype=torch.float32).reshape(
        1, 1, 8, 8, 8
    )
    values = (values + 0.005 * domain.contrast_index + 0.001 * split_index).clamp_max(
        1.0
    )
    values[..., :2, :, :] = 0.0
    return values


def _make_fixture(tmp_path: Path, *, include_prospective: bool = True) -> _Fixture:
    config = load_yaml_config(_CONFIG_PATH)
    records_by_split: dict[str, list[VolumeRecord]] = {
        "train": [],
        "validation": [],
    }
    volumes: dict[str, torch.Tensor] = {}
    fit: list[PhotometryFitVolume] = []
    for split_index, split in enumerate(("train", "validation")):
        for contrast in CONTRASTS:
            for field in FIELD_STRENGTHS_T:
                domain = Domain(field, contrast)
                case_id = f"R_{split}_{contrast.value}_{field:g}T"
                source_path = tmp_path / "sources" / f"{case_id}.bin"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(f"{case_id}-content".encode("utf-8"))
                record = VolumeRecord(
                    case_id=case_id,
                    image_path=source_path,
                    domain=domain,
                    subject_id=f"{split}-{contrast.value}-{field:g}",
                    split=split,
                    metadata={"prefix": "R"},
                )
                records_by_split[split].append(record)
                volume = _volume(domain, split_index)
                volumes[str(source_path)] = volume
                if split == "train":
                    fit.append(
                        PhotometryFitVolume(
                            volume=volume,
                            domain=domain,
                            record_identity=case_id,
                            subject_identity=record.subject_id,
                            metadata_prefix="R",
                            source_path_identity=str(source_path),
                            source_file_sha256=sha256_file(source_path),
                        )
                    )
    p_path = tmp_path / "sources" / "P_1234_T1w_01T.bin"
    p_path.write_bytes(b"prospective-must-not-be-read")
    if include_prospective:
        records_by_split["train"].append(
            VolumeRecord(
                case_id="P_1234_T1w_01T",
                image_path=p_path,
                domain=Domain(0.1, "T1w"),
                subject_id="1234",
                split="train",
                metadata={"prefix": "P"},
            )
        )
    photometry_config = {
        "contract": "stage2-photometry-variant-a-config-v1",
        "synthetic": True,
    }
    artifact = fit_frozen_photometry(
        fit,
        source_split_file_sha256=_FAKE_SPLIT_SHA,
        source_membership_fingerprint=_FAKE_MEMBERSHIP,
        source_recovery_fingerprint=_FAKE_RECOVERY,
        code_commit="synthetic-photometry-commit",
        code_provenance=_photometry_code_provenance(),
        resolved_config=photometry_config,
        num_quantiles=16,
    )
    qualification = {
        "contract_version": VARIANT_A_QUALIFICATION_CONTRACT_VERSION,
        "artifact_sha256": artifact.artifact_sha256,
        "source_split": {
            "file_sha256": _FAKE_SPLIT_SHA,
            "membership_fingerprint": _FAKE_MEMBERSHIP,
            "recovery_fingerprint": _FAKE_RECOVERY,
        },
        "resolved_config_sha256": artifact.provenance["resolved_config_sha256"],
        "vae_provenance": {
            "config_file_sha256": _VAE_CONFIG_SHA,
            "checkpoint_sha256": _VAE_CHECKPOINT_SHA,
            "encoder_statistic": "posterior_mean",
        },
        "failure_classification": [],
        "canonical_latent_bank_authorized": True,
    }
    qualification["result_sha256"] = sha256_json(qualification)
    return _Fixture(
        records_by_split={key: tuple(value) for key, value in records_by_split.items()},
        volumes=volumes,
        artifact=artifact,
        qualification=qualification,
        config=config,
        code_provenance=_code_provenance(),
        p_path=str(p_path),
    )


def _build_canonical(
    tmp_path: Path,
    fixture: _Fixture,
    *,
    resume: bool = False,
    loader=None,
) -> tuple[Path, dict]:
    out_dir = tmp_path / "canonical"
    volume_loader = loader or (lambda record: fixture.volumes[str(record.image_path)].clone())
    manifest = build_canonical_volume_artifact(
        artifact=fixture.artifact,
        qualification=fixture.qualification,
        records_by_split=fixture.records_by_split,
        config=CanonicalVolumeBuildConfig.from_mapping(fixture.config, out_dir=out_dir),
        resolved_config=fixture.config,
        source_split_file_sha256=_FAKE_SPLIT_SHA,
        source_membership_fingerprint=_FAKE_MEMBERSHIP,
        source_recovery_fingerprint=_FAKE_RECOVERY,
        photometry_artifact_file_sha256=_PHOTOMETRY_FILE_SHA,
        qualification_file_sha256=_QUALIFICATION_FILE_SHA,
        code_provenance=fixture.code_provenance,
        volume_loader=volume_loader,
        resume=resume,
    )
    return out_dir, manifest


def _build_bank(
    tmp_path: Path,
    fixture: _Fixture,
    canonical_dir: Path,
    encoder: _FrozenPosteriorMeanEncoder,
    *,
    resume: bool = False,
) -> tuple[Path, dict]:
    out_dir = tmp_path / "bank"
    manifest = build_photometry_factored_latent_bank(
        encoder=encoder,
        artifact=fixture.artifact,
        qualification=fixture.qualification,
        canonical_dir=canonical_dir,
        config=PhotometryFactoredLatentBankConfig.from_mapping(
            fixture.config, out_dir=out_dir
        ),
        resolved_config=fixture.config,
        photometry_artifact_file_sha256=_PHOTOMETRY_FILE_SHA,
        qualification_file_sha256=_QUALIFICATION_FILE_SHA,
        vae_config_sha256=_VAE_CONFIG_SHA,
        vae_checkpoint_sha256=_VAE_CHECKPOINT_SHA,
        code_provenance=fixture.code_provenance,
        device=torch.device("cpu"),
        resume=resume,
    )
    return out_dir, manifest


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_canonical_builder_excludes_every_p_identity_before_array_load(tmp_path) -> None:
    fixture = _make_fixture(tmp_path)
    loaded: list[str] = []

    def loader(record: VolumeRecord) -> torch.Tensor:
        loaded.append(str(record.image_path))
        assert str(record.image_path) != fixture.p_path
        return fixture.volumes[str(record.image_path)].clone()

    _, manifest = _build_canonical(tmp_path, fixture, loader=loader)
    assert len(loaded) == 30
    assert manifest["record_count"] == 30
    assert manifest["eligibility_proof"]["prospective_accepted_count"] == 0
    assert [item["record_identity"] for item in manifest["excluded_prospective_records"]] == [
        "P_1234_T1w_01T"
    ]
    assert all(item["cohort"] == "R" for item in manifest["records"])


def test_canonical_and_bank_exact_resume_are_no_clobber(tmp_path) -> None:
    fixture = _make_fixture(tmp_path)
    canonical_dir, _ = _build_canonical(tmp_path, fixture)
    canonical_before = _artifact_bytes(canonical_dir)

    def forbidden_loader(record: VolumeRecord) -> torch.Tensor:
        raise AssertionError(f"exact resume loaded source array {record.case_id}")

    _build_canonical(tmp_path, fixture, resume=True, loader=forbidden_loader)
    assert _artifact_bytes(canonical_dir) == canonical_before
    with pytest.raises(FileExistsError, match="overwrite"):
        _build_canonical(tmp_path, fixture, resume=False)

    encoder = _FrozenPosteriorMeanEncoder()
    bank_dir, _ = _build_bank(tmp_path, fixture, canonical_dir, encoder)
    assert encoder.calls == 30
    bank_before = _artifact_bytes(bank_dir)
    resumed_encoder = _FrozenPosteriorMeanEncoder()
    _build_bank(tmp_path, fixture, canonical_dir, resumed_encoder, resume=True)
    assert resumed_encoder.calls == 0
    assert _artifact_bytes(bank_dir) == bank_before
    with pytest.raises(FileExistsError, match="overwrite"):
        _build_bank(
            tmp_path,
            fixture,
            canonical_dir,
            _FrozenPosteriorMeanEncoder(),
            resume=False,
        )


def test_source_content_mutation_invalidates_resume_before_array_load(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    _build_canonical(tmp_path, fixture)
    mutated = fixture.records_by_split["train"][0].image_path
    Path(mutated).write_bytes(b"mutated-source-content")

    def forbidden_loader(record: VolumeRecord) -> torch.Tensor:
        raise AssertionError(f"mutation preflight loaded source array {record.case_id}")

    with pytest.raises(ValueError, match="exact resume"):
        _build_canonical(tmp_path, fixture, resume=True, loader=forbidden_loader)


def test_support_mask_round_trip_is_conservative_and_packed() -> None:
    support = torch.ones(1, 1, 8, 8, 8, dtype=torch.bool)
    support[..., 0, 0, 0] = False
    support[..., 4:6, 4:6, 4:6] = False
    downsampled = downsample_source_support(support, factor=2)
    assert downsampled.shape == (4, 4, 4)
    assert not downsampled[0, 0, 0]
    assert not downsampled[2, 2, 2]
    assert int(downsampled.sum()) == 62
    packed = pack_support_mask(downsampled)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 8
    assert torch.equal(unpack_support_mask(packed, downsampled.shape), downsampled)


def test_bank_stats_descriptors_and_all_domains_have_strict_roles(tmp_path) -> None:
    fixture = _make_fixture(tmp_path)
    canonical_dir, canonical_manifest = _build_canonical(tmp_path, fixture)
    bank_dir, bank = _build_bank(
        tmp_path, fixture, canonical_dir, _FrozenPosteriorMeanEncoder()
    )
    expected_domains = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    assert set(canonical_manifest["domain_counts"]["train"]) == expected_domains
    assert set(canonical_manifest["domain_counts"]["validation"]) == expected_domains
    assert set(bank["domain_counts"]["train"]) == expected_domains
    assert set(bank["domain_counts"]["validation"]) == expected_domains
    assert bank["contract_version"] == PHOTOMETRY_FACTORED_LATENT_BANK_VERSION
    assert bank["legacy_contract_mutation"] == "latent-bank-v1-unchanged"

    stats = json.loads((bank_dir / FACTORED_LATENT_STATS_FILE).read_text(encoding="utf-8"))
    assert stats["computed_over"] == {"cohort": "R", "split": "train"}
    assert stats["record_count"] == 15
    assert {item["record_identity"] for item in stats["records"]} == {
        item["record_identity"] for item in bank["records"] if item["split"] == "train"
    }
    descriptors = json.loads(
        (bank_dir / STRUCTURAL_DESCRIPTOR_MANIFEST).read_text(encoding="utf-8")
    )
    assert descriptors["contract_version"] == PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION
    assert descriptors["computed_over"] == {"cohort": "R", "split": "train"}
    assert descriptors["record_count"] == 15
    assert all(item["subject_group_identity"].startswith("R:") for item in descriptors["records"])
    assert all(item["paired_endpoint_or_target_input"] == "none" for item in descriptors["records"])
    assert all(item["split"] == "train" for item in descriptors["records"])

    first = torch.load(
        bank_dir / descriptors["records"][0]["path"],
        map_location="cpu",
        weights_only=False,
    )
    assert first["descriptor"].ndim == 1
    assert first["descriptor"].dtype == torch.float32
    assert first["input"] == "canonical_standardized_latent_only"
    assert all(
        not ({"target", "target_domain", "paired_endpoint"} & set(item))
        for item in descriptors["records"]
    )

    report = audit_photometry_factored_latent_bank(
        root=bank_dir,
        canonical_dir=canonical_dir,
        encoder=_FrozenPosteriorMeanEncoder(),
        artifact=fixture.artifact,
        qualification=fixture.qualification,
        resolved_config=fixture.config,
        photometry_artifact_file_sha256=_PHOTOMETRY_FILE_SHA,
        qualification_file_sha256=_QUALIFICATION_FILE_SHA,
        vae_config_sha256=_VAE_CONFIG_SHA,
        vae_checkpoint_sha256=_VAE_CHECKPOINT_SHA,
        device=torch.device("cpu"),
    )
    assert report["train_statistics_verified"] is True
    assert report["structural_descriptors_verified"] is True


def test_manifest_and_payload_hash_mutation_fail_closed(tmp_path) -> None:
    fixture = _make_fixture(tmp_path)
    canonical_dir, _ = _build_canonical(tmp_path, fixture)
    manifest_path = canonical_dir / "canonical_volume_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutated = json.loads(json.dumps(original))
    mutated["records"][0]["canonical_dtype"] = "float16"
    manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_canonical_volume_manifest(canonical_dir)
    manifest_path.write_text(
        json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    bank_dir, bank = _build_bank(
        tmp_path, fixture, canonical_dir, _FrozenPosteriorMeanEncoder()
    )
    record_path = bank_dir / bank["records"][0]["path"]
    payload = torch.load(record_path, map_location="cpu", weights_only=False)
    payload["latent"] = payload["latent"] + 1
    torch.save(payload, record_path)
    with pytest.raises(ValueError, match="file hash"):
        audit_photometry_factored_latent_bank(
            root=bank_dir,
            canonical_dir=canonical_dir,
            encoder=_FrozenPosteriorMeanEncoder(),
            artifact=fixture.artifact,
            qualification=fixture.qualification,
            resolved_config=fixture.config,
            photometry_artifact_file_sha256=_PHOTOMETRY_FILE_SHA,
            qualification_file_sha256=_QUALIFICATION_FILE_SHA,
            vae_config_sha256=_VAE_CONFIG_SHA,
            vae_checkpoint_sha256=_VAE_CHECKPOINT_SHA,
            device=torch.device("cpu"),
        )


def test_canonical_audit_recomputes_source_and_rejects_qualification_mutation(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    canonical_dir, _ = _build_canonical(tmp_path, fixture)
    report = audit_canonical_volume_artifact(
        root=canonical_dir,
        artifact=fixture.artifact,
        qualification=fixture.qualification,
        records_by_split=fixture.records_by_split,
        source_split_file_sha256=_FAKE_SPLIT_SHA,
        source_membership_fingerprint=_FAKE_MEMBERSHIP,
        source_recovery_fingerprint=_FAKE_RECOVERY,
        photometry_artifact_file_sha256=_PHOTOMETRY_FILE_SHA,
        qualification_file_sha256=_QUALIFICATION_FILE_SHA,
        volume_loader=lambda record: fixture.volumes[str(record.image_path)].clone(),
    )
    assert report["source_content_verified"] is True
    mutated = json.loads(json.dumps(fixture.qualification))
    mutated["canonical_latent_bank_authorized"] = False
    with pytest.raises(ValueError, match="result hash"):
        audit_canonical_volume_artifact(
            root=canonical_dir,
            artifact=fixture.artifact,
            qualification=mutated,
            records_by_split=fixture.records_by_split,
            source_split_file_sha256=_FAKE_SPLIT_SHA,
            source_membership_fingerprint=_FAKE_MEMBERSHIP,
            source_recovery_fingerprint=_FAKE_RECOVERY,
            photometry_artifact_file_sha256=_PHOTOMETRY_FILE_SHA,
            qualification_file_sha256=_QUALIFICATION_FILE_SHA,
            volume_loader=lambda record: fixture.volumes[str(record.image_path)].clone(),
        )


def test_structural_descriptor_is_deterministic_and_uses_standardized_latents() -> None:
    latent = torch.arange(2 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4, 4)
    support = torch.ones(4, 4, 4, dtype=torch.bool)
    support[0] = False
    first = structural_descriptor(latent, support)
    second = structural_descriptor(latent.clone(), support.clone())
    assert torch.equal(first, second)
    assert first.shape == (2 * (1 + 8 + 64) * 4,)
    assert torch.isfinite(first).all()


def test_existing_latent_bank_v1_reader_behavior_is_unchanged(tmp_path) -> None:
    assert LATENT_BANK_CONTRACT_VERSION == "latent-bank-v1"
    assert PHOTOMETRY_FACTORED_LATENT_BANK_VERSION != LATENT_BANK_CONTRACT_VERSION
    latent_dir = tmp_path / "legacy" / "train"
    latent_dir.mkdir(parents=True)
    payload = {
        "contract_version": LATENT_BANK_CONTRACT_VERSION,
        "case_id": "legacy_case",
        "subject_id": "legacy_subject",
        "split": "train",
        "domain": Domain(0.1, "T1w").to_dict(),
        "latent": torch.ones(2, 2, 2, 2),
    }
    torch.save(payload, latent_dir / "legacy_case.pt")
    index = LatentBankIndex(tmp_path / "legacy", "train")
    assert len(index) == 1
    assert torch.equal(index.load_latent(0), payload["latent"])
    assert not (tmp_path / "legacy" / FACTORED_LATENT_BANK_MANIFEST).exists()


def test_new_artifact_commands_are_exposed_in_help() -> None:
    from fieldbridge.cli import build_parser

    help_text = build_parser().format_help()
    for command in (
        "build-stage2-canonical-volumes",
        "audit-stage2-canonical-volumes",
        "build-photometry-factored-latent-bank",
        "audit-photometry-factored-latent-bank",
    ):
        assert command in help_text
