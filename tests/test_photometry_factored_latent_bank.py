from __future__ import annotations

import errno
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.latent_bank import LATENT_BANK_CONTRACT_VERSION
from fieldbridge.data.latent_bank_dataset import LatentBankIndex
from fieldbridge.data.photometry_factored_latent_bank import (
    FACTORED_LATENT_STATS_FILE,
    COMPLETE_DEPENDENCY_SUPPORT_DIAGNOSTIC_VERSION,
    GROUPNORM_DEPENDENCY_PROVENANCE_VERSION,
    LOCAL_VALID_CORE_SUPPORT_RULE,
    PHOTOMETRY_FACTORED_AUDIT_VERSION,
    PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION,
    PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
    PHOTOMETRY_FACTORED_RESUME_VERSION,
    STRUCTURAL_DESCRIPTOR_MANIFEST,
    MaskedChannelWelford,
    PhotometryFactoredLatentBankConfig,
    audit_photometry_factored_latent_bank,
    build_photometry_factored_latent_bank,
    derive_encoder_local_support_rule,
    pack_support_mask,
    propagate_encoder_complete_dependency_diagnostic,
    propagate_encoder_local_valid_core_support,
    receptive_field_source_bounds,
    standardize_supported_latent,
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
    AtomicPublicationUnavailable,
    COMPUTATIONAL_PROVENANCE_VERSION,
    DEPENDENCY_MAP_VERSION,
    REVIEWED_DEPENDENCY_MAP,
    atomic_torch_save_no_clobber,
    preflight_atomic_no_clobber_filesystem,
    storage_tensor_sha256,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    VARIANT_A_QUALIFICATION_CONTRACT_VERSION,
)
from fieldbridge.models.autoencoders.kl_vae import KLVAEEncoder

_CONFIG_PATH = Path("configs/experiment/stage2_canonical_artifacts_v2.yaml")
_FAKE_SPLIT_SHA = sha256_text("synthetic canonical split file")
_FAKE_MEMBERSHIP = sha256_text("synthetic membership")
_FAKE_RECOVERY = sha256_text("synthetic recovery")
_VAE_CONFIG_SHA = sha256_text("synthetic frozen VAE config")
_VAE_CHECKPOINT_SHA = sha256_text("synthetic frozen VAE checkpoint")
_PHOTOMETRY_FILE_SHA = sha256_text("synthetic photometry artifact file")
_QUALIFICATION_FILE_SHA = sha256_text("synthetic qualification file")


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
    dependency_map = {
        path: list(responsibilities)
        for path, responsibilities in sorted(REVIEWED_DEPENDENCY_MAP.items())
    }
    runtime = {
        "python_version": "3.synthetic",
        "python_implementation": "CPython",
        "platform": "synthetic-platform",
        "torch_version": "synthetic-torch",
        "numpy_version": "synthetic-numpy",
        "cuda_compiled_version": None,
        "cuda_available": False,
        "cudnn_version": None,
        "device": {"type": "cpu", "index": None, "name": None, "capability": None},
    }
    payload = {
        "contract_version": COMPUTATIONAL_PROVENANCE_VERSION,
        "git_head": "synthetic-commit",
        "checkout_clean": True,
        "dependency_map_version": DEPENDENCY_MAP_VERSION,
        "dependency_map": dependency_map,
        "dependency_map_sha256": sha256_json(
            {"version": DEPENDENCY_MAP_VERSION, "modules": dependency_map}
        ),
        "module_sha256": {path: sha256_text(path) for path in dependency_map},
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
    }
    payload["provenance_sha256"] = sha256_json(payload)
    return payload


def _photometry_code_provenance() -> dict:
    return {
        "git_head": "synthetic-photometry-commit",
        "checkout_clean": True,
        "module_sha256": {path: sha256_text(path) for path in PHOTOMETRY_SOURCE_MODULES},
    }


def _encoder(*, use_norm: bool = False, num_res_blocks: int = 1) -> KLVAEEncoder:
    torch.manual_seed(1234)
    encoder = KLVAEEncoder(
        in_channels=1,
        base_channels=2,
        latent_channels=2,
        spatial_dims=3,
        activation="silu",
        use_norm=use_norm,
        num_res_blocks=num_res_blocks,
    )
    return encoder.requires_grad_(False).eval()


def _volume(domain: Domain, split_index: int) -> torch.Tensor:
    values = torch.linspace(0.05, 0.85, 16**3, dtype=torch.float32).reshape(
        1, 1, 16, 16, 16
    )
    return (
        values
        + 0.004 * domain.contrast_index
        + 0.002 * FIELD_STRENGTHS_T.index(domain.field_strength_t)
        + 0.001 * split_index
    ).clamp_max(0.99)


def _make_fixture(tmp_path: Path, *, include_prospective: bool = True) -> _Fixture:
    config = load_yaml_config(_CONFIG_PATH)
    records_by_split: dict[str, list[VolumeRecord]] = {"train": [], "validation": []}
    volumes: dict[str, torch.Tensor] = {}
    fit: list[PhotometryFitVolume] = []
    for split_index, split in enumerate(("train", "validation")):
        for contrast in CONTRASTS:
            for field in FIELD_STRENGTHS_T:
                domain = Domain(field, contrast)
                case_id = f"R_{split}_{contrast.value}_{field:g}T"
                source_path = tmp_path / "sources" / f"{case_id}.bin"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(f"{case_id}-content".encode())
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


def _common(fixture: _Fixture, encoder: KLVAEEncoder) -> dict:
    return {
        "encoder": encoder,
        "artifact": fixture.artifact,
        "qualification": fixture.qualification,
        "records_by_split": fixture.records_by_split,
        "resolved_config": fixture.config,
        "source_split_file_sha256": _FAKE_SPLIT_SHA,
        "source_membership_fingerprint": _FAKE_MEMBERSHIP,
        "source_recovery_fingerprint": _FAKE_RECOVERY,
        "photometry_artifact_file_sha256": _PHOTOMETRY_FILE_SHA,
        "qualification_file_sha256": _QUALIFICATION_FILE_SHA,
        "vae_config_sha256": _VAE_CONFIG_SHA,
        "vae_checkpoint_sha256": _VAE_CHECKPOINT_SHA,
        "code_provenance": fixture.code_provenance,
        "source_shape_resolver": lambda record: fixture.volumes[str(record.image_path)].shape,
        "device": torch.device("cpu"),
    }


def _build(
    tmp_path: Path,
    fixture: _Fixture,
    encoder: KLVAEEncoder,
    *,
    resume: bool = False,
    loader=None,
    source_file_hasher=sha256_file,
) -> tuple[Path, dict]:
    out_dir = tmp_path / "bank"
    manifest = build_photometry_factored_latent_bank(
        **_common(fixture, encoder),
        config=PhotometryFactoredLatentBankConfig.from_mapping(
            fixture.config, out_dir=out_dir
        ),
        volume_loader=loader
        or (lambda record: fixture.volumes[str(record.image_path)].clone()),
        source_file_hasher=source_file_hasher,
        resume=resume,
    )
    return out_dir, manifest


def _audit(
    root: Path,
    fixture: _Fixture,
    encoder: KLVAEEncoder,
    *,
    code_provenance: dict | None = None,
    loader=None,
) -> dict:
    common = _common(fixture, encoder)
    common["code_provenance"] = code_provenance or fixture.code_provenance
    return audit_photometry_factored_latent_bank(
        **common,
        root=root,
        volume_loader=loader
        or (lambda record: fixture.volumes[str(record.image_path)].clone()),
    )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mutated_provenance(original: dict, module: str) -> dict:
    value = json.loads(json.dumps(original))
    value["module_sha256"][module] = sha256_text(f"mutated:{module}")
    value.pop("provenance_sha256")
    value["provenance_sha256"] = sha256_json(value)
    return value


def test_streamed_builder_rejects_every_p_before_hash_shape_or_array_load(tmp_path) -> None:
    fixture = _make_fixture(tmp_path)
    loaded: list[str] = []

    def loader(record: VolumeRecord) -> torch.Tensor:
        loaded.append(str(record.image_path))
        assert str(record.image_path) != fixture.p_path
        return fixture.volumes[str(record.image_path)].clone()

    def hasher(path: str | Path) -> str:
        assert str(path) != fixture.p_path
        return sha256_file(path)

    _, manifest = _build(
        tmp_path, fixture, _encoder(), loader=loader, source_file_hasher=hasher
    )
    assert len(loaded) == 30
    assert manifest["record_count"] == 30
    assert manifest["eligibility_proof"]["prospective_accepted_count"] == 0
    assert [item["record_identity"] for item in manifest["excluded_prospective_records"]] == [
        "P_1234_T1w_01T"
    ]
    assert manifest["canonical_stream"]["full_canonical_tensor_persisted"] is False
    assert not list((tmp_path / "bank").rglob("*canonical*"))


def test_streamed_exact_resume_is_byte_exact_and_does_not_load_arrays(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    root, _ = _build(tmp_path, fixture, _encoder())
    before = _artifact_bytes(root)

    def forbidden_loader(record: VolumeRecord) -> torch.Tensor:
        raise AssertionError(f"exact resume loaded {record.case_id}")

    _build(tmp_path, fixture, _encoder(), resume=True, loader=forbidden_loader)
    assert _artifact_bytes(root) == before
    with pytest.raises(FileExistsError, match="overwrite"):
        _build(tmp_path, fixture, _encoder(), resume=False)


def test_interrupted_partial_build_resumes_without_reencoding_published_record(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    calls = 0

    def interrupted_loader(record: VolumeRecord) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return fixture.volumes[str(record.image_path)].clone()

    with pytest.raises(RuntimeError, match="interruption"):
        _build(tmp_path, fixture, _encoder(), loader=interrupted_loader)
    published = list((tmp_path / "bank" / "latents").rglob("*.pt"))
    assert len(published) == 1
    first_bytes = published[0].read_bytes()
    resumed_loads: list[str] = []

    def resumed_loader(record: VolumeRecord) -> torch.Tensor:
        resumed_loads.append(record.case_id)
        return fixture.volumes[str(record.image_path)].clone()

    root, manifest = _build(
        tmp_path, fixture, _encoder(), resume=True, loader=resumed_loader
    )
    assert manifest["record_count"] == 30
    assert len(resumed_loads) == 29
    assert published[0].read_bytes() == first_bytes
    assert not list(root.rglob("*.tmp"))


def test_source_content_mutation_invalidates_resume_before_array_load(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    _build(tmp_path, fixture, _encoder())
    Path(fixture.records_by_split["train"][0].image_path).write_bytes(b"mutated")

    def forbidden_loader(record: VolumeRecord) -> torch.Tensor:
        raise AssertionError(f"mutation preflight loaded {record.case_id}")

    with pytest.raises(ValueError, match="exact resume"):
        _build(tmp_path, fixture, _encoder(), resume=True, loader=forbidden_loader)


@pytest.mark.parametrize(("num_res_blocks", "receptive_field"), [(1, 37), (2, 65)])
def test_conv_and_residual_valid_core_erosion_is_exact(
    num_res_blocks: int, receptive_field: int
) -> None:
    encoder = _encoder(use_norm=False, num_res_blocks=num_res_blocks)
    rule = derive_encoder_local_support_rule(encoder)
    assert rule["contract_version"] == LOCAL_VALID_CORE_SUPPORT_RULE
    assert rule["scope"] == "anatomical-spatial-validity"
    assert rule["convolutional_receptive_field_size"] == [receptive_field] * 3
    assert rule["output_stride"] == [4, 4, 4]

    for unsupported_index in ((40, 40, 40), (0, 0, 0)):
        source = torch.ones(1, 1, 80, 80, 80, dtype=torch.bool)
        source[(0, 0, *unsupported_index)] = False
        propagated = propagate_encoder_local_valid_core_support(source, encoder)
        for z in range(propagated.shape[0]):
            for y in range(propagated.shape[1]):
                for x in range(propagated.shape[2]):
                    bounds = receptive_field_source_bounds(
                        rule, (z, y, x), source.shape[-3:]
                    )
                    depends = all(
                        bounds[axis][0] <= unsupported_index[axis] <= bounds[axis][1]
                        for axis in range(3)
                    )
                    assert bool(propagated[z, y, x]) is (not depends)


def test_groupnorm_records_global_dependence_without_collapsing_local_support() -> None:
    encoder = _encoder(use_norm=True, num_res_blocks=1)
    rule = derive_encoder_local_support_rule(encoder)
    provenance = rule["groupnorm_dependency_provenance"]
    assert provenance["contract_version"] == GROUPNORM_DEPENDENCY_PROVENANCE_VERSION
    assert provenance["local_support_mask_action"] == (
        "record-separately-without-spatial-collapse"
    )
    assert len(provenance["modules"]) == 6
    assert all(
        set(module) == {"module_path", "type", "channels", "groups", "epsilon"}
        and module["type"].endswith(".GroupNorm")
        for module in provenance["modules"]
    )
    support = torch.ones(1, 1, 80, 80, 80, dtype=torch.bool)
    support[..., 40, 40, 40] = False
    local = propagate_encoder_local_valid_core_support(support, encoder)
    pointwise_norm_reference = propagate_encoder_local_valid_core_support(
        support, _encoder(use_norm=False, num_res_blocks=1)
    )
    assert bool(local.any())
    assert torch.equal(local, pointwise_norm_reference)
    diagnostic = propagate_encoder_complete_dependency_diagnostic(support, encoder)
    assert not bool(diagnostic.any())
    assert rule["optional_diagnostic"] == {
        "contract_version": COMPLETE_DEPENDENCY_SUPPORT_DIAGNOSTIC_VERSION,
        "operational": False,
        "blocking": False,
    }
    assert rule["normalization_statistical_independence_claim"] is False


def test_empty_complete_dependency_diagnostic_does_not_block_bank(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    fixture.config["latent_bank"]["complete_dependency_diagnostic"]["enabled"] = True
    for identity, volume in fixture.volumes.items():
        resized = torch.nn.functional.interpolate(
            volume, size=(24, 24, 24), mode="trilinear", align_corners=False
        )
        resized[..., 0, 0, 0] = 0
        fixture.volumes[identity] = resized
    root, manifest = _build(
        tmp_path, fixture, _encoder(use_norm=True, num_res_blocks=1)
    )
    assert manifest["complete_dependency_diagnostic_enabled"] is True
    assert all(
        entry["sidecar"]["local_valid_core_support_nonzero_count"] > 0
        and entry["sidecar"]["complete_dependency_diagnostic"]["empty"] is True
        and entry["sidecar"]["complete_dependency_diagnostic"]["blocking"] is False
        for entry in manifest["records"]
    )
    assert _audit(root, fixture, _encoder(use_norm=True, num_res_blocks=1))[
        "all_records_verified"
    ] is True


def test_actual_frozen_klvae_has_nonempty_local_valid_core_support() -> None:
    model = dict(load_yaml_config(Path("configs/experiment/stage1_vae.yaml"))["model"])
    assert model.pop("name") == "kl_vae"
    encoder = KLVAEEncoder(**model).requires_grad_(False).eval()
    support = torch.zeros(1, 1, 80, 80, 80, dtype=torch.bool)
    support[..., 4:-4, 4:-4, 4:-4] = True
    local = propagate_encoder_local_valid_core_support(support, encoder)
    rule = derive_encoder_local_support_rule(encoder)
    assert bool(local.any())
    assert rule["convolutional_receptive_field_size"] == [65, 65, 65]
    assert len(rule["groupnorm_dependency_provenance"]["modules"]) == 12


def test_local_valid_core_support_fails_closed_on_invalid_mask_or_graph() -> None:
    encoder = _encoder()
    with pytest.raises(ValueError, match="empty"):
        propagate_encoder_local_valid_core_support(
            torch.zeros(1, 1, 16, 16, 16, dtype=torch.bool), encoder
        )
    nonfinite = torch.ones(1, 1, 16, 16, 16)
    nonfinite[..., 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        propagate_encoder_local_valid_core_support(nonfinite, encoder)
    with pytest.raises(ValueError, match="requires"):
        propagate_encoder_local_valid_core_support(
            torch.ones(16, 16, dtype=torch.bool), encoder
        )
    unsupported = _encoder()
    unsupported.res1[0].act1 = torch.nn.MaxPool3d(2)
    with pytest.raises(ValueError, match="Unsupported encoder activation"):
        derive_encoder_local_support_rule(unsupported)


def test_packed_support_round_trip_seals_actual_mask() -> None:
    support = torch.ones(5, 4, 3, dtype=torch.bool)
    support[0, 0, 0] = False
    support[-1, -1, -1] = False
    packed = pack_support_mask(support)
    assert packed.dtype == torch.uint8
    assert packed.numel() == math.ceil(support.numel() / 8)
    restored = unpack_support_mask(packed, support.shape)
    assert torch.equal(restored, support)
    assert storage_tensor_sha256(restored) == storage_tensor_sha256(support)


def test_masked_welford_and_descriptor_ignore_arbitrary_unsupported_values() -> None:
    support = torch.zeros(4, 4, 4, dtype=torch.bool)
    support[1:3, 1:3, 1:3] = True
    first = torch.arange(2 * 4**3, dtype=torch.float32).reshape(2, 4, 4, 4)
    second = first * 1.5 + 3
    changed_first = first.clone()
    changed_second = second.clone()
    changed_first[:, ~support] = 1.0e20
    changed_second[:, ~support] = -1.0e20
    left = MaskedChannelWelford(2)
    right = MaskedChannelWelford(2)
    for original, changed in ((first, changed_first), (second, changed_second)):
        left.update(original, support)
        right.update(changed, support)
    left_stats = left.compute()
    assert right.compute() == left_stats
    standardized_left = standardize_supported_latent(
        first, support, left_stats["per_channel_mean"], left_stats["per_channel_std"]
    )
    standardized_right = standardize_supported_latent(
        changed_first,
        support,
        left_stats["per_channel_mean"],
        left_stats["per_channel_std"],
    )
    assert torch.equal(standardized_left, standardized_right)
    assert torch.equal(
        structural_descriptor(standardized_left, support),
        structural_descriptor(standardized_right, support),
    )


def test_masked_statistics_reject_empty_nonfinite_and_degenerate_values() -> None:
    accumulator = MaskedChannelWelford(1)
    with pytest.raises(ValueError, match="empty"):
        accumulator.update(torch.ones(1, 2, 2, 2), torch.zeros(2, 2, 2, dtype=torch.bool))
    bad = torch.ones(1, 2, 2, 2)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.update(bad, torch.ones(2, 2, 2, dtype=torch.bool))
    degenerate = MaskedChannelWelford(1)
    degenerate.update(torch.ones(1, 2, 2, 2), torch.ones(2, 2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="degenerate"):
        degenerate.compute()


def test_bank_statistics_domains_streaming_and_descriptor_boundary(tmp_path) -> None:
    fixture = _make_fixture(tmp_path)
    root, manifest = _build(tmp_path, fixture, _encoder())
    expected_domains = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    assert set(manifest["domain_counts"]["train"]) == expected_domains
    assert set(manifest["domain_counts"]["validation"]) == expected_domains
    assert manifest["encoding"]["strategy_requested"] == "full"
    assert manifest["encoding"]["path_used_required"] == "full"
    assert manifest["encoding"]["oom_fallback"] == "forbidden-hard-stop"
    assert manifest["resume_contract_version"] == PHOTOMETRY_FACTORED_RESUME_VERSION
    assert manifest["audit_contract_version"] == PHOTOMETRY_FACTORED_AUDIT_VERSION
    assert manifest["operational_support_rule"]["contract_version"] == (
        LOCAL_VALID_CORE_SUPPORT_RULE
    )
    assert manifest["operational_support_rule"][
        "normalization_statistical_independence_claim"
    ] is False
    assert manifest["complete_dependency_diagnostic_enabled"] is False
    assert manifest["storage_preflight"]["full_volume_bytes_avoided"] > 0
    assert manifest["storage_preflight"]["predicted_latent_bytes"] > 0
    assert all(
        entry["sidecar"]["encoding_path"]["path_used"] == "full"
        and entry["sidecar"]["canonical_persisted"] is False
        and entry["sidecar"]["operational_support_contract"]
        == LOCAL_VALID_CORE_SUPPORT_RULE
        and entry["sidecar"]["complete_dependency_diagnostic"] is None
        for entry in manifest["records"]
    )

    stats = json.loads((root / FACTORED_LATENT_STATS_FILE).read_text())
    assert stats["computed_over"] == {
        "cohort": "R",
        "split": "train",
        "cells": "encoder_local_valid_core_only",
    }
    assert stats["record_count"] == 15
    assert stats["algorithm"] == "channelwise-masked-welford-float64-v1"
    assert stats["minimum_channel_variance"] == 1.0e-12
    assert len(stats["per_channel_supported_count"]) == 2
    assert stats["total_supported_value_count"] == sum(
        stats["per_channel_supported_count"]
    )
    descriptors = json.loads((root / STRUCTURAL_DESCRIPTOR_MANIFEST).read_text())
    assert descriptors["record_count"] == 15
    assert descriptors["coupling_authorized"] is False
    assert (
        descriptors["qualification_required"]
        == PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION
    )
    assert descriptors["learned_disentanglement_claim"] == "forbidden"
    assert descriptors["qualification_requirements"] == {
        "retrospective_split": "validation",
        "subject_group_exclusion": "required",
        "subject_retrieval": "must-demonstrate-retained-instance-signal",
        "field_predictability": "must-demonstrate-reduction",
        "support_volume_shortcuts": "must-test-and-rule-out",
        "descriptor_stability": "must-demonstrate",
    }
    assert all(
        entry["sidecar"]["subject_group_identity"].startswith("R:")
        and entry["sidecar"]["coupling_authorized"] is False
        for entry in descriptors["records"]
    )
    report = _audit(root, fixture, _encoder())
    assert report["source_to_N_d_to_E_recomputed"] is True
    assert report["masked_train_statistics_verified"] is True
    assert report["structural_descriptors_verified"] is True
    assert report["contract_version"] == PHOTOMETRY_FACTORED_AUDIT_VERSION


def test_every_dependency_identity_mutation_invalidates_resume_and_audit(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    root, _ = _build(tmp_path, fixture, _encoder())
    for module in REVIEWED_DEPENDENCY_MAP:
        mutated = _mutated_provenance(fixture.code_provenance, module)
        original = fixture.code_provenance
        fixture.code_provenance = mutated
        try:
            with pytest.raises(ValueError, match="exact resume"):
                _build(tmp_path, fixture, _encoder(), resume=True)
        finally:
            fixture.code_provenance = original
        with pytest.raises(ValueError, match="resume/audit"):
            _audit(root, fixture, _encoder(), code_provenance=mutated)

    runtime_mutation = json.loads(json.dumps(fixture.code_provenance))
    runtime_mutation["runtime"]["python_version"] = "changed-runtime"
    runtime_mutation["runtime_sha256"] = sha256_json(runtime_mutation["runtime"])
    runtime_mutation.pop("provenance_sha256")
    runtime_mutation["provenance_sha256"] = sha256_json(runtime_mutation)
    original = fixture.code_provenance
    fixture.code_provenance = runtime_mutation
    try:
        with pytest.raises(ValueError, match="exact resume"):
            _build(tmp_path, fixture, _encoder(), resume=True)
    finally:
        fixture.code_provenance = original
    with pytest.raises(ValueError, match="resume/audit"):
        _audit(root, fixture, _encoder(), code_provenance=runtime_mutation)


def test_local_rule_and_groupnorm_provenance_mutations_invalidate_resume_and_audit(
    tmp_path,
) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    root, _ = _build(tmp_path, fixture, _encoder(use_norm=True, num_res_blocks=1))

    changed_local_graph = _encoder(use_norm=True, num_res_blocks=2)
    with pytest.raises(ValueError, match="exact resume"):
        _build(tmp_path, fixture, changed_local_graph, resume=True)
    with pytest.raises(ValueError, match="resume/audit"):
        _audit(root, fixture, changed_local_graph)

    changed_groupnorm = _encoder(use_norm=True, num_res_blocks=1)
    changed_groupnorm.res1[0].norm1.eps = 2.0e-5
    changed_rule = derive_encoder_local_support_rule(changed_groupnorm)
    original_rule = derive_encoder_local_support_rule(_encoder(use_norm=True, num_res_blocks=1))
    assert changed_rule["convolutional_receptive_field_size"] == original_rule[
        "convolutional_receptive_field_size"
    ]
    assert changed_rule["groupnorm_dependency_provenance_sha256"] != original_rule[
        "groupnorm_dependency_provenance_sha256"
    ]
    with pytest.raises(ValueError, match="exact resume"):
        _build(tmp_path, fixture, changed_groupnorm, resume=True)
    with pytest.raises(ValueError, match="resume/audit"):
        _audit(root, fixture, changed_groupnorm)


def test_payload_hash_mutation_fails_complete_audit(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    root, manifest = _build(tmp_path, fixture, _encoder())
    path = root / manifest["records"][0]["path"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["latent"] = payload["latent"] + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="file hash"):
        _audit(root, fixture, _encoder())


@pytest.mark.parametrize("error_number", [errno.ENOTSUP, errno.EPERM])
def test_filesystem_preflight_rejects_unsupported_hardlinks_before_processing(
    tmp_path, error_number
) -> None:
    def unsupported(source, destination) -> None:
        raise OSError(error_number, "unsupported")

    with pytest.raises(AtomicPublicationUnavailable, match="local scratch"):
        preflight_atomic_no_clobber_filesystem(
            tmp_path / "output", required_free_bytes=0, linker=unsupported
        )
    assert not list((tmp_path / "output").glob(".fieldbridge-atomic-probe*"))


def test_build_filesystem_failure_occurs_before_any_source_array_load(tmp_path) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)

    def unsupported(source, destination) -> None:
        raise OSError(errno.ENOTSUP, "unsupported")

    def forbidden_loader(record: VolumeRecord) -> torch.Tensor:
        raise AssertionError(f"filesystem preflight loaded {record.case_id}")

    out_dir = tmp_path / "bank"
    with pytest.raises(AtomicPublicationUnavailable, match="local scratch"):
        build_photometry_factored_latent_bank(
            **_common(fixture, _encoder()),
            config=PhotometryFactoredLatentBankConfig.from_mapping(
                fixture.config, out_dir=out_dir
            ),
            volume_loader=forbidden_loader,
            publication_linker=unsupported,
        )
    assert not list(out_dir.rglob("*.pt"))
    assert not list(out_dir.glob(".fieldbridge-atomic-probe*"))


def test_atomic_publication_handles_concurrent_destination_and_cleans_temporary(tmp_path) -> None:
    destination = tmp_path / "record.pt"

    def concurrent(source, target) -> None:
        Path(target).write_bytes(b"concurrent-writer")
        raise FileExistsError(errno.EEXIST, "exists")

    with pytest.raises(FileExistsError, match="overwrite"):
        atomic_torch_save_no_clobber(destination, {"value": torch.ones(1)}, linker=concurrent)
    assert destination.read_bytes() == b"concurrent-writer"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_publication_interruption_and_enotsup_leave_no_partial_destination(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "record.pt"
    original_save = torch.save

    def interrupted(payload, path) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_torch_save_no_clobber(destination, {"value": torch.ones(1)})
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(torch, "save", original_save)

    def unsupported(source, target) -> None:
        raise OSError(errno.ENOTSUP, "unsupported")

    with pytest.raises(AtomicPublicationUnavailable):
        atomic_torch_save_no_clobber(destination, {"value": torch.ones(1)}, linker=unsupported)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_full_encode_oom_is_hard_stop_without_tiled_fallback(tmp_path, monkeypatch) -> None:
    fixture = _make_fixture(tmp_path, include_prospective=False)
    encoder = _encoder()

    def oom(volume, domain):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(encoder, "encode_dist", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        _build(tmp_path, fixture, encoder)
    assert not list((tmp_path / "bank").rglob("*.pt"))
    assert not (tmp_path / "bank" / "photometry_factored_latent_bank_manifest.json").exists()


def test_primary_config_rejects_tiled_strategy() -> None:
    config = load_yaml_config(_CONFIG_PATH)
    config["latent_bank"]["strategy"] = "tiled"
    with pytest.raises(ValueError, match="full encoding"):
        PhotometryFactoredLatentBankConfig.from_mapping(config, out_dir="external")


def test_v2_config_rejects_old_support_resume_and_audit_plans() -> None:
    current = load_yaml_config(_CONFIG_PATH)
    mutations = (
        ("contract", "stage2-canonical-artifacts-config-v1"),
        ("latent_bank.contract", "photometry-factored-latent-bank-v1"),
        (
            "latent_bank.operational_support.contract",
            "frozen-encoder-dependency-propagation-v1",
        ),
        ("latent_bank.resume_contract", "photometry-factored-latent-bank-resume-v1"),
        ("latent_bank.audit_contract", "photometry-factored-latent-bank-audit-v1"),
    )
    for dotted, value in mutations:
        mutated = deepcopy(current)
        target = mutated
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        with pytest.raises(ValueError, match="incompatible"):
            PhotometryFactoredLatentBankConfig.from_mapping(mutated, out_dir="external")


def test_operational_support_contract_disclaims_normalization_statistical_independence() -> None:
    rule = derive_encoder_local_support_rule(_encoder(use_norm=True))
    assert rule["scope"] == "anatomical-spatial-validity"
    assert rule["normalization_statistical_independence_claim"] is False
    assert "does not establish independence" in rule["scope_statement"]
    assert rule["optional_diagnostic"]["operational"] is False

    for path in (
        Path("src/fieldbridge/data/photometry_factored_latent_bank.py"),
        Path("configs/experiment/stage2_canonical_artifacts_v2.yaml"),
        Path("docs/stage2_canonical_artifacts_runbook.md"),
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "operational_support_claims_normalization_independence: true" not in text


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


def test_streamed_artifact_commands_are_exposed_without_canonical_persistence_commands() -> None:
    from fieldbridge.cli import build_parser

    help_text = build_parser().format_help()
    for command in (
        "preflight-photometry-factored-latent-bank",
        "build-photometry-factored-latent-bank",
        "audit-photometry-factored-latent-bank",
    ):
        assert command in help_text
    assert "build-stage2-canonical-volumes" not in help_text
    assert "audit-stage2-canonical-volumes" not in help_text
    assert "train-stage2-field-graph" not in help_text
