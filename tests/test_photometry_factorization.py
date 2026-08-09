from __future__ import annotations

import copy
import inspect
import json
import os

import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_DUPLICATE_KNOT_RULE,
    PHOTOMETRY_SOURCE_MODULES,
    VARIANT_A_PROSPECTIVE_EXCLUSION_REASON,
    FrozenPhotometryArtifact,
    PhotometryFitVolume,
    all_photometry_domain_labels,
    canonical_tensor_sha256,
    classify_variant_a_cohort,
    fit_frozen_photometry,
    interpolate_fixed_grid,
    sha256_json,
    sha256_text,
)


def _code_provenance(commit: str = "synthetic-commit") -> dict:
    return {
        "git_head": commit,
        "checkout_clean": True,
        "module_sha256": {
            name: f"{index + 1:x}" * 64
            for index, name in enumerate(PHOTOMETRY_SOURCE_MODULES)
        },
    }


def _volume(field: float, contrast_index: int, offset: int = 0) -> torch.Tensor:
    values = torch.linspace(0.05, 0.95, 63, dtype=torch.float32)
    scale = 0.45 + 0.04 * FIELD_STRENGTHS_T.index(field) + 0.03 * contrast_index
    foreground = (values * scale + offset * 0.002).clamp_max(1.0)
    return torch.cat([torch.zeros(1), foreground]).reshape(4, 4, 4)


def _fit_items(*, unequal_counts: bool = False) -> list[PhotometryFitVolume]:
    items: list[PhotometryFitVolume] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            count = 1 + field_index % 3 if unequal_counts else 1
            for offset in range(count):
                identity = f"R_{contrast.value}_{field:g}_{offset}"
                items.append(
                    PhotometryFitVolume(
                        volume=_volume(field, contrast_index, offset),
                        domain=Domain(field, contrast),
                        record_identity=identity,
                        subject_identity=f"R-{contrast_index}-{field_index}-{offset}",
                        metadata_prefix="R",
                        source_path_identity=f"external/{identity}.nii.gz",
                        source_file_sha256=sha256_text(f"file-{identity}"),
                    )
                )
    return items


def _excluded_record(identity: str = "P_0100_T1w") -> dict:
    return {
        "record_identity": identity,
        "record_identity_sha256": sha256_text(identity),
        "subject_identity": "0100",
        "subject_group_identity": "P:0100",
        "metadata_prefix": "P",
        "cohort": "P",
        "split": "train",
        "source_path_identity_sha256": sha256_text(f"external/{identity}.nii.gz"),
        "reason": VARIANT_A_PROSPECTIVE_EXCLUSION_REASON,
    }


def _fit(
    *,
    items: list[PhotometryFitVolume] | None = None,
    excluded: tuple[dict, ...] = (),
) -> FrozenPhotometryArtifact:
    return fit_frozen_photometry(
        _fit_items() if items is None else items,
        source_split_file_sha256="a" * 64,
        source_membership_fingerprint="membership-v1",
        source_recovery_fingerprint="recovery-v3",
        code_commit="synthetic-commit",
        code_provenance=_code_provenance(),
        resolved_config={"contract": "stage2-photometry-variant-a-config-v1"},
        num_quantiles=17,
        excluded_prospective_records=excluded,
    )


def test_fit_requires_all_15_domains_and_seals_exact_equal_weights() -> None:
    artifact = _fit(items=_fit_items(unequal_counts=True))
    assert set(artifact.domain_templates) == set(all_photometry_domain_labels())
    assert len(artifact.domain_templates) == 15
    for template in artifact.domain_templates.values():
        assert template.per_volume_weight == pytest.approx(1.0 / template.volume_count)
    for contrast in CONTRASTS:
        canonical = artifact.canonical_templates[contrast.value]
        assert canonical.per_field_weight == pytest.approx(0.2)
        expected = torch.stack(
            [artifact.domain_templates[label].quantiles for label in canonical.domain_labels]
        ).mean(dim=0)
        torch.testing.assert_close(canonical.quantiles, expected, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="all 15 domains"):
        _fit(items=_fit_items()[:-1])


def test_artifact_is_byte_deterministic_and_seals_record_content(tmp_path) -> None:
    first = _fit(items=list(reversed(_fit_items())))
    second = _fit(items=_fit_items())
    first_path = first.save(tmp_path / "first.json")
    second_path = second.save(tmp_path / "second.json")
    assert first_path.read_bytes() == second_path.read_bytes()
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["artifact_sha256"] == first.artifact_sha256
    assert payload["provenance"]["eligibility_proof"] == {
        "accepted_count": 15,
        "all_cohort_R": True,
        "all_split_train": True,
        "forbidden_traveller_accepted_count": 0,
        "prospective_accepted_count": 0,
        "prospective_excluded_count": 0,
    }
    assert payload["provenance"]["excluded_prospective_records"] == []
    accepted = payload["provenance"]["accepted_records"]
    assert all(item["source_file_sha256"] for item in accepted)
    assert all(item["canonical_loaded_array_sha256"] for item in accepted)


def test_duplicate_knots_flat_inputs_and_endpoint_clamping_are_deterministic() -> None:
    source = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    target = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    mapped = interpolate_fixed_grid(values, source, target)
    # The duplicate source knot is collapsed to the arithmetic mean target (1.0).
    torch.testing.assert_close(
        mapped, torch.tensor([1.0, 1.0, 2.5, 4.0, 4.0], dtype=torch.float64)
    )
    flat = interpolate_fixed_grid(
        torch.tensor([-5.0, 2.0, 9.0]),
        torch.tensor([3.0, 3.0, 3.0]),
        torch.tensor([1.0, 2.0, 4.0]),
    )
    torch.testing.assert_close(flat, torch.full((3,), 7.0 / 3.0))
    assert PHOTOMETRY_DUPLICATE_KNOT_RULE in _fit().to_dict()["duplicate_knot_rule"]


def test_fixed_maps_are_monotonic_finite_and_preserve_exact_source_support() -> None:
    artifact = _fit()
    source_domain = Domain(0.1, "T1w")
    target_domain = Domain(7.0, "T1w")
    source = _volume(0.1, 0)
    canonical = artifact.normalize_source(source, source_domain)
    prediction = artifact.render_target(canonical, target_domain)
    assert torch.isfinite(prediction).all()
    assert torch.equal(canonical.support_mask, source != 0)
    assert bool((canonical.values[source == 0] == 0).all())
    assert bool((prediction[source == 0] == 0).all())
    ordered = prediction[source != 0][torch.argsort(source[source != 0])]
    assert bool((ordered[1:] >= ordered[:-1]).all())


def test_round_trip_uses_fixed_grids_without_runtime_prediction_cdf(monkeypatch) -> None:
    artifact = _fit()
    source = _volume(3.0, 1)
    domain = Domain(3.0, "T2w")
    expected = artifact.factorized_identity(source, domain, domain)

    def forbidden_quantile(*args, **kwargs):
        raise AssertionError("runtime quantile computation is forbidden")

    monkeypatch.setattr(torch, "quantile", forbidden_quantile)
    actual = artifact.factorized_identity(source, domain, domain)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert "target" not in inspect.signature(artifact.normalize_source).parameters
    assert "prediction" not in inspect.signature(artifact.normalize_source).parameters


def test_variant_a5_boundary_repeats_identical_in_memory_tensor_and_support() -> None:
    artifact = _fit()
    source = _volume(0.1, 0)
    first = artifact.normalize_source(source, Domain(0.1, "T1w"))
    second = artifact.normalize_source(source.clone(), Domain(0.1, "T1w"))
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.support_mask, second.support_mask)
    assert canonical_tensor_sha256(first.values) == canonical_tensor_sha256(second.values)
    assert canonical_tensor_sha256(first.support_mask) == canonical_tensor_sha256(
        second.support_mask
    )


def test_fit_rejects_validation_prospective_and_named_travellers() -> None:
    base = _fit_items()
    validation = list(base)
    item = validation[0]
    validation[0] = PhotometryFitVolume(
        volume=item.volume,
        domain=item.domain,
        record_identity=item.record_identity,
        subject_identity=item.subject_identity,
        metadata_prefix=item.metadata_prefix,
        source_path_identity=item.source_path_identity,
        source_file_sha256=item.source_file_sha256,
        split="validation",
        cohort="R",
    )
    with pytest.raises(ValueError, match="split=train"):
        _fit(items=validation)

    prospective = list(base)
    item = prospective[0]
    prospective[0] = PhotometryFitVolume(
        volume=item.volume,
        domain=item.domain,
        record_identity="P_0007_T1w_01",
        subject_identity="0007",
        metadata_prefix="P",
        source_path_identity=item.source_path_identity,
        source_file_sha256=item.source_file_sha256,
        split="train",
        cohort="P",
    )
    with pytest.raises(ValueError, match="traveller 0007"):
        _fit(items=prospective)

    other_p = list(base)
    item = other_p[0]
    other_p[0] = PhotometryFitVolume(
        volume=item.volume,
        domain=item.domain,
        record_identity="P_0010_T1w_01",
        subject_identity="0010",
        metadata_prefix="P",
        source_path_identity=item.source_path_identity,
        source_file_sha256=item.source_file_sha256,
        split="train",
        cohort="P",
    )
    with pytest.raises(ValueError, match="rejects every P record"):
        _fit(items=other_p)


@pytest.mark.parametrize("traveller", ("0006", "0007", "0009"))
def test_fit_explicitly_rejects_every_reserved_traveller(traveller: str) -> None:
    records = list(_fit_items())
    item = records[0]
    records[0] = PhotometryFitVolume(
        volume=item.volume,
        domain=item.domain,
        record_identity=f"P_{traveller}_T1w_01",
        subject_identity=traveller,
        metadata_prefix="P",
        source_path_identity=item.source_path_identity,
        source_file_sha256=item.source_file_sha256,
        split="train",
        cohort="P",
    )
    with pytest.raises(ValueError, match=f"traveller {traveller}"):
        _fit(items=records)


@pytest.mark.parametrize(
    ("case_id", "metadata_prefix", "cohort", "message"),
    [
        ("P_0010_T1w", "R", "R", "identity conflict"),
        ("R_0010_T1w", "P", "R", "identity conflict"),
        ("R_0010_T1w", None, "R", "metadata prefix"),
        ("unknown_0010", "R", "R", "R_ or P_"),
    ],
)
def test_fit_rejects_mislabeled_conflicting_and_missing_cohort_identities(
    case_id: str, metadata_prefix: str | None, cohort: str, message: str
) -> None:
    records = list(_fit_items())
    item = records[0]
    records[0] = PhotometryFitVolume(
        volume=item.volume,
        domain=item.domain,
        record_identity=case_id,
        subject_identity="0010",
        metadata_prefix=metadata_prefix,
        source_path_identity=item.source_path_identity,
        source_file_sha256=item.source_file_sha256,
        split="train",
        cohort=cohort,
    )
    with pytest.raises(ValueError, match=message):
        _fit(items=records)


def test_artifact_reload_reconciles_identity_namespace_independently() -> None:
    payload = _fit().to_dict()
    accepted = payload["provenance"]["accepted_records"]
    accepted[0]["record_identity"] = "P_0010_T1w"
    accepted[0]["record_identity_sha256"] = sha256_text("P_0010_T1w")
    payload["provenance"]["accepted_records_sha256"] = sha256_json(accepted)
    payload["artifact_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="identity conflict"):
        FrozenPhotometryArtifact.from_dict(payload)


def test_artifact_seals_and_revalidates_prospective_exclusion_evidence() -> None:
    artifact = _fit(excluded=(_excluded_record(),))
    provenance = artifact.provenance
    assert provenance["eligibility_proof"]["prospective_excluded_count"] == 1
    assert provenance["excluded_prospective_records"][0]["record_identity"] == "P_0100_T1w"
    assert provenance["excluded_prospective_records_sha256"] == sha256_json(
        provenance["excluded_prospective_records"]
    )

    mutated = artifact.to_dict()
    mutated["provenance"]["excluded_prospective_records"][0]["cohort"] = "R"
    mutated["provenance"]["excluded_prospective_records_sha256"] = sha256_json(
        mutated["provenance"]["excluded_prospective_records"]
    )
    mutated["artifact_sha256"] = sha256_json(
        {key: value for key, value in mutated.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="identity conflict"):
        FrozenPhotometryArtifact.from_dict(mutated)


def test_canonical_cohort_classifier_preserves_retrospective_subject_group() -> None:
    identity = classify_variant_a_cohort(
        case_identity="R_0042_T2w",
        metadata_prefix="R",
        supplied_cohort="R",
        subject_identity="0042",
        allowed_cohorts=("R",),
    )
    assert identity.subject_group_identity == "R:0042"


def test_artifact_rejects_version_hash_split_content_and_template_mutation() -> None:
    artifact = _fit()
    payload = artifact.to_dict()
    unsupported = copy.deepcopy(payload)
    unsupported["contract_version"] = "future"
    with pytest.raises(ValueError, match="contract mismatch"):
        FrozenPhotometryArtifact.from_dict(unsupported)

    with pytest.raises(ValueError, match="split file mismatch"):
        FrozenPhotometryArtifact.from_dict(
            payload, expected_split_file_sha256="b" * 64
        )
    with pytest.raises(ValueError, match="membership fingerprint mismatch"):
        FrozenPhotometryArtifact.from_dict(
            payload, expected_membership_fingerprint="changed"
        )

    content = copy.deepcopy(payload)
    content["provenance"]["accepted_records"][0]["source_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="accepted-record content hash mismatch"):
        FrozenPhotometryArtifact.from_dict(content)

    nonmonotone = copy.deepcopy(payload)
    label = next(iter(nonmonotone["domain_templates"]))
    nonmonotone["domain_templates"][label]["quantiles"][2] = -10.0
    with pytest.raises(ValueError, match="not monotonic"):
        FrozenPhotometryArtifact.from_dict(nonmonotone)

    nonfinite = copy.deepcopy(payload)
    nonfinite["probabilities"][2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        FrozenPhotometryArtifact.from_dict(nonfinite)


def test_artifact_refuses_overwrite_and_cleans_atomic_temporary_file(
    tmp_path, monkeypatch
) -> None:
    artifact = _fit()
    path = artifact.save(tmp_path / "artifact.json")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        artifact.save(path)

    target = tmp_path / "failed.json"

    def fail_replace(source, destination):
        del source, destination
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        artifact.save(target)
    assert not target.exists()
    assert not list(tmp_path.glob(".failed.json.*.tmp"))


def test_source_context_rejects_artifact_and_contrast_mismatch() -> None:
    first = _fit()
    second_payload = copy.deepcopy(first.to_dict())
    second_payload["provenance"]["source_membership_fingerprint"] = "other"
    second_payload["artifact_sha256"] = ""
    second_payload["artifact_sha256"] = sha256_json(
        {key: value for key, value in second_payload.items() if key != "artifact_sha256"}
    )
    second = FrozenPhotometryArtifact.from_dict(second_payload)
    source = _volume(0.1, 0)
    canonical = first.normalize_source(source, Domain(0.1, "T1w"))
    with pytest.raises(ValueError, match="different photometry artifact"):
        second.render_target(canonical, Domain(7.0, "T1w"))
    with pytest.raises(ValueError, match="same-contrast"):
        first.render_target(canonical, Domain(7.0, "T2w"))
