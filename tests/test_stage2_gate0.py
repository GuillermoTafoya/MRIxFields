from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.evaluation.stage2_gate0 import (
    DecodeSpec,
    ReferenceGateSpec,
    ResidualGateSpec,
    RobustNormSpec,
    StratumSpec,
    SweepSpec,
    assert_full_volume_bank,
    assert_subjects_excluded,
    compose_minus,
    evaluate_reference_gate,
    f16_quantization_energy,
    gate0_metric_fn,
    latent_rms,
    residual_energy_gate,
    robust_normalize,
    wrong_target_sweep,
)
from fieldbridge.models.translators.affine_baseline import fit_affine_baselines

CHANNELS = 2
LATENT_SHAPE = (CHANNELS, 4, 4, 4)
FIELDS = (0.1, 1.5, 3.0)
CONTRAST = Contrast.T1W


@dataclass(frozen=True)
class _Case:
    case_id: str
    subject_id: str
    domain: Domain
    latent_path: Path | None = None


def _cases(subjects=("0006", "0007")) -> list[_Case]:
    return [
        _Case(
            case_id=f"P_T1W_{field:g}T_{subject}",
            subject_id=subject,
            domain=Domain(field, CONTRAST),
        )
        for subject in subjects
        for field in FIELDS
    ]


def _latents(cases, *, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        case.case_id: torch.randn(LATENT_SHAPE, generator=generator) for case in cases
    }


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def test_full_volume_bank_guard_accepts_a_full_bank() -> None:
    provenance = assert_full_volume_bank(
        {
            "strategy_used": ["full"],
            "config": {"strategy": "full"},
            "vae_checkpoint_sha256": "abc",
            "roundtrip": {"mean_ssim3d": 0.97},
        }
    )
    assert provenance["strategy_used"] == ["full"]
    assert provenance["vae_checkpoint_sha256"] == "abc"
    assert provenance["roundtrip_mean_ssim3d"] == 0.97


@pytest.mark.parametrize("used", [["tiled"], ["full", "tiled"], [], ["reused_existing"]])
def test_full_volume_bank_guard_refuses_anything_else(used) -> None:
    with pytest.raises(ValueError, match="requires a full-volume latent bank"):
        assert_full_volume_bank({"strategy_used": used, "config": {"strategy": "tiled"}})


def test_frozen_subject_guard_refuses_0009() -> None:
    with pytest.raises(ValueError, match="frozen for Gate 0"):
        assert_subjects_excluded(["0006", "0009"], ["0009"])
    assert_subjects_excluded(["0006", "0007"], ["0009"])  # does not raise


# --------------------------------------------------------------------------------------
# Step 4: robust normalization
# --------------------------------------------------------------------------------------


def test_robust_normalization_cancels_a_pure_intensity_rescaling() -> None:
    pytest.importorskip("skimage.metrics")
    import numpy as np

    generator = torch.Generator().manual_seed(7)
    target = torch.rand(1, 1, 8, 24, 24, generator=generator).numpy().astype(np.float64)
    # A prediction that is structurally perfect but globally darker.
    prediction = np.clip(0.4 * target + 0.05, 0.0, 1.0)

    raw = gate0_metric_fn(
        torch.from_numpy(prediction),
        torch.from_numpy(target),
        metrics=("ssim",),
        device="cpu",
    )
    assert raw["ssim_robust"] > raw["ssim"]
    assert raw["ssim_robust"] > 0.99


def test_robust_normalization_is_disabled_by_spec() -> None:
    pytest.importorskip("skimage.metrics")
    generator = torch.Generator().manual_seed(3)
    target = torch.rand(1, 1, 4, 16, 16, generator=generator)
    out = gate0_metric_fn(
        target * 0.5,
        target,
        metrics=("ssim",),
        device="cpu",
        robust=RobustNormSpec(enabled=False),
    )
    assert "ssim_robust" not in out


def test_robust_normalize_survives_a_degenerate_percentile_window() -> None:
    import numpy as np

    constant = np.full((4, 4), 0.3)
    out = robust_normalize(constant, constant > 0.0, RobustNormSpec())
    assert np.all(out == 0.0)


def test_robust_normalize_maps_an_affine_rescaling_onto_the_same_array() -> None:
    """The property step 4 depends on, checked without the optional SSIM backend."""

    import numpy as np

    generator = torch.Generator().manual_seed(5)
    target = torch.rand(6, 6, 6, generator=generator).numpy().astype(np.float64)
    rescaled = 0.3 * target + 0.2
    mask = target > 0.0
    spec = RobustNormSpec()

    np.testing.assert_allclose(
        robust_normalize(rescaled, mask, spec),
        robust_normalize(target, mask, spec),
        atol=1e-9,
    )


def test_robust_normalize_uses_the_mask_for_its_percentiles() -> None:
    import numpy as np

    array = np.zeros((10, 10))
    array[5:, :] = np.linspace(0.5, 1.0, 50).reshape(5, 10)
    mask = array > 0.0
    masked = robust_normalize(array, mask, RobustNormSpec())
    unmasked = robust_normalize(array, None, RobustNormSpec())
    # With the background included, p1..p99 spans 0..1 and the foreground is compressed.
    assert masked[5:, :].std() > unmasked[5:, :].std()


# --------------------------------------------------------------------------------------
# Step 1: wrong-target sweep
# --------------------------------------------------------------------------------------


def test_sweep_flags_a_transport_that_ignores_the_target_field() -> None:
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=1)

    def identity(z, source, target):
        return z

    result = wrong_target_sweep(
        transport=identity, latents_by_case=latents, cases=cases, log=False
    )
    assert result["summary"]["responds_fraction"] == 0.0
    assert result["summary"]["mean_responsiveness"] == 0.0
    for case in result["cases"]:
        assert case["mean_output_spread"] == 0.0
        assert case["mean_real_spread"] > 0.0


def test_sweep_passes_an_oracle_that_returns_the_real_target_latent() -> None:
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=2)
    by_domain = {(c.subject_id, c.domain.field_strength_t): c.case_id for c in cases}

    def oracle(z, source, target):
        return latents[by_domain[("0006", target.field_strength_t)]]

    result = wrong_target_sweep(
        transport=oracle, latents_by_case=latents, cases=cases, log=False
    )
    assert result["summary"]["responds_fraction"] == 1.0
    assert result["summary"]["mean_responsiveness"] == pytest.approx(1.0)
    assert result["summary"]["monotone_fraction"] == pytest.approx(
        result["summary"]["real_monotone_fraction"]
    )


def test_sweep_threshold_is_config_driven() -> None:
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=3)
    by_domain = {(c.subject_id, c.domain.field_strength_t): c.case_id for c in cases}

    def weak(z, source, target):
        real = latents[by_domain[("0006", target.field_strength_t)]]
        return z + 0.05 * (real - z)

    lenient = wrong_target_sweep(
        transport=weak,
        latents_by_case=latents,
        cases=cases,
        spec=SweepSpec(responsiveness_min=0.01),
        log=False,
    )
    strict = wrong_target_sweep(
        transport=weak,
        latents_by_case=latents,
        cases=cases,
        spec=SweepSpec(responsiveness_min=0.9),
        log=False,
    )
    assert lenient["summary"]["responds_fraction"] == 1.0
    assert strict["summary"]["responds_fraction"] == 0.0


def test_sweep_can_exclude_the_identity_target() -> None:
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=4)
    result = wrong_target_sweep(
        transport=lambda z, s, t: z,
        latents_by_case=latents,
        cases=cases,
        spec=SweepSpec(include_identity_target=False),
        log=False,
    )
    for case in result["cases"]:
        assert case["source_field_t"] not in case["target_fields_t"]


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def test_compose_minus_of_a_method_with_itself_is_the_identity() -> None:
    z = torch.randn(LATENT_SHAPE)
    method = lambda z_, s, t: 3.0 * z_ + 1.0  # noqa: E731
    composed = compose_minus(method, method)
    torch.testing.assert_close(composed(z, Domain(0.1, CONTRAST), Domain(3.0, CONTRAST)), z)


def test_compose_minus_removes_the_subtracted_displacement() -> None:
    z = torch.randn(LATENT_SHAPE)
    src, tgt = Domain(0.1, CONTRAST), Domain(3.0, CONTRAST)
    primary = lambda z_, s, t: z_ + 5.0  # noqa: E731
    subtracted = lambda z_, s, t: z_ + 2.0  # noqa: E731
    composed = compose_minus(primary, subtracted)
    torch.testing.assert_close(composed(z, src, tgt), z + 3.0)


def test_latent_rms_is_zero_for_identical_latents() -> None:
    z = torch.randn(LATENT_SHAPE)
    assert latent_rms(z, z) == 0.0
    assert latent_rms(z + 2.0, z) == pytest.approx(2.0, rel=1e-5)


# --------------------------------------------------------------------------------------
# Step 5: residual energy gate
# --------------------------------------------------------------------------------------


def _affine_from_pool(latents, cases):
    return fit_affine_baselines(
        [(latents[c.case_id], c.domain) for c in cases], channels=CHANNELS
    )


def test_residual_gate_closes_when_the_relation_is_exactly_affine() -> None:
    """If z_t is literally a*z_s + b, the affine explains everything and there is no Gate 1."""

    cases = _cases()
    generator = torch.Generator().manual_seed(11)
    a = {field: 1.0 + index for index, field in enumerate(FIELDS)}
    b = {field: 0.5 * index for index, field in enumerate(FIELDS)}
    base = {
        subject: torch.randn(LATENT_SHAPE, generator=generator)
        for subject in ("0006", "0007")
    }
    latents = {
        case.case_id: a[case.domain.field_strength_t] * base[case.subject_id]
        + b[case.domain.field_strength_t]
        for case in cases
    }
    baselines = _affine_from_pool(latents, cases)

    result = residual_energy_gate(
        baselines={"all": baselines["all"]},
        latents_by_case=latents,
        cases=cases,
        log=False,
    )
    overall = result["baselines"]["all"]["overall"]
    assert overall["explained_fraction"] > 0.99
    assert result["verdict"]["decision"] == "gate0_close_generative_branch"


def test_residual_gate_proceeds_on_a_shared_learnable_residual() -> None:
    """A residual that is the SAME structure for both travellers is exactly what Gate 1 needs."""

    cases = _cases()
    generator = torch.Generator().manual_seed(12)
    base = {
        subject: torch.randn(LATENT_SHAPE, generator=generator)
        for subject in ("0006", "0007")
    }
    # One shared, field-pair-specific structure both subjects carry, large enough to dominate.
    shared = {
        field: 6.0 * torch.randn(LATENT_SHAPE, generator=generator) for field in FIELDS
    }
    latents = {
        case.case_id: base[case.subject_id] + shared[case.domain.field_strength_t]
        for case in cases
    }
    baselines = _affine_from_pool(latents, cases)

    result = residual_energy_gate(
        baselines={"all": baselines["all"]},
        latents_by_case=latents,
        cases=cases,
        log=False,
    )
    predictability = result["baselines"]["all"]["predictability"]
    assert predictability["median_predictable_fraction"] > 0.5
    assert predictability["control_num_comparisons"] > 0
    assert result["verdict"]["decision"] == "gate0_proceed_to_gate1"
    assert "1-vs-1" in result["verdict"]["caveat"]


def test_predictability_is_reported_against_the_anatomical_alignment_ceiling() -> None:
    """A voxelwise cross-subject cosine is capped by how well the two brains line up."""

    cases = _cases()
    latents = _latents(cases, seed=31)
    baselines = _affine_from_pool(latents, cases)
    result = residual_energy_gate(
        baselines={"all": baselines["all"]},
        latents_by_case=latents,
        cases=cases,
        log=False,
    )
    predictability = result["baselines"]["all"]["predictability"]
    alignment = predictability["anatomical_alignment"]

    # One cross-subject comparison per (contrast, field): 1 contrast x 3 fields.
    assert alignment["num_comparisons"] == len(FIELDS)
    assert set(alignment["per_domain_cosine"]) == {f"{f:g}T/{CONTRAST.value}" for f in FIELDS}
    assert alignment["predictable_fraction_ceiling"] == pytest.approx(
        alignment["mean_cosine"] ** 2
    )
    assert predictability["median_predictable_fraction_of_ceiling"] == pytest.approx(
        predictability["median_predictable_fraction"]
        / alignment["predictable_fraction_ceiling"]
    )
    # The ceiling is reported, never used to decide: the raw fraction drives the verdict.
    checks = result["verdict"]["checks"]["all"]
    assert checks["predictable_above_threshold"] == (
        checks["median_predictable_fraction"] >= result["spec"]["predictable_fraction_min"]
    )


def test_residual_gate_reports_the_quantization_floor_and_pair_count() -> None:
    cases = _cases()
    latents = _latents(cases, seed=13)
    baselines = _affine_from_pool(latents, cases)
    result = residual_energy_gate(
        baselines={"all": baselines["all"], "foreground": baselines["foreground"]},
        latents_by_case=latents,
        cases=cases,
        log=False,
    )
    # 2 subjects x 1 contrast x 3 fields -> 3*2 ordered cross-field pairs each.
    assert result["num_pairs"] == 12
    assert result["quantization_floor_energy"] > 0.0
    assert set(result["baselines"]) == {"all", "foreground"}
    for block in result["baselines"].values():
        assert block["anatomy_floor"]["num_comparisons"] > 0
        assert set(block["strata"]) == {"far_field_pairs", "near_field_pairs"}


def test_residual_gate_refuses_the_frozen_subject() -> None:
    cases = _cases(subjects=("0006", "0009"))
    latents = _latents(cases, seed=14)
    baselines = _affine_from_pool(latents, cases)
    with pytest.raises(ValueError, match="frozen for Gate 0"):
        residual_energy_gate(
            baselines={"all": baselines["all"]},
            latents_by_case=latents,
            cases=cases,
            log=False,
        )


def test_residual_gate_needs_two_travellers_for_the_predictable_fraction() -> None:
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=15)
    baselines = _affine_from_pool(latents, cases)
    with pytest.raises(ValueError, match="at least two paired travellers"):
        residual_energy_gate(
            baselines={"all": baselines["all"]},
            latents_by_case=latents,
            cases=cases,
            log=False,
        )


def test_f16_quantization_energy_is_small_and_positive() -> None:
    latent = torch.randn(LATENT_SHAPE) * 0.4
    energy = f16_quantization_energy(latent)
    assert 0.0 < energy < 1e-6


# --------------------------------------------------------------------------------------
# Step 3: the reference gate
# --------------------------------------------------------------------------------------


class _ToyDecoder(torch.nn.Module):
    """Upsamples a latent to an image; enough to exercise the decode/metric/strata plumbing."""

    downsample_factor = 2
    latent_channels = CHANNELS

    def decode(self, latent: torch.Tensor, domain: Domain) -> torch.Tensor:
        upsampled = torch.nn.functional.interpolate(
            latent, scale_factor=self.downsample_factor, mode="nearest"
        )
        return upsampled.mean(dim=1, keepdim=True).sigmoid()


def _records_for(cases) -> list[VolumeRecord]:
    return [
        VolumeRecord(
            case_id=case.case_id,
            image_path=f"synthetic/{case.case_id}.nii.gz",
            domain=case.domain,
            subject_id=case.subject_id,
        )
        for case in cases
    ]


def test_reference_gate_scores_every_method_and_stratifies_by_identity_difficulty() -> None:
    pytest.importorskip("skimage.metrics")
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=21)
    baselines = _affine_from_pool(latents, cases)
    decoder = _ToyDecoder().eval()

    def loader(record: VolumeRecord) -> torch.Tensor:
        generator = torch.Generator().manual_seed(abs(hash(record.case_id)) % (2**31))
        return torch.rand(1, 1, 8, 8, 8, generator=generator)

    methods = {
        "identity": lambda z, s, t: z,
        "affine": baselines["all"].transport,
    }
    result = evaluate_reference_gate(
        methods=methods,
        decoder=decoder,
        records=_records_for(cases),
        latents_by_case=latents,
        cases=cases,
        decode=DecodeSpec(block_size=(8, 8, 8), halo=(2, 2, 2), precision="float32"),
        device=torch.device("cpu"),
        spec=ReferenceGateSpec(
            metrics=("ssim", "nrmse"),
            strata=StratumSpec(catastrophic_identity_nrmse=0.0),  # force every pair catastrophic
        ),
        volume_loader=loader,
        log=False,
    )

    assert result["num_pairs"] == len(FIELDS) * (len(FIELDS) - 1)
    assert result["method_names"] == ["identity", "affine", "ceiling"]
    assert result["strata"]["catastrophic_identity"]["fraction_of_pairs"] == 1.0
    assert result["strata"]["ordinary"]["num_pairs"] == 0
    for name in result["method_names"]:
        metrics = result["overall"]["methods"][name]
        assert {"ssim", "nrmse", "ssim_robust"} <= set(metrics)
    # Every stratum's pair counts must add back up to the total.
    assert (
        result["strata"]["catastrophic_identity"]["num_pairs"]
        + result["strata"]["ordinary"]["num_pairs"]
        == result["num_pairs"]
    )


def _fake_metric_fn(prediction, target, *, metrics, device, robust=RobustNormSpec()):
    """Stand-in for the official metrics: exercises the gate plumbing without scikit-image."""

    difference = float((prediction.float() - target.float()).abs().mean())
    out = {"nrmse": difference, "ssim": 1.0 - difference}
    if robust.enabled:
        out["ssim_robust"] = 1.0 - 0.5 * difference
    return out


def test_reference_gate_plumbing_runs_without_the_official_metric_backend() -> None:
    cases = _cases(subjects=("0006",))
    latents = _latents(cases, seed=23)
    baselines = _affine_from_pool(latents, cases)
    decode_calls: list[str] = []

    class _CountingDecoder(_ToyDecoder):
        def decode(self, latent, domain):
            decode_calls.append(domain.label)
            return super().decode(latent, domain)

    def loader(record: VolumeRecord) -> torch.Tensor:
        generator = torch.Generator().manual_seed(abs(hash(record.case_id)) % (2**31))
        return torch.rand(1, 1, 8, 8, 8, generator=generator)

    result = evaluate_reference_gate(
        methods={
            "identity": lambda z, s, t: z,
            "affine": baselines["all"].transport,
            "affine_minus_affine": compose_minus(
                baselines["all"].transport, baselines["all"].transport
            ),
        },
        decoder=_CountingDecoder().eval(),
        records=_records_for(cases),
        latents_by_case=latents,
        cases=cases,
        decode=DecodeSpec(block_size=(8, 8, 8), halo=(2, 2, 2), precision="float32"),
        device=torch.device("cpu"),
        spec=ReferenceGateSpec(metrics=("ssim", "nrmse")),
        metric_fn=_fake_metric_fn,
        volume_loader=loader,
        log=False,
    )

    num_pairs = len(FIELDS) * (len(FIELDS) - 1)
    assert result["num_pairs"] == num_pairs
    # identity decodes are cached per case, not per pair: 3 cached cases + 2 decodes per pair.
    assert len(decode_calls) == len(FIELDS) + 2 * num_pairs
    # compose_minus(f, f) is the identity map, so it must score exactly like identity.
    for row in result["pairs"]:
        assert row["affine_minus_affine"] == row["identity"]
    assert set(result["by_contrast"]) == {CONTRAST.value}


def test_reference_gate_requires_an_identity_method() -> None:
    cases = _cases(subjects=("0006",))
    with pytest.raises(ValueError, match="requires an 'identity' method"):
        evaluate_reference_gate(
            methods={"affine": lambda z, s, t: z},
            decoder=_ToyDecoder().eval(),
            records=_records_for(cases),
            latents_by_case=_latents(cases, seed=22),
            cases=cases,
            decode=DecodeSpec(block_size=(8, 8, 8), halo=(2, 2, 2), precision="float32"),
            device=torch.device("cpu"),
            log=False,
        )
