"""Etapa-2 Gate 0: cheap diagnostics that decide whether the residual objective is worth A100 hours.

v2 scored held-out traveller 0006 at nRMSE 0.459 / SSIM 0.880 against identity 0.595 / 0.876 and
a frozen-VAE ceiling of 0.131 / 0.966. The nRMSE gap without an SSIM gap is the signature of a
global intensity rescaling, not of structure. Gate 0 tests that reading directly, before any
further training:

1. :func:`wrong_target_sweep`     — does the model respond to the target field at all, or does it
                                    emit the same volume for every requested target?
2. ``affine_baseline``            — the closed-form per-channel affine (separate module).
3. :func:`evaluate_reference_gate`— identity / affine / SB v2 / SB-minus-affine on the same 60
                                    pairs, same protocol as v2, with LPIPS this time.
4. robust-normalized SSIM         — a column of (3), separating "got the brightness right" from
                                    "got the structure right".
5. :func:`residual_energy_gate`   — the decisive one: is what the affine leaves behind actually
                                    learnable, or is it noise plus subject-specific anatomy?

Every table is reported both aggregated and stratified by identity difficulty, because the v2
aggregate was carried entirely by the handful of pairs where identity is catastrophic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.latent_bank import decode_latent, downsample_factor, load_volume
from fieldbridge.data.latent_bank_dataset import LatentStats
from fieldbridge.evaluation.mrixfields2026_official import (
    official_task3_lpips,
    official_task3_nrmse,
    official_task3_ssim,
)
from fieldbridge.evaluation.stage2_transport_eval import (
    DecodeSpec,
    TransportSamplerConfig,
    sample_transport,
    traveller_cases,
)
from fieldbridge.models.translators.affine_baseline import AffineLatentBaseline

GATE0_CONTRACT_VERSION = "stage2-gate0-v1"

# A raw-latent -> raw-latent map conditioned on the domain pair. Deliberately NOT a velocity:
# the affine and identity references have no ODE, so the gate composes finished latents and the
# ODE integration is an implementation detail of the SB reference only.
TransportFn = Callable[[torch.Tensor, Domain, Domain], torch.Tensor]


# --------------------------------------------------------------------------------------
# Provenance guards
# --------------------------------------------------------------------------------------


def assert_full_volume_bank(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse to run Gate 0 on a tiled-encode bank, and return the provenance to log.

    Tiled encode and full encode are different banks with different latents; mixing a tiled
    bank into a comparison against v2 numbers taken on a full bank would produce a difference
    that has nothing to do with the transport. This fails closed rather than warning, because
    nothing downstream can tell the two apart from the numbers alone.
    """

    used = list(manifest.get("strategy_used", []))
    configured = str((manifest.get("config") or {}).get("strategy", ""))
    if used != ["full"]:
        raise ValueError(
            f"Gate 0 requires a full-volume latent bank; this bank reports strategy_used="
            f"{used!r} (config.strategy={configured!r}). Point --bank-dir at the full bank."
        )
    return {
        "strategy_used": used,
        "config_strategy": configured,
        "vae_checkpoint_sha256": manifest.get("vae_checkpoint_sha256"),
        "git_commit": manifest.get("git_commit"),
        "counts": manifest.get("counts"),
        "roundtrip_mean_ssim3d": (manifest.get("roundtrip") or {}).get("mean_ssim3d"),
    }


def assert_subjects_excluded(subject_ids: Sequence[str], excluded: Sequence[str]) -> None:
    """Hard stop if a frozen subject (0009) appears anywhere in a Gate-0 computation."""

    violation = sorted(set(subject_ids) & set(excluded))
    if violation:
        raise ValueError(
            f"Subject(s) {violation} are frozen for Gate 0 and must not be touched. "
            "0009 is the untouched test traveller; spending it here forfeits the only clean "
            "confirmatory read the project has left."
        )


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RobustNormSpec:
    """Robust intensity normalization applied before the extra SSIM column (step 4).

    Each volume is rescaled by ITS OWN percentiles inside a mask taken from the target, so a
    prediction that is a pure intensity rescaling of the truth normalizes onto the truth and
    scores ~1. What survives is structure. Percentiles are taken inside the mask because the
    volumes are [0, 1] with a large near-zero background, which pins p1 at 0 otherwise.
    """

    low_percentile: float = 1.0
    high_percentile: float = 99.0
    mask_threshold: float = 0.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "mask_threshold": self.mask_threshold,
            "enabled": self.enabled,
        }


def robust_normalize(array: Any, mask: Any, spec: RobustNormSpec) -> Any:
    """Percentile-rescale ``array`` to [0, 1] using its own statistics inside ``mask``."""

    import numpy as np

    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    low, high = np.percentile(values, [spec.low_percentile, spec.high_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or (high - low) < 1e-8:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def gate0_metric_fn(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    metrics: Sequence[str],
    device: str,
    robust: RobustNormSpec = RobustNormSpec(),
) -> dict[str, float]:
    """Official Task-3 metrics plus the robust-normalized SSIM column."""

    import numpy as np

    pred = prediction.squeeze().detach().cpu().to(torch.float32).numpy().astype(np.float64)
    tgt = target.squeeze().detach().cpu().to(torch.float32).numpy().astype(np.float64)
    out: dict[str, float] = {}
    if "nrmse" in metrics:
        out["nrmse"] = official_task3_nrmse(pred, tgt)
    if "ssim" in metrics:
        out["ssim"] = official_task3_ssim(pred, tgt)
    if "lpips" in metrics:
        try:
            out["lpips"] = official_task3_lpips(pred, tgt, device=device)
        except ImportError:
            pass
    if robust.enabled and "ssim" in metrics:
        mask = tgt > robust.mask_threshold
        out["ssim_robust"] = official_task3_ssim(
            robust_normalize(pred, mask, robust), robust_normalize(tgt, mask, robust)
        )
    return out


def latent_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    """Root-mean-square distance in raw latent units. Absolute, so distances are comparable."""

    return float((a.to(torch.float32) - b.to(torch.float32)).square().mean().sqrt())


# --------------------------------------------------------------------------------------
# Step 1: wrong-target sweep
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """Thresholds for the step-1 verdict. Config-driven; no magic numbers in the logic."""

    # A case "responds" if the spread of its five outputs reaches this fraction of the spread
    # of the five REAL target latents. 0.10 says: move at least a tenth of what you should.
    responsiveness_min: float = 0.10
    # Include f_t == f_s in the sweep. It is the identity request, and a model that ignores
    # the target field answers it identically to every other request, which is the tell.
    include_identity_target: bool = True
    # Fraction of requests whose output must land nearest its OWN real target for the case to
    # count as directionally correct. Chance is 1/5 = 0.20 with five candidate fields, so this
    # has to clear chance by a margin to mean anything.
    correct_target_fraction_min: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return {
            "responsiveness_min": self.responsiveness_min,
            "include_identity_target": self.include_identity_target,
            "correct_target_fraction_min": self.correct_target_fraction_min,
            "chance_correct_target_fraction": None,  # filled per case from the field count
        }


@torch.inference_mode()
def wrong_target_sweep(
    *,
    transport: TransportFn,
    latents_by_case: Mapping[str, torch.Tensor],
    cases: Sequence[Any],
    spec: SweepSpec = SweepSpec(),
    log: bool = True,
) -> dict[str, Any]:
    """Fix a real source, request all five target fields, and see whether the output moves —
    and whether it moves in the right direction.

    Two separate questions, deliberately not conflated:

    - **Responsiveness.** Does the output change at all with the requested field? Reported as
      the spread of the five outputs against the spread of the five REAL target latents.
    - **Direction.** Does it change *correctly*? For each request, the output is assigned to
      the nearest of that subject's five real target latents — a target-domain classifier with
      no training and no extra data. Responsiveness without direction is a model that reacts to
      its conditioning by moving somewhere arbitrary, which is not translation.

    Latent-space only, no decode, so this is nearly free.
    """

    by_group: dict[tuple[str, str], dict[float, Any]] = {}
    for case in cases:
        key = (case.subject_id, Contrast.parse(case.domain.contrast).value)
        by_group.setdefault(key, {})[float(case.domain.field_strength_t)] = case

    rows: list[dict[str, Any]] = []
    for (subject_id, contrast), field_map in sorted(by_group.items()):
        fields = sorted(field_map)
        for source_field in fields:
            src = field_map[source_field]
            z_s = latents_by_case[src.case_id]
            targets = [
                f for f in fields if spec.include_identity_target or f != source_field
            ]
            outputs = {
                f: transport(z_s, src.domain, field_map[f].domain) for f in targets
            }
            reals = {f: latents_by_case[field_map[f].case_id] for f in targets}

            to_source = {f: latent_rms(outputs[f], z_s) for f in targets}
            real_to_source = {f: latent_rms(reals[f], z_s) for f in targets}
            pairwise = {
                f"{fi:g}|{fj:g}": latent_rms(outputs[fi], outputs[fj])
                for i, fi in enumerate(targets)
                for fj in targets[i + 1 :]
            }
            real_pairwise = {
                f"{fi:g}|{fj:g}": latent_rms(reals[fi], reals[fj])
                for i, fi in enumerate(targets)
                for fj in targets[i + 1 :]
            }

            spread = _mean(pairwise.values())
            real_spread = _mean(real_pairwise.values())
            responsiveness = spread / real_spread if real_spread > 0 else 0.0

            # Target-domain classification: assign each output to the nearest real target
            # latent of this same subject and contrast. Same anatomy throughout, so the only
            # thing distinguishing the candidates is the field.
            classification: list[dict[str, Any]] = []
            for requested in targets:
                distances = {other: latent_rms(outputs[requested], reals[other]) for other in targets}
                ordered_by_distance = sorted(targets, key=lambda f: distances[f])
                nearest = ordered_by_distance[0]
                wrong = [distances[f] for f in targets if f != requested]
                classification.append(
                    {
                        "requested_field_t": requested,
                        "nearest_real_field_t": nearest,
                        "correct": nearest == requested,
                        "rank_of_requested": ordered_by_distance.index(requested) + 1,
                        "distance_to_requested": distances[requested],
                        "mean_distance_to_wrong": _mean(wrong),
                        # Positive means the requested target is closer than the average wrong
                        # one; negative means the output actively resembles the wrong domain.
                        "margin": _mean(wrong) - distances[requested],
                    }
                )
            correct_fraction = _mean([float(item["correct"]) for item in classification])
            chance = 1.0 / len(targets)

            # Monotone in |log(f_t / f_s)|: the further the requested field, the further the
            # output should sit from the source.
            ordered = sorted(targets, key=lambda f: abs(math.log(f / source_field)))
            distances = [to_source[f] for f in ordered]
            monotone = all(b >= a for a, b in zip(distances, distances[1:]))
            real_monotone = all(
                b >= a
                for a, b in zip(
                    [real_to_source[f] for f in ordered], [real_to_source[f] for f in ordered][1:]
                )
            )

            row = {
                "subject_id": subject_id,
                "contrast": contrast,
                "source_field_t": source_field,
                "target_fields_t": targets,
                "distance_to_source": {f"{f:g}": to_source[f] for f in targets},
                "real_distance_to_source": {f"{f:g}": real_to_source[f] for f in targets},
                "pairwise_output_distance": pairwise,
                "pairwise_real_distance": real_pairwise,
                "mean_output_spread": spread,
                "mean_real_spread": real_spread,
                "responsiveness": responsiveness,
                "monotone_in_log_field_ratio": monotone,
                "real_monotone_in_log_field_ratio": real_monotone,
                "responds": responsiveness >= spec.responsiveness_min,
                "target_classification": classification,
                "correct_target_fraction": correct_fraction,
                "chance_correct_target_fraction": chance,
                "mean_classification_margin": _mean(
                    [item["margin"] for item in classification]
                ),
                "mean_rank_of_requested": _mean(
                    [item["rank_of_requested"] for item in classification]
                ),
                "directionally_correct": correct_fraction >= spec.correct_target_fraction_min,
            }
            rows.append(row)
            if log:
                print(
                    f"gate0_sweep {subject_id} {contrast} src={source_field:g}T "
                    f"R={responsiveness:.3f} responds={row['responds']} "
                    f"correct_target={correct_fraction:.2f} (chance {chance:.2f}) "
                    f"margin={row['mean_classification_margin']:+.4f} monotone={monotone}",
                    flush=True,
                )

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "num_cases": len(subset),
            "responds_fraction": _mean([float(r["responds"]) for r in subset]),
            "mean_responsiveness": _mean([r["responsiveness"] for r in subset]),
            "monotone_fraction": _mean(
                [float(r["monotone_in_log_field_ratio"]) for r in subset]
            ),
            "real_monotone_fraction": _mean(
                [float(r["real_monotone_in_log_field_ratio"]) for r in subset]
            ),
            "correct_target_fraction": _mean([r["correct_target_fraction"] for r in subset]),
            "chance_correct_target_fraction": _mean(
                [r["chance_correct_target_fraction"] for r in subset]
            ),
            "mean_classification_margin": _mean(
                [r["mean_classification_margin"] for r in subset]
            ),
            "mean_rank_of_requested": _mean([r["mean_rank_of_requested"] for r in subset]),
            "directionally_correct_fraction": _mean(
                [float(r["directionally_correct"]) for r in subset]
            ),
        }

    summary = block(rows)
    by_contrast = {
        contrast: block([r for r in rows if r["contrast"] == contrast])
        for contrast in sorted({r["contrast"] for r in rows})
    }
    if log:
        print(
            f"gate0_sweep DONE cases={summary['num_cases']} "
            f"responds={summary['responds_fraction']:.3f} "
            f"correct_target={summary['correct_target_fraction']:.3f} "
            f"(chance {summary['chance_correct_target_fraction']:.3f}) "
            f"margin={summary['mean_classification_margin']:+.4f}",
            flush=True,
        )
    return {
        "contract_version": GATE0_CONTRACT_VERSION,
        "spec": spec.to_dict(),
        "summary": summary,
        "by_contrast": by_contrast,
        "cases": rows,
    }


# --------------------------------------------------------------------------------------
# Step 3 + 4: the four-reference gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StratumSpec:
    """Difficulty stratification. v2's aggregate win lived entirely in the catastrophic stratum."""

    # Identity official nRMSE above this marks a pair where source and target intensity scales
    # are wildly mismatched. v1/v2 measured 13 of 60 traveller pairs above 1.0.
    catastrophic_identity_nrmse: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"catastrophic_identity_nrmse": self.catastrophic_identity_nrmse}


@dataclass(frozen=True, slots=True)
class ReferenceGateSpec:
    metrics: tuple[str, ...] = ("ssim", "nrmse", "lpips")
    robust: RobustNormSpec = field(default_factory=RobustNormSpec)
    strata: StratumSpec = field(default_factory=StratumSpec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": list(self.metrics),
            "robust": self.robust.to_dict(),
            "strata": self.strata.to_dict(),
        }


@torch.inference_mode()
def evaluate_reference_gate(
    *,
    methods: Mapping[str, TransportFn],
    decoder: Any,
    records: Sequence[VolumeRecord],
    latents_by_case: Mapping[str, torch.Tensor],
    cases: Sequence[Any],
    decode: DecodeSpec,
    device: torch.device,
    spec: ReferenceGateSpec = ReferenceGateSpec(),
    include_ceiling: bool = True,
    image_methods: Mapping[str, Callable[[torch.Tensor, Domain, Domain], torch.Tensor]] | None = None,
    metric_fn: Callable[..., dict[str, float]] = gate0_metric_fn,
    volume_loader: Callable[[VolumeRecord], torch.Tensor] = load_volume,
    on_pair: Callable[[dict[str, Any]], None] | None = None,
    completed_pairs: Sequence[Mapping[str, Any]] = (),
    log: bool = True,
) -> dict[str, Any]:
    """Score N named references on every traveller pair, one decode per latent reference.

    ``methods`` maps a name to a raw-latent transport. ``identity`` is required: the difficulty
    strata are defined by its official nRMSE.

    ``image_methods`` maps a name to a post-decode transform of the IDENTITY reconstruction.
    That is where the intensity baselines live — they are photometric maps of a volume the gate
    has already decoded, so they add no decode cost, and they test the shortcut hypothesis in
    the space where it is actually visible.

    ``on_pair`` is called with each completed row, and ``completed_pairs`` seeds the run with
    rows from a previous attempt. Together they make an hour of full-volume decoding resumable
    across a Colab disconnect instead of restartable from zero.
    """

    if "identity" not in methods:
        raise ValueError(
            "evaluate_reference_gate requires an 'identity' method: the difficulty strata are "
            "defined by identity's nRMSE, exactly as in the v1/v2 reads."
        )
    decoder = decoder.to(device).eval()
    factor = downsample_factor(decoder)
    record_by_case = {str(r.case_id): r for r in records}

    by_group: dict[tuple[str, str], dict[float, Any]] = {}
    for case in cases:
        key = (case.subject_id, Contrast.parse(case.domain.contrast).value)
        by_group.setdefault(key, {})[float(case.domain.field_strength_t)] = case

    decoded_cache: dict[str, torch.Tensor] = {}
    decode_paths: set[str] = set()

    def decode_raw(latent: torch.Tensor, domain: Domain) -> torch.Tensor:
        image, path_used = decode_latent(
            decoder,
            latent.unsqueeze(0) if latent.ndim == 4 else latent,
            domain,
            factor=factor,
            strategy=decode.strategy,
            block_size=decode.block_size,
            halo=decode.halo,
            precision=decode.precision,
        )
        decode_paths.add(path_used)
        return image

    def decoded_case(case: Any) -> torch.Tensor:
        if case.case_id not in decoded_cache:
            decoded_cache[case.case_id] = decode_raw(latents_by_case[case.case_id], case.domain)
        return decoded_cache[case.case_id]

    pairs: list[dict[str, Any]] = [dict(row) for row in completed_pairs]
    already_done = {
        (str(row["subject_id"]), str(row["contrast"]), float(row["source_field_t"]),
         float(row["target_field_t"]))
        for row in pairs
    }
    if pairs and log:
        print(f"gate0_ref resuming with {len(pairs)} pair(s) already scored", flush=True)

    for (subject_id, contrast), field_map in sorted(by_group.items()):
        fields = sorted(field_map)
        for field_s in fields:
            for field_t in fields:
                if field_s == field_t:
                    continue
                if (subject_id, contrast, field_s, field_t) in already_done:
                    continue
                src, tgt = field_map[field_s], field_map[field_t]
                target_image = volume_loader(record_by_case[tgt.case_id]).to(device)
                z_s = latents_by_case[src.case_id]

                row: dict[str, Any] = {
                    "subject_id": subject_id,
                    "contrast": contrast,
                    "source_field_t": field_s,
                    "target_field_t": field_t,
                    "log_field_ratio": math.log(field_t / field_s),
                }
                for name, method in methods.items():
                    if name == "identity":
                        # Served from the per-case decode cache: decode(z_source) does not
                        # depend on the requested target, so it is decoded once per case
                        # rather than once per pair. The method is not called at all.
                        image = decoded_case(src)
                    else:
                        z_out = method(z_s, src.domain, tgt.domain)
                        image = decode_raw(z_out, tgt.domain)
                        del z_out
                    row[name] = metric_fn(
                        image,
                        target_image,
                        metrics=spec.metrics,
                        device=device.type,
                        robust=spec.robust,
                    )
                    if name != "identity":
                        del image
                # Post-decode photometric maps of the identity reconstruction. No extra decode:
                # this is the same volume, intensity-remapped onto the target domain.
                for name, transform in (image_methods or {}).items():
                    row[name] = metric_fn(
                        transform(decoded_case(src), src.domain, tgt.domain),
                        target_image,
                        metrics=spec.metrics,
                        device=device.type,
                        robust=spec.robust,
                    )
                if include_ceiling:
                    row["ceiling"] = metric_fn(
                        decoded_case(tgt),
                        target_image,
                        metrics=spec.metrics,
                        device=device.type,
                        robust=spec.robust,
                    )
                pairs.append(row)
                if on_pair is not None:
                    on_pair(row)
                if log:
                    print(
                        f"gate0_ref {subject_id} {contrast} {field_s:g}T->{field_t:g}T "
                        + " ".join(
                            f"{name}={_fmt(row[name])}"
                            for name in _method_names(methods, image_methods, include_ceiling)
                        ),
                        flush=True,
                    )
                del target_image

    method_names = _method_names(methods, image_methods, include_ceiling)
    # Index-based split: two traveller pairs can legitimately produce identical metric dicts,
    # and a value-equality `not in` would then drop the duplicate out of `ordinary`.
    catastrophic_index = {
        index
        for index, row in enumerate(pairs)
        if row["identity"].get("nrmse", 0.0) > spec.strata.catastrophic_identity_nrmse
    }
    catastrophic = [row for index, row in enumerate(pairs) if index in catastrophic_index]
    ordinary = [row for index, row in enumerate(pairs) if index not in catastrophic_index]

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "num_pairs": len(subset),
            "methods": {
                name: _mean_metrics([row[name] for row in subset]) for name in method_names
            },
        }

    result = {
        "contract_version": GATE0_CONTRACT_VERSION,
        "spec": spec.to_dict(),
        "method_names": method_names,
        # Kept explicit so a diagnostic column can never be mistaken for, or substituted into,
        # the official Task-3 scoring. `ssim_robust` rescales both volumes by their own
        # percentiles first; it answers "is the structure right" and is not comparable with
        # any leaderboard number.
        "metric_roles": {
            "official": [m for m in spec.metrics],
            "diagnostic": ["ssim_robust"] if spec.robust.enabled else [],
        },
        "num_pairs": len(pairs),
        "subjects": sorted({row["subject_id"] for row in pairs}),
        "decode": {
            "strategy": decode.strategy,
            "path_used": sorted(decode_paths),
            "halo": list(decode.halo),
            "block_size": list(decode.block_size),
        },
        "overall": block(pairs),
        "strata": {
            "catastrophic_identity": {
                "definition": (
                    f"identity official nRMSE > {spec.strata.catastrophic_identity_nrmse}"
                ),
                "fraction_of_pairs": (len(catastrophic) / len(pairs)) if pairs else 0.0,
                **block(catastrophic),
            },
            "ordinary": block(ordinary),
        },
        "by_contrast": {
            contrast: block([row for row in pairs if row["contrast"] == contrast])
            for contrast in sorted({row["contrast"] for row in pairs})
        },
        "pairs": pairs,
    }
    if log:
        print(
            f"gate0_ref DONE pairs={len(pairs)} catastrophic={len(catastrophic)} "
            + " ".join(f"{n}={_fmt(result['overall']['methods'][n])}" for n in method_names),
            flush=True,
        )
    return result


# --------------------------------------------------------------------------------------
# Step 5: residual energy vs floor — the decisive gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidualGateSpec:
    """Verdict thresholds for step 5. All three must be read together; see the module docstring."""

    # Below this, what the affine leaves is at the level of the bank's own storage precision.
    residual_snr_min: float = 10.0
    # Fraction of one traveller's residual that the OTHER traveller's residual predicts. This
    # is what a network trained on the retrospective pool could hope to reproduce. With two
    # travellers this is a 1-vs-1 indicator, not a statistic; see the report.
    predictable_fraction_min: float = 0.05
    # Travellers that must never enter a Gate-0 computation.
    excluded_subjects: tuple[str, ...] = ("0009",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_snr_min": self.residual_snr_min,
            "predictable_fraction_min": self.predictable_fraction_min,
            "excluded_subjects": list(self.excluded_subjects),
        }


def f16_quantization_energy(latent: torch.Tensor) -> float:
    """Mean squared error committed by storing this latent as float16.

    The bank stores the deterministic posterior mean, so re-encoding the same volume returns
    the identical latent: there is no stochastic encode noise to measure. The only noise floor
    that exists *inside* the latent space is the storage quantization, bounded by half an ulp
    per value. That makes this a hard lower bound on any residual that could mean anything.
    """

    values = latent.to(torch.float32)
    half = values.to(torch.float16)
    ulp = (torch.nextafter(half, torch.full_like(half, float("inf"))) - half).to(torch.float32)
    return float((ulp / 2).square().mean())


@torch.inference_mode()
def residual_energy_gate(
    *,
    baselines: Mapping[str, AffineLatentBaseline],
    latents_by_case: Mapping[str, torch.Tensor],
    cases: Sequence[Any],
    spec: ResidualGateSpec = ResidualGateSpec(),
    log: bool = True,
) -> dict[str, Any]:
    """Energy and learnability of ``z_target - affine(z_source)`` on the paired travellers.

    Three questions, in increasing order of how much they matter:

    - How much of the identity displacement does the closed-form affine already explain?
    - Is what is left above the storage floor, and is it smaller than a pure anatomy mismatch
      (the same field pair with ANOTHER subject's target)?
    - Is what is left *predictable* — does one traveller's residual explain the other's? A
      residual can carry plenty of energy and still be unlearnable if it is subject-specific,
      which is the v2 failure repeated one level down.
    """

    subjects = sorted({case.subject_id for case in cases})
    assert_subjects_excluded(subjects, spec.excluded_subjects)
    if len(subjects) < 2:
        raise ValueError(
            f"The residual gate needs at least two paired travellers to estimate the "
            f"predictable fraction; got {subjects}."
        )

    by_group: dict[tuple[str, str], dict[float, Any]] = {}
    for case in cases:
        key = (case.subject_id, Contrast.parse(case.domain.contrast).value)
        by_group.setdefault(key, {})[float(case.domain.field_strength_t)] = case

    quantization_floor = _mean(
        [f16_quantization_energy(latents_by_case[case.case_id]) for case in cases]
    )

    # residuals[space][(contrast, f_s, f_t)][subject] = residual tensor
    residuals: dict[str, dict[tuple[str, float, float], dict[str, torch.Tensor]]] = {
        name: {} for name in baselines
    }
    rows: list[dict[str, Any]] = []

    for (subject_id, contrast), field_map in sorted(by_group.items()):
        fields = sorted(field_map)
        for field_s in fields:
            for field_t in fields:
                if field_s == field_t:
                    continue
                src, tgt = field_map[field_s], field_map[field_t]
                z_s = latents_by_case[src.case_id].to(torch.float32)
                z_t = latents_by_case[tgt.case_id].to(torch.float32)
                identity_energy = float((z_t - z_s).square().mean())

                row: dict[str, Any] = {
                    "subject_id": subject_id,
                    "contrast": contrast,
                    "source_field_t": field_s,
                    "target_field_t": field_t,
                    "abs_log_field_ratio": abs(math.log(field_t / field_s)),
                    "identity_energy": identity_energy,
                }
                for name, baseline in baselines.items():
                    residual = z_t - baseline.transport(z_s, src.domain, tgt.domain)
                    energy = float(residual.square().mean())
                    residuals[name].setdefault((contrast, field_s, field_t), {})[
                        subject_id
                    ] = residual
                    row[name] = {
                        "residual_energy": energy,
                        "explained_fraction": (
                            1.0 - energy / identity_energy if identity_energy > 0 else 0.0
                        ),
                        "residual_snr_vs_quantization": (
                            energy / quantization_floor if quantization_floor > 0 else float("inf")
                        ),
                    }
                rows.append(row)
                if log:
                    print(
                        f"gate0_residual {subject_id} {contrast} {field_s:g}T->{field_t:g}T "
                        f"E_id={identity_energy:.5f} "
                        + " ".join(
                            f"{n}: E={row[n]['residual_energy']:.5f} "
                            f"expl={row[n]['explained_fraction']:.3f}"
                            for n in baselines
                        ),
                        flush=True,
                    )

    alignment = _anatomical_alignment(latents_by_case, by_group, subjects)
    predictability = {
        name: _predictability(
            residuals[name], subjects, alignment=alignment, log=log, space=name
        )
        for name in baselines
    }
    anatomy_floor = {
        name: _anatomy_floor(residuals[name], latents_by_case, by_group, baselines[name])
        for name in baselines
    }

    def aggregate(name: str, subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "num_pairs": len(subset),
            "identity_energy": _mean([r["identity_energy"] for r in subset]),
            "residual_energy": _mean([r[name]["residual_energy"] for r in subset]),
            "explained_fraction": _mean([r[name]["explained_fraction"] for r in subset]),
            "residual_snr_vs_quantization": _mean(
                [r[name]["residual_snr_vs_quantization"] for r in subset]
            ),
        }

    median_ratio = _median([r["abs_log_field_ratio"] for r in rows])
    result: dict[str, Any] = {
        "contract_version": GATE0_CONTRACT_VERSION,
        "spec": spec.to_dict(),
        "subjects": subjects,
        "num_pairs": len(rows),
        "quantization_floor_energy": quantization_floor,
        "baselines": {},
        "pairs": rows,
    }
    for name in baselines:
        result["baselines"][name] = {
            "overall": aggregate(name, rows),
            "strata": {
                "far_field_pairs": {
                    "definition": f"|log(f_t/f_s)| > {median_ratio:.4f} (median split)",
                    **aggregate(name, [r for r in rows if r["abs_log_field_ratio"] > median_ratio]),
                },
                "near_field_pairs": aggregate(
                    name, [r for r in rows if r["abs_log_field_ratio"] <= median_ratio]
                ),
            },
            "by_contrast": {
                contrast: aggregate(name, [r for r in rows if r["contrast"] == contrast])
                for contrast in sorted({r["contrast"] for r in rows})
            },
            "anatomy_floor": anatomy_floor[name],
            "predictability": predictability[name],
        }

    result["verdict"] = _residual_verdict(result, spec)
    if log:
        print(f"gate0_residual DONE {json.dumps(result['verdict'], sort_keys=True)}", flush=True)
    return result


def _anatomical_alignment(
    latents_by_case: Mapping[str, torch.Tensor],
    by_group: Mapping[tuple[str, str], Mapping[float, Any]],
    subjects: Sequence[str],
) -> dict[str, Any]:
    """Voxelwise cosine between DIFFERENT subjects' latents in the SAME domain.

    Context for the predictable fraction, which is a voxelwise comparison across two brains:
    if the volumes were not in a shared space the comparison would be near-meaningless, and
    even in a shared space, registration error and anatomical variation attenuate any
    genuinely shared effect. This measures that attenuation instead of assuming it.

    **This is a descriptive reference scale, not a ceiling.** A low cross-subject residual
    cosine is consistent with "no transferable field effect" AND with "a transferable effect
    that voxelwise cosine cannot see across unaligned anatomy". Nothing here separates those
    two, and the number must not be read as an upper bound on what a model could learn — a
    model can exploit spatially adaptive structure that a fixed voxelwise inner product
    cannot. Treat it as the scale against which the residual cosine is small or not, and
    nothing more.
    """

    cosines: list[float] = []
    per_domain: dict[str, float] = {}
    contrasts = sorted({contrast for _, contrast in by_group})
    for contrast in contrasts:
        fields = sorted(
            {
                field
                for (subject, group_contrast), field_map in by_group.items()
                if group_contrast == contrast
                for field in field_map
            }
        )
        for field in fields:
            available = [
                by_group[(subject, contrast)][field]
                for subject in subjects
                if (subject, contrast) in by_group and field in by_group[(subject, contrast)]
            ]
            for index in range(len(available)):
                for other in available[index + 1 :]:
                    value = _cosine(
                        latents_by_case[available[index].case_id].reshape(-1),
                        latents_by_case[other.case_id].reshape(-1),
                    )
                    cosines.append(value)
                    per_domain[f"{field:g}T/{contrast}"] = value
    mean_cosine = _mean(cosines)
    return {
        "mean_cosine": mean_cosine,
        # Squared, to be comparable with the squared-cosine predictable fraction. Named a
        # reference scale, not a ceiling: see this function's docstring.
        "predictable_fraction_reference_scale": mean_cosine * mean_cosine,
        "num_comparisons": len(cosines),
        "per_domain_cosine": per_domain,
        "interpretation": (
            "Descriptive reference scale for the cross-subject residual cosine. NOT an upper "
            "bound on learnability: a low value is equally consistent with no transferable "
            "field effect and with a transferable effect that voxelwise cosine cannot detect "
            "across imperfectly aligned anatomy."
        ),
    }


def _predictability(
    residuals: Mapping[tuple[str, float, float], Mapping[str, torch.Tensor]],
    subjects: Sequence[str],
    *,
    alignment: Mapping[str, Any],
    log: bool,
    space: str,
) -> dict[str, Any]:
    """Leave-one-subject-out: how much of one traveller's residual the other's predicts.

    For a fixed (contrast, f_s, f_t), the optimal scalar reconstruction of subject B's residual
    from subject A's is ``alpha* r_A`` with ``alpha* = <r_A, r_B>/||r_A||^2``, whose explained
    variance is exactly ``cos^2(r_A, r_B)``. That squared cosine IS the ceiling on what a model
    that learned a shared per-field-pair residual could transfer to a new subject.

    The control uses A's residual for a DIFFERENT field pair: if the aligned and misaligned
    cosines are the same, the shared component is a generic artifact, not a field effect.
    """

    aligned: list[dict[str, Any]] = []
    control: list[float] = []
    keys = sorted(residuals)
    for key in keys:
        per_subject = residuals[key]
        available = [s for s in subjects if s in per_subject]
        if len(available) < 2:
            continue
        a, b = per_subject[available[0]].reshape(-1), per_subject[available[1]].reshape(-1)
        cosine = _cosine(a, b)
        aligned.append(
            {
                "contrast": key[0],
                "source_field_t": key[1],
                "target_field_t": key[2],
                "cosine": cosine,
                "predictable_fraction": cosine * cosine,
            }
        )
        for other in keys:
            if other == key or other[0] != key[0]:
                continue
            other_subject = residuals[other].get(available[0])
            if other_subject is None:
                continue
            control.append(_cosine(other_subject.reshape(-1), b))

    cosines = [item["cosine"] for item in aligned]
    fractions = [item["predictable_fraction"] for item in aligned]
    reference_scale = float(alignment["predictable_fraction_reference_scale"])
    median_fraction = _median(fractions)
    summary = {
        "space": space,
        "num_field_pairs": len(aligned),
        "mean_cosine": _mean(cosines),
        "median_cosine": _median(cosines),
        "mean_predictable_fraction": _mean(fractions),
        "median_predictable_fraction": median_fraction,
        "control_mean_cosine_mismatched_field_pair": _mean(control),
        "control_num_comparisons": len(control),
        "anatomical_alignment": dict(alignment),
        # The raw fraction expressed relative to the cross-subject alignment scale. Descriptive
        # context, not a normalized score: the denominator is a reference scale, not a bound.
        "median_predictable_fraction_vs_reference_scale": (
            median_fraction / reference_scale if reference_scale > 0 else 0.0
        ),
        "per_field_pair": aligned,
    }
    if log:
        print(
            f"gate0_predictability[{space}] pairs={len(aligned)} "
            f"median_cos={summary['median_cosine']:.4f} "
            f"median_predictable={median_fraction:.4f} "
            f"vs_reference={summary['median_predictable_fraction_vs_reference_scale']:.4f} "
            f"alignment_cos={alignment['mean_cosine']:.4f} "
            f"control_cos={summary['control_mean_cosine_mismatched_field_pair']:.4f}",
            flush=True,
        )
    return summary


def _anatomy_floor(
    residuals: Mapping[tuple[str, float, float], Mapping[str, torch.Tensor]],
    latents_by_case: Mapping[str, torch.Tensor],
    by_group: Mapping[tuple[str, str], Mapping[float, Any]],
    baseline: AffineLatentBaseline,
) -> dict[str, Any]:
    """Energy of a residual built from the WRONG subject's target — pure anatomy mismatch.

    Sets the scale at which a residual stops being about field strength. A real residual whose
    energy approaches this is dominated by anatomy, not by the field change.
    """

    energies: list[float] = []
    for (contrast, field_s, field_t), per_subject in sorted(residuals.items()):
        subjects = sorted(per_subject)
        for index, subject in enumerate(subjects):
            other = subjects[(index + 1) % len(subjects)]
            if other == subject:
                continue
            src_case = by_group[(subject, contrast)][field_s]
            wrong_tgt_case = by_group[(other, contrast)][field_t]
            z_s = latents_by_case[src_case.case_id].to(torch.float32)
            z_t_wrong = latents_by_case[wrong_tgt_case.case_id].to(torch.float32)
            mapped = baseline.transport(z_s, src_case.domain, wrong_tgt_case.domain)
            energies.append(float((z_t_wrong - mapped).square().mean()))
    return {"mean_energy": _mean(energies), "num_comparisons": len(energies)}


def _residual_verdict(result: Mapping[str, Any], spec: ResidualGateSpec) -> dict[str, Any]:
    """Mechanical reading of the thresholds. The decision stays with the human."""

    checks: dict[str, Any] = {}
    proceed_any = False
    for name, block in result["baselines"].items():
        snr = block["overall"]["residual_snr_vs_quantization"]
        predictable = block["predictability"]["median_predictable_fraction"]
        anatomy = block["anatomy_floor"]["mean_energy"]
        energy = block["overall"]["residual_energy"]
        above_storage_floor = snr >= spec.residual_snr_min
        learnable = predictable >= spec.predictable_fraction_min
        below_anatomy_floor = energy < anatomy
        checks[name] = {
            "residual_snr_vs_quantization": snr,
            "above_storage_floor": above_storage_floor,
            "median_predictable_fraction": predictable,
            "predictable_above_threshold": learnable,
            # Reported, NOT part of the decision rule, and not a normalized score. The
            # alignment scale was measured after the first run exposed the confound, and
            # moving a pre-declared threshold onto a statistic chosen once the numbers were
            # visible is the exact failure mode the pre-registration exists to prevent.
            "median_predictable_fraction_vs_reference_scale": block["predictability"][
                "median_predictable_fraction_vs_reference_scale"
            ],
            "anatomical_alignment_cosine": block["predictability"]["anatomical_alignment"][
                "mean_cosine"
            ],
            "residual_energy": energy,
            "anatomy_floor_energy": anatomy,
            "residual_below_anatomy_floor": below_anatomy_floor,
            "proceed": bool(above_storage_floor and learnable),
        }
        proceed_any = proceed_any or checks[name]["proceed"]
    return {
        "decision": "gate0_proceed_to_gate1" if proceed_any else "gate0_close_generative_branch",
        "rationale": (
            "At least one affine baseline leaves a residual that is both above the storage "
            "floor and partly predictable across travellers."
            if proceed_any
            else "No affine baseline leaves a residual that is both above the storage floor "
            "and predictable from one traveller to the other."
        ),
        "checks": checks,
        "caveat": (
            "The predictable fraction is estimated from two paired travellers (1-vs-1). It is "
            "an indicator, not a statistic, and cannot be given a confidence interval. It is "
            "also a voxelwise statistic across imperfectly aligned anatomy, so a low value "
            "does not by itself establish that no transferable field effect exists — only "
            "that this measurement does not see one. This verdict is one input among the "
            "Gate-0 diagnostics, not a standalone decision."
        ),
    }


# --------------------------------------------------------------------------------------
# Per-contrast diagnostics: why T2w and T2-FLAIR look worse
# --------------------------------------------------------------------------------------


@torch.inference_mode()
def coupling_quality_by_contrast(
    *,
    train_index: Any,
    train_descriptors: torch.Tensor,
    cases: Sequence[Any],
    latents_by_case: Mapping[str, torch.Tensor],
    stats: LatentStats,
    pool_size: int,
    nn_candidates: int = 5,
    log: bool = True,
) -> dict[str, Any]:
    """How good the ``nn`` unpaired coupling's pseudo-pairs are, per contrast.

    For each traveller source and each other field, find the nearest training-pool records of
    the target domain in descriptor space and report that distance. If T2/T2-FLAIR sources sit
    systematically further from their retrieved partner than T1w sources do, the coupling is
    handing the transport a worse pseudo-pair for those contrasts, and any per-contrast
    difference downstream is at least partly a data-association problem rather than a modelling
    one. Descriptor distance is the coupling's own cost, so this is what the coupling saw —
    not a proxy for it.
    """

    # The coupling's own pooling, imported rather than reimplemented: a second copy that
    # drifted would silently measure a different space than the one the coupling ranked in.
    from fieldbridge.training.stage2_transport import _spatial_pool

    records = list(train_index.records)
    by_domain: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        by_domain.setdefault(record.domain.label, []).append(position)

    rows: list[dict[str, Any]] = []
    for case in cases:
        contrast = Contrast.parse(case.domain.contrast).value
        source_descriptor = _spatial_pool(
            stats.normalize(latents_by_case[case.case_id].unsqueeze(0)), pool_size
        ).reshape(-1)
        for target_label, positions in sorted(by_domain.items()):
            if not target_label.endswith(f"/{contrast}"):
                continue
            if target_label == case.domain.label:
                continue
            candidates = train_descriptors[positions].to(source_descriptor.device)
            distances = torch.cdist(
                source_descriptor.reshape(1, -1), candidates.reshape(len(positions), -1)
            ).reshape(-1)
            best = torch.topk(distances, k=min(nn_candidates, distances.numel()), largest=False)
            rows.append(
                {
                    "subject_id": case.subject_id,
                    "contrast": contrast,
                    "source_field_t": float(case.domain.field_strength_t),
                    "target_domain": target_label,
                    "pool_size": len(positions),
                    "nearest_distance": float(best.values[0]),
                    "mean_candidate_distance": float(best.values.mean()),
                    "median_pool_distance": float(distances.median()),
                }
            )
            if log:
                print(
                    f"gate0_coupling {case.subject_id} {case.domain.label} -> {target_label} "
                    f"nearest={rows[-1]['nearest_distance']:.4f} "
                    f"pool_median={rows[-1]['median_pool_distance']:.4f} n={len(positions)}",
                    flush=True,
                )

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "num_retrievals": len(subset),
            "mean_nearest_distance": _mean([r["nearest_distance"] for r in subset]),
            "mean_candidate_distance": _mean([r["mean_candidate_distance"] for r in subset]),
            "mean_pool_median_distance": _mean([r["median_pool_distance"] for r in subset]),
            # How much closer the retrieved partner is than a random pool member. Near 1.0 means
            # the retrieval found nothing special and the coupling is effectively random.
            "retrieval_advantage": (
                _mean([r["nearest_distance"] for r in subset])
                / _mean([r["median_pool_distance"] for r in subset])
                if subset and _mean([r["median_pool_distance"] for r in subset]) > 0
                else 0.0
            ),
        }

    return {
        "contract_version": GATE0_CONTRACT_VERSION,
        "pool_size": pool_size,
        "nn_candidates": nn_candidates,
        "overall": block(rows),
        "by_contrast": {
            contrast: block([r for r in rows if r["contrast"] == contrast])
            for contrast in sorted({r["contrast"] for r in rows})
        },
        "retrievals": rows,
    }


def per_contrast_diagnostics(
    *,
    reference: Mapping[str, Any] | None = None,
    sweep: Mapping[str, Any] | None = None,
    residual: Mapping[str, Any] | None = None,
    coupling: Mapping[str, Any] | None = None,
    residual_baseline: str = "foreground",
) -> dict[str, Any]:
    """Assemble the per-contrast evidence table from whichever Gate-0 outputs are available.

    The question this answers: T2w and T2-FLAIR look worse — is that a Stage-1 ceiling, a
    coupling-quality problem, or a shortcut that only works for T1w? Each column comes from a
    different diagnostic, so a contrast that is bad in only one column is a different problem
    from one that is bad in all of them.

    ``loss_contribution`` is deliberately absent: it needs per-contrast loss/gradient
    instrumentation inside the training loop, which Gate 0 does not run.
    """

    contrasts: set[str] = set()
    for payload, key in ((reference, "by_contrast"), (sweep, "by_contrast"), (coupling, "by_contrast")):
        if payload:
            contrasts.update(payload.get(key, {}))
    if residual:
        block = residual.get("baselines", {}).get(residual_baseline, {})
        contrasts.update(block.get("by_contrast", {}))

    table: dict[str, dict[str, Any]] = {}
    for contrast in sorted(contrasts):
        entry: dict[str, Any] = {}
        if reference and contrast in reference.get("by_contrast", {}):
            methods = reference["by_contrast"][contrast]["methods"]
            entry["stage1_ceiling"] = methods.get("ceiling")
            entry["identity"] = methods.get("identity")
            for name in ("sb_v2", "affine", "identity_histogram_matched", "identity_robust_affine"):
                if name in methods:
                    entry[name] = methods[name]
        if coupling and contrast in coupling.get("by_contrast", {}):
            entry["coupling"] = coupling["by_contrast"][contrast]
        if sweep and contrast in sweep.get("by_contrast", {}):
            block = sweep["by_contrast"][contrast]
            entry["target_conditioning"] = {
                "responds_fraction": block.get("responds_fraction"),
                "correct_target_fraction": block.get("correct_target_fraction"),
                "chance_correct_target_fraction": block.get("chance_correct_target_fraction"),
                "mean_classification_margin": block.get("mean_classification_margin"),
            }
        if residual:
            block = residual.get("baselines", {}).get(residual_baseline, {})
            if contrast in block.get("by_contrast", {}):
                entry["residual"] = block["by_contrast"][contrast]
        table[contrast] = entry

    return {
        "contract_version": GATE0_CONTRACT_VERSION,
        "residual_baseline": residual_baseline,
        "sources_present": {
            "reference_gate": reference is not None,
            "wrong_target_sweep": sweep is not None,
            "residual_gate": residual is not None,
            "coupling_quality": coupling is not None,
        },
        "not_measured": {
            "loss_contribution": (
                "Needs per-contrast loss/gradient instrumentation inside the Stage-2 training "
                "loop. Gate 0 runs no training, so this row cannot be filled here."
            )
        },
        "by_contrast": table,
    }


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def build_sb_transport(
    translator: Any,
    stats: LatentStats,
    sampler: TransportSamplerConfig,
) -> TransportFn:
    """Wrap the trained velocity field as a raw-latent transport (normalize, integrate, denormalize)."""

    def transport(z: torch.Tensor, source: Domain, target: Domain) -> torch.Tensor:
        batched = z.unsqueeze(0) if z.ndim == 4 else z
        z1 = sample_transport(translator, stats.normalize(batched), source, target, sampler)
        out = stats.denormalize(z1)
        return out[0] if z.ndim == 4 else out

    return transport


def compose_minus(primary: TransportFn, subtracted: TransportFn) -> TransportFn:
    """``z + (primary(z) - subtracted(z))`` — primary's displacement with the other's removed.

    **This is a diagnostic decomposition, not a model variant.** Subtracting one displacement
    from another is not guaranteed to land on the latent manifold the decoder was trained on,
    so its decoded output can be off-distribution and its official metrics are not a claim
    about a system anyone could deploy. It answers exactly one question — how much of SB's
    displacement survives once the closed-form rescaling is taken out — and it must never be
    reported as a candidate model or promoted on its score.

    The literal difference of the two output latents would be worse still: that leaves the
    manifold outright (a near-zero-mean difference field, not a brain), which is why the
    displacement form is used instead.
    """

    def transport(z: torch.Tensor, source: Domain, target: Domain) -> torch.Tensor:
        return z + (primary(z, source, target) - subtracted(z, source, target))

    return transport


def load_raw_latents(cases: Sequence[Any], device: torch.device) -> dict[str, torch.Tensor]:
    """case_id -> raw (C, x, y, z) float32 latent on ``device``."""

    out: dict[str, torch.Tensor] = {}
    for case in cases:
        payload = torch.load(case.latent_path, map_location="cpu")
        out[str(case.case_id)] = payload["latent"].to(torch.float32).to(device)
    return out


def _method_names(
    methods: Mapping[str, Any],
    image_methods: Mapping[str, Any] | None,
    include_ceiling: bool,
) -> list[str]:
    return (
        list(methods)
        + list(image_methods or {})
        + (["ceiling"] if include_ceiling else [])
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if float(denominator) > 0 else 0.0


def _mean(values: Any) -> float:
    items = [float(v) for v in values]
    return float(sum(items) / len(items)) if items else 0.0


def _median(values: Any) -> float:
    items = sorted(float(v) for v in values)
    if not items:
        return 0.0
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return 0.5 * (items[middle - 1] + items[middle])


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    return {
        key: float(
            sum(row[key] for row in rows if key in row)
            / max(1, sum(key in row for row in rows))
        )
        for key in sorted(keys)
    }


def _fmt(metrics: Mapping[str, float]) -> str:
    return "{" + " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())) + "}"


def write_gate0_result(result: Mapping[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = [
    "GATE0_CONTRACT_VERSION",
    "ReferenceGateSpec",
    "ResidualGateSpec",
    "RobustNormSpec",
    "StratumSpec",
    "SweepSpec",
    "TransportFn",
    "assert_full_volume_bank",
    "assert_subjects_excluded",
    "build_sb_transport",
    "compose_minus",
    "coupling_quality_by_contrast",
    "evaluate_reference_gate",
    "f16_quantization_energy",
    "gate0_metric_fn",
    "latent_rms",
    "load_raw_latents",
    "per_contrast_diagnostics",
    "residual_energy_gate",
    "robust_normalize",
    "traveller_cases",
    "wrong_target_sweep",
    "write_gate0_result",
]
