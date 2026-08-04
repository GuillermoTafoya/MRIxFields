"""Etapa-2 transport evaluation gate: decode transported latents and score them.

The transport train loss (flow-matching MSE) floors by construction for unpaired OT-CFM and is
*not* a quality signal. The only ground truth for the field-to-field task is the prospective
travellers (``P_`` cases): the same subject imaged at multiple field strengths, same contrast.
This module integrates the learned velocity field into a target-field latent, decodes it with the
frozen VAE, and scores it against the real target volume with the official Task-3 metrics.

Three comparable readouts per pair (all share the same decode path, so differences isolate the
transport step):

- ``transport``: decode(denorm(ODE(v_theta) from z0_norm))            — the model.
- ``identity`` : decode(source latent)                               — no-transport baseline.
- ``ceiling``  : decode(target latent)                               — the frozen-VAE ceiling.

If ``transport`` does not sit clearly above ``identity`` (toward ``ceiling``), the transport is
not learning the field change, regardless of what the training loss did.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

Solver = Literal["euler", "heun"]
METHODS = ("transport", "identity", "ceiling")
DEFAULT_METRICS = ("ssim", "nrmse", "lpips")


@dataclass(frozen=True, slots=True)
class TransportSamplerConfig:
    """Deterministic probability-flow ODE sampler for the OT-CFM velocity field.

    ``euler`` is one forward eval per step; ``heun`` is the 2nd-order trapezoid (two evals/step)
    and needs far fewer steps for the near-straight OT-CFM paths. (Schrodinger-bridge SDE sampling
    is a separate follow-up; this integrates the learned field ``v_theta`` as a flow ODE.)
    """

    solver: Solver = "heun"
    n_steps: int = 20


@dataclass(frozen=True, slots=True)
class DecodeSpec:
    """Decode parameters for the gate. Full-volume by default; tiling is the fallback.

    The tiled path is an approximation, not an equivalent: the decoder's GroupNorm makes it a
    non-local function of the latent, so no halo makes tiled decode match full decode (see
    ``latent_bank.decode_latent``). It costs ~0.04 of official nRMSE, applied equally to the
    transport, identity and ceiling columns.

    ``halo`` deliberately does NOT come from the bank manifest. The bank's halo is an *encode*
    parameter — and when the bank was built with ``strategy="full"`` it was never even used,
    yet it was still being inherited here, which is how the run-C gate silently decoded at a
    4-latent-voxel halo, roughly half the decoder's receptive field.
    """

    block_size: tuple[int, int, int] = (128, 128, 128)
    halo: tuple[int, int, int] = (64, 64, 64)
    precision: Literal["float32", "bfloat16"] = "bfloat16"
    strategy: Literal["auto", "full", "tiled"] = "auto"

    @classmethod
    def from_bank_manifest(cls, manifest: Mapping[str, Any], **overrides: Any) -> "DecodeSpec":
        """Take only ``precision`` from the bank; decode geometry is a decode-side choice."""

        config = manifest.get("config", {}) if isinstance(manifest, Mapping) else {}
        defaults = cls()
        spec = {
            "block_size": defaults.block_size,
            "halo": defaults.halo,
            "precision": config.get("precision", defaults.precision),
            "strategy": defaults.strategy,
        }
        spec.update({key: value for key, value in overrides.items() if value is not None})
        return cls(
            block_size=tuple(int(v) for v in spec["block_size"]),  # type: ignore[arg-type]
            halo=tuple(int(v) for v in spec["halo"]),  # type: ignore[arg-type]
            precision=spec["precision"],  # type: ignore[arg-type]
            strategy=spec["strategy"],  # type: ignore[arg-type]
        )


@torch.inference_mode()
def sample_transport(
    translator: Any,
    z0: torch.Tensor,
    source_domain: Domain,
    target_domain: Domain,
    cfg: TransportSamplerConfig,
) -> torch.Tensor:
    """Integrate dz/dt = v_theta(z_t, t, c_s, c_t) from t=0 (z0) to t=1 in standardized space."""

    if z0.ndim != 5:
        raise ValueError(f"sample_transport expects a (1,C,x,y,z) latent, got {tuple(z0.shape)}.")
    if cfg.n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    batch = z0.shape[0]
    dom_s = [source_domain] * batch
    dom_t = [target_domain] * batch
    dt = 1.0 / cfg.n_steps
    z = z0
    for step in range(cfg.n_steps):
        t0 = torch.full((batch,), step * dt, device=z0.device, dtype=torch.float32)
        v0 = translator(z, dom_s, dom_t, t0)
        if cfg.solver == "euler":
            z = z + dt * v0
            continue
        t1 = torch.full((batch,), min((step + 1) * dt, 1.0), device=z0.device, dtype=torch.float32)
        v1 = translator(z + dt * v0, dom_s, dom_t, t1)
        z = z + 0.5 * dt * (v0 + v1)
    return z


def official_metric_fn(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    metrics: Sequence[str],
    device: str,
) -> dict[str, float]:
    """Official Task-3 metrics on a decoded (1,1,X,Y,Z) prediction vs target volume.

    LPIPS is skipped (not failed) when the optional ``lpips`` package is unavailable, so the gate
    still runs on boxes without the ``official-evaluation`` extra.
    """

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
    return out


@dataclass(frozen=True, slots=True)
class _TravellerCase:
    case_id: str
    subject_id: str
    domain: Domain
    latent_path: Path
    image_record: VolumeRecord
    split: str | None = None


def _traveller_cases(
    records: Sequence[VolumeRecord],
    manifest: Mapping[str, Any],
    bank_dir: Path,
    subjects: Sequence[str] | None,
    split_of_case: Mapping[str, str] | None = None,
) -> list[_TravellerCase]:
    """Join split records (real target volumes) to bank latents by case_id, keeping travellers."""

    latent_path_by_case: dict[str, Path] = {}
    for entry in manifest.get("records", []):
        latent_path_by_case[str(entry["case_id"])] = bank_dir / entry["path"]
    subject_filter = set(subjects) if subjects else None
    cases: list[_TravellerCase] = []
    for record in records:
        # Only prospective (P_) travellers are cross-field paired. This filter ALWAYS applies:
        # retrospective and prospective share numeric subject_ids (e.g. R_..._0007 and P_..._0007
        # are different people), so ``subjects`` can only further restrict *within* prospective —
        # it must never let a retrospective same-number subject into a traveller's field group.
        is_prospective = str(record.case_id).startswith("P_") or (
            isinstance(record.metadata, Mapping) and record.metadata.get("prefix") == "P"
        )
        if not is_prospective:
            continue
        if subject_filter is not None and record.subject_id not in subject_filter:
            continue
        latent_path = latent_path_by_case.get(str(record.case_id))
        if latent_path is None:
            continue  # traveller present in the split but not encoded in this bank
        cases.append(
            _TravellerCase(
                case_id=str(record.case_id),
                subject_id=str(record.subject_id),
                domain=record.domain,
                latent_path=latent_path,
                image_record=record,
                split=(split_of_case or {}).get(str(record.case_id)),
            )
        )
    return cases


# Public alias: Gate 0 reuses this join rather than reimplementing it. The prospective-only
# filter and the cohort caveat it encodes are leakage-critical, and two copies of that logic
# is exactly how they drift apart.
traveller_cases = _traveller_cases


def _load_raw_latent(path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    return payload["latent"].to(torch.float32).unsqueeze(0).to(device)


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    return {
        key: float(sum(row[key] for row in rows if key in row) / max(1, sum(key in row for row in rows)))
        for key in sorted(keys)
    }


@torch.inference_mode()
def evaluate_transport_travellers(
    *,
    translator: Any,
    decoder: Any,
    records: Sequence[VolumeRecord],
    bank_manifest: Mapping[str, Any],
    bank_dir: Path,
    stats: LatentStats,
    sampler: TransportSamplerConfig,
    decode: DecodeSpec,
    device: torch.device,
    metrics: Sequence[str] = DEFAULT_METRICS,
    subjects: Sequence[str] | None = None,
    split_of_case: Mapping[str, str] | None = None,
    training_splits: Sequence[str] = ("train",),
    allow_training_subjects: bool = False,
    field_pairs: Sequence[tuple[float, float]] | None = None,
    metric_fn: Callable[..., dict[str, float]] = official_metric_fn,
    volume_loader: Callable[[VolumeRecord], torch.Tensor] = load_volume,
    log: bool = True,
) -> dict[str, Any]:
    """Score every same-subject same-contrast cross-field traveller pair. See module docstring."""

    translator = translator.to(device).eval()
    decoder = decoder.to(device).eval()
    factor = downsample_factor(decoder)
    cases = _traveller_cases(records, bank_manifest, bank_dir, subjects, split_of_case)
    if not cases:
        raise ValueError(
            "No traveller (prospective) cases found that are also encoded in the bank. "
            "Pass --subjects, or check the split/bank."
        )

    # Fail closed on scoring a subject the transport was trained on. With the `nn` coupling a
    # traveller in the training pool retrieves its OWN volume at the target field as the nearest
    # neighbour — the same anatomy is trivially nearest — so it receives real paired supervision
    # and any score on it measures memorization. Nothing in the numbers distinguishes that from a
    # clean read, which is exactly why this has to be refused rather than warned about.
    subject_splits = {case.subject_id: case.split for case in cases}
    contaminated = sorted(
        subject for subject, split in subject_splits.items() if split in set(training_splits)
    )
    if contaminated and not allow_training_subjects:
        raise ValueError(
            f"Refusing to score subject(s) {contaminated}: they are in the transport's training "
            f"split(s) {list(training_splits)}, so the result measures memorization, not "
            "generalization. Score a held-out traveller, or pass allow_training_subjects=True "
            "(--allow-training-subjects) if an optimistic upper bound is what you want."
        )

    by_group: dict[tuple[str, Contrast], dict[float, _TravellerCase]] = {}
    for case in cases:
        key = (case.subject_id, Contrast.parse(case.domain.contrast))
        by_group.setdefault(key, {})[float(case.domain.field_strength_t)] = case

    decoded_cache: dict[str, torch.Tensor] = {}
    decode_paths: set[str] = set()

    def decoded_from_latent(case: _TravellerCase) -> torch.Tensor:
        if case.case_id not in decoded_cache:
            latent = _load_raw_latent(case.latent_path, device)
            decoded_cache[case.case_id], _ = decode_latent(
                decoder, latent, case.domain, factor=factor, strategy=decode.strategy,
                block_size=decode.block_size, halo=decode.halo, precision=decode.precision,
            )
        return decoded_cache[case.case_id]

    pairs: list[dict[str, Any]] = []
    for (subject_id, contrast), field_map in sorted(by_group.items(), key=lambda kv: (kv[0][0], kv[0][1].value)):
        fields = sorted(field_map)
        candidate_pairs = field_pairs if field_pairs is not None else [
            (fs, ft) for fs in fields for ft in fields if fs != ft
        ]
        for field_s, field_t in candidate_pairs:
            if field_s not in field_map or field_t not in field_map:
                continue
            src, tgt = field_map[field_s], field_map[field_t]
            target_image = volume_loader(tgt.image_record).to(device)

            z0_norm = stats.normalize(_load_raw_latent(src.latent_path, device))
            z1_norm = sample_transport(translator, z0_norm, src.domain, tgt.domain, sampler)
            transport_image, decode_path = decode_latent(
                decoder, stats.denormalize(z1_norm), tgt.domain, factor=factor,
                strategy=decode.strategy, block_size=decode.block_size, halo=decode.halo,
                precision=decode.precision,
            )
            decode_paths.add(decode_path)

            row = {
                "subject_id": subject_id,
                "contrast": contrast.value,
                "source_field_t": field_s,
                "target_field_t": field_t,
                "transport": metric_fn(transport_image, target_image, metrics=metrics, device=device.type),
                "identity": metric_fn(decoded_from_latent(src), target_image, metrics=metrics, device=device.type),
                "ceiling": metric_fn(decoded_from_latent(tgt), target_image, metrics=metrics, device=device.type),
            }
            pairs.append(row)
            if log:
                t = row["transport"]
                i = row["identity"]
                print(
                    f"transport_eval {subject_id} {contrast.value} {field_s}T->{field_t}T "
                    f"transport={_fmt(t)} identity={_fmt(i)} ceiling={_fmt(row['ceiling'])}",
                    flush=True,
                )
            del target_image, transport_image, z0_norm, z1_norm

    overall = {method: _mean_metrics([row[method] for row in pairs]) for method in METHODS}
    by_contrast: dict[str, dict[str, dict[str, float]]] = {}
    for contrast_value in sorted({row["contrast"] for row in pairs}):
        subset = [row for row in pairs if row["contrast"] == contrast_value]
        by_contrast[contrast_value] = {m: _mean_metrics([row[m] for row in subset]) for m in METHODS}

    result = {
        "num_pairs": len(pairs),
        "subjects": sorted({row["subject_id"] for row in pairs}),
        # Provenance, so a contaminated read is never mistaken for a clean one after the fact.
        "subject_splits": {k: v for k, v in sorted(subject_splits.items())},
        "scored_training_subjects": contaminated,
        "metrics": list(metrics),
        "sampler": {"solver": sampler.solver, "n_steps": sampler.n_steps},
        # "tiled" here means the numbers carry a decode approximation; see decode_latent.
        "decode": {"strategy": decode.strategy, "path_used": sorted(decode_paths),
                   "halo": list(decode.halo), "block_size": list(decode.block_size)},
        "overall": overall,
        "by_contrast": by_contrast,
        "pairs": pairs,
    }
    if log:
        print(
            f"transport_eval DONE pairs={len(pairs)} "
            f"transport={_fmt(overall['transport'])} identity={_fmt(overall['identity'])} "
            f"ceiling={_fmt(overall['ceiling'])}",
            flush=True,
        )
    return result


def _fmt(metrics: Mapping[str, float]) -> str:
    return "{" + " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())) + "}"


def write_transport_eval(result: Mapping[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = [
    "TransportSamplerConfig",
    "DecodeSpec",
    "traveller_cases",
    "sample_transport",
    "official_metric_fn",
    "evaluate_transport_travellers",
    "write_transport_eval",
]
