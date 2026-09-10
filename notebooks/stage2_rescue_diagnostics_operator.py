"""One-click R-only Stage-2 rescue diagnostic operator.

This file orchestrates checked-in diagnostic functions.  It never trains a scientific
model, opens P:0006, or touches P:0009.  The only optimizer use is the explicitly
isolated synthetic micro-overfit gate.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.nn import functional as F


TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
CHECKPOINT_SHA256 = "09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021"
BANK_ARCHIVE_SHA256 = "78d323c02ceccdfcb054307da3c9e14575210869d22cade6c5ecd4afa4baf8d5"
BANK_TREE_SHA256 = "f9cb09bfa177a3e389f87f087b0d756a2709e2054559a39c85e8272d5e1cfaa3"
BANK_ARTIFACT_SHA256 = "8081ce89a0eac1522b4fb28cd7919de4a4ecf1d5af72552d141a0ee9b9944194"

DRIVE_ROOT = Path("/content/drive/MyDrive/MRIxFields2026")
GATE01_ROOT = Path("/content/drive/MyDrive/MRIxFields2026/Gate01Private_8012a3f")
OUTPUT_ROOT = Path("/content/drive/MyDrive/MRIxFields2026/UnifiedStage2_1ca2b4a_01")
STAGE2_ROOT = OUTPUT_ROOT / "stage2_unified_v7"
BANK_NAMESPACE = STAGE2_ROOT / "bank_8081ce89a0ea"
TRAINING_NAMESPACE = BANK_NAMESPACE / "implementation_82633d66e5ea"
BANK_ARCHIVE = OUTPUT_ROOT / "photometry_factored_latent_bank_v2.tar"
PHOTOMETRY_ARTIFACT = OUTPUT_ROOT / "stage2_photometry_factorization_v1.json"
VAE_CHECKPOINT = DRIVE_ROOT / "vae_kl_vae_best.pt"
VAE_CONFIG = GATE01_ROOT / "stage1-run-c.yaml"
LOCAL_SCRATCH = Path("/content/stage2_gate01_recovery_v8_scratch")
LOCAL_BANK_ARCHIVE = LOCAL_SCRATCH / "photometry_factored_latent_bank_v2.tar"
LOCAL_BANK_ROOT = LOCAL_SCRATCH / "photometry_factored_latent_bank_v2"
DIAGNOSTIC_ROOT = OUTPUT_ROOT / "stage2_rescue_2026_09_01" / "diagnostics_v1"

if "REPO_DIR" not in globals():
    REPO_DIR = Path(__file__).resolve().parents[1]

PILOT_ROOT = TRAINING_NAMESPACE / "unified_full_objective_pilot_200"
PILOT_ATTEMPT = PILOT_ROOT / "scientific_attempts" / "attempt-0001"
CHECKPOINT = PILOT_ATTEMPT / "checkpoints" / "stage2_unified_full_step000000200.pt"
RESOLVED_CONFIG = PILOT_ROOT / "resolved_config.json"


def _progress(payload: Mapping[str, object]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _require_input_files() -> None:
    required = (
        BANK_ARCHIVE,
        PHOTOMETRY_ARTIFACT,
        VAE_CHECKPOINT,
        VAE_CONFIG,
        CHECKPOINT,
        RESOLVED_CONFIG,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError({"missing_sealed_inputs": missing})
    if DIAGNOSTIC_ROOT == TRAINING_NAMESPACE or TRAINING_NAMESPACE in DIAGNOSTIC_ROOT.parents:
        raise RuntimeError("Rescue diagnostics may not write inside a sealed namespace.")


def _fixed_positions(index, count: int) -> list[int]:
    selected: list[int] = []
    for contrast in Contrast:
        candidates = [
            position
            for position, record in enumerate(index.records)
            if record.domain.contrast == contrast
        ]
        if candidates:
            selected.append(candidates[0])
        if len(selected) == count:
            break
    if len(selected) < count:
        for position in range(len(index.records)):
            if position not in selected:
                selected.append(position)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("The fixed R-only diagnostic source inventory is incomplete.")
    return selected


def _load_or_run(path: Path, producer):
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        verify_self_hash(payload)
        print(json.dumps({"stage": path.stem, "status": "exact_resume"}), flush=True)
        return payload
    payload = producer()
    write_diagnostic_json(path, payload)
    print(json.dumps({"stage": path.stem, "status": payload["status"]}), flush=True)
    return payload


def _vae_kwargs(model_config: Mapping[str, object], component: str) -> dict[str, object]:
    shared = {
        "base_channels",
        "latent_channels",
        "spatial_dims",
        "activation",
        "use_norm",
        "num_res_blocks",
    }
    kwargs = {key: value for key, value in model_config.items() if key in shared}
    if component == "decoder" and "out_channels" in model_config:
        kwargs["out_channels"] = model_config["out_channels"]
    if component == "decoder" and "output_activation" in model_config:
        kwargs["output_activation"] = model_config["output_activation"]
    if component == "decoder" and "domain_conditioning_dim" in model_config:
        kwargs["domain_conditioning_dim"] = model_config["domain_conditioning_dim"]
    return kwargs


source_dir = str(REPO_DIR / "src")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, source_dir)

from fieldbridge.data.domains import Contrast, Domain, FIELD_STRENGTHS_T
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import SourceCanonicalizedVolume, sha256_file
from fieldbridge.evaluation.stage2_step200_inference_audit import (
    preflight_frozen_stage1_run_c_config,
    preflight_reviewed_photometry_namespace_artifact,
    verify_frozen_stage1_vae_bank_provenance,
    verify_reviewed_photometry_bank_provenance,
)
from fieldbridge.evaluation.stage2_rescue_diagnostics import (
    LatentDiagnosticRecord,
    aggregate_conditioning_diagnostics,
    aggregate_gradient_diagnostics,
    aggregate_target_sweeps,
    bank_domain_latent_centroids,
    bank_identifiability_features,
    build_rescue_scorecard,
    diagnose_conditioning_plumbing,
    diagnose_off_manifold_latent_drift,
    diagnose_per_term_gradients,
    diagnose_real_domain_identifiability,
    render_rescue_scorecard_markdown,
    run_synthetic_micro_overfit,
    same_source_five_target_sweep,
    verify_self_hash,
    write_diagnostic_json,
)
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    copy_verified_stage2_bank_tar_to_local,
    preflight_stage2_local_disk_capacity,
    restore_verified_stage2_bank_tar,
)
from fieldbridge.models.discriminators import DomainProjectionDiscriminator
from fieldbridge.models.factory import build_decoder, build_translator
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.stage2_unified import (
    UNIFIED_RESUME_CONTRACT,
    UnifiedStage2Config,
    _DomainPools,
    _sample_training_batch,
    integrate_transport,
)


_require_input_files()
if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
    raise ValueError("The step-200 checkpoint raw SHA-256 changed.")
if not torch.cuda.is_available():
    raise RuntimeError("Full-volume rescue diagnostics require a CUDA runtime.")
device = torch.device("cuda")
properties = torch.cuda.get_device_properties(device)
if "A100" not in properties.name or int(properties.total_memory) < 79 * 1024**3:
    raise RuntimeError("Run-all requires an authenticated NVIDIA A100 80 GB runtime.")
print(
    json.dumps(
        {
            "stage": "hardware_and_boundary_preflight",
            "status": "pass",
            "gpu": properties.name,
            "training_authorized": False,
            "synthetic_micro_overfit_only": True,
            "P0006_accessed": False,
            "P0009_accessed": False,
        },
        sort_keys=True,
    ),
    flush=True,
)

capacity = preflight_stage2_local_disk_capacity(
    BANK_ARCHIVE,
    LOCAL_SCRATCH,
    local_archive_path=LOCAL_BANK_ARCHIVE,
    local_bank_root=LOCAL_BANK_ROOT,
    expected_extracted_bytes=12_873_486_620,
)
print(json.dumps({"stage": "local_capacity", **capacity.to_dict()}, sort_keys=True), flush=True)
local_archive = copy_verified_stage2_bank_tar_to_local(
    BANK_ARCHIVE,
    LOCAL_BANK_ARCHIVE,
    expected_archive_sha256=BANK_ARCHIVE_SHA256,
    progress_callback=_progress,
)
restore = restore_verified_stage2_bank_tar(
    local_archive,
    LOCAL_BANK_ROOT,
    expected_archive_sha256=BANK_ARCHIVE_SHA256,
    expected_tree_sha256=BANK_TREE_SHA256,
    expected_bank_artifact_sha256=BANK_ARTIFACT_SHA256,
    expected_file_count=3312,
    expected_total_bytes=12_873_486_620,
    progress_callback=_progress,
)
print(
    json.dumps(
        {
            "stage": "reviewed_R_bank_restore",
            "status": "pass",
            "restored_from_tar": restore.restored_from_tar,
            "P_records_loaded": 0,
        },
        sort_keys=True,
    ),
    flush=True,
)

resolved = json.loads(RESOLVED_CONFIG.read_text(encoding="utf-8"))
cfg = UnifiedStage2Config.from_mapping(resolved)
model_config = dict(resolved["model"])
model_name = str(model_config.pop("name", "flow_matching_latent"))
translator = build_translator(model_name, **model_config)
checkpoint = load_checkpoint(CHECKPOINT, map_location="cpu")
metadata = checkpoint.get("_meta")
if (
    checkpoint.get("contract_version") != UNIFIED_RESUME_CONTRACT
    or checkpoint.get("training_cursor") != 200
    or not isinstance(metadata, Mapping)
    or metadata.get("git_commit") != TRAINING_EVIDENCE_COMMIT
):
    raise ValueError("Step-200 checkpoint evidence identity changed.")
translator.load_state_dict(checkpoint["translator"], strict=True)

train_index = PhotometryFactoredLatentBankIndex(
    LOCAL_BANK_ROOT, "train", expected_artifact_sha256=BANK_ARTIFACT_SHA256
)
validation_index = PhotometryFactoredLatentBankIndex(
    LOCAL_BANK_ROOT, "validation", expected_artifact_sha256=BANK_ARTIFACT_SHA256
)
stats = FactoredLatentStats.from_bank(LOCAL_BANK_ROOT)
critic_input_channels = int(stats.mean.numel()) + 1 if cfg.critic_space == "latent" else 2
critic = DomainProjectionDiscriminator(critic_input_channels, cfg.critic_channels)
critic.load_state_dict(checkpoint["critic"], strict=True)
del checkpoint

vae_preflight = preflight_frozen_stage1_run_c_config(VAE_CONFIG)
verify_frozen_stage1_vae_bank_provenance(
    vae_preflight, vae_checkpoint_path=VAE_CHECKPOINT, bank_dir=LOCAL_BANK_ROOT
)
vae_model = dict(vae_preflight.parsed_config["model"])
decoder = build_decoder("kl_vae", **_vae_kwargs(vae_model, "decoder"))
vae_state = load_checkpoint(VAE_CHECKPOINT, map_location="cpu")
decoder.load_state_dict(vae_state["decoder"], strict=True)
del vae_state
photometry_preflight = preflight_reviewed_photometry_namespace_artifact(
    PHOTOMETRY_ARTIFACT
)
photometry_provenance = verify_reviewed_photometry_bank_provenance(
    photometry_preflight, bank_dir=LOCAL_BANK_ROOT
)
artifact = photometry_provenance.preflight.artifact

translator.to(device).eval()
critic.to(device).eval()
decoder.to(device).eval().requires_grad_(False)
DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
fixed_validation = _fixed_positions(validation_index, 3)


def conditioning_producer():
    rows = []
    for position in fixed_validation:
        latent, support, domains, _ = validation_index.load_batch([position])
        source = domains[0]
        targets = [field for field in FIELD_STRENGTHS_T if field != source.field_strength_t]
        normalized = stats.normalize(latent.to(device), support.to(device))
        rows.append(
            diagnose_conditioning_plumbing(
                translator,
                normalized,
                [source],
                [Domain(targets[0], source.contrast)],
                [Domain(targets[-1], source.contrast)],
                time_values=torch.tensor([0.5], device=device, dtype=normalized.dtype),
            )
        )
        del latent, support, normalized
    return aggregate_conditioning_diagnostics(rows)


conditioning = _load_or_run(
    DIAGNOSTIC_ROOT / "conditioning_plumbing_v1.json", conditioning_producer
)


def identifiability_producer():
    train_features, train_domains = bank_identifiability_features(train_index, stats)
    validation_features, validation_domains = bank_identifiability_features(
        validation_index, stats
    )
    return diagnose_real_domain_identifiability(
        train_features,
        train_domains,
        validation_features,
        validation_domains,
        include_fifteen_way=True,
    )


identifiability = _load_or_run(
    DIAGNOSTIC_ROOT / "real_domain_identifiability_v1.json", identifiability_producer
)


def sweep_producer():
    rows = []
    target_reference_centroids = bank_domain_latent_centroids(train_index, stats)
    for position in fixed_validation:
        latent, support, domains, _ = validation_index.load_batch([position])
        source_domain = domains[0]
        normalized = stats.normalize(latent.to(device), support.to(device))
        source_support = support.to(device)

        def renderer(decoded: torch.Tensor, target: Domain) -> torch.Tensor:
            support_image = F.interpolate(
                source_support.float(), size=decoded.shape[2:], mode="nearest"
            ).bool()
            context = SourceCanonicalizedVolume(
                values=decoded[0],
                support_mask=support_image[0],
                source_domain=source_domain,
                artifact_sha256=artifact.artifact_sha256,
            )
            return artifact.render_target(context, target).unsqueeze(0)

        rows.append(
            same_source_five_target_sweep(
                translator,
                decoder,
                normalized,
                source_support,
                source_domain,
                stats,
                renderer=renderer,
                target_reference_centroids=target_reference_centroids,
                integration_steps=4,
                solver="heun",
            )
        )
        del latent, support, normalized, source_support
        torch.cuda.empty_cache()
    return aggregate_target_sweeps(rows)


target_sweep = _load_or_run(
    DIAGNOSTIC_ROOT / "same_source_five_target_sweep_v1.json", sweep_producer
)


def drift_producer():
    real_records = []
    for position in range(min(30, len(train_index))):
        latent, support, domains, _ = train_index.load_batch([position])
        real_records.append(
            LatentDiagnosticRecord(
                latent=latent,
                support=support,
                source_domain=domains[0],
                target_domain=domains[0],
                contrast=domains[0].contrast,
            )
        )
    generated_records = []
    with torch.inference_mode():
        for position in fixed_validation:
            latent, support, domains, _ = validation_index.load_batch([position])
            source_domain = domains[0]
            normalized = stats.normalize(latent.to(device), support.to(device))
            for field in FIELD_STRENGTHS_T:
                target = Domain(field, source_domain.contrast)
                generated = integrate_transport(
                    translator,
                    normalized,
                    [source_domain],
                    [target],
                    steps=4,
                    solver="heun",
                )
                generated_records.append(
                    LatentDiagnosticRecord(
                        latent=stats.denormalize(generated).cpu(),
                        support=support,
                        source_domain=source_domain,
                        target_domain=target,
                        contrast=source_domain.contrast,
                    )
                )
            del latent, support, normalized, generated
    return diagnose_off_manifold_latent_drift(real_records, generated_records, stats)


drift = _load_or_run(
    DIAGNOSTIC_ROOT / "off_manifold_latent_drift_v1.json", drift_producer
)


def gradient_producer():
    sampler = torch.Generator().manual_seed(20_260_901)
    pools = _DomainPools.from_index(train_index)
    pools.require_all_domains()
    rows = []
    for _ in range(2):
        batch = _sample_training_batch(train_index, pools, stats, cfg, device, sampler)
        rows.append(
            diagnose_per_term_gradients(
                cfg,
                translator,
                critic,
                decoder,
                batch,
                stats,
                seed=20_260_901 + len(rows),
            )
        )
        del batch
        gc.collect()
        torch.cuda.empty_cache()
    return aggregate_gradient_diagnostics(rows)


gradients = _load_or_run(
    DIAGNOSTIC_ROOT / "per_term_gradient_diagnostics_v1.json", gradient_producer
)


def micro_overfit_producer():
    tiny = build_translator(
        "flow_matching_latent",
        latent_channels=1,
        hidden_channels=(8,),
        bottleneck_channels=16,
        cond_dim=16,
        time_embed_dim=16,
        spatial_dims=2,
        use_norm=False,
        pad_to_multiple=True,
        zero_init_output=False,
    )
    return run_synthetic_micro_overfit(
        tiny, mode="velocity", steps=200, learning_rate=0.02, device=device
    )


micro_overfit = _load_or_run(
    DIAGNOSTIC_ROOT / "synthetic_micro_overfit_v1.json", micro_overfit_producer
)

scorecard = build_rescue_scorecard(
    {
        "conditioning_plumbing": conditioning,
        "real_domain_identifiability": identifiability,
        "target_sweep": target_sweep,
        "off_manifold_drift": drift,
        "gradient_health": gradients,
        "micro_overfit": micro_overfit,
    }
)
scorecard_path = DIAGNOSTIC_ROOT / "stage2_rescue_scorecard_v1.json"
write_diagnostic_json(scorecard_path, scorecard, resume=scorecard_path.exists())
markdown_path = DIAGNOSTIC_ROOT / "stage2_rescue_scorecard_v1.md"
markdown = render_rescue_scorecard_markdown(scorecard)
if markdown_path.exists():
    if markdown_path.read_text(encoding="utf-8") != markdown:
        raise FileExistsError("Existing Markdown scorecard differs; refusing to clobber.")
else:
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
print(json.dumps(scorecard, indent=2, sort_keys=True), flush=True)
print(
    json.dumps(
        {
            "stage": "complete",
            "diagnostic_root": str(DIAGNOSTIC_ROOT),
            "scientific_training_invoked": False,
            "P0006_accessed": False,
            "P0009_accessed": False,
            "architecture_verdict": "human_decision_required",
        },
        sort_keys=True,
    ),
    flush=True,
)
