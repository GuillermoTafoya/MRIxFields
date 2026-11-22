"""Authenticated inputs for the one-direction P:0006 decoder-context parity replay."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

import stage2_zero_transport_operator as op
from fieldbridge.data.domains import Domain
from fieldbridge.data.photometry_factorization import canonical_tensor_sha256, sha256_file
from fieldbridge.evaluation import stage2_step200_inference_audit as audit
from fieldbridge.evaluation.low_memory_vae import enable_low_memory
from fieldbridge.evaluation.stage2_gate01 import _load_verified_array, load_gate01_input_manifest
from fieldbridge.evaluation.stage2_gate01_calibration import PosthocTargetCalibrator
from fieldbridge.evaluation.stage2_gate01_protocol import Gate01ProtocolLock
from fieldbridge.evaluation.stage2_step200_inference_audit import (
    _apply_shape_bound_calibrator,
    adapt_full_volume_layout,
    build_frozen_step200_inference_plan,
    preflight_frozen_p0006_scientific_role,
)
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    P0006_IDENTITY_SHA256,
    _load_completed_stage2_config,
    _verified_protocol_file,
)
from fieldbridge.evaluation.zero_transport_diagnostic import _ieee_encoder_convolutions
from fieldbridge.models.factory import build_translator
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.stage2_unified import UNIFIED_RESUME_CONTRACT


ARRAY_KEYS = {
    "source_image": "source_image_sha256", "source_support_mask": "source_support_sha256",
    "raw_identity": "raw_identity_sha256", "target": "target_sha256",
}
SOURCE_KEYS = ("source_image", "source_support_mask", "raw_identity")
SENTINEL_CASE_ID = "gate01-f526a76463b585e3c13d0b16"


@dataclass(frozen=True)
class Direction:
    case_id: str
    source_domain: Domain
    target_domain: Domain
    wrong_domain: Domain
    seed: int
    ordinal: int
    array_sha256: dict
    calibrated_identity_sha256: str


@dataclass
class Cohort:
    provenance: dict
    calibrator: object
    manifest: object
    rows: dict


@dataclass
class SourceArrays:
    source_image: torch.Tensor
    support_mask: torch.Tensor
    raw_identity: torch.Tensor
    source_domain: Domain


def as_plan_row(descriptor):
    return {
        "case_id": descriptor.case_id,
        "source_domain": descriptor.source_domain.label,
        "target_domain": descriptor.target_domain.label,
        "wrong_domain": descriptor.wrong_domain.label,
        "seed": descriptor.seed,
        "ordinal": descriptor.ordinal,
        "array_sha256": dict(descriptor.array_sha256),
        "calibrated_identity_sha256": descriptor.calibrated_identity_sha256,
    }


def _pilot_descriptor(manifest, receipts, seeds):
    """Read metadata for one locked case. Open no arrays from sibling directions."""
    rows = [row for row in manifest.case_specs if row["case_id"] == SENTINEL_CASE_ID]
    matches = [(i, receipt) for i, receipt in enumerate(receipts, 1)
               if receipt["case_identity"] == SENTINEL_CASE_ID]
    if len(rows) != 1 or len(matches) != 1:
        raise ValueError("The archived pilot case must have exactly one row and receipt.")
    row = rows[0]
    ordinal, receipt = matches[0]
    if ordinal != 12 or len(seeds) < ordinal:
        raise ValueError("The archived pilot seed schedule is missing or reordered.")
    seed = seeds[ordinal - 1]
    source = Domain.from_dict(row["source_domain"])
    target = Domain.from_dict(row["target_domain"])
    if (
        row["traveller_identity_sha256"] != P0006_IDENTITY_SHA256
        or source != Domain(3, "T1w") or target != Domain(7, "T1w")
        or seed["index"] != ordinal or seed["seed"] != 1091568062
        or seed["source_image_sha256"] != receipt["source_image_sha256"]
        or any(row[key] != receipt[key] or row[key] != seed[key]
               for key in ("source_domain", "target_domain"))
    ):
        raise ValueError("Pilot participant, direction, or direction-specific seed changed.")
    hashes = {name: receipt[digest] for name, digest in ARRAY_KEYS.items()}
    if any(row[name]["sha256"] != digest for name, digest in hashes.items()):
        raise ValueError("Archived source, mask, reconstruction or target hash changed.")
    spec = Direction(SENTINEL_CASE_ID, source, target, Domain(0.1, "T1w"), seed["seed"],
                     ordinal, hashes, receipt["calibrated_identity_sha256"])
    return spec, row


def read_p0006_metadata(drive_root):
    """Authenticate the frozen Gate 0.1 study, roles and calibrator before opening arrays."""
    root = Path(drive_root)
    recovery = (root / "UnifiedStage2_1ca2b4a_01/stage2_unified_v7/bank_8081ce89a0ea"
                / "recovery_training_82633d66e5ea_operator_d3949c3591df")
    role = preflight_frozen_p0006_scientific_role(
        recovery / "stage2_gate01_p0006_development_validation_evaluation_only_protocol_v4.json",
        recovery / "stage2_long_run_evaluation_readiness_v4.json",
    )
    protocol = role.protocol
    plan = build_frozen_step200_inference_plan(protocol).payload
    archive = (root / "Gate01Private_8012a3f").resolve()
    if Path(protocol["archive_identity"]["logical_root"]).resolve() != archive:
        raise ValueError("Frozen Gate 0.1 archive differs from the mounted archive.")
    lock = Gate01ProtocolLock.load(_verified_protocol_file(protocol, "protocol_lock"))
    calibrator = PosthocTargetCalibrator.load(
        _verified_protocol_file(protocol, "calibrator"), expected_split_fingerprint=lock.split_fingerprint,
        expected_template_sha256=lock.calibrator_template_sha256, expected_artifact_sha256=lock.calibrator_artifact_sha256,
    )
    manifest, info = load_gate01_input_manifest(
        _verified_protocol_file(protocol, "gate01_manifest"), protocol_lock=lock, calibrator=calibrator)
    if manifest.root != archive or info["evidence_scope"]["traveller_identity_sha256"] != P0006_IDENTITY_SHA256:
        raise ValueError("Frozen Gate 0.1 archive or participant identity mismatch.")
    provenance = {
        "participant": "P:0006", "role": role.readiness["evaluation_role"],
        "protocol_file_sha256": role.protocol_file_sha256, "protocol_sha256": role.protocol_sha256,
        "readiness_file_sha256": role.readiness_file_sha256, "readiness_sha256": role.readiness_sha256,
        "gate_manifest_sha256": info["sha256"], "calibrator_artifact_sha256": calibrator.artifact_sha256,
        "source_support_threshold": manifest.support_threshold, "number_of_acquisitions": 15, "number_of_cases": 60,
    }
    rows = {row["case_id"]: row for row in manifest.case_specs}
    return Cohort(provenance, calibrator, manifest, rows), protocol["case_receipts"], plan["case_seeds"]


def load_pilot_inputs(drive_root):
    """Authenticate the Gate 0.1 metadata and load only the original paired pilot."""
    cohort, receipts, seeds = read_p0006_metadata(drive_root)
    descriptor, row = _pilot_descriptor(cohort.manifest, receipts, seeds)
    cohort.rows = {descriptor.case_id: row}
    source = load_source(cohort, descriptor)
    target = load_target(cohort, descriptor, tuple(source.source_image.shape))
    baseline = calibrated_source_reconstruction(cohort, descriptor, source)
    return cohort, descriptor, source, target, baseline


def _array(cohort, descriptor, key):
    array, digest = _load_verified_array(cohort.manifest.root, cohort.rows[descriptor.case_id][key],
                                         key, mask=(key == "source_support_mask"))
    if digest != descriptor.array_sha256[key]:
        raise ValueError(f"Loaded array identity differs for {descriptor.case_id}/{key}.")
    return array


def load_source(cohort, descriptor):
    image, mask, stage1 = (_array(cohort, descriptor, key) for key in SOURCE_KEYS)
    for name, array in (("source_support_mask", mask), ("raw_identity", stage1)):
        adapt_full_volume_layout(array, role=name, expected_shape=tuple(image.shape))
    if not torch.equal(image.abs() > cohort.manifest.support_threshold, mask):
        raise ValueError("Gate 0.1 source support differs from the frozen source-derived threshold.")
    return SourceArrays(image, mask, stage1, descriptor.source_domain)


def load_target(cohort, descriptor, expected_shape):
    target = _array(cohort, descriptor, "target")
    adapt_full_volume_layout(target, role="target", expected_shape=tuple(expected_shape))
    return target


def calibrated_source_reconstruction(cohort, descriptor, source):
    if source.source_domain != descriptor.source_domain:
        raise ValueError("Shared source was reused for a different acquisition domain.")
    calibrated = _apply_shape_bound_calibrator(
        cohort.calibrator, source.raw_identity, descriptor.target_domain, source.support_mask,
        expected_shape=tuple(source.source_image.shape), role="calibrated Stage 1 source reconstruction")
    if canonical_tensor_sha256(calibrated) != descriptor.calibrated_identity_sha256:
        raise ValueError("Calibrated Stage 1 source reconstruction differs from the frozen receipt.")
    return calibrated


# The historical audit authenticates this complete container even though only
# the generator's weights are used. Optimizer and critic objects are never built.
STEP200_KEYS = frozenset("""
contract_version run_fingerprint validation_plan_sha256 selection_rule_sha256
training_cursor translator critic generator_optimizer critic_optimizer
generator_scheduler critic_scheduler scaler sampler_rng torch_rng cuda_rng
python_rng numpy_rng history_prefix_bytes history_prefix_sha256 validation_selection
pilot_report in_progress_pilot_rows _meta
""".split())


TRANSLATOR_STATE_SHA256 = "dec52f8958c2dbc103c78d62eb6a9cd39b8bf638f2c222ec063b9d42379853ad"
RESOLVED_CONFIG_SHA256 = "10f967832e1f632dfe23a8c2b007f2affd8e7bafeeda6a5d3605ff37ce8f2463"


def load_translator(checkpoint_path, resolved_config_path, *, device,
                    expected_config_sha256=RESOLVED_CONFIG_SHA256):
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(resolved_config_path)
    if config_path.name != "resolved_config.json" or config_path.is_symlink():
        raise ValueError("Expected the frozen resolved_config.json artifact.")
    raw_config = config_path.read_bytes()
    digest = hashlib.sha256(raw_config).hexdigest()
    if digest != expected_config_sha256:
        raise ValueError("Frozen step-200 resolved configuration SHA-256 mismatch.")
    config = json.loads(raw_config)
    expected = dict(device="auto", batch_size=1, precision="bf16",
                    integration_steps=4, integration_solver="heun")
    if not isinstance(config.get("training"), Mapping) or any(
        config["training"].get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Frozen training or integration settings changed.")
    replay = _load_completed_stage2_config(config_path.parent)
    if sha256_file(config_path) != digest:
        raise ValueError("Resolved configuration changed during validation.")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("Expected a regular frozen checkpoint file.")
    if sha256_file(checkpoint_path) != audit.CHECKPOINT_SHA256:
        raise ValueError("Step-200 checkpoint file SHA-256 mismatch.")
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    if set(state) != STEP200_KEYS:
        raise ValueError("Step-200 checkpoint top-level inventory mismatch.")
    meta = state["_meta"]
    if (state["contract_version"] != UNIFIED_RESUME_CONTRACT
            or state["training_cursor"] != 200
            or state["run_fingerprint"] != audit.RUN_FINGERPRINT
            or not isinstance(meta, Mapping)
            or meta.get("git_commit") != audit.TRAINING_EVIDENCE_COMMIT
            or meta.get("config") != replay.effective_config):
        raise ValueError("Frozen training cursor, fingerprint, commit, or configuration mismatch.")
    if not isinstance(config.get("model"), Mapping) or not isinstance(state["translator"], Mapping):
        raise ValueError("Malformed frozen translator specification or weights.")
    model_spec = dict(config["model"])
    name = model_spec.pop("name", "flow_matching_latent")
    translator = build_translator(name, **model_spec)
    translator.load_state_dict(state["translator"], strict=True)
    state_digest = audit._module_state_sha256(translator)
    if state_digest != TRANSLATOR_STATE_SHA256:
        raise ValueError("Loaded translator differs from the August inference state fingerprint.")
    history = {
        "translator_state_sha256": state_digest,
        "checkpoint_sha256": audit.CHECKPOINT_SHA256,
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "resolved_config_sha256": digest,
        "declared_training_sha256": replay.declared_normalized_sha256,
        "effective_training_sha256": replay.effective_normalized_sha256,
        "training_commit": audit.TRAINING_EVIDENCE_COMMIT,
        "run_fingerprint": state["run_fingerprint"],
        "training_cursor": state["training_cursor"],
        "model": config["model"],
        "integration_steps": 4,
        "integration_solver": "heun",
        "parameter_count": sum(p.numel() for p in translator.parameters()),
    }
    del state, meta
    gc.collect()
    return translator.eval().requires_grad_(False).to(device), history


def _compare(actual, reference, *, rmse_abs, rmse_rel, max_abs, max_rel):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError(f"Output mismatch: {actual.shape}/{actual.dtype} vs "
                         f"{reference.shape}/{reference.dtype}")
    actual = actual.detach().to(device="cpu", dtype=torch.float64)
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    diff = actual - reference
    error = diff.square().mean().sqrt().item()
    maximum = diff.abs().max().item()
    error_limit = rmse_abs + rmse_rel * reference.square().mean().sqrt().item()
    max_limit = max_abs + max_rel * reference.abs().max().item()
    if not all(math.isfinite(v) for v in (error, maximum, error_limit, max_limit)):
        raise ValueError("Non-finite value in the elementwise numerical check")
    return {
        "rmse": error, "max_abs": maximum, "rmse_limit": error_limit,
        "max_abs_limit": max_limit, "pass": error <= error_limit and maximum <= max_limit,
    }


@torch.inference_mode()
def qualify_local_operator(encoder, decoder, domain, device):
    """Compare native and offloaded inference with identical weights and inputs."""
    device = torch.device(device)
    seed = 39413
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand((1, 1, 36, 40, 44), generator=generator)
    x[..., :3, :, :] = 0
    x[..., 15:21, 10:28, 12:34] *= 0.1
    x = x.to(device)

    def encode():
        with _ieee_encoder_convolutions(device), torch.autocast(device.type, enabled=False):
            return encoder.encode_dist(x, [domain])

    def decode(z):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return decoder.decode(z, [domain])

    native_mean, native_logvar = encode()
    native_decode = decode(native_mean)
    slab_depth = 8
    enable_low_memory(encoder, decoder, slab_depth=slab_depth)
    low_mean, low_logvar = encode()
    low_decode = decode(native_mean)
    # The second decode includes propagation of the encoder differences.
    low_round_trip = decode(low_mean)

    fp32 = dict(rmse_abs=1e-4, rmse_rel=1e-4, max_abs=1e-3, max_rel=1e-3)
    bf16 = dict(rmse_abs=0.003, rmse_rel=0.005, max_abs=0.04, max_rel=0.02)
    checks = {
        "encoder_mean": _compare(low_mean, native_mean, **fp32),
        "encoder_logvar": _compare(low_logvar, native_logvar, **fp32),
        "decoder_fixed_latent": _compare(low_decode, native_decode, **bf16),
        "round_trip": _compare(low_round_trip, native_decode, **bf16),
    }
    if not all(record["pass"] for record in checks.values()):
        raise ValueError("Same-device synthetic check failed; " + json.dumps(checks, sort_keys=True))
    return {"status": "passed", "source": "synthetic", "seed": seed, "input_shape": list(x.shape),
            "slab_depth": slab_depth, "checks": checks}


def _stage(staged):
    pilot = (op.OUTPUT_ROOT / "stage2_unified_v7" / "bank_8081ce89a0ea" / "implementation_82633d66e5ea" /
             "unified_full_objective_pilot_200")
    resolved = pilot / "resolved_config.json"
    checkpoint = pilot / "scientific_attempts" / "attempt-0001" / "checkpoints" / "stage2_unified_full_step000000200.pt"
    for source, expected in ((resolved, RESOLVED_CONFIG_SHA256),
            (checkpoint,"09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021")):
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Required frozen step-200 input missing or non-regular: {source.name}")
        staged_path = staged / source.name
        shutil.copyfile(source, staged_path)
        if op.sha256_file(staged_path) != expected:
            raise ValueError(f"Staged step-200 identity mismatch: {source.name}")
    op.read_bank_metadata(op.BANK_ARCHIVE, staged)
    for source in (op.CONFIG, op.PHOTOMETRY, op.CHECKPOINT):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Required frozen input missing or non-regular: {source.name}")
        shutil.copyfile(source, staged / source.name)
    print("Shared bank metadata and model inputs are ready.", flush=True)


def load_qualified_codec(staged, bank_dir, device, mark):
    """Load the frozen Run C codec, qualify its low-memory path and authenticate bank statistics."""
    mark("Authenticate frozen bank, Run C encoder/decoder and training photometry")
    config = op.preflight_frozen_stage1_run_c_config(staged / op.CONFIG.name)
    photo = op.preflight_reviewed_photometry_namespace_artifact(staged / op.PHOTOMETRY.name)
    bank = op.PhotometryFactoredLatentBankIndex(bank_dir, "validation", expected_artifact_sha256=op.BANK_SHA256)
    manifest = dict(bank.manifest)
    manifest.pop("artifact_sha256")
    if op.sha256_json(manifest) != op.BANK_SHA256:
        raise ValueError("Frozen bank manifest digest mismatch.")
    stats_info = manifest["latent_statistics"]
    if op.sha256_file(bank_dir / op.FACTORED_LATENT_STATS_FILE) != stats_info["file_sha256"]:
        raise ValueError("Latent statistics digest mismatch.")
    stats = op.FactoredLatentStats.from_bank(bank_dir)
    if stats.artifact_sha256 != stats_info["artifact_sha256"]:
        raise ValueError("Latent statistics canonical digest mismatch.")
    vae = op.verify_frozen_stage1_vae_bank_provenance(config, vae_checkpoint_path=staged / op.CHECKPOINT.name, bank_dir=bank_dir)
    photo = op.verify_reviewed_photometry_bank_provenance(photo, bank_dir=bank_dir)
    model = config.parsed_config["model"]
    encoder = op.build_encoder("kl_vae", **op._kl_vae_kwargs(model, "encoder"))
    decoder = op.build_decoder("kl_vae", **op._kl_vae_kwargs(model, "decoder"))
    op._require_linear_decoder_output_activation(decoder)
    state = op.load_checkpoint(staged / op.CHECKPOINT.name, map_location="cpu")
    encoder.load_state_dict(state["encoder"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    del state
    encoder.to(device).requires_grad_(False).eval()
    decoder.to(device).requires_grad_(False).eval()
    mark("Qualify the full-volume low-memory operator")
    parity = qualify_local_operator(encoder, decoder, op.Domain(3.0, "T1w"), device)
    support_rule = manifest["operational_support_rule"]["rule_sha256"]
    meta = dict(latent_statistics=stats_info, vae_sha256=vae.checkpoint_file_sha256,
                vae_config=config.parsed_config, frozen_vae=vae.sanitized_provenance(),
                split=manifest["source_split"], bank_artifact_sha256=op.BANK_SHA256,
                bank_archive_sha256=op.BANK_ARCHIVE_SHA256,
                canonical_photometry=photo.sanitized_provenance(),
                support_rule_sha256=support_rule, same_device_parity=parity)
    return encoder, decoder, photo.preflight.artifact, stats, support_rule, meta


def stage_and_load_models(staged, device, mark):
    """Stage the immutable pilot inputs and qualify the archived host-staged full-volume VAE."""
    _stage(staged)
    encoder, decoder, artifact, stats, rule, meta = load_qualified_codec(staged, staged, device, mark)
    translator, translator_info = load_translator(staged / "stage2_unified_full_step000000200.pt",
                                                  staged / "resolved_config.json", device=device)
    return encoder, decoder, translator, artifact, stats, rule, {**meta, "translator_provenance": translator_info}
