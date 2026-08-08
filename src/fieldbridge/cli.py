"""Command-line interface for FieldBridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from fieldbridge.config import dump_yaml_config, load_yaml_config
from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.vae_splits import (
    build_vae_splits,
    load_vae_splits,
    save_vae_splits,
    summarize_vae_splits,
    vae_splits_fingerprint,
    vae_splits_recovery_fingerprint_v3,
)
from fieldbridge.data.datasets import (
    ALL_DOMAINS,
    ImageTransform,
    ManifestVolumeDataset,
    StreamingPatchDataset,
    collate_raw_batches,
)
from fieldbridge.data.pseudo_pairs import (
    PseudoPairSliceDataset,
    collate_pseudo_pair_slices,
    make_field_balanced_sampler,
)
from fieldbridge.data.patch_bank import (
    PatchBankDataset,
    build_patch_bank,
    patch_bank_size,
)
from fieldbridge.data.manifests import audit_manifest, load_manifest
from fieldbridge.data.preprocessing import SlicePreprocessingSpec, from_model_range, selected_slice_indices
from fieldbridge.data.sampling import domain_oversampling_weights, field_balanced_weights
from fieldbridge.data.sources import nifti_image_loader
from fieldbridge.data.transforms import StratifiedCropConfig, assert_official_unit_range
from fieldbridge.data.volume_splits import (
    audit_volume_splits,
    build_volume_splits,
    load_volume_splits,
    save_volume_splits,
    summarize_volume_splits,
    validate_pseudo_pair_manifest_records,
    volume_splits_fingerprint,
)
from fieldbridge.models.diffusion.denoising_unet import DenoisingUNet
from fieldbridge.models.factory import build_decoder, build_encoder, build_translator
from fieldbridge.official.data_manifest import (
    audit_mrixfields_manifest,
    read_manifest_jsonl,
    scan_mrixfields_data_root,
    write_manifest_jsonl,
)
from fieldbridge.official.mrixfields2026 import spec_as_dict
from fieldbridge.official.submissions import (
    build_submission_zip,
    validate_submission_dir,
)
from fieldbridge.evaluation.pseudo_pairs import PseudoPairEvalConfig, evaluate_pseudo_pairs
from fieldbridge.evaluation.mrixfields2026_official import (
    evaluate_official_task3_directory,
)
from fieldbridge.evaluation.intensity_baselines import (
    fit_image_intensity_baselines,
    reference_volume_records,
)
from fieldbridge.evaluation.stage2_gate01 import (
    evaluate_gate01,
    fixed_montage_specifications,
    gate01_code_provenance,
    gate01_selection_fingerprint,
    load_gate01_input_manifest,
    write_gate01_outputs,
)
from fieldbridge.evaluation.stage2_gate01_montage import (
    Gate01MontageCollector,
    render_gate01_montages,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    PosthocTargetCalibrator,
    RESPLIT_FINGERPRINT,
    TrainingTemplateVolume,
    fit_posthoc_target_calibrator,
    gate01_calibrator_code_provenance,
)
from fieldbridge.evaluation.stage2_gate01_protocol import Gate01ProtocolLock
from fieldbridge.evaluation.stage2_gate01_builder import (
    assert_gate01_external_path,
    build_gate01_private_manifest,
    write_gate01_protocol_lock,
)
from fieldbridge.evaluation.stage2_gate01_producer import (
    prepare_gate01_prospective_selection,
    produce_gate01_private_artifacts,
    write_gate01_producer_spec,
)
from fieldbridge.evaluation.board_score import (
    aggregate_task3_board,
    board_units_from_payload,
    evaluate_task3_board_directory,
    format_board_table,
    format_rank_sum_table,
    rank_submissions,
    submissions_from_payload,
)
from fieldbridge.evaluation.stage1_full_volume_audit import (
    AuditRuntime,
    audit_stage1_checkpoint,
    checkpoint_public_metadata,
    freeze_stage1_audit_selection,
    load_and_validate_stage1_audit_selection,
    prepare_audit_root,
    resolve_audit_commit,
    run_synthetic_stage1_audit_smoke,
    sha256_file,
    write_audit_comparison,
)
from fieldbridge.evaluation.stage1_report import run_stage1_eval
from fieldbridge.data.latent_bank import (
    LatentBankConfig,
    build_latent_bank,
    decode_latent,
    encode_latent,
    load_volume,
)
from fieldbridge.training.checkpoints import load_checkpoint, resolve_git_commit
from fieldbridge.training.pseudo_pair_epochs import (
    PSEUDO_PAIR_PIPELINE_VERSION,
    PseudoPairEpochConfig,
    train_pseudo_pair_epochs,
)
from fieldbridge.training.smoke_train import SmokeTrainConfig, run_smoke_train
from fieldbridge.training.stage1_vae import (
    Stage1VAEConfig,
    preflight_stage1_resume,
    run_stage1_vae_train,
)
from fieldbridge.training.stage2_diffuser import Stage2DiffuserConfig, run_stage2_diffuser_train
from fieldbridge.training.stage2_transport import Stage2TransportConfig, run_stage2_transport_train
from fieldbridge.data.latent_bank_dataset import LatentBankIndex, LatentStats
from fieldbridge.data.resplit import resplit_file
from fieldbridge.data.photometry_factorization import (
    FORBIDDEN_TRAVELLER_IDS,
    PHOTOMETRY_CLAMPING_RULE,
    PHOTOMETRY_DUPLICATE_KNOT_RULE,
    PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
    PHOTOMETRY_FACTORIZATION_CONFIG_VERSION,
    PHOTOMETRY_FACTORIZATION_SEMANTICS,
    PHOTOMETRY_INTERPOLATION_RULE,
    PHOTOMETRY_SUPPORT_POLICY,
    FrozenPhotometryArtifact,
    PhotometryFitVolume,
    assert_variant_a_external_path,
    capture_photometry_code_provenance,
    fit_frozen_photometry,
    reject_target_or_prediction_derived_fields,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    BaselineEvaluationCase,
    ContinuityReference,
    OfficialQualificationMetrics,
    QualificationVolume,
    VariantAQualificationThresholds,
    evaluate_factorized_identity_case,
    qualify_variant_a,
    write_variant_a_result,
)
from fieldbridge.evaluation.stage2_transport_eval import (
    DecodeSpec,
    TransportSamplerConfig,
    evaluate_transport_travellers,
    write_transport_eval,
)
from fieldbridge.training.train_loop import TrainLoopConfig, run_train_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fieldbridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-train", help="Run synthetic CPU smoke training.")
    smoke.add_argument("--config", type=Path, default=Path("configs/experiment/smoke.yaml"))
    smoke.add_argument("--steps", type=int, default=None)
    smoke.add_argument("--batch-size", type=int, default=None)
    smoke.add_argument("--seed", type=int, default=None)
    smoke.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    train = subparsers.add_parser("train", help="Run the configurable Etapa 2 translator training loop.")
    train.add_argument("--config", type=Path, default=Path("configs/experiment/smoke.yaml"))
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest of real NIfTI volumes (requires the 'nifti' extra) to use "
        "instead of the synthetic loader. Never commit a real manifest to the repo.",
    )
    train.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    train_pseudo = subparsers.add_parser(
        "train-pseudo-pairs",
        help="Train the epoch-based T2-FLAIR pseudo-pair conditional U-Net baseline.",
    )
    train_pseudo.add_argument("--config", type=Path, default=Path("configs/experiment/pseudo_pair_t2flair_pilot.yaml"))
    train_pseudo.add_argument("--manifest", type=Path, default=None)
    train_pseudo.add_argument("--sequence", default=None)
    train_pseudo.add_argument("--source-field", type=float, default=None)
    train_pseudo.add_argument("--target-fields", nargs="+", type=float, default=None)
    train_pseudo.add_argument("--train-volumes-per-field", type=int, default=None)
    train_pseudo.add_argument("--val-volumes-per-field", type=int, default=None)
    train_pseudo.add_argument("--test-volumes-per-field", type=int, default=None)
    train_pseudo.add_argument("--slices-per-volume", type=int, default=None)
    train_pseudo.add_argument("--slice-start", type=int, default=None)
    train_pseudo.add_argument("--slice-end", type=int, default=None)
    train_pseudo.add_argument("--slice-axis", choices=("x", "y", "z"), default=None)
    train_pseudo.add_argument("--output-height", type=int, default=None)
    train_pseudo.add_argument("--output-width", type=int, default=None)
    train_pseudo.add_argument("--epochs", type=int, default=None)
    train_pseudo.add_argument("--batch-size", type=int, default=None)
    train_pseudo.add_argument("--workers", type=int, default=None)
    train_pseudo.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=None)
    train_pseudo.add_argument("--checkpoint-dir", type=Path, default=None)
    train_pseudo.add_argument("--resume-checkpoint", type=Path, default=None)
    train_pseudo.add_argument("--seed", type=int, default=None)
    train_pseudo.add_argument("--max-pilot-records", type=int, default=None)
    train_pseudo.add_argument("--split-json", type=Path, default=None)
    train_pseudo.add_argument(
        "--preflight",
        "--dry-run",
        dest="preflight",
        action="store_true",
        help="Validate manifest/splits/datasets and print counts without optimization.",
    )
    train_pseudo.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    eval_pseudo = subparsers.add_parser(
        "eval-pseudo-pairs",
        help="Evaluate degraded and predicted pseudo-pair baselines from a checkpoint.",
    )
    eval_pseudo.add_argument("--config", type=Path, default=Path("configs/experiment/pseudo_pair_t2flair_pilot.yaml"))
    eval_pseudo.add_argument("--manifest", type=Path, default=None)
    eval_pseudo.add_argument("--checkpoint", type=Path, required=True)
    eval_pseudo.add_argument("--split-json", type=Path, default=None)
    eval_pseudo.add_argument("--split", choices=("validation", "test"), default="test")
    eval_pseudo.add_argument("--batch-size", type=int, default=None)
    eval_pseudo.add_argument("--workers", type=int, default=None)
    eval_pseudo.add_argument("--slice-axis", choices=("x", "y", "z"), default=None)
    eval_pseudo.add_argument("--seed", type=int, default=None)
    eval_pseudo.add_argument("--max-pilot-records", type=int, default=None)
    eval_pseudo.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    train_stage1_vae = subparsers.add_parser(
        "train-stage1-vae", help="Run Etapa 1 VAE-only training (KLVAEEncoder/Decoder, SSIM+nRMSE+LPIPS+KL)."
    )
    train_stage1_vae.add_argument("--config", type=Path, default=Path("configs/experiment/stage1_vae.yaml"))
    train_stage1_vae.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest of real NIfTI volumes (requires the 'nifti' extra). Required unless "
        "--patch-bank or --split-json is given. No synthetic fallback for this stage — never "
        "commit a real manifest to the repo.",
    )
    train_stage1_vae.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="VAE split file (see build-vae-splits). When given, trains on the 'train' split "
        "and validates per-epoch on the 'validation' split (history.jsonl + best checkpoint). "
        "Takes precedence over --manifest.",
    )
    train_stage1_vae.add_argument("--steps", type=int, default=None)
    train_stage1_vae.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of full passes over the manifest. Endpoint precedence is "
        "--epochs > config training.epochs > explicit --steps/config steps.",
    )
    train_stage1_vae.add_argument("--batch-size", type=int, default=None)
    train_stage1_vae.add_argument(
        "--patches-per-volume",
        type=int,
        default=None,
        help="Random patches drawn per volume before it is dropped (overrides "
        "data.patches_per_volume). Higher = fewer disk reads per training step.",
    )
    train_stage1_vae.add_argument("--seed", type=int, default=None)
    train_stage1_vae.add_argument(
        "--patch-bank",
        type=Path,
        default=None,
        help="Train from a prebuilt patch bank (see build-patch-bank) loaded into RAM "
        "instead of streaming volumes from disk. Zero per-epoch I/O; --manifest is ignored.",
    )
    train_stage1_vae.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    build_bank = subparsers.add_parser(
        "build-patch-bank",
        help="Preprocess a manifest into a reusable float16 patch bank (read each volume "
        "from disk once). Resumable and read-error tolerant.",
    )
    build_bank.add_argument("--config", type=Path, default=Path("configs/experiment/stage1_vae.yaml"))
    build_bank.add_argument("--manifest", type=Path, required=True, help="Manifest of real NIfTI volumes.")
    build_bank.add_argument("--out", type=Path, required=True, help="Output bank directory (created/resumed).")
    build_bank.add_argument(
        "--patches-per-volume",
        type=int,
        default=None,
        help="Patches to extract per volume (default: data.patches_per_volume from config). "
        "This fixes the bank size: num_volumes * ppv * patch_bytes.",
    )
    build_bank.add_argument("--seed", type=int, default=None)

    build_splits = subparsers.add_parser(
        "build-vae-splits",
        help="Build a subject-level, domain-stratified train/validation/test split for the "
        "Etapa 1 VAE from a manifest, and write it (with a leakage audit + fingerprint) to JSON.",
    )
    build_splits.add_argument("--manifest", type=Path, required=True, help="Manifest of real NIfTI volumes.")
    build_splits.add_argument("--out", type=Path, required=True, help="Output split JSON (contains real paths — do not commit).")
    build_splits.add_argument("--train-frac", type=float, default=0.8)
    build_splits.add_argument("--val-frac", type=float, default=0.1)
    build_splits.add_argument("--test-frac", type=float, default=0.1)
    build_splits.add_argument("--seed", type=int, default=13)
    build_splits.add_argument("--json", action="store_true", help="Emit the split summary as JSON.")

    eval_stage1_vae = subparsers.add_parser(
        "eval-stage1-vae",
        help="Deterministic reconstruction eval + diagnostic plots for a stage-1 VAE checkpoint.",
        description="Deterministic reconstruction eval + diagnostic plots for a stage-1 VAE checkpoint. "
        'Real Stage-1 evaluation requires: pip install -e ".[nifti,evaluation]"',
    )
    eval_stage1_vae.add_argument("--checkpoint", type=Path, required=True, help="Trained VAE checkpoint (.pt).")
    eval_stage1_vae.add_argument("--config", type=Path, default=Path("configs/experiment/stage1_vae.yaml"))
    eval_stage1_vae.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest of real NIfTI volumes to reconstruct. Required unless --split-json is "
        'given. Install requirements with: pip install -e ".[nifti,evaluation]"',
    )
    eval_stage1_vae.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="VAE split file (see build-vae-splits); evaluate the --split subset. Use "
        "--split test for the final held-out report. Takes precedence over --manifest.",
    )
    eval_stage1_vae.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Which subset of --split-json to evaluate (default: test).",
    )
    eval_stage1_vae.add_argument("--out", type=Path, required=True, help="Output directory for metrics + plots.")
    eval_stage1_vae.add_argument("--num-samples", type=int, default=4)
    eval_stage1_vae.add_argument(
        "--per-field-contrast",
        "--per-domain",
        dest="per_field_contrast",
        action="store_true",
        help="Reconstruct one volume per distinct field-strength and contrast pair instead "
        "of the first N in manifest order. --per-domain remains a compatibility alias.",
    )
    eval_stage1_vae.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding-window overlap fraction in [0, 1) (default 0.5). Overlap + Hann "
        "blending removes the panel seams from non-overlapping tiles.",
    )
    eval_stage1_vae.add_argument(
        "--per-domain-samples",
        type=int,
        default=1,
        help="Volumes kept per distinct field/contrast pair when selecting by domain "
        "(default 1). Implies --per-field-contrast selection.",
    )
    eval_stage1_vae.add_argument(
        "--oversample-field",
        type=float,
        default=None,
        help="Field strength (T) to over-represent, e.g. 0.1 for ultra-low-field. Its "
        "per-pair cap becomes per-domain-samples * oversample-factor.",
    )
    eval_stage1_vae.add_argument(
        "--oversample-factor",
        type=int,
        default=3,
        help="Multiplier applied to --oversample-field's per-pair cap (default 3).",
    )
    eval_stage1_vae.add_argument(
        "--eval-seed",
        type=int,
        default=13,
        help="Seed recorded for provenance (selection is loader-order-deterministic).",
    )
    eval_stage1_vae.add_argument(
        "--no-latent-stats",
        dest="latent_stats",
        action="store_false",
        help="Skip the posterior-collapse latent statistics.",
    )
    eval_stage1_vae.add_argument(
        "--metrics-raw",
        type=Path,
        default=None,
        help="Optional metrics_raw.json from training to also render the loss curve.",
    )

    select_stage1_audit = subparsers.add_parser(
        "select-stage1-vae-audit",
        help="Freeze the deterministic 60-volume, 15-domain Stage-1 test selection.",
    )
    select_stage1_audit.add_argument("--split-json", type=Path, required=True)
    select_stage1_audit.add_argument(
        "--private-out",
        type=Path,
        required=True,
        help="Private selection JSON with rerun identities/paths; keep outside Git.",
    )
    select_stage1_audit.add_argument(
        "--sanitized-out",
        type=Path,
        required=True,
        help="Sanitized selection report containing anonymous domain/case slots only.",
    )
    select_stage1_audit.add_argument("--seed", type=int, default=13)

    audit_stage1 = subparsers.add_parser(
        "audit-stage1-vae",
        help="Audit one or more Stage-1 checkpoints on a frozen 60-volume selection.",
        description="Full-volume posterior-mean audit. Do not run concurrently with Stage-1 "
        "training on the same GPU.",
    )
    audit_stage1.add_argument("--split-json", type=Path, required=True)
    audit_stage1.add_argument("--selection", type=Path, required=True)
    audit_stage1.add_argument("--config", type=Path, default=Path("configs/experiment/stage1_vae.yaml"))
    audit_stage1.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Checkpoint label and private path; repeat to compare independently evaluated checkpoints.",
    )
    audit_stage1.add_argument("--out", type=Path, required=True)
    audit_stage1.add_argument("--overlap", type=float, default=0.5)
    audit_stage1.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    audit_stage1.add_argument(
        "--precision", choices=("float32", "amp-bfloat16"), default="float32"
    )
    audit_stage1.add_argument("--resume", action="store_true")

    smoke_stage1_audit = subparsers.add_parser(
        "smoke-stage1-audit",
        help="Run the complete 15-domain audit orchestration on synthetic CPU volumes.",
    )
    smoke_stage1_audit.add_argument("--out", type=Path, required=True)

    train_stage2_diffuser = subparsers.add_parser(
        "train-stage2-diffuser",
        help="Run Etapa 1's conditional latent diffuser training (VAE frozen by default).",
    )
    train_stage2_diffuser.add_argument("--config", type=Path, default=Path("configs/experiment/stage2_diffuser.yaml"))
    train_stage2_diffuser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest of real NIfTI volumes (requires the 'nifti' extra). No synthetic "
        "fallback for this stage — never commit a real manifest to the repo.",
    )
    train_stage2_diffuser.add_argument("--steps", type=int, default=None)
    train_stage2_diffuser.add_argument("--batch-size", type=int, default=None)
    train_stage2_diffuser.add_argument(
        "--patches-per-volume",
        type=int,
        default=None,
        help="Random patches drawn per volume before it is dropped (overrides "
        "data.patches_per_volume). Higher = fewer disk reads per training step.",
    )
    train_stage2_diffuser.add_argument("--seed", type=int, default=None)
    train_stage2_diffuser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    train_transport = subparsers.add_parser(
        "train-stage2-transport",
        help="Train the latent transport v_theta (OT-CFM / Schrodinger bridge) on the latent bank.",
    )
    train_transport.add_argument(
        "--config", type=Path, default=Path("configs/experiment/stage2_transport_fm_v1.yaml")
    )
    train_transport.add_argument("--bank-dir", type=Path, required=True, help="Latent bank directory.")
    train_transport.add_argument(
        "--latent-stats", type=Path, default=None, help="Defaults to <bank-dir>/latent_stats.json."
    )
    train_transport.add_argument("--checkpoint-dir", type=Path, default=None)
    train_transport.add_argument("--steps", type=int, default=None)
    train_transport.add_argument("--batch-size", type=int, default=None)
    train_transport.add_argument("--seed", type=int, default=None)
    train_transport.add_argument("--coupling", choices=("independent", "ot"), default=None)
    train_transport.add_argument("--bridge", choices=("ot_cfm", "schrodinger"), default=None)
    train_transport.add_argument("--resume-from", type=Path, default=None)
    train_transport.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    train_transport.add_argument("--val", action="store_true", help="Enable validation-split flow loss + best checkpoint.")
    train_transport.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    eval_transport = subparsers.add_parser(
        "eval-stage2-transport",
        help="Decode transported latents and score them vs the real target on the traveller pairs.",
    )
    eval_transport.add_argument(
        "--config", type=Path, default=Path("configs/experiment/stage2_transport_fm_v1.yaml"),
        help="Transport config providing the translator model kwargs.",
    )
    eval_transport.add_argument("--bank-dir", type=Path, required=True, help="Latent bank directory.")
    eval_transport.add_argument("--split-json", type=Path, required=True, help="VAE split JSON (real target volumes).")
    eval_transport.add_argument("--transport-checkpoint", type=Path, required=True, help="Trained v_theta checkpoint.")
    eval_transport.add_argument(
        "--vae-config", type=Path, default=Path("configs/experiment/stage1_vae_v2_fgw_freebits.yaml"),
        help="VAE config providing the kl_vae decoder kwargs.",
    )
    eval_transport.add_argument("--vae-checkpoint", type=Path, required=True, help="Frozen VAE checkpoint (.pt).")
    eval_transport.add_argument("--latent-stats", type=Path, default=None, help="Defaults to <bank-dir>/latent_stats.json.")
    eval_transport.add_argument("--out", type=Path, default=None, help="Write the full per-pair JSON here.")
    eval_transport.add_argument("--solver", choices=("euler", "heun"), default="heun")
    eval_transport.add_argument("--n-steps", type=int, default=20, help="ODE integration steps.")
    eval_transport.add_argument(
        "--metrics", nargs="+", default=["ssim", "nrmse", "lpips"], choices=("ssim", "nrmse", "lpips"),
    )
    eval_transport.add_argument(
        "--subjects", nargs="+", default=None,
        help="Restrict to these subject_ids (default: all prospective travellers in the split).",
    )
    eval_transport.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    resplit_parser = subparsers.add_parser(
        "resplit-travellers",
        help="Move whole subjects between train/validation/test arrays of a split JSON (no re-encode).",
    )
    resplit_parser.add_argument("--split-json", type=Path, required=True, help="Input split JSON.")
    resplit_parser.add_argument("--out", type=Path, required=True, help="Output split JSON (must differ from input).")
    resplit_parser.add_argument("--subjects", nargs="+", required=True, help="Subject ids to move, e.g. 0006 0007.")
    resplit_parser.add_argument("--to", dest="to_split", choices=("train", "validation", "test"), required=True)

    build_latent_bank_parser = subparsers.add_parser(
        "build-latent-bank",
        help="Encode the split volumes to a frozen-VAE latent bank for Stage-2 transport.",
    )
    build_latent_bank_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/stage1_vae_v2_fgw_freebits.yaml"),
        help="VAE config providing the kl_vae model kwargs (and an optional latent_bank section).",
    )
    build_latent_bank_parser.add_argument("--split-json", type=Path, required=True)
    build_latent_bank_parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen VAE checkpoint (.pt).")
    build_latent_bank_parser.add_argument("--out", type=Path, required=True, help="Latent bank output dir (gitignored WORK_DIR).")
    build_latent_bank_parser.add_argument("--strategy", choices=("tiled", "full", "auto"), default=None)
    build_latent_bank_parser.add_argument("--store-dtype", choices=("float16", "float32"), default=None)
    build_latent_bank_parser.add_argument("--precision", choices=("float32", "bfloat16"), default=None)
    build_latent_bank_parser.add_argument(
        "--block-size", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
        help="Core tile size (voxels, multiples of the VAE downsample factor).",
    )
    build_latent_bank_parser.add_argument(
        "--halo", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
        help="Halo of real neighbours around each tile (voxels, multiples of the factor).",
    )
    build_latent_bank_parser.add_argument("--roundtrip-samples", type=int, default=None)
    build_latent_bank_parser.add_argument("--seed", type=int, default=None)
    build_latent_bank_parser.add_argument("--overwrite", action="store_true")
    build_latent_bank_parser.add_argument(
        "--splits", nargs="+", default=None, help="Splits to encode (default train validation test)."
    )
    build_latent_bank_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    print_config = subparsers.add_parser("print-config", help="Print a YAML config.")
    print_config.add_argument("--config", type=Path, default=Path("configs/experiment/smoke.yaml"))

    audit = subparsers.add_parser("audit-manifest", help="Validate manifest structure and paths.")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--strict-paths", action="store_true")

    subparsers.add_parser(
        "mrixfields2026-print-spec",
        help="Print official MRIxFields2026 constants, task specs, and validation IDs.",
    )

    official_audit = subparsers.add_parser(
        "mrixfields2026-audit-submission",
        help="Validate an MRIxFields2026 submission tree at path/filename level.",
    )
    official_audit.add_argument("--root", type=Path, required=True)
    official_audit.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    official_audit.add_argument(
        "--strict-segmentation",
        dest="strict_segmentation",
        action="store_true",
        default=True,
    )
    official_audit.add_argument(
        "--allow-missing-seg",
        dest="strict_segmentation",
        action="store_false",
    )
    official_audit.add_argument("--allow-extra-files", action="store_true")
    official_audit.add_argument("--json", action="store_true", help="Emit JSON output.")

    official_zip = subparsers.add_parser(
        "mrixfields2026-zip-submission",
        help="Validate and zip an MRIxFields2026 submission with taskN/ at archive root.",
    )
    official_zip.add_argument("--submission-root", type=Path, required=True)
    official_zip.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    official_zip.add_argument("--out", type=Path, required=True)
    official_zip.add_argument(
        "--strict-segmentation",
        dest="strict_segmentation",
        action="store_true",
        default=True,
    )
    official_zip.add_argument(
        "--allow-missing-seg",
        dest="strict_segmentation",
        action="store_false",
    )

    build_manifest = subparsers.add_parser(
        "mrixfields2026-build-manifest",
        help="Scan an extracted MRIxFields2026 data root and write a JSONL audit manifest.",
    )
    build_manifest.add_argument("--data-root", type=Path, required=True)
    build_manifest.add_argument("--out", type=Path, required=True)
    build_manifest.add_argument("--split", action="append", default=None)
    build_manifest.add_argument("--inspect-payload", action="store_true")
    build_manifest.add_argument("--json", action="store_true", help="Emit JSON output.")

    audit_data = subparsers.add_parser(
        "mrixfields2026-audit-data",
        help="Audit an MRIxFields2026 JSONL manifest or extracted data root.",
    )
    audit_source = audit_data.add_mutually_exclusive_group(required=True)
    audit_source.add_argument("--manifest", type=Path)
    audit_source.add_argument("--data-root", type=Path)
    audit_data.add_argument("--inspect-payload", action="store_true")
    audit_data.add_argument("--json", action="store_true", help="Emit JSON output.")

    official_task3_eval = subparsers.add_parser(
        "mrixfields2026-evaluate-task3",
        help="Run the source-pinned official Task-3 metrics over matched NIfTI directories.",
    )
    official_task3_eval.add_argument("--pred-dir", type=Path, required=True)
    official_task3_eval.add_argument("--target-dir", type=Path, required=True)
    official_task3_eval.add_argument(
        "--metrics",
        nargs="+",
        choices=("nrmse", "ssim", "lpips"),
        default=("nrmse", "ssim", "lpips"),
    )
    official_task3_eval.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="LPIPS device; published fallback uses CPU when CUDA is unavailable.",
    )
    official_task3_eval.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path; the same payload is always printed.",
    )

    board = subparsers.add_parser(
        "mrixfields2026-task3-board",
        help=(
            "Aggregate official Task-3 metrics into the leaderboard-shaped per-contrast "
            "table + rank-sum. One of --pred-dir/--board-json/--summaries-json."
        ),
    )
    board_source = board.add_mutually_exclusive_group(required=True)
    board_source.add_argument(
        "--pred-dir",
        type=Path,
        help="Prediction tree <contrast>/<subtask>/*.nii.gz; requires --target-dir.",
    )
    board_source.add_argument(
        "--board-json",
        type=Path,
        help="Precomputed per-unit case metrics ({'units': [...]}) to aggregate.",
    )
    board_source.add_argument(
        "--summaries-json",
        type=Path,
        help="Submission means ({'submissions': {name: {nrmse,ssim,lpips}}}) to rank.",
    )
    board.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Target tree matching --pred-dir layout (required with --pred-dir).",
    )
    board.add_argument(
        "--metrics",
        nargs="+",
        choices=("nrmse", "ssim", "lpips"),
        default=("nrmse", "ssim", "lpips"),
    )
    board.add_argument(
        "--contrast-weighting",
        choices=("balanced", "pooled"),
        default="balanced",
        help="Overall mean weights contrasts equally (balanced) or cases equally (pooled).",
    )
    board.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="LPIPS device for --pred-dir evaluation.",
    )
    board.add_argument(
        "--rank-as",
        default=None,
        help="With --pred-dir/--board-json, add our aggregate under this name to the ranking.",
    )
    board.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")

    fit_stage2_photometry = subparsers.add_parser(
        "fit-stage2-photometry",
        help="Fit the fixed Variant-A photometry artifact from retrospective train records only.",
    )
    fit_stage2_photometry.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/stage2_photometry_factorization_a_v1.yaml"),
    )
    fit_stage2_photometry.add_argument("--split-json", type=Path, required=True)
    fit_stage2_photometry.add_argument("--out", type=Path, required=True)
    fit_stage2_photometry.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    fit_stage2_photometry.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Progress interval; logs to stderr and never changes JSON output.",
    )

    audit_stage2_photometry = subparsers.add_parser(
        "audit-stage2-photometry",
        help=(
            "Qualify fixed photometry and frozen-VAE compatibility on retrospective "
            "validation records."
        ),
    )
    audit_stage2_photometry.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/stage2_photometry_factorization_a_v1.yaml"),
    )
    audit_stage2_photometry.add_argument("--split-json", type=Path, required=True)
    audit_stage2_photometry.add_argument("--artifact", type=Path, required=True)
    audit_stage2_photometry.add_argument("--vae-config", type=Path, required=True)
    audit_stage2_photometry.add_argument("--vae-checkpoint", type=Path, required=True)
    audit_stage2_photometry.add_argument("--continuity-reference", type=Path, required=True)
    audit_stage2_photometry.add_argument("--gate01-result", type=Path, required=True)
    audit_stage2_photometry.add_argument("--out", type=Path, required=True)
    audit_stage2_photometry.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    audit_stage2_photometry.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    audit_stage2_photometry.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Progress interval; logs to stderr and never changes JSON output.",
    )

    eval_stage2_photometry = subparsers.add_parser(
        "eval-stage2-photometry-baseline",
        help=(
            "Evaluate one fixed-map directed identity case with a hash-verified "
            "Gate-0.1 continuity reference."
        ),
    )
    eval_stage2_photometry.add_argument("--artifact", type=Path, required=True)
    eval_stage2_photometry.add_argument(
        "--source-array", type=Path, required=True, help="External .npy source volume."
    )
    eval_stage2_photometry.add_argument(
        "--target-array",
        type=Path,
        required=True,
        help="External .npy metric target; never used for photometry or support.",
    )
    eval_stage2_photometry.add_argument("--source-field", type=float, required=True)
    eval_stage2_photometry.add_argument(
        "--source-contrast", choices=("T1w", "T2w", "T2-FLAIR"), required=True
    )
    eval_stage2_photometry.add_argument("--target-field", type=float, required=True)
    eval_stage2_photometry.add_argument(
        "--target-contrast", choices=("T1w", "T2w", "T2-FLAIR"), required=True
    )
    eval_stage2_photometry.add_argument("--case-id", required=True)
    eval_stage2_photometry.add_argument("--selection-id", required=True)
    eval_stage2_photometry.add_argument("--continuity-reference", type=Path, required=True)
    eval_stage2_photometry.add_argument("--gate01-result", type=Path, required=True)
    eval_stage2_photometry.add_argument("--out", type=Path, required=True)
    eval_stage2_photometry.add_argument(
        "--metrics",
        nargs="+",
        choices=("nrmse", "ssim", "lpips"),
        default=("nrmse", "ssim", "lpips"),
    )
    eval_stage2_photometry.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    fit_intensity = subparsers.add_parser(
        "fit-intensity-baseline",
        help=(
            "Fit the frozen Gate-0 v1 robust-affine and histogram baselines from "
            "training records."
        ),
    )
    fit_intensity.add_argument("--split-json", type=Path, required=True)
    fit_intensity.add_argument(
        "--out", type=Path, required=True, help="Output stem; one JSON is written per mode."
    )
    fit_intensity.add_argument("--per-domain", type=int, default=8)
    fit_intensity.add_argument("--num-quantiles", type=int, default=256)
    fit_intensity.add_argument("--mask-threshold", type=float, default=0.0)
    fit_intensity.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    fit_gate01 = subparsers.add_parser(
        "fit-gate01-target-calibrator",
        help=(
            "Fit the frozen Gate-0.1 post-hoc target-CDF calibrator from all "
            "retrospective training records."
        ),
    )
    fit_gate01.add_argument("--split-json", type=Path, required=True)
    fit_gate01.add_argument("--out", type=Path, required=True)
    fit_gate01.add_argument("--num-quantiles", type=int, default=256)
    fit_gate01.add_argument("--mask-threshold", type=float, default=0.0)
    fit_gate01.add_argument("--low-probability", type=float, default=0.01)
    fit_gate01.add_argument("--high-probability", type=float, default=0.99)
    fit_gate01.add_argument(
        "--training-cohort-identity",
        required=True,
        help="Auditable label/hash identifying the retrospective training cohort.",
    )
    fit_gate01.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    lock_gate01 = subparsers.add_parser(
        "lock-gate01-protocol",
        help="Hash-seal an independent external Gate-0.1 protocol specification.",
    )
    lock_gate01.add_argument("--spec", type=Path, required=True)
    lock_gate01.add_argument("--out", type=Path, required=True)

    fingerprint_gate01 = subparsers.add_parser(
        "fingerprint-gate01-selection",
        help="Hash a sanitized, predeclared Gate-0.1 selection descriptor list.",
    )
    fingerprint_gate01.add_argument("--selection", type=Path, required=True)

    select_gate01 = subparsers.add_parser(
        "prepare-gate01-private-selection",
        help=(
            "Resolve one external prospective traveller to a hash-only frozen "
            "15-acquisition Gate-0.1 selection."
        ),
    )
    select_gate01.add_argument("--split-json", type=Path, required=True)
    select_gate01.add_argument("--traveller-subject-id", required=True)
    select_gate01.add_argument("--out", type=Path, required=True)

    spec_gate01 = subparsers.add_parser(
        "lock-gate01-producer-spec",
        help=(
            "Hash-seal every external input, selected source array, and strict full-volume "
            "inference setting for the private producer."
        ),
    )
    for option in (
        "selection",
        "split-json",
        "bank-source-split-json",
        "bank-dir",
        "stage1-config",
        "stage1-checkpoint",
        "sb-v2-config",
        "sb-v2-checkpoint",
        "protocol-lock",
    ):
        spec_gate01.add_argument(f"--{option}", type=Path, required=True)
    spec_gate01.add_argument("--solver", choices=("euler", "heun"), required=True)
    spec_gate01.add_argument("--n-steps", type=int, required=True)
    spec_gate01.add_argument("--decode-block-size", type=int, nargs=3, required=True)
    spec_gate01.add_argument("--decode-halo", type=int, nargs=3, required=True)
    spec_gate01.add_argument(
        "--decode-precision", choices=("float32", "bfloat16"), required=True
    )
    spec_gate01.add_argument("--deterministic-seed", type=int, default=0)
    spec_gate01.add_argument("--out", type=Path, required=True)

    produce_gate01 = subparsers.add_parser(
        "produce-gate01-private-artifacts",
        help=(
            "Deterministically and resumably infer the complete external 15/60/180 "
            "Gate-0.1 artifact graph."
        ),
    )
    for option in (
        "spec",
        "selection",
        "split-json",
        "bank-source-split-json",
        "bank-dir",
        "stage1-config",
        "stage1-checkpoint",
        "sb-v2-config",
        "sb-v2-checkpoint",
        "protocol-lock",
        "output-dir",
        "state-dir",
    ):
        produce_gate01.add_argument(f"--{option}", type=Path, required=True)
    produce_gate01.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    produce_gate01.add_argument("--resume", action="store_true")

    build_gate01 = subparsers.add_parser(
        "build-gate01-private-manifest",
        help=(
            "Verify frozen external arrays and resumably build the private Gate-0.1 "
            "prediction manifest without running inference."
        ),
    )
    build_gate01.add_argument("--plan", type=Path, required=True)
    build_gate01.add_argument("--producer-spec", type=Path, required=True)
    build_gate01.add_argument("--producer-state", type=Path, required=True)
    build_gate01.add_argument("--protocol-lock", type=Path, required=True)
    build_gate01.add_argument("--calibrator", type=Path, required=True)
    build_gate01.add_argument("--out", type=Path, required=True)
    build_gate01.add_argument("--state", type=Path, required=True)
    build_gate01.add_argument("--resume", action="store_true")

    gate01 = subparsers.add_parser(
        "gate01-equal-photometry",
        help=(
            "Evaluate raw/calibrated identity, raw/calibrated SB-v2, and the "
            "Stage-1 ceiling on strict external full-volume predictions."
        ),
    )
    gate01.add_argument("--manifest", type=Path, required=True)
    gate01.add_argument("--calibrator", type=Path, required=True)
    gate01.add_argument(
        "--protocol-lock",
        type=Path,
        default=None,
        help="Independent external protocol lock; mandatory for scientific mode.",
    )
    gate01.add_argument(
        "--metrics",
        nargs="+",
        choices=("nrmse", "ssim", "lpips"),
        default=("nrmse", "ssim", "lpips"),
    )
    gate01.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    gate01.add_argument(
        "--include-robust-affine",
        action="store_true",
        help="Add robust-affine-calibrated identity/SB as diagnostic-only methods.",
    )
    gate01.add_argument("--out", type=Path, default=None)
    gate01.add_argument("--markdown-out", type=Path, default=None)
    gate01.add_argument("--contract-out", type=Path, default=None)
    gate01.add_argument(
        "--montage-dir",
        type=Path,
        default=None,
        help="Render the frozen deterministic montage PNGs and provenance manifest.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "smoke-train":
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        if args.seed is not None:
            config["seed"] = args.seed
        result = run_smoke_train(SmokeTrainConfig.from_mapping(config))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"smoke-train completed: steps={result.steps} final_loss={result.final_loss:.6f}")
        return 0

    if args.command == "train":
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        if args.seed is not None:
            config["seed"] = args.seed
        model_config = _model_config(config)
        model_name = str(model_config.get("name", "identity"))
        encoder_name, encoder_kwargs = _component_config(model_config, "encoder", model_name)
        decoder_name, decoder_kwargs = _component_config(model_config, "decoder", model_name)
        translator_name, translator_kwargs = _component_config(model_config, "translator", model_name)
        encoder = build_encoder(encoder_name, **encoder_kwargs)
        decoder = build_decoder(decoder_name, **decoder_kwargs)
        translator = build_translator(translator_name, **translator_kwargs)
        loop_config = TrainLoopConfig.from_mapping(config)
        loader = (
            _build_manifest_loader(
                args.manifest,
                batch_size=loop_config.batch_size,
                shuffle=True,
                num_workers=_num_workers(config),
            )
            if args.manifest
            else None
        )
        result = run_train_loop(loop_config, encoder=encoder, decoder=decoder, translator=translator, loader=loader)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"train completed: steps={result.steps} final_loss={result.final_loss:.6f}")
        return 0

    if args.command == "train-pseudo-pairs":
        config = _load_optional_config(args.config)
        _apply_pseudo_pair_overrides(config, args, training=True)
        manifest_path = _pseudo_pair_manifest_path(config)
        records = _pseudo_pair_records(config, manifest_path)
        preprocessing = _pseudo_pair_preprocessing(config)
        slice_count = len(selected_slice_indices(preprocessing))
        split_path = _pseudo_pair_split_path(config)
        splits = _build_or_load_pseudo_pair_splits(config, records, split_path)
        leakage_audit = audit_volume_splits(splits)
        leakage_audit.raise_for_leakage()
        split_summary = summarize_volume_splits(splits, slices_per_volume=slice_count)
        save_volume_splits(splits, split_path)
        split_sha256 = volume_splits_fingerprint(splits)
        checkpoint_dir = _pseudo_pair_checkpoint_dir(config)
        if checkpoint_dir is None:
            raise ValueError("train-pseudo-pairs requires training.checkpoint_dir or --checkpoint-dir.")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _override(config, "training", "checkpoint_dir", str(checkpoint_dir))
        if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")

        train_config = PseudoPairEpochConfig.from_mapping(config)
        train_dataset = PseudoPairSliceDataset(
            splits.train,
            image_loader=nifti_image_loader,
            source_field=_pseudo_pair_source_field(config),
            sequence=_pseudo_pair_sequence(config),
            preprocessing=preprocessing,
            mode="train",
            seed=train_config.seed,
            cache_size=_pseudo_pair_cache_size(config),
        )
        val_dataset = PseudoPairSliceDataset(
            splits.validation,
            image_loader=nifti_image_loader,
            source_field=_pseudo_pair_source_field(config),
            sequence=_pseudo_pair_sequence(config),
            preprocessing=preprocessing,
            mode="validation",
            seed=train_config.seed,
            cache_size=_pseudo_pair_cache_size(config),
        )
        test_dataset = PseudoPairSliceDataset(
            splits.test,
            image_loader=nifti_image_loader,
            source_field=_pseudo_pair_source_field(config),
            sequence=_pseudo_pair_sequence(config),
            preprocessing=preprocessing,
            mode="test",
            seed=train_config.seed,
            cache_size=_pseudo_pair_cache_size(config),
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("train-pseudo-pairs produced empty train or validation slice splits.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            sampler=make_field_balanced_sampler(train_dataset, seed=train_config.seed),
            num_workers=_num_workers(config),
            collate_fn=collate_pseudo_pair_slices,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=_num_workers(config),
            collate_fn=collate_pseudo_pair_slices,
        )
        steps_per_epoch = len(train_loader)
        _print_pseudo_pair_summary(split_summary, batch_size=train_config.batch_size, steps_per_epoch=steps_per_epoch)
        run_metadata = {
            "manifest_validation": _pseudo_pair_manifest_validation(config, records).to_dict(),
            "split_json": str(split_path),
            "split_sha256": split_sha256,
            "split_summary": split_summary,
            "preprocessing": preprocessing.to_dict(),
        }
        if args.preflight:
            payload = _pseudo_pair_preflight_payload(
                manifest_path=manifest_path,
                split_path=split_path,
                split_sha256=split_sha256,
                split_summary=split_summary,
                leakage_audit=leakage_audit.to_dict(),
                preprocessing=preprocessing,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                batch_size=train_config.batch_size,
                num_workers=_num_workers(config),
                manifest_validation=run_metadata["manifest_validation"],
            )
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        model = _build_pseudo_pair_translator(config)
        result = train_pseudo_pair_epochs(
            train_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            run_metadata=run_metadata,
        )
        payload = {
            **result.to_dict(),
            "split_json": str(split_path),
            "split_summary": split_summary,
            "steps_per_epoch": steps_per_epoch,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                "train-pseudo-pairs completed: "
                f"epochs={result.epochs_completed} global_step={result.global_step} "
                f"best_checkpoint={result.best_checkpoint} last_checkpoint={result.last_checkpoint}"
            )
        return 0

    if args.command == "eval-pseudo-pairs":
        config = _load_optional_config(args.config)
        _apply_pseudo_pair_overrides(config, args, training=False)
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state = load_checkpoint(args.checkpoint)
        if state.get("trainer") != "pseudo_pair_epochs":
            raise ValueError("Checkpoint is not compatible with eval-pseudo-pairs.")
        if int(state.get("pseudo_pair_pipeline_version", 1)) < PSEUDO_PAIR_PIPELINE_VERSION:
            raise ValueError(
                "Checkpoint was produced before the pseudo-pair loss/axis correction; "
                "rerun train-pseudo-pairs from scratch before evaluation."
            )
        manifest_path = _pseudo_pair_manifest_path(config)
        records = _pseudo_pair_records(config, manifest_path)
        preprocessing = _pseudo_pair_preprocessing(config)
        split_path = _pseudo_pair_split_path(config, checkpoint=args.checkpoint)
        if split_path.exists():
            splits = load_volume_splits(split_path)
        else:
            splits = _build_or_load_pseudo_pair_splits(config, records, split_path)
            save_volume_splits(splits, split_path)
        split_sha256 = volume_splits_fingerprint(splits)
        checkpoint_split_sha256 = (
            state.get("run_metadata", {}).get("split_sha256")
            if isinstance(state.get("run_metadata"), Mapping)
            else None
        )
        if checkpoint_split_sha256 is not None and checkpoint_split_sha256 != split_sha256:
            raise ValueError(
                "Checkpoint split identity does not match the loaded split JSON: "
                f"{checkpoint_split_sha256} != {split_sha256}."
            )
        records_for_eval = splits.records_for(args.split)
        if not records_for_eval:
            raise ValueError(f"eval-pseudo-pairs split {args.split!r} is empty.")
        model = _build_pseudo_pair_translator(config)
        model.load_state_dict(state["model"])
        eval_config = PseudoPairEvalConfig.from_mapping(config)
        dataset = PseudoPairSliceDataset(
            records_for_eval,
            image_loader=nifti_image_loader,
            source_field=_pseudo_pair_source_field(config),
            sequence=_pseudo_pair_sequence(config),
            preprocessing=preprocessing,
            mode="test" if args.split == "test" else "validation",
            seed=int(config.get("seed", 13)),
            cache_size=_pseudo_pair_cache_size(config),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("training", {}).get("batch_size", 8))
            if isinstance(config.get("training", {}), Mapping)
            else 8,
            shuffle=False,
            num_workers=_num_workers(config),
            collate_fn=collate_pseudo_pair_slices,
        )
        payload = evaluate_pseudo_pairs(model, loader, eval_config)
        payload["checkpoint"] = str(args.checkpoint)
        payload["split_json"] = str(split_path)
        payload["split"] = args.split
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "train-stage1-vae":
        if args.patch_bank is None and args.manifest is None and args.split_json is None:
            raise ValueError(
                "train-stage1-vae requires one of --manifest, --split-json, or --patch-bank."
            )
        config = _load_optional_config(args.config)
        _override(config, "training", "batch_size", args.batch_size)
        _override(config, "data", "patches_per_volume", args.patches_per_volume)
        # A split provides its own train/validation record lists; --manifest / --patch-bank
        # keep the original no-validation behavior.
        split = load_vae_splits(args.split_json) if args.split_json is not None else None
        if split is not None:
            _override(
                config,
                "training",
                "split_fingerprint",
                vae_splits_recovery_fingerprint_v3(split),
            )
        # Resolve the endpoint only after the split determines an exact pass length.
        # Precedence is CLI --epochs > config training.epochs > explicit steps.
        if args.patch_bank is not None:
            num_volumes, ppv = patch_bank_size(args.patch_bank)
            batch_size = int(config.get("training", {}).get("batch_size", 2))
            steps_per_epoch = max(1, -(-num_volumes * ppv // max(1, batch_size)))
        elif split is not None:
            steps_per_epoch = _steps_per_epoch_for_volumes(config, len(split.train))
        else:
            steps_per_epoch = _steps_per_epoch(config, args.manifest)
        _override(config, "training", "steps_per_epoch", steps_per_epoch)
        _resolve_stage1_endpoint(
            config,
            steps_per_epoch=steps_per_epoch,
            cli_epochs=args.epochs,
            cli_steps=args.steps,
        )
        _resolve_stage1_epoch_dependent_schedules(
            config, steps_per_epoch=steps_per_epoch
        )
        if args.seed is not None:
            config["seed"] = args.seed
        _preflight_stage1_v3_records(config, split)
        stage_config = Stage1VAEConfig.from_mapping(config)
        preflight_stage1_resume(stage_config)
        model_config = _model_config(config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        val_loader: "DataLoader[RawBatch] | None" = None
        if args.patch_bank is not None:
            loader = DataLoader(
                PatchBankDataset(args.patch_bank),
                batch_size=stage_config.batch_size,
                shuffle=True,
                num_workers=_num_workers(config),
                collate_fn=collate_raw_batches,
            )
        elif split is not None:
            loader = _build_streaming_patch_loader_from_records(
                split.train, batch_size=stage_config.batch_size, config=config, num_workers=_num_workers(config)
            )
            if split.validation:
                # seed_offset so the val loader draws different patches than train from the same base seed.
                val_loader = _build_streaming_patch_loader_from_records(
                    split.validation,
                    batch_size=stage_config.batch_size,
                    config=config,
                    num_workers=_num_workers(config),
                    seed_offset=10_000,
                    apply_field_balance=False,
                )
        else:
            loader = _build_streaming_patch_loader(
                args.manifest,
                batch_size=stage_config.batch_size,
                config=config,
                num_workers=_num_workers(config),
            )
        result = run_stage1_vae_train(
            stage_config, encoder=encoder, decoder=decoder, loader=loader, val_loader=val_loader
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"train-stage1-vae completed: steps={result.steps} final_loss={result.final_loss:.6f}")
        return 0

    if args.command == "build-vae-splits":
        records = load_manifest(args.manifest).records
        splits = build_vae_splits(
            records,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        save_vae_splits(splits, args.out)
        summary = summarize_vae_splits(splits)
        summary["fingerprint"] = vae_splits_fingerprint(splits)
        summary["recovery_fingerprint_v3"] = (
            vae_splits_recovery_fingerprint_v3(splits)
        )
        summary["out"] = str(args.out)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            counts = {name: summary["splits"][name]["num_records"] for name in ("train", "validation", "test")}
            print(
                f"build-vae-splits wrote {args.out} (fingerprint {summary['fingerprint'][:12]}): "
                f"train={counts['train']} validation={counts['validation']} test={counts['test']}"
            )
        return 0

    if args.command == "build-patch-bank":
        config = _load_optional_config(args.config)
        seed = int(args.seed) if args.seed is not None else int(config.get("seed", 13))
        ppv = int(args.patches_per_volume) if args.patches_per_volume is not None else _patches_per_volume(config)
        manifest = load_manifest(args.manifest)
        result = build_patch_bank(
            manifest.records,
            image_loader=nifti_image_loader,
            out_dir=args.out,
            patch_size=_data_patch_size(config) or (64, 64, 64),
            patches_per_volume=ppv,
            seed=seed,
            logger=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(
            f"build-patch-bank done: wrote={result.num_volumes_written} skipped={result.num_volumes_skipped} "
            f"failed={result.num_volumes_failed} total_patches={result.total_patches} out={result.out_dir}"
        )
        return 0

    if args.command == "select-stage1-vae-audit":
        splits = load_vae_splits(args.split_json)
        payload = freeze_stage1_audit_selection(
            splits,
            private_path=args.private_out,
            sanitized_path=args.sanitized_out,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "total_volumes": payload["total_volumes"],
                    "domain_count": len(payload["domains"]),
                    "selection_fingerprint": payload["selection_fingerprint"],
                    "split_fingerprint": payload["split_fingerprint"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "audit-stage1-vae":
        import torch

        config = _load_optional_config(args.config)
        splits = load_vae_splits(args.split_json)
        selection = load_and_validate_stage1_audit_selection(args.selection, splits=splits)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device(
            "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
        )
        data_config = _data_mapping(config)
        training_config = config.get("training", {})
        if not isinstance(training_config, Mapping):
            training_config = {}
        runtime = AuditRuntime(
            patch_size=_eval_patch_size(config),
            overlap=float(args.overlap),
            foreground_threshold=float(data_config.get("foreground_threshold", 0.0)),
            precision=args.precision,
            seed=int(selection["seed"]),
            latent_active_kl_threshold=float(training_config.get("latent_active_kl_threshold", 0.01)),
        )
        root_contract = prepare_audit_root(
            args.out,
            selection=selection,
            audit_commit=resolve_audit_commit(),
            config_sha256=sha256_file(args.config),
            runtime=runtime,
            device=device,
        )
        checkpoint_specs = [_parse_labeled_path(spec) for spec in args.checkpoint]
        labels = [label for label, _ in checkpoint_specs]
        if len(labels) != len(set(labels)):
            raise ValueError("Checkpoint labels must be unique within one audit invocation.")
        model_config = _model_config(config)
        summaries: list[dict[str, Any]] = []
        for index, (label, checkpoint_path) in enumerate(checkpoint_specs, start=1):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
            decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
            state = load_checkpoint(checkpoint_path)
            encoder.load_state_dict(state["encoder"])
            decoder.load_state_dict(state["decoder"])
            metadata = checkpoint_public_metadata(state, encoder=encoder, decoder=decoder)
            checkpoint_slot = f"checkpoint-{index:02d}"
            print(
                f"stage1_audit checkpoint={checkpoint_slot} state=validated_loading device={device} "
                f"precision={runtime.precision}",
                flush=True,
            )
            summary = audit_stage1_checkpoint(
                encoder=encoder,
                decoder=decoder,
                volume_loader=lambda record: nifti_image_loader(record.image_path, record),
                selection=selection,
                out_dir=args.out / "checkpoints" / checkpoint_slot,
                checkpoint_slot=checkpoint_slot,
                checkpoint_label=label,
                checkpoint_sha256=sha256_file(checkpoint_path),
                checkpoint_metadata=metadata,
                root_contract=root_contract,
                runtime=runtime,
                device=device,
                resume=bool(args.resume),
                progress_path=args.out / "run_progress_sanitized.json",
            )
            summaries.append(summary)
            del encoder, decoder, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
        comparison = write_audit_comparison(
            args.out, checkpoint_summaries=summaries, root_contract=root_contract
        )
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0

    if args.command == "smoke-stage1-audit":
        payload = run_synthetic_stage1_audit_smoke(args.out)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-stage1-vae":
        import torch

        config = _load_optional_config(args.config)
        model_config = _model_config(config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        if args.manifest is None and args.split_json is None:
            raise ValueError("eval-stage1-vae requires --manifest or --split-json.")
        state = load_checkpoint(args.checkpoint)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        # Full-volume reconstruction: official [0, 1] volumes passed through unchanged
        # (no rescaling, per the official format), and NO random crop — the sliding window
        # in run_stage1_eval tiles the whole volume itself.
        if args.split_json is not None:
            records = load_vae_splits(args.split_json).records_for(args.split)
            if not records:
                raise ValueError(f"Split '{args.split}' in {args.split_json} is empty.")
            loader = _build_manifest_loader_from_records(
                records, batch_size=1, transform=assert_official_unit_range, shuffle=False
            )
        else:
            loader = _build_manifest_loader(
                args.manifest,
                batch_size=1,
                transform=assert_official_unit_range,
                shuffle=False,
            )
        patch_size = _eval_patch_size(config)
        loss_curve = _load_loss_curve(args.metrics_raw)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_config = config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}
        payload = run_stage1_eval(
            encoder=encoder,
            decoder=decoder,
            loader=loader,
            patch_size=patch_size,
            out_dir=args.out,
            num_samples=args.num_samples,
            per_field_contrast=args.per_field_contrast,
            per_domain_samples=args.per_domain_samples,
            oversample_field=args.oversample_field,
            oversample_factor=args.oversample_factor,
            eval_seed=args.eval_seed,
            compute_latent_stats=args.latent_stats,
            latent_active_kl_threshold=float(training_config.get("latent_active_kl_threshold", 0.01)),
            overlap=args.overlap,
            device=device,
            lpips_num_slices=int(training_config.get("lpips_num_slices", 8)),
            loss_curve=loss_curve,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "build-latent-bank":
        import torch

        config = _load_optional_config(args.config)
        model_config = _model_config(config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        state = load_checkpoint(args.checkpoint)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)

        splits = load_vae_splits(args.split_json)
        bank_config = LatentBankConfig.from_mapping(config, out_dir=args.out)
        overrides = {
            "strategy": args.strategy,
            "store_dtype": args.store_dtype,
            "precision": args.precision,
            "block_size": tuple(args.block_size) if args.block_size is not None else None,
            "halo": tuple(args.halo) if args.halo is not None else None,
            "roundtrip_samples": args.roundtrip_samples,
            "seed": args.seed,
            "splits": tuple(args.splits) if args.splits is not None else None,
        }
        applied = {key: value for key, value in overrides.items() if value is not None}
        if args.overwrite:
            applied["overwrite"] = True
        bank_config = replace(bank_config, **applied)

        records_by_split = {
            name: splits.records_for(name) for name in bank_config.splits
        }
        device = _resolve_device(args.device)
        manifest = build_latent_bank(
            encoder=encoder,
            decoder=decoder,
            records_by_split=records_by_split,
            config=bank_config,
            device=device,
            checkpoint_sha256=sha256_file(args.checkpoint),
            git_commit=resolve_git_commit(),
            vae_config_path=str(args.config),
        )
        summary = {
            "counts": manifest["counts"],
            "strategy_used": manifest["strategy_used"],
            "latent_std": manifest["latent_stats"]["global_std"],
            "roundtrip_mean_ssim3d": manifest["roundtrip"]["mean_ssim3d"],
            "out_dir": str(args.out),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "train-stage2-diffuser":
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        _override(config, "data", "patches_per_volume", args.patches_per_volume)
        if args.seed is not None:
            config["seed"] = args.seed
        model_config = _model_config(config)
        unet = DenoisingUNet(**{key: value for key, value in model_config.items() if key != "name"})
        vae_model_config = config.get("vae_model", {})
        if not isinstance(vae_model_config, Mapping):
            raise ValueError("Config section 'vae_model' must be a mapping.")
        encoder = build_encoder("kl_vae", **dict(vae_model_config))
        stage_config = Stage2DiffuserConfig.from_mapping(config)
        loader = _build_streaming_patch_loader(
            args.manifest,
            batch_size=stage_config.batch_size,
            config=config,
            num_workers=_num_workers(config),
        )
        result = run_stage2_diffuser_train(stage_config, unet=unet, encoder=encoder, loader=loader)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"train-stage2-diffuser completed: steps={result.steps} final_loss={result.final_loss:.6f}")
        return 0

    if args.command == "train-stage2-transport":
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        _override(config, "training", "coupling", args.coupling)
        _override(config, "training", "bridge", args.bridge)
        _override(config, "training", "device", args.device)
        if args.seed is not None:
            config["seed"] = args.seed
            _override(config, "training", "seed", args.seed)
        if args.checkpoint_dir is not None:
            _override(config, "training", "checkpoint_dir", str(args.checkpoint_dir))
        if args.resume_from is not None:
            _override(config, "training", "resume_from", str(args.resume_from))

        model_config = _model_config(config)
        translator_name = str(model_config.get("name", "flow_matching_latent"))
        translator_kwargs = {key: value for key, value in model_config.items() if key != "name"}
        translator = build_translator(translator_name, **translator_kwargs)

        stats_path = args.latent_stats or (args.bank_dir / "latent_stats.json")
        stats = LatentStats.from_json(stats_path)
        train_index = LatentBankIndex(args.bank_dir, "train")
        val_index = LatentBankIndex(args.bank_dir, "validation") if args.val else None

        stage_config = Stage2TransportConfig.from_mapping(config)
        result = run_stage2_transport_train(
            stage_config,
            translator=translator,
            train_index=train_index,
            stats=stats,
            val_index=val_index,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(
                f"train-stage2-transport completed: steps={result.steps} "
                f"final_loss={result.final_loss:.6f} sec_per_step={result.seconds_per_step:.3f} "
                f"best_val={result.best_val}"
            )
        return 0

    if args.command == "fit-stage2-photometry":
        config = _load_variant_a_config(args.config)
        split_path = assert_variant_a_external_path(args.split_json)
        output_path = assert_variant_a_external_path(args.out)
        splits = load_vae_splits(split_path)
        split_file_sha256 = sha256_file(split_path)
        membership_fingerprint = vae_splits_fingerprint(splits)
        recovery_fingerprint = vae_splits_recovery_fingerprint_v3(splits)
        records = [record for record in splits.train if _variant_a_record_cohort(record) == "R"]
        if not records:
            raise ValueError("Variant-A fitting found no retrospective R/train records.")
        device = _resolve_device(args.device)
        log_every = _nonnegative_log_every(args.log_every)

        def fit_stream():
            for index, record in enumerate(records, start=1):
                if log_every and (index == 1 or index % log_every == 0 or index == len(records)):
                    print(
                        f"variant_a_fit record={index}/{len(records)} domain={record.domain.label}",
                        file=sys.stderr,
                        flush=True,
                    )
                yield PhotometryFitVolume(
                    volume=load_volume(record).to(device),
                    domain=record.domain,
                    record_identity=str(record.case_id),
                    subject_identity=record.subject_id,
                    source_path_identity=str(record.image_path),
                    source_file_sha256=sha256_file(record.image_path),
                    split="train",
                    cohort="R",
                )

        artifact = fit_frozen_photometry(
            fit_stream(),
            source_split_file_sha256=split_file_sha256,
            source_membership_fingerprint=membership_fingerprint,
            source_recovery_fingerprint=recovery_fingerprint,
            code_commit=resolve_git_commit(),
            code_provenance=capture_photometry_code_provenance(),
            resolved_config=config,
            num_quantiles=int(config["photometry"]["num_quantiles"]),
        )
        artifact.save(output_path)
        print(
            json.dumps(
                {
                    "out": str(output_path),
                    "contract_version": artifact.to_dict()["contract_version"],
                    "artifact_sha256": artifact.artifact_sha256,
                    "source_split_file_sha256": split_file_sha256,
                    "source_membership_fingerprint": membership_fingerprint,
                    "source_recovery_fingerprint": recovery_fingerprint,
                    "accepted_records": len(records),
                    "excluded_prospective_records": sum(
                        _variant_a_record_cohort(record) == "P" for record in splits.train
                    ),
                    "domain_volume_counts": artifact.provenance["domain_volume_counts"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "audit-stage2-photometry":
        import torch

        config = _load_variant_a_config(args.config)
        split_path = assert_variant_a_external_path(args.split_json)
        artifact_path = assert_variant_a_external_path(args.artifact)
        checkpoint_path = assert_variant_a_external_path(args.vae_checkpoint)
        continuity_path = assert_variant_a_external_path(args.continuity_reference)
        gate01_path = assert_variant_a_external_path(args.gate01_result)
        output_path = assert_variant_a_external_path(args.out)
        splits = load_vae_splits(split_path)
        split_file_sha256 = sha256_file(split_path)
        membership_fingerprint = vae_splits_fingerprint(splits)
        recovery_fingerprint = vae_splits_recovery_fingerprint_v3(splits)
        artifact = FrozenPhotometryArtifact.load(
            artifact_path,
            expected_split_file_sha256=split_file_sha256,
            expected_membership_fingerprint=membership_fingerprint,
            expected_recovery_fingerprint=recovery_fingerprint,
        )
        continuity = ContinuityReference.load(
            continuity_path, source_result_path=gate01_path
        )
        records = [
            record
            for record in splits.validation
            if _variant_a_record_cohort(record) == "R"
        ]
        if not records:
            raise ValueError("Variant-A qualification found no retrospective R/validation records.")

        vae_config_path = Path(args.vae_config).resolve()
        if not vae_config_path.is_file():
            raise FileNotFoundError(
                f"Variant-A frozen-VAE config not found: {vae_config_path}"
            )
        vae_config = load_yaml_config(vae_config_path)
        model_config = _model_config(vae_config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        state = load_checkpoint(checkpoint_path)
        try:
            encoder.load_state_dict(state["encoder"], strict=True)
            decoder.load_state_dict(state["decoder"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise ValueError(
                "Variant-A qualification could not load the frozen VAE checkpoint."
            ) from exc
        encoder.requires_grad_(False).eval()
        decoder.requires_grad_(False).eval()
        device = _resolve_device(args.device)
        encoder.to(device)
        decoder.to(device)
        precision = str(args.precision)

        def vae_roundtrip(volume, domain):
            latent, encode_path = encode_latent(
                encoder,
                volume.to(device),
                domain,
                strategy="full",
                block_size=(64, 64, 64),
                halo=(0, 0, 0),
                precision=precision,
            )
            reconstructed, decode_path = decode_latent(
                decoder,
                latent,
                domain,
                factor=int(getattr(decoder, "downsample_factor")),
                strategy="full",
                block_size=(64, 64, 64),
                halo=(0, 0, 0),
                precision=precision,
                clamp=True,
            )
            if encode_path != "full" or decode_path != "full":
                raise ValueError("Variant-A VAE preflight requires exact full encode/decode.")
            return reconstructed

        log_every = _nonnegative_log_every(args.log_every)

        def progress(index: int, total: int | None, identity: str) -> None:
            del total
            if log_every and (index == 1 or index % log_every == 0 or index == len(records)):
                print(
                    f"variant_a_qualification record={index}/{len(records)} "
                    f"identity_sha256={hashlib.sha256(identity.encode('utf-8')).hexdigest()}",
                    file=sys.stderr,
                    flush=True,
                )

        def qualification_stream():
            for record in records:
                yield QualificationVolume(
                    volume=load_volume(record).to(device),
                    domain=record.domain,
                    record_identity=str(record.case_id),
                    subject_identity=record.subject_id,
                    source_path_identity=str(record.image_path),
                    source_file_sha256=sha256_file(record.image_path),
                    split="validation",
                    cohort="R",
                )

        thresholds = VariantAQualificationThresholds.from_mapping(config)
        metric_names = tuple(config["evaluation"]["metrics"])
        result = qualify_variant_a(
            artifact,
            qualification_stream(),
            thresholds=thresholds,
            resolved_config=config,
            source_split_file_sha256=split_file_sha256,
            source_membership_fingerprint=membership_fingerprint,
            source_recovery_fingerprint=recovery_fingerprint,
            vae_roundtrip=vae_roundtrip,
            vae_provenance={
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "config_file_sha256": sha256_file(vae_config_path),
                "encoder_class": type(encoder).__name__,
                "decoder_class": type(decoder).__name__,
                "encoder_statistic": "posterior_mean",
                "encode_strategy": "full",
                "decode_strategy": "full",
                "precision": precision,
                "device_type": device.type,
            },
            continuity=continuity,
            metric_function=OfficialQualificationMetrics(
                metrics=metric_names, device=device.type
            ),
            progress=progress,
        )
        write_variant_a_result(output_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-stage2-photometry-baseline":
        paths = {
            "artifact": assert_variant_a_external_path(args.artifact),
            "source": assert_variant_a_external_path(args.source_array),
            "target": assert_variant_a_external_path(args.target_array),
            "continuity": assert_variant_a_external_path(args.continuity_reference),
            "gate01": assert_variant_a_external_path(args.gate01_result),
            "out": assert_variant_a_external_path(args.out),
        }
        artifact = FrozenPhotometryArtifact.load(paths["artifact"])
        continuity = ContinuityReference.load(
            paths["continuity"], source_result_path=paths["gate01"]
        )
        source = _load_variant_a_npy(paths["source"])
        target = _load_variant_a_npy(paths["target"])
        case = BaselineEvaluationCase(
            case_identity=str(args.case_id),
            selection_identity=str(args.selection_id),
            source=source,
            target=target,
            source_domain=Domain(args.source_field, args.source_contrast),
            target_domain=Domain(args.target_field, args.target_contrast),
        )
        result = evaluate_factorized_identity_case(
            artifact,
            case,
            continuity=continuity,
            metric_function=OfficialQualificationMetrics(
                metrics=tuple(args.metrics), device=args.device
            ),
        )
        write_variant_a_result(paths["out"], result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "fit-intensity-baseline":
        splits = load_vae_splits(args.split_json)
        records = list(splits.train) + list(splits.validation) + list(splits.test)
        split_of_case = {
            str(record.case_id): split_name
            for split_name in ("train", "validation", "test")
            for record in splits.records_for(split_name)
        }
        chosen = reference_volume_records(
            records,
            split_of_case=split_of_case,
            per_domain=args.per_domain,
        )
        if not chosen:
            raise SystemExit(
                "No training-split records found to fit Gate-0 intensity references."
            )
        device = _resolve_device(args.device)

        def legacy_stream():
            for record in chosen:
                yield load_volume(record).to(device), record.domain

        baselines = fit_image_intensity_baselines(
            legacy_stream(),
            num_quantiles=args.num_quantiles,
            mask_threshold=args.mask_threshold,
            provenance={
                "split_fingerprint": vae_splits_fingerprint(splits),
                "per_domain": args.per_domain,
            },
            log=True,
        )
        written = {}
        for mode, baseline in baselines.items():
            path = args.out.with_name(
                f"{args.out.stem}_{mode}{args.out.suffix or '.json'}"
            )
            baseline.save(path)
            written[mode] = str(path)
        print(
            json.dumps(
                {"written": written, "reference_volumes": len(chosen)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "fit-gate01-target-calibrator":
        assert_gate01_external_path(args.split_json, repo_root=None)
        assert_gate01_external_path(args.out, repo_root=None)
        splits = load_vae_splits(args.split_json)
        split_fingerprint = vae_splits_fingerprint(splits)
        if split_fingerprint != RESPLIT_FINGERPRINT:
            raise SystemExit(
                "Gate 0.1 requires the frozen resplit fingerprint; "
                f"found {split_fingerprint}, expected {RESPLIT_FINGERPRINT}."
            )
        retrospective = []
        for record in splits.train:
            prefix = str(
                record.metadata.get("prefix", str(record.case_id).split("_", 1)[0])
            )
            if prefix == "R":
                retrospective.append(record)
        if not retrospective:
            raise SystemExit(
                "No retrospective training records were found for Gate 0.1 fitting."
            )
        device = _resolve_device(args.device)

        def template_stream():
            for record in retrospective:
                yield TrainingTemplateVolume(
                    volume=load_volume(record).to(device),
                    domain=record.domain,
                    record_identity=str(record.case_id),
                    split="train",
                    cohort="R",
                )

        calibrator = fit_posthoc_target_calibrator(
            template_stream(),
            split_fingerprint=split_fingerprint,
            training_cohort_identity=args.training_cohort_identity,
            code_commit=resolve_git_commit(),
            code_provenance=gate01_calibrator_code_provenance(),
            num_quantiles=args.num_quantiles,
            mask_threshold=args.mask_threshold,
            low_probability=args.low_probability,
            high_probability=args.high_probability,
            extra_config={
                "split_recovery_fingerprint_v3": (
                    vae_splits_recovery_fingerprint_v3(splits)
                )
            },
        )
        calibrator.save(args.out)
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "contract_version": calibrator.to_dict()["contract_version"],
                    "artifact_sha256": calibrator.artifact_sha256,
                    "template_sha256": calibrator.template_sha256,
                    "split_fingerprint": calibrator.split_fingerprint,
                    "reference_volumes": len(retrospective),
                    "domain_volume_counts": calibrator.provenance[
                        "domain_volume_counts"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "lock-gate01-protocol":
        lock = write_gate01_protocol_lock(args.spec, args.out)
        print(json.dumps(lock.summary(), indent=2, sort_keys=True))
        return 0

    if args.command == "fingerprint-gate01-selection":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if not isinstance(selection, list):
            raise SystemExit("Gate 0.1 selection descriptor root must be a list.")
        print(gate01_selection_fingerprint(selection))
        return 0

    if args.command == "prepare-gate01-private-selection":
        result = prepare_gate01_prospective_selection(
            args.split_json, args.traveller_subject_id, args.out
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "lock-gate01-producer-spec":
        result = write_gate01_producer_spec(
            selection_path=args.selection,
            split_path=args.split_json,
            bank_source_split_path=args.bank_source_split_json,
            bank_dir=args.bank_dir,
            stage1_config_path=args.stage1_config,
            stage1_checkpoint_path=args.stage1_checkpoint,
            sb_v2_config_path=args.sb_v2_config,
            sb_v2_checkpoint_path=args.sb_v2_checkpoint,
            protocol_lock_path=args.protocol_lock,
            sampler=TransportSamplerConfig(solver=args.solver, n_steps=args.n_steps),
            decode=DecodeSpec(
                block_size=tuple(args.decode_block_size),
                halo=tuple(args.decode_halo),
                precision=args.decode_precision,
            ),
            deterministic_seed=args.deterministic_seed,
            out_path=args.out,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "produce-gate01-private-artifacts":
        result = produce_gate01_private_artifacts(
            spec_path=args.spec,
            selection_path=args.selection,
            split_path=args.split_json,
            bank_source_split_path=args.bank_source_split_json,
            bank_dir=args.bank_dir,
            stage1_config_path=args.stage1_config,
            stage1_checkpoint_path=args.stage1_checkpoint,
            sb_v2_config_path=args.sb_v2_config,
            sb_v2_checkpoint_path=args.sb_v2_checkpoint,
            protocol_lock_path=args.protocol_lock,
            output_dir=args.output_dir,
            state_dir=args.state_dir,
            device=args.device,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "build-gate01-private-manifest":
        result = build_gate01_private_manifest(
            args.plan,
            args.producer_spec,
            args.producer_state,
            args.protocol_lock,
            args.calibrator,
            args.out,
            args.state,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "gate01-equal-photometry":
        protocol_lock = (
            Gate01ProtocolLock.load(args.protocol_lock)
            if args.protocol_lock is not None
            else None
        )
        calibrator = PosthocTargetCalibrator.load(
            args.calibrator,
            expected_split_fingerprint=RESPLIT_FINGERPRINT,
            expected_template_sha256=(
                protocol_lock.calibrator_template_sha256
                if protocol_lock is not None
                else None
            ),
            expected_artifact_sha256=(
                protocol_lock.calibrator_artifact_sha256
                if protocol_lock is not None
                else None
            ),
        )
        if protocol_lock is not None:
            protocol_lock.assert_calibrator(calibrator)
        cases, input_metadata = load_gate01_input_manifest(
            args.manifest,
            protocol_lock=protocol_lock,
            calibrator=calibrator,
        )
        if input_metadata["execution_mode"] == "scientific" and args.montage_dir is None:
            raise ValueError(
                "Scientific Gate 0.1 mode requires --montage-dir for frozen rendering."
            )
        montage_collector = (
            Gate01MontageCollector(fixed_montage_specifications())
            if args.montage_dir is not None
            else None
        )
        code_provenance = gate01_code_provenance()
        result = evaluate_gate01(
            cases,
            calibrator=calibrator,
            artifact_provenance=input_metadata["artifact_provenance"],
            code_commit=resolve_git_commit(),
            evidence_scope=input_metadata["evidence_scope"],
            input_manifest_sha256=input_metadata["sha256"],
            producer_receipt=input_metadata["producer_receipt"],
            split_provenance=input_metadata["split_provenance"],
            support_threshold=input_metadata["support_threshold"],
            execution_mode=input_metadata["execution_mode"],
            selection_fingerprint_sha256=input_metadata[
                "selection_fingerprint_sha256"
            ],
            code_provenance=code_provenance,
            protocol_lock=protocol_lock,
            metrics=tuple(args.metrics),
            device=args.device,
            include_robust_affine=args.include_robust_affine,
            case_observer=(
                montage_collector.observe if montage_collector is not None else None
            ),
        )
        if montage_collector is not None and args.montage_dir is not None:
            montage = render_gate01_montages(
                montage_collector,
                args.montage_dir,
                require_complete=input_metadata["execution_mode"] == "scientific",
            )
            result["montage_rendering"] = montage
            result["contract"]["montage_rendering"] = {
                "contract_version": montage["contract_version"],
                "manifest_sha256": montage["manifest_sha256"],
                "complete_frozen_selection": montage[
                    "complete_frozen_selection"
                ],
            }
        written = write_gate01_outputs(
            result,
            json_path=args.out,
            markdown_path=args.markdown_out,
            contract_path=args.contract_out,
        )
        printable = dict(result)
        printable["written"] = written
        print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.command == "eval-stage2-transport":
        import torch

        config = _load_optional_config(args.config)
        model_config = _model_config(config)
        translator_name = str(model_config.get("name", "flow_matching_latent"))
        translator_kwargs = {key: value for key, value in model_config.items() if key != "name"}
        translator = build_translator(translator_name, **translator_kwargs)
        translator.load_state_dict(load_checkpoint(args.transport_checkpoint)["translator"])

        vae_config = _load_optional_config(args.vae_config)
        vae_model_config = _model_config(vae_config)
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(vae_model_config, "decoder"))
        vae_state = load_checkpoint(args.vae_checkpoint)
        decoder.load_state_dict(vae_state["decoder"])
        decoder.requires_grad_(False)

        stats_path = args.latent_stats or (args.bank_dir / "latent_stats.json")
        stats = LatentStats.from_json(stats_path)
        bank_manifest = json.loads((args.bank_dir / "latent_bank_manifest.json").read_text(encoding="utf-8"))
        splits = load_vae_splits(args.split_json)
        records = list(splits.train) + list(splits.validation) + list(splits.test)
        device = _resolve_device(args.device)
        result = evaluate_transport_travellers(
            translator=translator,
            decoder=decoder,
            records=records,
            bank_manifest=bank_manifest,
            bank_dir=args.bank_dir,
            stats=stats,
            sampler=TransportSamplerConfig(solver=args.solver, n_steps=args.n_steps),
            decode=DecodeSpec.from_bank_manifest(bank_manifest),
            device=device,
            metrics=tuple(args.metrics),
            subjects=args.subjects,
        )
        if args.out is not None:
            write_transport_eval(result, args.out)
        print(json.dumps({k: result[k] for k in ("num_pairs", "subjects", "metrics", "sampler", "overall", "by_contrast")},
                         indent=2, sort_keys=True))
        return 0

    if args.command == "resplit-travellers":
        summary = resplit_file(args.split_json, args.out, args.subjects, args.to_split)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "print-config":
        print(dump_yaml_config(load_yaml_config(args.config)))
        return 0

    if args.command == "audit-manifest":
        report = audit_manifest(load_manifest(args.manifest), strict_paths=args.strict_paths)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1

    if args.command == "mrixfields2026-print-spec":
        print(json.dumps(spec_as_dict(), indent=2))
        return 0

    if args.command == "mrixfields2026-audit-submission":
        report = validate_submission_dir(
            args.root,
            args.task,
            strict_segmentation=args.strict_segmentation,
            allow_extra_files=args.allow_extra_files,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1

    if args.command == "mrixfields2026-zip-submission":
        validation = validate_submission_dir(
            args.submission_root,
            args.task,
            strict_segmentation=args.strict_segmentation,
        )
        if validation.ok:
            out_path = build_submission_zip(
                args.submission_root,
                args.task,
                args.out,
                validate_first=False,
                strict_segmentation=args.strict_segmentation,
            )
            payload = {"out": str(out_path), "validation": validation.to_dict()}
        else:
            payload = {"out": str(args.out), "validation": validation.to_dict()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if validation.ok else 1

    if args.command == "mrixfields2026-build-manifest":
        records = scan_mrixfields_data_root(
            args.data_root,
            splits=args.split,
            include_payload_metadata=args.inspect_payload,
        )
        write_manifest_jsonl(records, args.out)
        report = audit_mrixfields_manifest(records)
        payload = {"out": str(args.out), "audit": report.to_dict()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.ok else 1

    if args.command == "mrixfields2026-audit-data":
        if args.manifest is not None:
            records = read_manifest_jsonl(args.manifest)
        else:
            records = scan_mrixfields_data_root(
                args.data_root,
                include_payload_metadata=args.inspect_payload,
            )
        report = audit_mrixfields_manifest(records)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1

    if args.command == "mrixfields2026-evaluate-task3":
        payload = evaluate_official_task3_directory(
            args.pred_dir,
            args.target_dir,
            metrics=args.metrics,
            device=args.device,
        )
        serialized = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0

    if args.command == "mrixfields2026-task3-board":
        return _run_task3_board(args)

    raise ValueError(f"Unknown command: {args.command}")


def _run_task3_board(args: argparse.Namespace) -> int:
    if args.summaries_json is not None:
        payload_in = json.loads(Path(args.summaries_json).read_text(encoding="utf-8"))
        submissions = submissions_from_payload(payload_in)
        ranking = rank_submissions(submissions, metrics=args.metrics)
        payload: dict[str, Any] = {
            "ranking": ranking.to_dict(),
            "submissions": submissions,
        }
        table = format_rank_sum_table(ranking)
    else:
        if args.pred_dir is not None:
            if args.target_dir is None:
                raise ValueError("--pred-dir requires --target-dir.")
            aggregate, provenance = evaluate_task3_board_directory(
                args.pred_dir,
                args.target_dir,
                metrics=args.metrics,
                contrast_weighting=args.contrast_weighting,
                device=args.device,
            )
        else:
            payload_in = json.loads(Path(args.board_json).read_text(encoding="utf-8"))
            units = board_units_from_payload(payload_in)
            aggregate = aggregate_task3_board(
                units,
                metrics=args.metrics,
                contrast_weighting=args.contrast_weighting,
            )
            provenance = None
        payload = {"board": aggregate.to_dict()}
        if provenance is not None:
            payload["metric_contract"] = provenance["OFFICIAL_TASK3_METRIC_CONTRACT"]
            payload["provenance"] = provenance
        if args.rank_as is not None:
            payload["submission"] = {args.rank_as: dict(aggregate.overall.means)}
        table = format_board_table(aggregate)

    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    print(table)
    print(serialized)
    return 0


def _build_manifest_loader(
    manifest_path: Path,
    *,
    batch_size: int,
    transform: ImageTransform | None = assert_official_unit_range,
    shuffle: bool = False,
    num_workers: int = 0,
) -> "DataLoader[RawBatch]":
    # shuffle defaults to False so non-training callers (audits, eval) keep manifest order;
    # training paths pass shuffle=True — the previous fixed-order loader meant a short run
    # only ever saw the first N records.
    return _build_manifest_loader_from_records(
        load_manifest(manifest_path).records,
        batch_size=batch_size,
        transform=transform,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _build_manifest_loader_from_records(
    records: "Sequence[VolumeRecord]",
    *,
    batch_size: int,
    transform: ImageTransform | None = assert_official_unit_range,
    shuffle: bool = False,
    num_workers: int = 0,
) -> "DataLoader[RawBatch]":
    dataset = ManifestVolumeDataset(records, image_loader=nifti_image_loader, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_raw_batches,
    )


def _build_streaming_patch_loader(
    manifest_path: Path,
    *,
    batch_size: int,
    config: Mapping[str, Any],
    num_workers: int = 0,
) -> "DataLoader[RawBatch]":
    """Training loader that streams `data.patches_per_volume` random patches per volume,
    reading each volume from disk once per pass — see StreamingPatchDataset.

    Replaces the per-patch full-volume re-read of `_build_manifest_loader`, which starved
    the GPU on Drive-FUSE. `shuffle=False` because the dataset owns the shuffling (a
    manifest larger than RAM can't be map-indexed+shuffled cheaply); `num_workers` defaults
    to 0 so a single reader hits Drive sequentially (avoiding Drive-FUSE's concurrent-read
    crashes).
    """

    return _build_streaming_patch_loader_from_records(
        load_manifest(manifest_path).records, batch_size=batch_size, config=config, num_workers=num_workers
    )


def _build_streaming_patch_loader_from_records(
    records: "Sequence[VolumeRecord]",
    *,
    batch_size: int,
    config: Mapping[str, Any],
    num_workers: int = 0,
    seed_offset: int = 0,
    apply_field_balance: bool = True,
) -> "DataLoader[RawBatch]":
    """Same streaming loader, from an explicit record list (a split) instead of a manifest.

    `seed_offset` shifts the dataset's shuffle/crop seed so a validation loader draws
    different patches than the train loader from the same base seed. `apply_field_balance`
    is False for the validation loader so val metrics stay on the natural field distribution
    (honest + comparable across runs); field balancing (data.field_balance) is a TRAIN-only
    resampling.
    """

    base_seed = int(config.get("seed", 0)) if isinstance(config.get("seed", 0), int) else 0
    dataset = StreamingPatchDataset(
        records,
        image_loader=nifti_image_loader,
        patch_size=_data_patch_size(config),
        patches_per_volume=_patches_per_volume(config),
        volume_transform=assert_official_unit_range,
        seed=base_seed + seed_offset,
        crop_config=_crop_config(config),
        foreground_threshold=_foreground_threshold(config),
        sampling_weights=_field_sampling_weights(config, records) if apply_field_balance else None,
        joint_domain_balance=(
            _joint_domain_balance_enabled(config) if apply_field_balance else False
        ),
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_raw_batches,
    }
    if num_workers > 0:
        # persistent_workers is required, not just an optimization: this is an
        # IterableDataset that increments its own `_pass` to reshuffle each epoch. Without
        # persistence the DataLoader re-pickles the dataset (with `_pass=0`) for fresh
        # workers every epoch, replaying the same volume order and crops every epoch. It
        # also amortizes Windows `spawn` worker startup over the whole run instead of
        # paying it per epoch. prefetch_factor lets each worker read ahead so the ~231MB
        # NIfTI decode overlaps GPU compute (the num_workers=0 stall we are removing).
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = _prefetch_factor(config)
    return DataLoader(dataset, **loader_kwargs)


def _apply_pseudo_pair_overrides(config: dict[str, Any], args: argparse.Namespace, *, training: bool) -> None:
    _override(config, "data", "manifest", str(args.manifest) if getattr(args, "manifest", None) else None)
    _override(config, "data", "sequence", getattr(args, "sequence", None))
    _override(config, "data", "source_field", getattr(args, "source_field", None))
    _override(config, "data", "target_fields", getattr(args, "target_fields", None))
    _override(config, "data", "train_volumes_per_field", getattr(args, "train_volumes_per_field", None))
    _override(config, "data", "val_volumes_per_field", getattr(args, "val_volumes_per_field", None))
    _override(config, "data", "test_volumes_per_field", getattr(args, "test_volumes_per_field", None))
    _override(config, "data", "max_pilot_records", getattr(args, "max_pilot_records", None))
    _override(config, "data", "split_json", str(args.split_json) if getattr(args, "split_json", None) else None)
    _override_nested(config, "data", "preprocessing", "slice_start", getattr(args, "slice_start", None))
    _override_nested(config, "data", "preprocessing", "slice_end", getattr(args, "slice_end", None))
    _override_nested(config, "data", "preprocessing", "slice_axis", getattr(args, "slice_axis", None))
    _override_nested(
        config,
        "data",
        "preprocessing",
        "slices_per_volume",
        getattr(args, "slices_per_volume", None),
    )
    _override_nested(config, "data", "preprocessing", "output_height", getattr(args, "output_height", None))
    _override_nested(config, "data", "preprocessing", "output_width", getattr(args, "output_width", None))
    _override(config, "training", "batch_size", getattr(args, "batch_size", None))
    _override(config, "training", "num_workers", getattr(args, "workers", None))
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed
    if training:
        _override(config, "training", "epochs", getattr(args, "epochs", None))
        _override(config, "training", "lr", getattr(args, "lr", None))
        _override(
            config,
            "training",
            "checkpoint_dir",
            str(args.checkpoint_dir) if getattr(args, "checkpoint_dir", None) else None,
        )
        _override(
            config,
            "training",
            "resume_from",
            str(args.resume_checkpoint) if getattr(args, "resume_checkpoint", None) else None,
        )


def _pseudo_pair_manifest_path(config: Mapping[str, Any]) -> Path:
    data_config = _data_mapping(config)
    value = data_config.get("manifest")
    if not value:
        raise ValueError("Pseudo-pair commands require data.manifest or --manifest.")
    path = Path(os.path.expandvars(str(value)))
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path


def _pseudo_pair_records(config: Mapping[str, Any], manifest_path: Path) -> list[Any]:
    records = list(load_manifest(manifest_path).records)
    max_records = _data_mapping(config).get("max_pilot_records")
    if max_records is not None:
        records = records[: int(max_records)]
    _pseudo_pair_manifest_validation(config, records).raise_for_errors()
    return records


def _pseudo_pair_manifest_validation(config: Mapping[str, Any], records: list[Any]):
    return validate_pseudo_pair_manifest_records(
        records,
        sequence=_pseudo_pair_sequence(config),
        target_fields=_pseudo_pair_target_fields(config),
    )


def _pseudo_pair_preprocessing(config: Mapping[str, Any]) -> SlicePreprocessingSpec:
    data_config = _data_mapping(config)
    preprocessing = data_config.get("preprocessing", {})
    return SlicePreprocessingSpec.from_mapping(preprocessing)


def _pseudo_pair_sequence(config: Mapping[str, Any]) -> str:
    return str(_data_mapping(config).get("sequence", "T2-FLAIR"))


def _pseudo_pair_source_field(config: Mapping[str, Any]) -> float:
    return float(_data_mapping(config).get("source_field", 0.1))


def _pseudo_pair_target_fields(config: Mapping[str, Any]) -> tuple[float, ...]:
    raw = _data_mapping(config).get("target_fields", (1.5, 3.0, 5.0, 7.0))
    return tuple(float(field) for field in raw)


def _pseudo_pair_cache_size(config: Mapping[str, Any]) -> int:
    return int(_data_mapping(config).get("cache_size", 2))


def _pseudo_pair_checkpoint_dir(config: Mapping[str, Any]) -> Path | None:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        return None
    value = training.get("checkpoint_dir")
    return Path(os.path.expandvars(str(value))) if value else None


def _pseudo_pair_split_path(config: Mapping[str, Any], *, checkpoint: Path | None = None) -> Path:
    data_config = _data_mapping(config)
    configured = data_config.get("split_json")
    if configured:
        return Path(os.path.expandvars(str(configured)))
    checkpoint_dir = _pseudo_pair_checkpoint_dir(config)
    if checkpoint_dir is not None:
        return checkpoint_dir / "volume_splits.json"
    if checkpoint is not None:
        return checkpoint.parent / "volume_splits.json"
    return Path("volume_splits.json")


def _build_or_load_pseudo_pair_splits(config: Mapping[str, Any], records: list[Any], path: Path):
    if path.exists():
        return load_volume_splits(path)
    data_config = _data_mapping(config)
    return build_volume_splits(
        records,
        sequence=_pseudo_pair_sequence(config),
        target_fields=_pseudo_pair_target_fields(config),
        train_volumes_per_field=int(data_config.get("train_volumes_per_field", 16)),
        val_volumes_per_field=int(data_config.get("val_volumes_per_field", 4)),
        test_volumes_per_field=int(data_config.get("test_volumes_per_field", 4)),
        seed=int(config.get("seed", 13)),
    )


def _build_pseudo_pair_translator(config: Mapping[str, Any]):
    model_config = _model_config(config)
    nested = model_config.get("translator")
    if isinstance(nested, Mapping):
        translator_config = dict(nested)
        default_name = str(model_config.get("name", "conditional_unet_field_translator"))
    else:
        translator_config = {
            key: value
            for key, value in model_config.items()
            if key not in {"encoder", "decoder", "translator", "variant"}
        }
        default_name = "conditional_unet_field_translator"
    name = str(translator_config.pop("name", default_name))
    return build_translator(name, **translator_config)


def _print_pseudo_pair_summary(
    summary: Mapping[str, Any],
    *,
    batch_size: int,
    steps_per_epoch: int,
) -> None:
    train_summary = summary["splits"]["train"]
    print(
        "pseudo_pair split summary: "
        f"train_volumes={train_summary['volumes']} train_slices={train_summary['slices']} "
        f"batch_size={batch_size} steps_per_epoch={steps_per_epoch}",
        file=sys.stderr,
        flush=True,
    )


def _pseudo_pair_preflight_payload(
    *,
    manifest_path: Path,
    split_path: Path,
    split_sha256: str,
    split_summary: Mapping[str, Any],
    leakage_audit: Mapping[str, Any],
    preprocessing: SlicePreprocessingSpec,
    train_dataset: PseudoPairSliceDataset,
    val_dataset: PseudoPairSliceDataset,
    test_dataset: PseudoPairSliceDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int,
    num_workers: int,
    manifest_validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "manifest": str(manifest_path),
        "manifest_validation": dict(manifest_validation),
        "split_json": str(split_path),
        "split_sha256": split_sha256,
        "leakage_audit": dict(leakage_audit),
        "split_summary": split_summary,
        "datasets": {
            "train": _pseudo_pair_dataset_summary(train_dataset),
            "validation": _pseudo_pair_dataset_summary(val_dataset),
            "test": _pseudo_pair_dataset_summary(test_dataset),
        },
        "batch_size": batch_size,
        "num_workers": num_workers,
        "steps_per_epoch": len(train_loader),
        "validation_batches": len(val_loader),
        "preprocessing": {
            **preprocessing.to_dict(),
            "raw_volume_order": "C,X,Y,Z",
            "slice_plane": _pseudo_pair_slice_plane(preprocessing),
            "selected_slice_indices": list(selected_slice_indices(preprocessing)),
        },
    }


def _pseudo_pair_slice_plane(preprocessing: SlicePreprocessingSpec) -> str:
    if preprocessing.slice_axis == "x":
        return "Y,Z"
    if preprocessing.slice_axis == "y":
        return "X,Z"
    return "X,Y"


def _pseudo_pair_dataset_summary(dataset: PseudoPairSliceDataset) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(dataset),
        "volumes": len(dataset.records),
        "slices_per_volume": len(dataset.slice_indices),
        "fields": {},
    }
    for record in dataset.records:
        field_label = f"{record.domain.field_strength_t:g}T"
        summary["fields"][field_label] = int(summary["fields"].get(field_label, 0)) + len(dataset.slice_indices)
    if len(dataset) == 0:
        return summary
    sample = dataset[0]
    x_high_01 = from_model_range(sample.x_high, dataset.preprocessing.model_range)
    x_low_01 = from_model_range(sample.x_low, dataset.preprocessing.model_range)
    summary["sample"] = {
        "record_id": sample.record_id,
        "target_domain": sample.target_domain.label,
        "source_domain": sample.source_domain.label,
        "slice_index": sample.slice_index,
        "x_high_shape": list(sample.x_high.shape),
        "x_low_shape": list(sample.x_low.shape),
        "mask_shape": list(sample.mask.shape),
        "x_high_01_min": float(x_high_01.min().item()),
        "x_high_01_max": float(x_high_01.max().item()),
        "x_low_01_min": float(x_low_01.min().item()),
        "x_low_01_max": float(x_low_01.max().item()),
        "geometry": sample.geometry.to_dict(),
    }
    return summary


def _data_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("Config section 'data' must be a mapping.")
    return data_config


def _load_optional_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_yaml_config(path)
    return {}


def _load_variant_a_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    if config.get("contract") != PHOTOMETRY_FACTORIZATION_CONFIG_VERSION:
        raise ValueError("Variant-A config contract is incompatible.")
    if config.get("variant") != "A":
        raise ValueError("Variant-A commands accept variant=A only.")
    if config.get("scope") != "frozen_photometry_and_factorized_identity_only":
        raise ValueError("Variant-A config scope is incompatible.")
    photometry = config.get("photometry")
    if not isinstance(photometry, Mapping):
        raise ValueError("Variant-A config requires a photometry mapping.")
    expected = {
        "artifact_contract": PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
        "semantics": PHOTOMETRY_FACTORIZATION_SEMANTICS,
        "num_quantiles": 256,
        "canonical_scope": "contrast",
        "contrasts": ["T1w", "T2w", "T2-FLAIR"],
        "fields_t": [0.1, 1.5, 3.0, 5.0, 7.0],
        "within_domain_weighting": "equal_per_eligible_volume",
        "across_fields_weighting": "exactly_equal_0.2",
        "interpolation": PHOTOMETRY_INTERPOLATION_RULE,
        "duplicate_knot_rule": PHOTOMETRY_DUPLICATE_KNOT_RULE,
        "clamping": PHOTOMETRY_CLAMPING_RULE,
        "support": PHOTOMETRY_SUPPORT_POLICY,
        "runtime_statistics": "none",
    }
    mismatches = {
        key: {"expected": value, "actual": photometry.get(key)}
        for key, value in expected.items()
        if photometry.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Variant-A config changes approved photometry defaults: {mismatches}.")
    eligibility = config.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise ValueError("Variant-A config requires an eligibility mapping.")
    if eligibility.get("fit") != {"cohort": "R", "split": "train"}:
        raise ValueError("Variant-A fit eligibility must be exactly R/train.")
    if eligibility.get("qualification") != {"cohort": "R", "split": "validation"}:
        raise ValueError("Variant-A qualification eligibility must be exactly R/validation.")
    if eligibility.get("prospective_records") != "reject":
        raise ValueError("Variant-A prospective records must be rejected.")
    if tuple(eligibility.get("forbidden_travellers", ())) != tuple(
        sorted(FORBIDDEN_TRAVELLER_IDS)
    ):
        raise ValueError("Variant-A must explicitly reject travellers 0006, 0007, and 0009.")
    thresholds = VariantAQualificationThresholds.from_mapping(config)
    if thresholds.to_dict() != VariantAQualificationThresholds().to_dict():
        raise ValueError(
            "Variant-A v1 qualification thresholds must match the reviewed proposed defaults."
        )
    qualification = config.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("Variant-A config requires a qualification mapping.")
    if qualification.get("threshold_status") != (
        "versioned_proposed_defaults_requiring_guillermo_simon_signoff"
    ):
        raise ValueError("Variant-A threshold status is incompatible.")
    if qualification.get("failure_classes") != [
        "photometry_factorization_failure",
        "canonical_vae_distribution_shift_failure",
    ]:
        raise ValueError("Variant-A qualification failure classes are incompatible.")
    if qualification.get("canonical_latent_bank_authorization") != (
        "require_all_checks_pass"
    ):
        raise ValueError("Variant-A latent-bank authorization gate is incompatible.")
    if qualification.get("gate01_substitution") != "forbidden":
        raise ValueError("Variant-A must forbid Gate 0.1 substitution.")
    if config.get("vae_preflight") != {
        "encoder_statistic": "posterior_mean",
        "decode_strategy": "full",
        "output_clamp": [0.0, 1.0],
        "comparison": "same_retrospective_validation_record",
    }:
        raise ValueError("Variant-A frozen-VAE preflight is incompatible.")
    reject_target_or_prediction_derived_fields(config)
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Variant-A config requires an evaluation mapping.")
    if evaluation.get("result_contract") != (
        "stage2-photometry-dual-baseline-result-v1"
    ) or evaluation.get("continuity_reference_contract") != (
        "stage2-photometry-continuity-reference-v1"
    ):
        raise ValueError("Variant-A evaluation contracts are incompatible.")
    methods = evaluation.get("methods")
    if methods != [
        "fixed_map_factorized_identity",
        "gate01_posthoc_calibrated_identity",
        "raw_identity",
        "stage1_reconstruction_ceiling",
    ]:
        raise ValueError("Variant-A dual-baseline method identities are incompatible.")
    return config


def _variant_a_record_cohort(record: VolumeRecord) -> str:
    prefix = record.metadata.get("prefix") if isinstance(record.metadata, Mapping) else None
    if prefix is None:
        case_id = str(record.case_id)
        prefix = case_id.split("_", 1)[0] if "_" in case_id else ""
    cohort = str(prefix).upper()
    if cohort not in {"R", "P"}:
        raise ValueError(
            f"Variant-A record {record.case_id!r} has unknown cohort prefix {prefix!r}."
        )
    return cohort


def _nonnegative_log_every(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("--log-every must be non-negative.")
    return result


def _load_variant_a_npy(path: Path) -> "Any":
    import numpy as np
    import torch

    if path.suffix.lower() != ".npy":
        raise ValueError("Variant-A baseline evaluation accepts external .npy arrays only.")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load Variant-A array {path}: {exc}") from exc
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Variant-A array {path} must use a floating dtype.")
    if array.ndim < 3 or not np.isfinite(array).all():
        raise ValueError(f"Variant-A array {path} must be a finite full volume.")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _data_patch_size(config: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Configured training patch size, or None if unset (train on whole volumes — used by
    the small no-crop manifests in tests; real configs always set data.patch_size)."""

    data_config = config.get("data", {})
    patch_size = data_config.get("patch_size") if isinstance(data_config, Mapping) else None
    if patch_size is None:
        return None
    return tuple(int(p) for p in patch_size)


def _eval_patch_size(config: Mapping[str, Any]) -> tuple[int, int, int]:
    data_config = config.get("data", {})
    patch_size = data_config.get("patch_size") if isinstance(data_config, Mapping) else None
    if patch_size is None:
        patch_size = [64, 64, 64]
    return tuple(int(p) for p in patch_size)  # type: ignore[return-value]


def _patches_per_volume(config: Mapping[str, Any]) -> int:
    data_config = config.get("data", {})
    value = data_config.get("patches_per_volume") if isinstance(data_config, Mapping) else None
    return int(value) if value is not None else 1


def _crop_config(config: Mapping[str, Any]) -> StratifiedCropConfig | None:
    """`data.stratified_crop` -> StratifiedCropConfig, or None to keep uniform cropping."""

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return None
    return StratifiedCropConfig.from_mapping(data_config.get("stratified_crop"))


def _foreground_threshold(config: Mapping[str, Any]) -> float:
    """Intensity above which a voxel counts as foreground for stratified cropping.

    0.0 on the official [0, 1] data: background is exactly 0, so `> 0` is the mask.
    """

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return 0.0
    value = data_config.get("foreground_threshold")
    return float(value) if value is not None else 0.0


def _field_sampling_weights(
    config: Mapping[str, Any], records: "Sequence[VolumeRecord]"
) -> list[float] | None:
    """`data.field_balance` -> per-record streaming sampling weights, or None for uniform order.

    Config shape (default absent => uniform `randperm`, unchanged):

        data:
          field_balance:
            enabled: true
            mode: inverse_frequency          # equalize the 5 fields per pass (default)
            # boost_by_field: {5.0: 2.0}     # used only when mode: boost (explicit multipliers)

    `inverse_frequency` derives the ratios from the split's own field counts (no magic numbers);
    `boost` takes an explicit multiplier map via `domain_oversampling_weights`.
    """

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return None
    spec = data_config.get("field_balance")
    if not isinstance(spec, Mapping) or not spec.get("enabled", False):
        return None
    mode = str(spec.get("mode", "inverse_frequency"))
    if mode == "inverse_frequency":
        return field_balanced_weights(records)
    if mode == "boost":
        boost = {float(k): float(v) for k, v in dict(spec.get("boost_by_field", {})).items()}
        return domain_oversampling_weights(records, boost_by_field=boost)
    raise ValueError(
        f"data.field_balance.mode must be 'inverse_frequency' or 'boost', got {mode!r}."
    )


def _joint_domain_balance_enabled(config: Mapping[str, Any]) -> bool:
    """Whether the additive 15-way domain/subject-balanced schedule is enabled."""

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return False
    spec = data_config.get("joint_domain_balance")
    if spec is None:
        return False
    if isinstance(spec, bool):
        return spec
    if not isinstance(spec, Mapping):
        raise ValueError("data.joint_domain_balance must be a boolean or mapping.")
    mode = str(spec.get("mode", "field_contrast_subject"))
    if mode != "field_contrast_subject":
        raise ValueError(
            "data.joint_domain_balance.mode must be 'field_contrast_subject', "
            f"got {mode!r}."
        )
    return bool(spec.get("enabled", False))


def _steps_per_epoch(config: Mapping[str, Any], manifest_path: Path) -> int:
    """ceil(num_volumes * patches_per_volume / batch_size). Reads only manifest metadata
    (no image arrays), so it's cheap even though the loader re-reads it."""
    return _steps_per_epoch_for_volumes(config, len(load_manifest(manifest_path).records))


def _steps_per_epoch_for_volumes(config: Mapping[str, Any], num_volumes: int) -> int:
    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 2)) if isinstance(training, Mapping) else 2
    patches_per_volume = _patches_per_volume(config)
    batch_size = max(1, batch_size)
    workers = max(0, _num_workers(config))
    if workers <= 1:
        return max(1, -(-(num_volumes * patches_per_volume) // batch_size))

    # IterableDataset automatic batching is performed independently in each worker.
    # Account for each worker's possible final partial batch; using one global ceil can
    # declare an epoch early and then skip prefetched exposures on iterator restart.
    base, remainder = divmod(int(num_volumes), workers)
    batches = 0
    for worker_index in range(workers):
        worker_volumes = base + (1 if worker_index < remainder else 0)
        if worker_volumes:
            batches += -(-(worker_volumes * patches_per_volume) // batch_size)
    return max(1, batches)


def _resolve_stage1_endpoint(
    config: dict[str, Any],
    *,
    steps_per_epoch: int,
    cli_epochs: int | None,
    cli_steps: int | None,
) -> None:
    """Resolve one absolute Stage-1 endpoint after the pass length is known.

    Precedence is intentionally strict: ``--epochs`` wins, then
    ``training.epochs``, then an explicit CLI/config step endpoint.  Epoch endpoints
    always resolve to complete passes.  Step endpoints are accepted only when they also
    end on a pass boundary for validation-enabled v3 runs.
    """

    if steps_per_epoch <= 0:
        raise ValueError("Stage-1 steps_per_epoch must be positive before endpoint resolution.")
    training = config.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training config must be a mapping.")

    source: str
    epochs_value: int | None
    if cli_epochs is not None:
        epochs_value = int(cli_epochs)
        source = "cli_epochs"
    elif training.get("epochs") is not None:
        epochs_value = int(training["epochs"])
        source = "config_epochs"
    else:
        epochs_value = None
        source = "cli_steps" if cli_steps is not None else "config_steps"

    if epochs_value is not None:
        if epochs_value <= 0:
            raise ValueError("Stage-1 epochs must be positive.")
        endpoint_steps = epochs_value * int(steps_per_epoch)
        training["epochs"] = epochs_value
    else:
        raw_steps = cli_steps if cli_steps is not None else training.get("steps")
        if raw_steps is None:
            raise ValueError(
                "Stage-1 requires training.epochs, --epochs, training.steps, or --steps."
            )
        endpoint_steps = int(raw_steps)
        if endpoint_steps <= 0:
            raise ValueError("Stage-1 steps must be positive.")

    validation_enabled = bool(
        training.get("require_all_validation_domains", False)
        or _joint_domain_balance_enabled(config)
    )
    if validation_enabled and endpoint_steps % int(steps_per_epoch) != 0:
        raise ValueError(
            "Validation-enabled Stage-1 endpoints must contain complete epochs: "
            f"steps={endpoint_steps}, steps_per_epoch={steps_per_epoch}."
        )
    if validation_enabled and endpoint_steps < int(steps_per_epoch):
        raise ValueError(
            "Validation-enabled Stage-1 endpoint cannot reach the first epoch boundary: "
            f"steps={endpoint_steps}, steps_per_epoch={steps_per_epoch}."
        )

    training["steps"] = endpoint_steps
    training["endpoint_total_steps"] = endpoint_steps
    training["endpoint_source"] = source
    training["resolved_epochs"] = endpoint_steps // int(steps_per_epoch)


def _preflight_stage1_v3_records(
    config: Mapping[str, Any], splits: Any | None
) -> None:
    """Fail closed on absent or incomplete 15-domain train/validation splits."""

    training = config.get("training", {})
    require_validation = bool(
        isinstance(training, Mapping)
        and training.get("require_all_validation_domains", False)
    )
    joint_balance = _joint_domain_balance_enabled(config)
    if not (require_validation or joint_balance):
        return
    if splits is None:
        raise ValueError(
            "Stage-1 v3 requires a verified --split-json with train and validation records."
        )

    expected = {domain.label for domain in ALL_DOMAINS}
    for split_name in ("train", "validation"):
        records = tuple(getattr(splits, split_name, ()))
        if not records:
            raise ValueError(f"Stage-1 v3 {split_name} split is empty.")
        observed = {record.domain.label for record in records}
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise ValueError(
                f"Stage-1 v3 {split_name} split must contain all 15 domains; "
                f"missing={missing}, unexpected={unexpected}."
            )


def _resolve_stage1_epoch_dependent_schedules(
    config: dict[str, Any], *, steps_per_epoch: int
) -> None:
    """Resolve warm-up and scheduler lengths after the final epoch endpoint."""

    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training config must be a mapping.")
    if "kl_warmup_epochs" in training:
        _override(
            config,
            "training",
            "kl_warmup_steps",
            int(training["kl_warmup_epochs"]) * int(steps_per_epoch),
        )
    training = config.get("training", {})
    if (
        isinstance(training, Mapping)
        and str(training.get("lr_schedule", "constant")) == "cosine"
        and int(training.get("lr_cosine_total_steps", 0)) == 0
    ):
        _override(
            config,
            "training",
            "lr_cosine_total_steps",
            int(training.get("endpoint_total_steps", training.get("steps", 0))),
        )


def _load_loss_curve(path: Path | None) -> list[float] | None:
    # The loss curve is a nice-to-have overlay: an empty/malformed metrics file (e.g. a
    # training run whose stdout never reached the redirect) must not crash the eval.
    if path is None or not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"warning: could not parse loss curve from {path} (invalid JSON); skipping.")
        return None
    losses = data.get("losses") if isinstance(data, Mapping) else None
    return [float(x) for x in losses] if isinstance(losses, list) else None


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("--checkpoint must use LABEL=PATH syntax.")
    return label.strip(), Path(os.path.expandvars(raw_path.strip()))


def _num_workers(config: Mapping[str, Any]) -> int:
    training = config.get("training", {})
    if isinstance(training, Mapping) and "num_workers" in training:
        return int(training["num_workers"])
    return 0


def _prefetch_factor(config: Mapping[str, Any]) -> int:
    """DataLoader read-ahead per worker (only used when num_workers>0)."""
    training = config.get("training", {})
    if isinstance(training, Mapping) and "prefetch_factor" in training:
        return int(training["prefetch_factor"])
    return 4


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("model", {})
    if not isinstance(value, Mapping):
        raise ValueError("Config section 'model' must be a mapping.")
    return dict(value)


def _component_config(
    model_config: Mapping[str, Any],
    component: str,
    default_name: str,
) -> tuple[str, dict[str, Any]]:
    nested = model_config.get(component, {})
    if nested is None:
        nested_config: dict[str, Any] = {}
    elif isinstance(nested, Mapping):
        nested_config = dict(nested)
    else:
        raise ValueError(f"model.{component} must be a mapping when provided.")

    name = str(nested_config.pop("name", default_name))
    top_level_kwargs = _top_level_component_kwargs(model_config, component, default_name)

    if component in {"encoder", "decoder"} and default_name != "identity":
        return name, {**top_level_kwargs, **nested_config}
    if component == "translator" and default_name == "identity":
        return name, {**top_level_kwargs, **nested_config}
    return name, nested_config


def _top_level_component_kwargs(
    model_config: Mapping[str, Any],
    component: str,
    default_name: str,
) -> dict[str, Any]:
    top_level = {
        key: value
        for key, value in model_config.items()
        if key not in {"name", "variant", "encoder", "decoder", "translator"}
    }
    if default_name == "cnn_autoencoder" and component in {"encoder", "decoder"}:
        shared_keys = {"spatial_dims", "hidden_channels", "latent_channels", "activation", "use_norm"}
        encoder_keys = shared_keys | {"in_channels"}
        decoder_keys = shared_keys | {"out_channels", "final_activation"}
        allowed = encoder_keys if component == "encoder" else decoder_keys
        return {key: value for key, value in top_level.items() if key in allowed}
    return top_level


_KL_VAE_SHARED_KEYS = {
    "base_channels",
    "latent_channels",
    "spatial_dims",
    "activation",
    "use_norm",
    "num_res_blocks",
}


def _resolve_device(choice: str) -> "Any":
    import torch

    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    use_cuda = choice == "cuda" or (choice == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def _kl_vae_kwargs(model_config: Mapping[str, Any], component: str) -> dict[str, Any]:
    kwargs = {key: value for key, value in model_config.items() if key in _KL_VAE_SHARED_KEYS}
    if component == "encoder" and "in_channels" in model_config:
        kwargs["in_channels"] = model_config["in_channels"]
    if component == "decoder" and "out_channels" in model_config:
        kwargs["out_channels"] = model_config["out_channels"]
    # Decoder-only: the encoder has no output head.
    if component == "decoder" and "output_activation" in model_config:
        kwargs["output_activation"] = model_config["output_activation"]
    if component == "decoder" and "domain_conditioning_dim" in model_config:
        kwargs["domain_conditioning_dim"] = model_config["domain_conditioning_dim"]
    return kwargs


def _override(config: dict[str, Any], section: str, key: str, value: Any | None) -> None:
    if value is None:
        return
    section_config = config.setdefault(section, {})
    if not isinstance(section_config, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    section_config[key] = value


def _override_nested(
    config: dict[str, Any],
    section: str,
    nested: str,
    key: str,
    value: Any | None,
) -> None:
    if value is None:
        return
    section_config = config.setdefault(section, {})
    if not isinstance(section_config, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    nested_config = section_config.setdefault(nested, {})
    if not isinstance(nested_config, dict):
        raise ValueError(f"Config section {section}.{nested} must be a mapping.")
    nested_config[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
