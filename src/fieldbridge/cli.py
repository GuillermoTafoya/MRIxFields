"""Command-line interface for FieldBridge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from fieldbridge.config import dump_yaml_config, load_yaml_config
from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import (
    ImageTransform,
    ManifestVolumeDataset,
    StreamingPatchDataset,
    collate_raw_batches,
)
from fieldbridge.data.patch_bank import (
    PatchBankDataset,
    build_patch_bank,
    patch_bank_size,
)
from fieldbridge.data.manifests import audit_manifest, load_manifest
from fieldbridge.data.sources import nifti_image_loader
from fieldbridge.data.transforms import normalize_percentile_clip_to_unit_range
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
from fieldbridge.evaluation.stage1_report import run_stage1_eval
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.smoke_train import SmokeTrainConfig, run_smoke_train
from fieldbridge.training.stage1_vae import Stage1VAEConfig, run_stage1_vae_train
from fieldbridge.training.stage2_diffuser import Stage2DiffuserConfig, run_stage2_diffuser_train
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

    train_stage1_vae = subparsers.add_parser(
        "train-stage1-vae", help="Run Etapa 1 VAE-only training (KLVAEEncoder/Decoder, SSIM+nRMSE+LPIPS+KL)."
    )
    train_stage1_vae.add_argument("--config", type=Path, default=Path("configs/experiment/stage1_vae.yaml"))
    train_stage1_vae.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest of real NIfTI volumes (requires the 'nifti' extra). Required unless "
        "--patch-bank is given. No synthetic fallback for this stage — never commit a real "
        "manifest to the repo.",
    )
    train_stage1_vae.add_argument("--steps", type=int, default=None)
    train_stage1_vae.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of full passes over the manifest. Overrides --steps (steps = "
        "epochs * ceil(num_volumes * patches_per_volume / batch_size)).",
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

    eval_stage1_vae = subparsers.add_parser(
        "eval-stage1-vae",
        help="Deterministic reconstruction eval + diagnostic plots for a stage-1 VAE checkpoint.",
    )
    eval_stage1_vae.add_argument("--checkpoint", type=Path, required=True, help="Trained VAE checkpoint (.pt).")
    eval_stage1_vae.add_argument("--config", type=Path, default=Path("configs/experiment/stage1_vae.yaml"))
    eval_stage1_vae.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest of real NIfTI volumes to reconstruct (requires the 'nifti' extra).",
    )
    eval_stage1_vae.add_argument("--out", type=Path, required=True, help="Output directory for metrics + plots.")
    eval_stage1_vae.add_argument("--num-samples", type=int, default=4)
    eval_stage1_vae.add_argument(
        "--per-domain",
        action="store_true",
        help="Reconstruct one volume per distinct field strength (0.1T..7T) instead of the "
        "first N in manifest order.",
    )
    eval_stage1_vae.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding-window overlap fraction in [0, 1) (default 0.5). Overlap + Hann "
        "blending removes the panel seams from non-overlapping tiles.",
    )
    eval_stage1_vae.add_argument(
        "--metrics-raw",
        type=Path,
        default=None,
        help="Optional metrics_raw.json from training to also render the loss curve.",
    )

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

    if args.command == "train-stage1-vae":
        if args.patch_bank is None and args.manifest is None:
            raise ValueError("train-stage1-vae requires --manifest, or --patch-bank to train from a prebuilt bank.")
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        _override(config, "data", "patches_per_volume", args.patches_per_volume)
        # Compute steps_per_epoch so the loop can log epoch/step-in-epoch (both loaders are
        # length-less to the config). --epochs, if given, sets steps from it.
        if args.patch_bank is not None:
            num_volumes, ppv = patch_bank_size(args.patch_bank)
            batch_size = int(config.get("training", {}).get("batch_size", 2))
            steps_per_epoch = max(1, -(-num_volumes * ppv // max(1, batch_size)))
        else:
            steps_per_epoch = _steps_per_epoch(config, args.manifest)
        _override(config, "training", "steps_per_epoch", steps_per_epoch)
        if args.epochs is not None:
            _override(config, "training", "steps", args.epochs * steps_per_epoch)
        if args.seed is not None:
            config["seed"] = args.seed
        model_config = _model_config(config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        stage_config = Stage1VAEConfig.from_mapping(config)
        if args.patch_bank is not None:
            loader = DataLoader(
                PatchBankDataset(args.patch_bank),
                batch_size=stage_config.batch_size,
                shuffle=True,
                num_workers=_num_workers(config),
                collate_fn=collate_raw_batches,
            )
        else:
            loader = _build_streaming_patch_loader(
                args.manifest,
                batch_size=stage_config.batch_size,
                config=config,
                num_workers=_num_workers(config),
            )
        result = run_stage1_vae_train(stage_config, encoder=encoder, decoder=decoder, loader=loader)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"train-stage1-vae completed: steps={result.steps} final_loss={result.final_loss:.6f}")
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

    if args.command == "eval-stage1-vae":
        import torch

        config = _load_optional_config(args.config)
        model_config = _model_config(config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        state = load_checkpoint(args.checkpoint)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        # Full-volume reconstruction: normalize to [-1, 1] like training, but NO random
        # crop (the sliding window in run_stage1_eval tiles the whole volume itself).
        loader = _build_manifest_loader(
            args.manifest,
            batch_size=1,
            transform=normalize_percentile_clip_to_unit_range,
            shuffle=False,
        )
        patch_size = _eval_patch_size(config)
        loss_curve = _load_loss_curve(args.metrics_raw)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = run_stage1_eval(
            encoder=encoder,
            decoder=decoder,
            loader=loader,
            patch_size=patch_size,
            out_dir=args.out,
            num_samples=args.num_samples,
            per_domain=args.per_domain,
            overlap=args.overlap,
            device=device,
            lpips_num_slices=int(config.get("training", {}).get("lpips_num_slices", 8))
            if isinstance(config.get("training", {}), Mapping)
            else 8,
            loss_curve=loss_curve,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
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

    raise ValueError(f"Unknown command: {args.command}")


def _build_manifest_loader(
    manifest_path: Path,
    *,
    batch_size: int,
    transform: ImageTransform | None = normalize_percentile_clip_to_unit_range,
    shuffle: bool = False,
    num_workers: int = 0,
) -> "DataLoader[RawBatch]":
    # shuffle defaults to False so non-training callers (audits, eval) keep manifest order;
    # training paths pass shuffle=True — the previous fixed-order loader meant a short run
    # only ever saw the first N records.
    manifest = load_manifest(manifest_path)
    dataset = ManifestVolumeDataset(manifest.records, image_loader=nifti_image_loader, transform=transform)
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

    manifest = load_manifest(manifest_path)
    dataset = StreamingPatchDataset(
        manifest.records,
        image_loader=nifti_image_loader,
        patch_size=_data_patch_size(config),
        patches_per_volume=_patches_per_volume(config),
        volume_transform=normalize_percentile_clip_to_unit_range,
        seed=int(config.get("seed", 0)) if isinstance(config.get("seed", 0), int) else 0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_raw_batches,
    )


def _load_optional_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_yaml_config(path)
    return {}


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


def _steps_per_epoch(config: Mapping[str, Any], manifest_path: Path) -> int:
    """ceil(num_volumes * patches_per_volume / batch_size). Reads only manifest metadata
    (no image arrays), so it's cheap even though the loader re-reads it."""
    num_volumes = len(load_manifest(manifest_path).records)
    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 2)) if isinstance(training, Mapping) else 2
    patches = num_volumes * _patches_per_volume(config)
    return max(1, -(-patches // max(1, batch_size)))


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


def _num_workers(config: Mapping[str, Any]) -> int:
    training = config.get("training", {})
    if isinstance(training, Mapping) and "num_workers" in training:
        return int(training["num_workers"])
    return 0


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


_KL_VAE_SHARED_KEYS = {"base_channels", "latent_channels", "spatial_dims", "activation", "use_norm", "num_res_blocks"}


def _kl_vae_kwargs(model_config: Mapping[str, Any], component: str) -> dict[str, Any]:
    kwargs = {key: value for key, value in model_config.items() if key in _KL_VAE_SHARED_KEYS}
    if component == "encoder" and "in_channels" in model_config:
        kwargs["in_channels"] = model_config["in_channels"]
    if component == "decoder" and "out_channels" in model_config:
        kwargs["out_channels"] = model_config["out_channels"]
    return kwargs


def _override(config: dict[str, Any], section: str, key: str, value: Any | None) -> None:
    if value is None:
        return
    section_config = config.setdefault(section, {})
    if not isinstance(section_config, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    section_config[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
