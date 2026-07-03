"""Command-line interface for FieldBridge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from fieldbridge.config import dump_yaml_config, load_yaml_config
from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import ImageTransform, ManifestVolumeDataset, collate_raw_batches
from fieldbridge.data.manifests import audit_manifest, load_manifest
from fieldbridge.data.sources import nifti_image_loader
from fieldbridge.data.transforms import compose, normalize_percentile_clip_to_unit_range, random_crop
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
        required=True,
        help="Manifest of real NIfTI volumes (requires the 'nifti' extra). No synthetic "
        "fallback for this stage — never commit a real manifest to the repo.",
    )
    train_stage1_vae.add_argument("--steps", type=int, default=None)
    train_stage1_vae.add_argument("--batch-size", type=int, default=None)
    train_stage1_vae.add_argument("--seed", type=int, default=None)
    train_stage1_vae.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

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
        loader = _build_manifest_loader(args.manifest, batch_size=loop_config.batch_size) if args.manifest else None
        result = run_train_loop(loop_config, encoder=encoder, decoder=decoder, translator=translator, loader=loader)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"train completed: steps={result.steps} final_loss={result.final_loss:.6f}")
        return 0

    if args.command == "train-stage1-vae":
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        if args.seed is not None:
            config["seed"] = args.seed
        model_config = _model_config(config)
        encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model_config, "encoder"))
        decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model_config, "decoder"))
        stage_config = Stage1VAEConfig.from_mapping(config)
        loader = _build_manifest_loader(
            args.manifest, batch_size=stage_config.batch_size, transform=_manifest_transform(config)
        )
        result = run_stage1_vae_train(stage_config, encoder=encoder, decoder=decoder, loader=loader)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"train-stage1-vae completed: steps={result.steps} final_loss={result.final_loss:.6f}")
        return 0

    if args.command == "train-stage2-diffuser":
        config = _load_optional_config(args.config)
        _override(config, "training", "steps", args.steps)
        _override(config, "training", "batch_size", args.batch_size)
        if args.seed is not None:
            config["seed"] = args.seed
        model_config = _model_config(config)
        unet = DenoisingUNet(**{key: value for key, value in model_config.items() if key != "name"})
        vae_model_config = config.get("vae_model", {})
        if not isinstance(vae_model_config, Mapping):
            raise ValueError("Config section 'vae_model' must be a mapping.")
        encoder = build_encoder("kl_vae", **dict(vae_model_config))
        stage_config = Stage2DiffuserConfig.from_mapping(config)
        loader = _build_manifest_loader(
            args.manifest, batch_size=stage_config.batch_size, transform=_manifest_transform(config)
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
) -> "DataLoader[RawBatch]":
    manifest = load_manifest(manifest_path)
    dataset = ManifestVolumeDataset(manifest.records, image_loader=nifti_image_loader, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_batches)


def _load_optional_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_yaml_config(path)
    return {}


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


def _manifest_transform(config: Mapping[str, Any]) -> ImageTransform:
    """Percentile-clip normalization, plus a random spatial patch crop if
    data.patch_size is set in the config — required for full 3D volumes at typical
    resolutions (e.g. 364x436x364), where decoding back toward full resolution with
    enough latent channels for the diffuser OOMs on essentially any GPU otherwise. See
    random_crop's docstring.
    """

    data_config = config.get("data", {})
    patch_size = data_config.get("patch_size") if isinstance(data_config, Mapping) else None
    if patch_size is None:
        return normalize_percentile_clip_to_unit_range
    return compose([normalize_percentile_clip_to_unit_range, partial(random_crop, patch_size=patch_size)])


_KL_VAE_SHARED_KEYS = {"base_channels", "latent_channels", "spatial_dims", "activation", "use_norm"}


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
