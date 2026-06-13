"""Command-line interface for CLB-Field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clbfield.config import dump_yaml_config, load_yaml_config
from clbfield.data.manifests import audit_manifest, load_manifest
from clbfield.official.mrixfields2026 import spec_as_dict
from clbfield.official.submissions import (
    build_submission_zip,
    validate_submission_dir,
)
from clbfield.training.smoke_train import SmokeTrainConfig, run_smoke_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clbfield")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-train", help="Run synthetic CPU smoke training.")
    smoke.add_argument("--config", type=Path, default=Path("configs/experiment/smoke.yaml"))
    smoke.add_argument("--steps", type=int, default=None)
    smoke.add_argument("--batch-size", type=int, default=None)
    smoke.add_argument("--seed", type=int, default=None)
    smoke.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

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

    raise ValueError(f"Unknown command: {args.command}")


def _load_optional_config(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_yaml_config(path)
    return {}


def _override(config: dict[str, Any], section: str, key: str, value: Any | None) -> None:
    if value is None:
        return
    section_config = config.setdefault(section, {})
    if not isinstance(section_config, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    section_config[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
