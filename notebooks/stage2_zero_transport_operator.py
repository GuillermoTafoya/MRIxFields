"""Run the three retrospective zero-transport controls in a pinned A100 Colab checkout."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
from itertools import count
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.latent_bank import load_volume
from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentRecord,
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factored_latent_bank import (
    FACTORED_LATENT_BANK_MANIFEST,
    FACTORED_LATENT_STATS_FILE,
)
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json
from fieldbridge.data.stage2_canonical_volume import safe_relative_path
from fieldbridge.evaluation.stage2_step200_inference_audit import (
    _kl_vae_kwargs,
    _require_linear_decoder_output_activation,
    preflight_frozen_stage1_run_c_config,
    preflight_reviewed_photometry_namespace_artifact,
    verify_frozen_stage1_vae_bank_provenance,
    verify_reviewed_photometry_bank_provenance,
)
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import (
    copy_verified_stage2_bank_tar_to_local,
)
from fieldbridge.evaluation.zero_transport_diagnostic import run_zero_transport_scan
from fieldbridge.models.factory import build_decoder, build_encoder
from fieldbridge.official.mrixfields2026 import parse_mrixfields_filename
from fieldbridge.training.checkpoints import load_checkpoint

DRIVE = Path("/content/drive/MyDrive/MRIxFields2026")
OUTPUT_ROOT = DRIVE / "UnifiedStage2_1ca2b4a_01"
EXPERIMENT = "stage2_zero_transport_v1"
BANK_ARCHIVE = OUTPUT_ROOT / "photometry_factored_latent_bank_v2.tar"
BANK_ARCHIVE_SHA256 = "78d323c02ceccdfcb054307da3c9e14575210869d22cade6c5ecd4afa4baf8d5"
BANK_SHA256 = "8081ce89a0eac1522b4fb28cd7919de4a4ecf1d5af72552d141a0ee9b9944194"
CONFIG = DRIVE / "Gate01Private_8012a3f" / "stage1-run-c.yaml"
CHECKPOINT = DRIVE / "vae_kl_vae_best.pt"
PHOTOMETRY = OUTPUT_ROOT / "stage2_photometry_factorization_v1.json"
REQUESTED = (
    ("R:0015", Domain(0.1, "T1w")),
    ("R:0398", Domain(3.0, "T2w")),
    ("R:0796", Domain(7.0, "T2-FLAIR")),
)
SEED = 0


def verified_commit(repo: Path) -> str:
    expected = os.environ.get("FIELDBRIDGE_IMPLEMENTATION_COMMIT", "")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", expected) or actual != expected:
        raise ValueError("The checkout differs from the pinned implementation commit.")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True
    ):
        raise ValueError("The implementation checkout must be clean.")
    return actual


def archive_progress(event: dict[str, Any]) -> None:
    processed = event["bytes_processed"] / 1024**3
    total = event["total_bytes"] / 1024**3
    print(f"Bank archive {event['mode']}: {processed:.2f}/{total:.2f} GiB", flush=True)


def read_bank_metadata(archive: Path, metadata_dir: Path) -> None:
    """Read the two authenticated JSON files only. Never deserialize bank tensors."""
    local = copy_verified_stage2_bank_tar_to_local(
        archive, Path("/content") / EXPERIMENT / archive.name,
        expected_archive_sha256=BANK_ARCHIVE_SHA256, progress_callback=archive_progress,
    )
    with tarfile.open(local.local_archive_path, "r:") as bundle:
        members = bundle.getmembers()
        for filename in (FACTORED_LATENT_BANK_MANIFEST, FACTORED_LATENT_STATS_FILE):
            names = {filename, f"photometry_factored_latent_bank_v2/{filename}"}
            matches = [m for m in members if m.name.removeprefix("./") in names]
            if len(matches) != 1 or not matches[0].isreg():
                raise ValueError(f"Expected one regular bank metadata member for {filename}.")
            stream = bundle.extractfile(matches[0])
            if stream is None:
                raise ValueError(f"Missing bank metadata {filename}.")
            with stream:
                (metadata_dir / filename).write_bytes(stream.read())


def select_sources(
    index: PhotometryFactoredLatentBankIndex, root: Path
) -> list[tuple[FactoredLatentRecord, VolumeRecord]]:
    """Resolve only the three authorized R/validation records before reading image data."""
    selected = []
    for group, domain in REQUESTED:
        matches = [
            item for item in index.records
            if item.subject_group_id == group and item.domain == domain
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one sealed R/validation record for {group} {domain.label}.")
        item = matches[0]
        filename = item.case_id + ".nii.gz"
        parsed = parse_mrixfields_filename(filename)
        if (
            item.split != "validation" or item.sidecar["cohort"] != "R"
            or parsed.prefix != "R" or parsed.is_segmentation or f"R:{parsed.subject_id}" != group
            or Domain(float(parsed.field[:-1]), parsed.modality) != domain
        ):
            raise ValueError(f"Bank identity, filename, or split disagrees for {group}.")
        path = safe_relative_path(root, f"{parsed.modality}/{parsed.field}/{filename}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing sealed R source {path}.")
        selected.append((item, VolumeRecord(
            case_id=item.case_id, image_path=path, domain=domain,
            subject_id=parsed.subject_id, split="validation", metadata={"prefix": "R"},
        )))
    for item, volume in selected:
        if sha256_file(volume.image_path) != item.sidecar["source_file_sha256"]:
            raise ValueError(f"Raw NIfTI digest disagrees with the bank for {item.case_id}.")
    return selected


def new_run_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for number in count(1):
        path = parent / f"run-{number:04d}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise AssertionError("Unreachable")


def write_result(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if os.path.lexists(path):
        raise FileExistsError(path)
    with NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False,
    ) as output:
        partial = Path(output.name)
    try:
        data = text.encode("utf-8")
        partial.write_bytes(data)
        if partial.read_bytes() != data:
            raise OSError(f"Drive readback failed for {path.name}.")
        # This invocation exclusively owns its new, numbered run directory.
        if os.path.lexists(path):
            raise FileExistsError(path)
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    commit = verified_commit(Path(__file__).resolve().parents[1])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA A100 80 GB runtime is required.")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_properties(device)
    if "A100" not in gpu.name or gpu.total_memory < 79 * 1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"Expected a BF16-capable A100 80 GB. Found {gpu.name}.")
    config = preflight_frozen_stage1_run_c_config(CONFIG)
    photometry = preflight_reviewed_photometry_namespace_artifact(PHOTOMETRY)
    with TemporaryDirectory(prefix="zero-transport-bank-") as scratch:
        bank_dir = Path(scratch)
        read_bank_metadata(BANK_ARCHIVE, bank_dir)
        index = PhotometryFactoredLatentBankIndex(bank_dir, "validation", expected_artifact_sha256=BANK_SHA256)
        manifest = index.manifest
        manifest_body = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
        if sha256_json(manifest_body) != BANK_SHA256:
            raise ValueError("Bank manifest content digest differs from the frozen artifact.")
        stat_info = manifest["latent_statistics"]
        if sha256_file(bank_dir / FACTORED_LATENT_STATS_FILE) != stat_info["file_sha256"]:
            raise ValueError("Frozen latent statistics file digest mismatch.")
        stats = FactoredLatentStats.from_bank(bank_dir)
        if stats.artifact_sha256 != stat_info["artifact_sha256"]:
            raise ValueError("Frozen latent statistics artifact digest mismatch.")
        vae = verify_frozen_stage1_vae_bank_provenance(
            config, vae_checkpoint_path=CHECKPOINT, bank_dir=bank_dir,
        )
        photo = verify_reviewed_photometry_bank_provenance(photometry, bank_dir=bank_dir)
        selected = select_sources(index, DRIVE / "Data" / "Training_retrospective")
    model = config.parsed_config["model"]
    encoder = build_encoder("kl_vae", **_kl_vae_kwargs(model, "encoder"))
    decoder = build_decoder("kl_vae", **_kl_vae_kwargs(model, "decoder"))
    _require_linear_decoder_output_activation(decoder)
    state = load_checkpoint(CHECKPOINT, map_location="cpu")
    encoder.load_state_dict(state["encoder"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    checkpoint_version = state.get("checkpoint_version") or f"sha256:{vae.checkpoint_file_sha256}"
    del state
    encoder.to(device).requires_grad_(False).eval()
    decoder.to(device).requires_grad_(False).eval()
    support_rule_sha256 = manifest["operational_support_rule"]["rule_sha256"]
    run = new_run_directory(OUTPUT_ROOT / EXPERIMENT / commit)
    provenance = {
        "contract": EXPERIMENT, "implementation_commit": commit, "seed": SEED,
        "hardware": {"name": gpu.name, "bytes": gpu.total_memory},
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cpu_threads": torch.get_num_threads(),
        "bank_archive_sha256": BANK_ARCHIVE_SHA256, "bank_artifact_sha256": index.artifact_sha256,
        "latent_statistics": stat_info, "split": manifest["source_split"],
        "support_rule_sha256": support_rule_sha256,
        "vae": vae.sanitized_provenance(), "vae_config": config.parsed_config,
        "checkpoint_version": checkpoint_version, "photometry": photo.sanitized_provenance(),
        "raw_source_cohort": "R", "raw_source_split": "validation",
        "encoder_precision": "float32", "decoder_autocast_dtype": "bfloat16",
    }
    print(f"Drive output: {run}", flush=True)
    deltas = {}
    for number, (item, volume) in enumerate(selected, start=1):
        print(f"Starting scan {number}/{len(selected)}: {item.subject_group_id} {item.domain.label}", flush=True)
        source = load_volume(volume)
        details = run_zero_transport_scan(
            source, item.domain, artifact=photo.preflight.artifact, encoder=encoder,
            decoder=decoder, stats=stats, support_rule_sha256=support_rule_sha256, device=device,
            expected_sidecar=item.sidecar,
        )
        del source
        result = {
            **provenance, **details, "subject_group_id": item.subject_group_id,
            "case_id": item.case_id, "source_file_sha256": item.sidecar["source_file_sha256"],
            "source_loaded_array_sha256": item.sidecar["source_loaded_array_sha256"],
        }
        write_result(run / f"{item.case_id}.json", result)
        delta = details["primary_b_minus_a_rmse"]
        deltas[item.subject_group_id] = delta
        print(f"{item.subject_group_id} {item.domain.label}: B-A foreground RMSE {delta:+.9g}", flush=True)
    summary = {
        **provenance, "primary_b_minus_a_rmse_by_subject": deltas,
        "scan_files": [f"{item.case_id}.json" for item, _ in selected],
    }
    write_result(run / "summary.json", summary)
    print(json.dumps({"run": str(run), "primary_b_minus_a_rmse": deltas}), flush=True)


if __name__ == "__main__":
    main()
