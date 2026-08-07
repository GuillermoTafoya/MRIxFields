"""Precompute a latent bank from a frozen Stage-1 KL-VAE for Stage-2 transport training.

The VAE is the pipeline ceiling and is frozen here; Stage-2 trains on these latents instead
of decoding volumes every step. Design choices (documented because they are real choices):

- One coherent full-volume latent is stored per record — the deterministic posterior *mean*
  ``encode_dist(volume)[0]``, no sampling. That single latent is what a transport network
  should move, rather than the seam-blended tiles the Stage-1 recon eval produces.
- Full-volume encode of the official 364x436x364 shape can exceed 16 GB, and full-volume
  *decode* certainly does (it is why the recon eval tiles). So the default encode is
  **halo-tiled**: each core block is encoded with a receptive-field halo of real neighbours
  and the halo is cropped away in latent space, giving a result that matches a single
  full-volume encode away from block faces while keeping peak memory bounded by one block.
  ``strategy="full"`` does a single forward when memory allows; ``strategy="auto"`` tries
  full first and falls back to tiled on CUDA OOM.
- Stored latents are raw (not normalized). Per-channel mean/std over the TRAIN split are
  written to ``latent_stats.json`` for Stage-2 to standardize with; the decode path
  un-normalizes. Our latent std is ~0.7, not 1.0, so this matters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Domain
from fieldbridge.data.sources import nifti_image_loader
from fieldbridge.data.transforms import assert_official_unit_range
from fieldbridge.evaluation.metrics import ssim3d
from fieldbridge.training.train_loop import assert_frozen

LATENT_BANK_CONTRACT_VERSION = "latent-bank-v1"
_DATA_RANGE = 1.0  # official [0, 1] volumes

EncodeStrategy = Literal["tiled", "full", "auto"]
StoreDtype = Literal["float16", "float32"]
Precision = Literal["float32", "bfloat16"]


@dataclass(frozen=True, slots=True)
class LatentBankConfig:
    """Config for building the latent bank. All knobs config/CLI driven, no magic numbers."""

    out_dir: Path
    strategy: EncodeStrategy = "tiled"
    store_dtype: StoreDtype = "float16"
    precision: Precision = "float32"
    # Core block and halo (voxels) for tiled encode/decode; must be multiples of the VAE
    # downsample factor (validated at run time against the loaded encoder).
    block_size: tuple[int, int, int] = (128, 128, 128)
    halo: tuple[int, int, int] = (32, 32, 32)
    roundtrip_samples: int = 6
    seed: int = 13
    overwrite: bool = False
    splits: tuple[str, ...] = ("train", "validation", "test")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, out_dir: Path) -> "LatentBankConfig":
        section = data.get("latent_bank", {})
        section = dict(section) if isinstance(section, Mapping) else {}
        defaults = cls(out_dir=out_dir)

        def pick(key: str, current: Any) -> Any:
            return section.get(key, current)

        return cls(
            out_dir=out_dir,
            strategy=pick("strategy", defaults.strategy),
            store_dtype=pick("store_dtype", defaults.store_dtype),
            precision=pick("precision", defaults.precision),
            block_size=tuple(pick("block_size", defaults.block_size)),  # type: ignore[arg-type]
            halo=tuple(pick("halo", defaults.halo)),  # type: ignore[arg-type]
            roundtrip_samples=int(pick("roundtrip_samples", defaults.roundtrip_samples)),
            seed=int(pick("seed", defaults.seed)),
            overwrite=bool(pick("overwrite", defaults.overwrite)),
            splits=tuple(pick("splits", defaults.splits)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "strategy": self.strategy,
            "store_dtype": self.store_dtype,
            "precision": self.precision,
            "block_size": list(self.block_size),
            "halo": list(self.halo),
            "roundtrip_samples": self.roundtrip_samples,
            "seed": self.seed,
            "overwrite": self.overwrite,
            "splits": list(self.splits),
        }


def _store_dtype(name: StoreDtype) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def _autocast(device: torch.device, precision: Precision):
    if precision == "bfloat16":
        if device.type != "cuda":
            raise ValueError("bfloat16 encode precision requires CUDA.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def _is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def downsample_factor(encoder: Any) -> int:
    factor = getattr(encoder, "downsample_factor", None)
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError(
            "Encoder must expose a positive integer 'downsample_factor'; "
            f"got {factor!r}."
        )
    return factor


def _core_blocks(dim: int, block: int, factor: int) -> list[tuple[int, int]]:
    """Non-overlapping core blocks covering [0, dim); every start and size is a multiple of factor."""

    if dim % factor != 0:
        raise ValueError(f"Volume extent {dim} must be a multiple of downsample factor {factor}.")
    if block % factor != 0 or block <= 0:
        raise ValueError(f"block_size {block} must be a positive multiple of downsample factor {factor}.")
    starts = list(range(0, dim, block))
    return [(start, min(block, dim - start)) for start in starts]


@torch.inference_mode()
def encode_latent(
    encoder: Any,
    volume: torch.Tensor,
    domain: Domain,
    *,
    strategy: EncodeStrategy,
    block_size: Sequence[int],
    halo: Sequence[int],
    precision: Precision,
) -> tuple[torch.Tensor, str]:
    """Encode a (1,1,X,Y,Z) volume to its (1,C,X/f,Y/f,Z/f) posterior-mean latent.

    Returns the latent and the strategy actually used ("full" or "tiled").
    """

    if volume.ndim != 5:
        raise ValueError(f"encode_latent expects (1,1,X,Y,Z), got {tuple(volume.shape)}.")
    device = volume.device
    if strategy in ("full", "auto"):
        try:
            with _autocast(device, precision):
                mean, _ = encoder.encode_dist(volume, domain)
            return mean.float(), "full"
        except RuntimeError as error:
            if strategy == "full" or not _is_oom(error):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return _encode_tiled(encoder, volume, domain, block_size=block_size, halo=halo, precision=precision), "tiled"


def _encode_tiled(
    encoder: Any,
    volume: torch.Tensor,
    domain: Domain,
    *,
    block_size: Sequence[int],
    halo: Sequence[int],
    precision: Precision,
) -> torch.Tensor:
    factor = downsample_factor(encoder)
    device = volume.device
    _, _, depth, height, width = volume.shape
    bd, bh, bw = (int(b) for b in block_size)
    hd, hh, hw = (int(h) for h in halo)
    for name, value in (("halo", (hd, hh, hw)),):
        if any(v % factor != 0 or v < 0 for v in value):
            raise ValueError(f"{name} values must be non-negative multiples of factor {factor}: {value}.")

    latent_channels = int(getattr(encoder, "latent_channels"))
    latent = torch.zeros(
        (1, latent_channels, depth // factor, height // factor, width // factor),
        device=device,
        dtype=torch.float32,
    )
    for z, bz in _core_blocks(depth, bd, factor):
        for y, by in _core_blocks(height, bh, factor):
            for x, bw_ in _core_blocks(width, bw, factor):
                z0, z1 = max(0, z - hd), min(depth, z + bz + hd)
                y0, y1 = max(0, y - hh), min(height, y + by + hh)
                x0, x1 = max(0, x - hw), min(width, x + bw_ + hw)
                block = volume[..., z0:z1, y0:y1, x0:x1]
                with _autocast(device, precision):
                    mean, _ = encoder.encode_dist(block, domain)
                mean = mean.float()
                # Crop the halo away in latent space (all offsets are multiples of factor).
                lz, ly, lx = (z - z0) // factor, (y - y0) // factor, (x - x0) // factor
                latent[
                    ...,
                    z // factor : (z + bz) // factor,
                    y // factor : (y + by) // factor,
                    x // factor : (x + bw_) // factor,
                ] = mean[..., lz : lz + bz // factor, ly : ly + by // factor, lx : lx + bw_ // factor]
    return latent


@torch.inference_mode()
def decode_latent_tiled(
    decoder: Any,
    latent: torch.Tensor,
    domain: Domain,
    *,
    factor: int,
    block_size: Sequence[int],
    halo: Sequence[int],
    precision: Precision,
    clamp: bool = True,
) -> torch.Tensor:
    """Decode a (1,C,x,y,z) latent to a (1,1,X,Y,Z) image, tile by tile with a halo.

    Mirrors the halo-tiled encode: decode each latent core block with a latent-space halo of
    real neighbours, crop the halo in image space, and place. Bounded by one block's memory.
    """

    if latent.ndim != 5:
        raise ValueError(f"decode_latent_tiled expects (1,C,x,y,z), got {tuple(latent.shape)}.")
    device = latent.device
    _, _, ld, lh, lw = latent.shape
    bd, bh, bw = (int(b) // factor for b in block_size)
    hd, hh, hw = (int(h) // factor for h in halo)
    image = torch.zeros((1, 1, ld * factor, lh * factor, lw * factor), device=device, dtype=torch.float32)
    for z, bz in _core_blocks_latent(ld, bd):
        for y, by in _core_blocks_latent(lh, bh):
            for x, bx in _core_blocks_latent(lw, bw):
                z0, z1 = max(0, z - hd), min(ld, z + bz + hd)
                y0, y1 = max(0, y - hh), min(lh, y + by + hh)
                x0, x1 = max(0, x - hw), min(lw, x + bx + hw)
                block = latent[..., z0:z1, y0:y1, x0:x1]
                with _autocast(device, precision):
                    decoded = decoder.decode(block, domain)
                decoded = decoded.float()
                # Offset of the core within the decoded (haloed) image block.
                iz, iy, ix = (z - z0) * factor, (y - y0) * factor, (x - x0) * factor
                image[
                    ...,
                    z * factor : (z + bz) * factor,
                    y * factor : (y + by) * factor,
                    x * factor : (x + bx) * factor,
                ] = decoded[
                    ...,
                    iz : iz + bz * factor,
                    iy : iy + by * factor,
                    ix : ix + bx * factor,
                ]
    return image.clamp(0.0, 1.0) if clamp else image


@torch.inference_mode()
def decode_latent(
    decoder: Any,
    latent: torch.Tensor,
    domain: Domain,
    *,
    factor: int,
    strategy: EncodeStrategy,
    block_size: Sequence[int],
    halo: Sequence[int],
    precision: Precision,
    clamp: bool = True,
) -> tuple[torch.Tensor, str]:
    """Decode a latent and report whether the full or approximate path was used.

    This restores the reviewed Gate-0 behavior from ``d3476b9``.  In particular,
    ``strategy="full"`` propagates an OOM and can never fall back to tiled decode.
    ``auto`` retains the explicitly requested legacy fallback behavior for callers
    outside the frozen Gate-0.1 producer.
    """

    if latent.ndim != 5:
        raise ValueError(f"decode_latent expects (1,C,x,y,z), got {tuple(latent.shape)}.")
    device = latent.device
    if strategy in ("full", "auto"):
        try:
            with _autocast(device, precision):
                image = decoder.decode(latent, domain)
            image = image.float()
            return (image.clamp(0.0, 1.0) if clamp else image), "full"
        except RuntimeError as error:
            if strategy == "full" or not _is_oom(error):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
    tiled = decode_latent_tiled(
        decoder,
        latent,
        domain,
        factor=factor,
        block_size=block_size,
        halo=halo,
        precision=precision,
        clamp=clamp,
    )
    return tiled, "tiled"


def _core_blocks_latent(dim: int, block: int) -> list[tuple[int, int]]:
    if block <= 0:
        raise ValueError(f"latent block must be positive, got {block}.")
    starts = list(range(0, dim, block))
    return [(start, min(block, dim - start)) for start in starts]


class _ChannelStats:
    """Streaming per-channel mean/std over an arbitrary number of (C, ...) latents."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._sum = torch.zeros(channels, dtype=torch.float64)
        self._sumsq = torch.zeros(channels, dtype=torch.float64)
        self._count = 0

    def update(self, latent: torch.Tensor) -> None:
        # latent: (1, C, x, y, z) or (C, x, y, z)
        tensor = latent.detach().to(torch.float64).cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        flat = tensor.reshape(self.channels, -1)
        self._sum += flat.sum(dim=1)
        self._sumsq += flat.square().sum(dim=1)
        self._count += flat.shape[1]

    def compute(self) -> dict[str, Any]:
        if self._count == 0:
            raise ValueError("No latents accumulated for channel statistics.")
        mean = self._sum / self._count
        var = (self._sumsq / self._count) - mean.square()
        std = var.clamp_min(0.0).sqrt()
        return {
            "per_channel_mean": [float(v) for v in mean],
            "per_channel_std": [float(v) for v in std],
            "global_mean": float(mean.mean()),
            "global_std": float(std.mean()),
            "voxels_per_channel": int(self._count),
            "channels": int(self.channels),
        }


def _safe_case_id(case_id: str) -> str:
    cleaned = str(case_id).strip().replace("/", "_").replace("\\", "_")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Cannot derive a safe filename from case_id {case_id!r}.")
    return cleaned


def load_volume(record: VolumeRecord) -> torch.Tensor:
    """Load a record's NIfTI as a (1,1,X,Y,Z) float32 volume on the official [0,1] contract."""

    volume = nifti_image_loader(record.image_path, record)  # (1, X, Y, Z)
    volume = assert_official_unit_range(volume)
    return volume.unsqueeze(0)


def build_latent_bank(
    *,
    encoder: Any,
    decoder: Any,
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    config: LatentBankConfig,
    device: torch.device,
    checkpoint_sha256: str,
    git_commit: str,
    vae_config_path: str,
    volume_loader=load_volume,
    log: bool = True,
) -> dict[str, Any]:
    """Encode every record to its latent, write per-record files + latent_stats.json + manifest.

    The encoder/decoder must already be frozen (asserted here). Deterministic (posterior mean,
    fixed seed). Idempotent: existing latent files are skipped unless ``config.overwrite``.
    """

    assert_frozen(encoder)
    assert_frozen(decoder)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    torch.manual_seed(config.seed)

    factor = downsample_factor(encoder)
    for value in (*config.block_size, *config.halo):
        if int(value) % factor != 0:
            raise ValueError(
                f"block_size/halo must be multiples of the VAE downsample factor {factor}; got {value}."
            )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_dtype = _store_dtype(config.store_dtype)
    train_stats = _ChannelStats(int(getattr(encoder, "latent_channels")))

    manifest: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    strategy_used: set[str] = set()
    roundtrip_records: list[tuple[VolumeRecord, Path]] = []

    for split in config.splits:
        records = list(records_by_split.get(split, ()))
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        counts[split] = {"total": len(records), "encoded": 0, "skipped": 0}
        for index, record in enumerate(records):
            case_id = _safe_case_id(record.case_id)
            path = split_dir / f"{case_id}.pt"
            if path.exists() and not config.overwrite:
                counts[split]["skipped"] += 1
                payload = torch.load(path, map_location="cpu")
                if split == "train":
                    train_stats.update(payload["latent"].to(torch.float32))
                _append_manifest(manifest, payload, path, out_dir)
                if split != "train" and len(roundtrip_records) < config.roundtrip_samples:
                    roundtrip_records.append((record, path))
                continue

            volume = volume_loader(record).to(device)
            latent, used = encode_latent(
                encoder,
                volume,
                record.domain,
                strategy=config.strategy,
                block_size=config.block_size,
                halo=config.halo,
                precision=config.precision,
            )
            strategy_used.add(used)
            latent_cpu = latent[0].to(store_dtype).cpu()
            payload = {
                "contract_version": LATENT_BANK_CONTRACT_VERSION,
                "case_id": record.case_id,
                "subject_id": record.subject_id,
                "split": split,
                "domain": record.domain.to_dict(),
                "latent": latent_cpu,
                "latent_shape": list(latent_cpu.shape),
                "source_shape": list(volume.shape[1:]),
                "downsample_factor": factor,
                "store_dtype": config.store_dtype,
                "encode_strategy": used,
                "encode_precision": config.precision,
                "vae_checkpoint_sha256": checkpoint_sha256,
                "vae_config_path": vae_config_path,
                "git_commit": git_commit,
            }
            _atomic_torch_save(payload, path)
            counts[split]["encoded"] += 1
            if split == "train":
                train_stats.update(latent[0].to(torch.float32))
            _append_manifest(manifest, payload, path, out_dir)
            if split != "train" and len(roundtrip_records) < config.roundtrip_samples:
                roundtrip_records.append((record, path))
            if log and (index == 0 or (index + 1) % 25 == 0 or index + 1 == len(records)):
                print(
                    f"latent_bank split={split} {index + 1}/{len(records)} case={case_id} "
                    f"strategy={used} latent_shape={list(latent_cpu.shape)}",
                    flush=True,
                )
            del volume, latent

    stats = train_stats.compute()
    stats_payload = {
        "contract_version": LATENT_BANK_CONTRACT_VERSION,
        "computed_over": "train",
        "vae_checkpoint_sha256": checkpoint_sha256,
        "git_commit": git_commit,
        **stats,
    }
    _write_json(out_dir / "latent_stats.json", stats_payload)

    roundtrip = _run_roundtrip(
        decoder=decoder,
        records=roundtrip_records,
        factor=factor,
        config=config,
        device=device,
        volume_loader=volume_loader,
        log=log,
    )

    bank_manifest = {
        "contract_version": LATENT_BANK_CONTRACT_VERSION,
        "config": config.to_dict(),
        "vae_checkpoint_sha256": checkpoint_sha256,
        "git_commit": git_commit,
        "vae_config_path": vae_config_path,
        "counts": counts,
        "strategy_used": sorted(strategy_used) or ["reused_existing"],
        "latent_stats": stats_payload,
        "roundtrip": roundtrip,
        "records": manifest,
    }
    _write_json(out_dir / "latent_bank_manifest.json", bank_manifest)
    if log:
        print(
            f"latent_bank DONE counts={counts} std={stats['global_std']:.4f} "
            f"roundtrip_ssim3d={roundtrip.get('mean_ssim3d')}",
            flush=True,
        )
    return bank_manifest


def _run_roundtrip(
    *,
    decoder: Any,
    records: Sequence[tuple[VolumeRecord, Path]],
    factor: int,
    config: LatentBankConfig,
    device: torch.device,
    volume_loader,
    log: bool,
) -> dict[str, Any]:
    """Decode a handful of stored latents and report SSIM3D vs the original volume."""

    per_case: list[dict[str, Any]] = []
    for record, path in records:
        payload = torch.load(path, map_location="cpu")
        latent = payload["latent"].to(torch.float32).unsqueeze(0).to(device)
        recon = decode_latent_tiled(
            decoder,
            latent,
            record.domain,
            factor=factor,
            block_size=config.block_size,
            halo=config.halo,
            precision=config.precision,
        )
        target = volume_loader(record).to(device)
        value = float(ssim3d(recon, target, data_range=_DATA_RANGE))
        per_case.append(
            {"case_id": record.case_id, "domain": record.domain.label, "ssim3d": value}
        )
        if log:
            print(f"latent_bank roundtrip case={record.case_id} ssim3d={value:.4f}", flush=True)
        del latent, recon, target
    mean_ssim = (
        float(sum(item["ssim3d"] for item in per_case) / len(per_case)) if per_case else None
    )
    return {"mean_ssim3d": mean_ssim, "per_case": per_case}


def _append_manifest(
    manifest: list[dict[str, Any]], payload: Mapping[str, Any], path: Path, out_dir: Path
) -> None:
    manifest.append(
        {
            "case_id": payload["case_id"],
            "subject_id": payload.get("subject_id"),
            "split": payload["split"],
            "domain": payload["domain"],
            "latent_shape": payload["latent_shape"],
            "source_shape": payload["source_shape"],
            "path": path.relative_to(out_dir).as_posix(),
        }
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "LATENT_BANK_CONTRACT_VERSION",
    "LatentBankConfig",
    "build_latent_bank",
    "decode_latent",
    "encode_latent",
    "decode_latent_tiled",
    "downsample_factor",
    "load_volume",
]
