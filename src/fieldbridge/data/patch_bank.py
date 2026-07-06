"""Precomputed patch bank: read each manifest volume from (slow, network) storage exactly
once, extract normalized patches, and persist them as compact shards.

Rationale: the streaming loader re-reads every ~231MB NIfTI from Drive-FUSE *each epoch*
(~3-4h of I/O per pass, unavoidable while reading raw volumes). Training the same VAE for
many epochs pays that tax over and over. Building a bank once turns the raw dataset into a
small, reusable float16 patch set (e.g. ~31GB at 32 patches/volume, vs the ~100GB raw)
that lives on Drive and loads fully into RAM for compute-bound training.

The build is resumable and read-error tolerant on purpose: it runs unattended for hours
against Drive, so a transient read failure or a disconnect must not force a restart from
zero. One shard file per volume, an append-only JSONL index, and skip-if-already-done make
re-runs pick up where they left off.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.datasets import ImageLoader, ImageTransform
from fieldbridge.data.domains import Domain
from fieldbridge.data.transforms import normalize_percentile_clip_to_unit_range, random_crop

_META_NAME = "bank_meta.json"
_INDEX_NAME = "bank_index.jsonl"
_FAILURES_NAME = "bank_failures.jsonl"
_SHARD_DIR = "shards"


@dataclass(frozen=True, slots=True)
class PatchBankBuildResult:
    num_volumes_written: int
    num_volumes_skipped: int
    num_volumes_failed: int
    total_patches: int
    out_dir: Path


def _read_done_indices(index_path: Path) -> set[int]:
    if not index_path.exists():
        return set()
    done: set[int] = set()
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        done.add(int(json.loads(line)["vol_index"]))
    return done


def build_patch_bank(
    records: Sequence[VolumeRecord],
    *,
    image_loader: ImageLoader,
    out_dir: Path | str,
    patch_size: Sequence[int],
    patches_per_volume: int,
    volume_transform: ImageTransform | None = normalize_percentile_clip_to_unit_range,
    seed: int = 13,
    max_read_retries: int = 2,
    log_every: int = 25,
    logger=None,
) -> PatchBankBuildResult:
    """Extract `patches_per_volume` normalized float16 crops from each volume into `out_dir`.

    Resumable: volumes already present in the JSONL index are skipped. Read failures are
    retried `max_read_retries` times, then logged to bank_failures.jsonl and skipped (they
    are simply absent from the bank, not zero-filled, so training never sees garbage).
    Patch positions are seeded per volume (seed + vol_index) so a re-processed volume yields
    identical patches.
    """

    out_dir = Path(out_dir)
    shard_dir = out_dir / _SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    patch = tuple(int(p) for p in patch_size)
    index_path = out_dir / _INDEX_NAME
    failures_path = out_dir / _FAILURES_NAME

    _write_meta(out_dir, patch, patches_per_volume, seed)
    done = _read_done_indices(index_path)

    def _log(message: str) -> None:
        if logger is not None:
            logger(message)

    written = skipped = failed = 0
    total_patches = 0
    for vol_index, record in enumerate(records):
        if vol_index in done:
            skipped += 1
            continue

        volume = _load_with_retries(image_loader, record, max_read_retries, _log)
        if volume is None:
            failed += 1
            _record_failure(failures_path, vol_index, record.case_id)
            continue

        # Wrap processing + Drive writes too: a transient write error (Drive-FUSE) must not
        # kill a multi-hour unattended build. A skipped volume is absent from the index, so
        # the next resumed run reprocesses it (it is not marked done).
        try:
            if volume_transform is not None:
                volume = volume_transform(volume)
            torch.manual_seed(seed + vol_index)  # reproducible + resume-consistent crops
            patches = torch.stack(
                [random_crop(volume, patch_size=patch) for _ in range(patches_per_volume)], dim=0
            )
            shard_rel = f"{_SHARD_DIR}/{vol_index:06d}.npy"
            np.save(out_dir / shard_rel, patches.to(torch.float16).numpy())

            domain = record.domain if isinstance(record.domain, Domain) else Domain.from_dict(dict(record.domain))
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "vol_index": vol_index,
                            "case_id": record.case_id,
                            "shard": shard_rel,
                            "domain": domain.to_dict(),
                            "num_patches": int(patches_per_volume),
                        }
                    )
                    + "\n"
                )
        except Exception as exc:  # noqa: BLE001 - Drive write / crop errors must not abort the run
            _log(f"patch-bank: processing failed for {record.case_id}: {exc}")
            failed += 1
            _record_failure(failures_path, vol_index, record.case_id)
            continue
        written += 1
        total_patches += patches_per_volume
        if log_every and (written % log_every == 0):
            _log(f"patch-bank: wrote {written} volumes ({skipped} skipped, {failed} failed) -> {out_dir}")

    return PatchBankBuildResult(
        num_volumes_written=written,
        num_volumes_skipped=skipped,
        num_volumes_failed=failed,
        total_patches=total_patches,
        out_dir=out_dir,
    )


def _record_failure(failures_path: Path, vol_index: int, case_id: str) -> None:
    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"vol_index": vol_index, "case_id": case_id}) + "\n")


def _load_with_retries(image_loader: ImageLoader, record: VolumeRecord, retries: int, log) -> torch.Tensor | None:
    for attempt in range(retries + 1):
        try:
            return image_loader(record.image_path, record)
        except Exception as exc:  # noqa: BLE001 - Drive I/O raises OSError/ConnectionError variants
            log(f"patch-bank: read failed for {record.case_id} (attempt {attempt + 1}/{retries + 1}): {exc}")
    return None


def _write_meta(out_dir: Path, patch: tuple[int, ...], patches_per_volume: int, seed: int) -> None:
    meta_path = out_dir / _META_NAME
    meta = {"patch_size": list(patch), "patches_per_volume": int(patches_per_volume), "seed": int(seed)}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("patch_size") != meta["patch_size"] or existing.get("patches_per_volume") != meta[
            "patches_per_volume"
        ]:
            raise ValueError(
                f"Existing bank at {out_dir} was built with {existing}, incompatible with requested {meta}. "
                "Use a fresh --out directory."
            )
        return
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


class PatchBankDataset(Dataset[RawBatch]):
    """Map-style dataset over a prebuilt patch bank, loaded fully into RAM.

    All shards are read into memory once at construction (~31GB float16 for a 32-patch,
    1894-volume bank — fits the Colab high-RAM runtime), so training does zero disk I/O and
    is compute-bound. Every indexed volume contributes exactly `patches_per_volume` patches
    (failed volumes are absent from the index), so patch i maps to volume i // ppv.
    """

    def __init__(self, bank_dir: Path | str) -> None:
        bank_dir = Path(bank_dir)
        meta = json.loads((bank_dir / _META_NAME).read_text(encoding="utf-8"))
        self.patches_per_volume = int(meta["patches_per_volume"])
        index_path = bank_dir / _INDEX_NAME
        entries = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not entries:
            raise ValueError(f"Patch bank at {bank_dir} has an empty index — nothing to train on.")

        self._volumes: list[torch.Tensor] = []
        self._domains: list[Domain] = []
        self._case_ids: list[str] = []
        for entry in entries:
            shard = torch.from_numpy(np.load(bank_dir / entry["shard"]))
            if shard.shape[0] != self.patches_per_volume:
                raise ValueError(
                    f"Shard {entry['shard']} has {shard.shape[0]} patches, expected {self.patches_per_volume}."
                )
            self._volumes.append(shard)  # kept float16 in RAM; cast per-item on access
            self._domains.append(Domain.from_dict(entry["domain"]))
            self._case_ids.append(entry["case_id"])

    def __len__(self) -> int:
        return len(self._volumes) * self.patches_per_volume

    def __getitem__(self, index: int) -> RawBatch:
        vol = index // self.patches_per_volume
        local = index % self.patches_per_volume
        domain = self._domains[vol]
        return RawBatch(
            image=self._volumes[vol][local].float(),
            source_domain=domain,
            target_domain=domain,
            metadata={"case_id": self._case_ids[vol]},
        )


def patch_bank_size(bank_dir: Path | str) -> tuple[int, int]:
    """Return (num_volumes, patches_per_volume) from a bank's meta + index without loading
    shards — used by the CLI to compute steps_per_epoch."""
    bank_dir = Path(bank_dir)
    meta = json.loads((bank_dir / _META_NAME).read_text(encoding="utf-8"))
    num_volumes = sum(1 for line in (bank_dir / _INDEX_NAME).read_text(encoding="utf-8").splitlines() if line.strip())
    return num_volumes, int(meta["patches_per_volume"])
