"""Read side of the latent bank: indexed latents + standardization stats for Stage-2.

The bank stores raw posterior-mean latents plus ``latent_stats.json`` (per-channel mean/std
over the TRAIN split). Transport trains in standardized space; the decode path un-normalizes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fieldbridge.data.domains import Domain


@dataclass(frozen=True, slots=True)
class LatentStats:
    """Per-channel standardization statistics (channel-first latents)."""

    mean: torch.Tensor  # (C,)
    std: torch.Tensor  # (C,)

    @classmethod
    def from_json(cls, path: str | Path, *, eps: float = 1e-6) -> "LatentStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mean = torch.tensor([float(v) for v in payload["per_channel_mean"]], dtype=torch.float32)
        std = torch.tensor([float(v) for v in payload["per_channel_std"]], dtype=torch.float32)
        return cls(mean=mean, std=std.clamp_min(eps))

    def _broadcast(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tensor is (C, ...) or (B, C, ...); align channel dim.
        channel_dim = 1 if tensor.ndim >= 5 or (tensor.ndim == 4 and tensor.shape[0] != self.mean.numel()) else 0
        shape = [1] * tensor.ndim
        shape[channel_dim] = self.mean.numel()
        mean = self.mean.to(tensor.device, tensor.dtype).reshape(shape)
        std = self.std.to(tensor.device, tensor.dtype).reshape(shape)
        return mean, std

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(latent)
        return (latent - mean) / std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(latent)
        return latent * std + mean


@dataclass(frozen=True, slots=True)
class LatentRecord:
    case_id: str
    path: Path
    domain: Domain
    subject_id: str | None


def split_assignment_from_json(path: str | Path) -> dict[str, str]:
    """case_id -> split name, read from a VAE split JSON.

    Lets a resplit take effect without re-encoding. The bank bakes each record's split into
    its manifest at build time, so moving a subject between splits in the JSON is otherwise a
    silent no-op for everything that reads the bank — which is every Stage-2 consumer.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    assignment: dict[str, str] = {}
    for name in ("train", "validation", "test"):
        for record in splits.get(name, []):
            assignment[str(record["case_id"])] = name
    if not assignment:
        raise ValueError(f"Split file {path} has no records under train/validation/test.")
    return assignment


class LatentBankIndex:
    """Index of one split's latent files, with lazy per-record loading.

    ``split_json`` overrides the split recorded in the bank manifest. Pass it whenever the
    split has been revised since the bank was built; without it a resplit is silently ignored.
    """

    def __init__(
        self,
        bank_dir: str | Path,
        split: str,
        *,
        split_json: str | Path | None = None,
    ) -> None:
        self.bank_dir = Path(bank_dir)
        self.split = split
        self.split_override = (
            split_assignment_from_json(split_json) if split_json is not None else None
        )
        self.records: list[LatentRecord] = self._load_index()
        if not self.records:
            source = f" under {split_json}" if split_json is not None else ""
            raise ValueError(
                f"Latent bank {self.bank_dir} has no records for split {split!r}{source}."
            )

    def _split_of(self, case_id: str, manifest_split: str | None) -> str | None:
        if self.split_override is None:
            return manifest_split
        return self.split_override.get(case_id)

    def _load_index(self) -> list[LatentRecord]:
        manifest_path = self.bank_dir / "latent_bank_manifest.json"
        records: list[LatentRecord] = []
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest.get("records", []):
                case_id = str(entry["case_id"])
                if self._split_of(case_id, entry.get("split")) != self.split:
                    continue
                records.append(
                    LatentRecord(
                        case_id=case_id,
                        path=self.bank_dir / entry["path"],
                        domain=Domain.from_dict(entry["domain"]),
                        subject_id=entry.get("subject_id"),
                    )
                )
            if records:
                return records
        # Fallback: scan the split directory and read each payload's domain.
        for path in sorted((self.bank_dir / self.split).glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            records.append(
                LatentRecord(
                    case_id=str(payload["case_id"]),
                    path=path,
                    domain=Domain.from_dict(payload["domain"]),
                    subject_id=payload.get("subject_id"),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def domains(self) -> list[Domain]:
        return [record.domain for record in self.records]

    def load_latent(self, index: int) -> torch.Tensor:
        payload = torch.load(self.records[index].path, map_location="cpu")
        return payload["latent"].to(torch.float32)

    def load_batch(self, indices: Sequence[int]) -> tuple[torch.Tensor, list[Domain]]:
        latents = torch.stack([self.load_latent(i) for i in indices], dim=0)
        domains = [self.records[i].domain for i in indices]
        return latents, domains

    def summary(self) -> dict[str, Any]:
        by_domain: dict[str, int] = {}
        for record in self.records:
            by_domain[record.domain.label] = by_domain.get(record.domain.label, 0) + 1
        return {"split": self.split, "count": len(self.records), "by_domain": by_domain}


__all__ = ["LatentStats", "LatentRecord", "LatentBankIndex"]
