"""Strict read side for the retrospective photometry-factored latent bank.

The manifest and every sidecar are classified before a tensor payload is opened.  This
is intentionally separate from ``latent_bank_dataset``: latent-bank-v1 remains an
immutable compatibility surface and is never silently reinterpreted as canonical data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fieldbridge.data.domains import Domain
from fieldbridge.data.photometry_factored_latent_bank import (
    FACTORED_LATENT_STATS_FILE,
    LOCAL_VALID_CORE_SUPPORT_RULE,
    PHOTOMETRY_FACTORED_LATENT_STATS_VERSION,
    _load_latent_record,
    load_photometry_factored_latent_bank_manifest,
    unpack_support_mask,
)
from fieldbridge.data.stage2_canonical_volume import safe_relative_path


@dataclass(frozen=True, slots=True)
class FactoredLatentStats:
    mean: torch.Tensor
    std: torch.Tensor
    supported_count: torch.Tensor
    artifact_sha256: str

    @classmethod
    def from_bank(cls, bank_dir: str | Path) -> "FactoredLatentStats":
        payload = json.loads(
            (Path(bank_dir) / FACTORED_LATENT_STATS_FILE).read_text(encoding="utf-8")
        )
        if payload.get("contract_version") != PHOTOMETRY_FACTORED_LATENT_STATS_VERSION:
            raise ValueError("Factored latent-statistics contract mismatch.")
        computed = payload.get("computed_over", {})
        if computed != {
            "cohort": "R",
            "split": "train",
            "cells": "encoder_local_valid_core_only",
        }:
            raise ValueError("Factored statistics are not supported-cell R/train statistics.")
        if payload.get("operational_support_contract") != LOCAL_VALID_CORE_SUPPORT_RULE:
            raise ValueError("Factored statistics use an incompatible support rule.")
        mean = torch.tensor(payload["per_channel_mean"], dtype=torch.float32)
        std = torch.tensor(payload["per_channel_std"], dtype=torch.float32)
        count = torch.tensor(payload["per_channel_supported_count"], dtype=torch.int64)
        if mean.ndim != 1 or mean.shape != std.shape or mean.shape != count.shape:
            raise ValueError("Factored latent-statistics channel shapes are inconsistent.")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("Factored latent statistics contain non-finite values.")
        if bool((std <= 0).any()) or bool((count <= 0).any()):
            raise ValueError("Factored latent statistics are empty or degenerate.")
        return cls(mean, std, count, str(payload["artifact_sha256"]))

    def _broadcast(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_dim = 1 if tensor.ndim == 5 else 0
        shape = [1] * tensor.ndim
        shape[channel_dim] = int(self.mean.numel())
        return (
            self.mean.to(tensor.device, tensor.dtype).reshape(shape),
            self.std.to(tensor.device, tensor.dtype).reshape(shape),
        )

    def normalize(self, latent: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(latent)
        mask = _broadcast_support(support, latent)
        return ((latent - mean) / std).masked_fill(~mask, 0.0)

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(latent)
        return latent * std + mean


@dataclass(frozen=True, slots=True)
class FactoredLatentRecord:
    case_id: str
    subject_group_id: str
    domain: Domain
    split: str
    path: Path
    resume_key: str
    sidecar: dict[str, Any]


class PhotometryFactoredLatentBankIndex:
    """Lazy strict index over bank-v2 train or validation records."""

    def __init__(
        self,
        bank_dir: str | Path,
        split: str,
        *,
        expected_artifact_sha256: str | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("Unified Stage-2 permits only R/train and R/validation.")
        self.bank_dir = Path(bank_dir)
        self.split = split
        self.manifest = load_photometry_factored_latent_bank_manifest(
            self.bank_dir, expected_artifact_sha256=expected_artifact_sha256
        )
        self.artifact_sha256 = str(self.manifest["artifact_sha256"])
        self.records = self._classify_manifest_before_tensor_load()
        if not self.records:
            raise ValueError(f"Factored bank has no eligible R/{split} records.")

    def _classify_manifest_before_tensor_load(self) -> list[FactoredLatentRecord]:
        records: list[FactoredLatentRecord] = []
        for entry in self.manifest["records"]:
            sidecar = dict(entry.get("sidecar", {}))
            # The authoritative manifest loader already classifies every identity as R
            # and validates the complete sidecar before this method can load a payload.
            if sidecar.get("cohort") != "R":
                raise ValueError("Prospective identity reached the factored-bank index.")
            role = str(sidecar.get("split", ""))
            if role not in {"train", "validation"}:
                raise ValueError("Forbidden split role reached unified Stage-2.")
            if role != self.split:
                continue
            records.append(
                FactoredLatentRecord(
                    case_id=str(sidecar["record_identity"]),
                    subject_group_id=str(sidecar["subject_group_identity"]),
                    domain=Domain.from_dict(dict(sidecar["domain"])),
                    split=role,
                    path=safe_relative_path(self.bank_dir, str(entry["path"])),
                    resume_key=str(sidecar["resume_key"]),
                    sidecar=sidecar,
                )
            )
        return sorted(records, key=lambda item: item.case_id)

    def __len__(self) -> int:
        return len(self.records)

    def domains(self) -> list[Domain]:
        return [record.domain for record in self.records]

    def load(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        payload = _load_latent_record(record.path, expected_resume_key=record.resume_key)
        latent = payload["latent"].to(torch.float32)
        support = unpack_support_mask(
            payload["packed_local_valid_core_support"],
            payload["sidecar"]["local_valid_core_support_shape"],
        )
        if tuple(support.shape) != tuple(latent.shape[1:]) or not bool(support.any()):
            raise ValueError("Operational local-valid-core support is empty or misaligned.")
        return latent, support

    def load_batch(
        self, indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, list[Domain], list[FactoredLatentRecord]]:
        loaded = [self.load(int(index)) for index in indices]
        shapes = {tuple(item[0].shape) for item in loaded}
        if len(shapes) != 1:
            raise ValueError("A unified Stage-2 minibatch requires identical latent shapes.")
        latents = torch.stack([item[0] for item in loaded])
        supports = torch.stack([item[1] for item in loaded])[:, None]
        records = [self.records[int(index)] for index in indices]
        return latents, supports, [item.domain for item in records], records


def _broadcast_support(support: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    mask = support.to(device=latent.device, dtype=torch.bool)
    if latent.ndim == 4 and mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if latent.ndim == 5 and mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != latent.ndim:
        raise ValueError("Support mask rank does not match latent rank.")
    return mask.expand_as(latent)


__all__ = [
    "FactoredLatentRecord",
    "FactoredLatentStats",
    "PhotometryFactoredLatentBankIndex",
]
