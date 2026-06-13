"""Domain definitions for MRI field strength and contrast."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Any

import torch

from clbfield.official.mrixfields2026 import internal_modality_from_official, normalize_modality

FIELD_STRENGTHS_T: tuple[float, ...] = (0.1, 1.5, 3.0, 5.0, 7.0)
FIELD_MIN_T = min(FIELD_STRENGTHS_T)
FIELD_MAX_T = max(FIELD_STRENGTHS_T)


class Contrast(str, Enum):
    """Supported MRI contrast labels."""

    T1W = "T1w"
    T2W = "T2w"
    T2_FLAIR = "T2-FLAIR"

    @classmethod
    def parse(cls, value: str | "Contrast") -> "Contrast":
        if isinstance(value, cls):
            return value
        for contrast in cls:
            if value == contrast.value or value == contrast.name:
                return contrast
        try:
            internal_label = internal_modality_from_official(normalize_modality(str(value)))
        except ValueError:
            internal_label = ""
        for contrast in cls:
            if internal_label == contrast.value:
                return contrast
        raise ValueError(f"Unsupported contrast {value!r}. Expected one of {contrast_values()}.")


CONTRASTS: tuple[Contrast, ...] = (Contrast.T1W, Contrast.T2W, Contrast.T2_FLAIR)


def contrast_values() -> tuple[str, ...]:
    return tuple(contrast.value for contrast in CONTRASTS)


def validate_field_strength(field_strength_t: float) -> float:
    value = float(field_strength_t)
    if not any(isclose(value, allowed, rel_tol=0.0, abs_tol=1e-6) for allowed in FIELD_STRENGTHS_T):
        raise ValueError(f"Unsupported field strength {value}. Expected one of {FIELD_STRENGTHS_T}.")
    return value


@dataclass(frozen=True, slots=True)
class Domain:
    """MRI acquisition domain represented by field strength and contrast."""

    field_strength_t: float
    contrast: Contrast | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_strength_t", validate_field_strength(self.field_strength_t))
        object.__setattr__(self, "contrast", Contrast.parse(self.contrast))

    @property
    def contrast_index(self) -> int:
        return CONTRASTS.index(Contrast.parse(self.contrast))

    @property
    def label(self) -> str:
        return f"{self.field_strength_t:g}T/{Contrast.parse(self.contrast).value}"

    def field_encoding(
        self,
        *,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return a continuous scalar encoding for field strength."""

        value = self.field_strength_t
        if normalize:
            value = (value - FIELD_MIN_T) / (FIELD_MAX_T - FIELD_MIN_T)
        return torch.tensor([value], dtype=dtype, device=device)

    def contrast_encoding(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return a categorical one-hot encoding for contrast."""

        encoded = torch.zeros(len(CONTRASTS), dtype=dtype, device=device)
        encoded[self.contrast_index] = 1
        return encoded

    def conditioning_vector(
        self,
        *,
        normalize_field: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Concatenate field and contrast encodings into a single vector."""

        return torch.cat(
            [
                self.field_encoding(normalize=normalize_field, dtype=dtype, device=device),
                self.contrast_encoding(dtype=dtype, device=device),
            ],
            dim=0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"field_strength_t": self.field_strength_t, "contrast": Contrast.parse(self.contrast).value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Domain":
        field = data.get("field_strength_t", data.get("field_strength"))
        if field is None:
            raise ValueError("Domain mapping must include field_strength_t.")
        return cls(field_strength_t=float(field), contrast=Contrast.parse(str(data["contrast"])))


def domain_from_any(value: Domain | dict[str, Any]) -> Domain:
    if isinstance(value, Domain):
        return value
    if isinstance(value, dict):
        return Domain.from_dict(value)
    raise TypeError(f"Cannot parse Domain from {type(value).__name__}.")
