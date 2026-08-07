"""Deterministic rendering for the frozen Gate 0.1 montage specification."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fieldbridge.data.domains import Contrast


@dataclass(slots=True)
class Gate01MontageCollector:
    """Retain only predeclared 2-D slices while full volumes stream past."""

    specification: Mapping[str, Any]
    selected: dict[tuple[str, float, float], dict[str, Any]] = field(
        default_factory=dict
    )

    def observe(self, case: Any, predictions: Mapping[str, torch.Tensor]) -> None:
        contrast = Contrast.parse(case.target_domain.contrast).value
        key = (
            contrast,
            float(case.source_domain.field_strength_t),
            float(case.target_domain.field_strength_t),
        )
        allowed = {
            (
                item_contrast,
                float(pair["source_field_t"]),
                float(pair["target_field_t"]),
            )
            for item_contrast in self.specification["contrasts"]
            for pair in self.specification["directed_pairs_per_contrast"]
        }
        if key not in allowed:
            return
        volumes = {"target": case.target, **dict(predictions)}
        display_order = tuple(self.specification["display_order"])
        arrays = {
            name: _volume_array(volumes[name], name) for name in display_order
        }
        depth = next(iter(arrays.values())).shape[-1]
        slice_indices = [
            int(round(float(position) * (depth - 1)))
            for position in self.specification["relative_slice_positions"]
        ]
        slices = {
            name: [
                np.ascontiguousarray(array[..., index], dtype=np.float32)
                for index in slice_indices
            ]
            for name, array in arrays.items()
        }
        values = [panel for method in display_order for panel in slices[method]]
        display_min = min(float(panel.min()) for panel in values)
        display_max = max(float(panel.max()) for panel in values)
        self.selected[key] = {
            "case_identity_sha256": case.case_identity_sha256,
            "array_sha256": dict(sorted(case.array_sha256.items())),
            "slice_indices": slice_indices,
            "display_range": [display_min, display_max],
            "slices": slices,
        }


def render_gate01_montages(
    collector: Gate01MontageCollector,
    out_dir: str | Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Render fixed grayscale PNGs and a hash-linked provenance manifest."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = {
        (
            contrast,
            float(pair["source_field_t"]),
            float(pair["target_field_t"]),
        )
        for contrast in collector.specification["contrasts"]
        for pair in collector.specification["directed_pairs_per_contrast"]
    }
    if require_complete and set(collector.selected) != expected:
        missing = sorted(expected - set(collector.selected))
        raise ValueError(f"Scientific Gate 0.1 montage selection is incomplete: {missing}.")

    entries: list[dict[str, Any]] = []
    methods = tuple(collector.specification["display_order"])
    for key, item in sorted(collector.selected.items()):
        contrast, source, target = key
        rows = len(item["slice_indices"])
        panels = item["slices"]
        panel_shape = panels[methods[0]][0].shape
        height, width = panel_shape
        canvas = np.zeros(
            (rows * height + rows - 1, len(methods) * width + len(methods) - 1),
            dtype=np.uint8,
        )
        low, high = item["display_range"]
        for row_index in range(rows):
            for column_index, method in enumerate(methods):
                normalized = _normalize_panel(panels[method][row_index], low, high)
                y = row_index * (height + 1)
                x = column_index * (width + 1)
                canvas[y : y + height, x : x + width] = normalized
        filename = (
            f"{contrast.replace('-', '_')}_{source:g}T-to-{target:g}T.png"
        )
        encoded = _encode_grayscale_png(canvas)
        _write_bytes_atomic(root / filename, encoded)
        entries.append(
            {
                "contrast": contrast,
                "source_field_t": source,
                "target_field_t": target,
                "case_identity_sha256": item["case_identity_sha256"],
                "array_sha256": item["array_sha256"],
                "png": filename,
                "png_sha256": hashlib.sha256(encoded).hexdigest(),
                "slice_indices": item["slice_indices"],
                "display_range": item["display_range"],
                "grid": {
                    "rows": "slice_indices",
                    "columns": list(methods),
                    "separator_pixels": 1,
                    "interpolation": "none",
                },
            }
        )

    spec_json = json.dumps(
        collector.specification, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest = {
        "contract_version": "gate01-deterministic-montage-render-v1",
        "renderer": "stdlib grayscale PNG; filter=none; zlib level=9; no timestamps",
        "specification_sha256": hashlib.sha256(spec_json).hexdigest(),
        "complete_frozen_selection": set(collector.selected) == expected,
        "entries": entries,
    }
    serialized = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    _write_bytes_atomic(root / "montage_manifest.json", serialized)
    return {
        **manifest,
        "manifest": "montage_manifest.json",
        "manifest_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _volume_array(value: torch.Tensor, name: str) -> np.ndarray:
    array = value.detach().cpu().to(torch.float32).numpy().squeeze()
    if array.ndim != 3:
        raise ValueError(f"Gate 0.1 montage {name} must resolve to one 3-D volume.")
    if not np.isfinite(array).all():
        raise ValueError(f"Gate 0.1 montage {name} contains non-finite values.")
    return array


def _normalize_panel(panel: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros(panel.shape, dtype=np.uint8)
    scaled = np.clip((panel.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _encode_grayscale_png(image: np.ndarray) -> bytes:
    height, width = image.shape
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


__all__ = ["Gate01MontageCollector", "render_gate01_montages"]
