"""Full-volume comparison and scoring from the retrospective zero-velocity bypass audit."""

from __future__ import annotations

import math

import numpy as np


def summarize_difference(totals, group, first, second, regions):
    """Accumulate squared/absolute differences across pooled voxels or channel-cells."""
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("Expected two same-shape [C,D,H,W] arrays.")
    for name, mask in regions.items():
        if mask.dtype != np.bool_ or mask.shape != first.shape[1:]:
            raise ValueError("Boolean region must match the spatial grid.")
        key = group + "/" + name
        agg = totals.setdefault(key, [0, 0.0, 0.0, 0.0])
        for a, b in zip(first, second, strict=True):
            # Slice chunks bound temporary memory on full-sized MR volumes.
            for start in range(0, len(a), 16):
                slab = slice(start, start + 16)
                selection = mask[slab]
                diff = np.subtract(a[slab][selection], b[slab][selection], dtype=np.float64)
                if diff.size:
                    if not np.isfinite(diff).all():
                        raise ValueError("Nonfinite diagnostic difference.")
                    agg[0] += int(diff.size)
                    agg[1] += float(np.abs(diff).sum())
                    agg[2] += float(np.square(diff).sum())
                    agg[3] = max(agg[3], float(np.abs(diff).max()))


def finish(totals):
    result = {}
    for key, (n, absolute, squared, peak) in sorted(totals.items()):
        result[key] = dict(
            count=n,
            mae=absolute / n if n else None,
            rmse=math.sqrt(squared / n) if n else None,
            max_abs=peak if n else None,
        )
    return result


def image_regions(core, shape, foreground=None):
    """Project latent cells to the decoder grid; this is a spatial proxy only."""
    projected = core
    for axis, (out_size, in_size) in enumerate(zip(shape, core.shape, strict=True)):
        if out_size < in_size or out_size % in_size:
            raise ValueError("Output and latent grids lack an exact integer scale.")
        projected = np.repeat(projected, out_size // in_size, axis=axis)
    if projected.shape != tuple(shape):
        raise ValueError("Projected core has incompatible spatial shape.")
    if foreground is None:
        return dict(projected_core=projected, outside_projected_core=~projected)
    if foreground.shape != tuple(shape) or foreground.dtype != np.bool_:
        raise ValueError("Canonical foreground must be boolean on the exact output grid.")
    return dict(
        foreground=foreground,
        foreground_core=foreground & projected,
        foreground_rim=foreground & ~projected,
        background=~foreground,
    )
