"""Tolerant checkpoint loading for warm-starting from external (e.g. MAISI/Pinaya) weights."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn


def load_state_dict_tolerant(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    log: Callable[[str], None] | None = None,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load `state_dict` into `module` tolerating missing/unexpected/shape-mismatched keys.

    `nn.Module.load_state_dict(..., strict=False)` tolerates missing and unexpected
    keys, but a key present in both with a *different shape* still raises
    `RuntimeError: size mismatch` even with `strict=False`. External checkpoints
    (MAISI/Pinaya) are exactly the case where same-named tensors are likely to differ
    in shape (e.g. different `latent_channels`), so shape-mismatched keys are filtered
    out here before calling `load_state_dict`, and logged separately from the
    missing/unexpected keys `load_state_dict` itself reports.
    """

    logger = log if log is not None else print
    module_state = module.state_dict()

    filtered: dict[str, torch.Tensor] = {}
    shape_mismatched: list[str] = []
    for key, value in state_dict.items():
        if key in module_state and module_state[key].shape != value.shape:
            shape_mismatched.append(key)
            continue
        filtered[key] = value

    result = module.load_state_dict(filtered, strict=False)

    if shape_mismatched:
        logger(f"load_state_dict_tolerant: skipped {len(shape_mismatched)} shape-mismatched keys: {shape_mismatched}")
    if result.missing_keys:
        logger(f"load_state_dict_tolerant: {len(result.missing_keys)} missing keys: {list(result.missing_keys)}")
    if result.unexpected_keys:
        logger(
            f"load_state_dict_tolerant: {len(result.unexpected_keys)} unexpected keys: {list(result.unexpected_keys)}"
        )
    return result
