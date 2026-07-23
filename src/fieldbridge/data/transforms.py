"""Small tensor transforms for synthetic and injected-loader datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def identity_transform(image: torch.Tensor) -> torch.Tensor:
    return image


def normalize_zero_mean_unit_variance(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (image - image.mean()) / (image.std().clamp_min(eps))


_QUANTILE_ELEMENT_LIMIT = 2**24  # torch.quantile's practical size limit on some backends

# Slack on the official [0, 1] contract before `assert_official_unit_range` fires. Covers
# float32 round-off from the challenge's own preprocessing, not a genuinely rescaled volume.
_UNIT_RANGE_TOLERANCE = 1e-4


def assert_official_unit_range(image: torch.Tensor) -> torch.Tensor:
    """Pass the volume through unchanged after checking the official [0, 1] contract.

    The MRIxFields2026 data ships pre-normalized to [0, 1] and the official format
    forbids rescaling intensity in training *or* evaluation (see the `mrixfields-format`
    skill). So the correct volume transform is a no-op — this exists to make that a
    checked no-op rather than an unchecked one: a volume that is not already in [0, 1]
    means the loader, the manifest, or the preprocessing is wrong, and silently training
    on it would produce metrics that are not comparable to the challenge leaderboard.

    Use this, not `normalize_percentile_clip_to_unit_range`, for official challenge data.
    """

    low, high = float(image.min()), float(image.max())
    limit_low, limit_high = -_UNIT_RANGE_TOLERANCE, 1.0 + _UNIT_RANGE_TOLERANCE
    if low < limit_low or high > limit_high:
        raise ValueError(
            f"Volume violates the official [0, 1] intensity contract: range "
            f"[{low:.6f}, {high:.6f}]. MRIxFields2026 data is pre-normalized to [0, 1] and "
            "must not be rescaled; check the loader/manifest instead of renormalizing here."
        )
    return image


def normalize_percentile_clip_to_unit_range(
    image: torch.Tensor,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip to [lower_percentile, upper_percentile] then affine-map to [-1, 1].

    DO NOT use on official MRIxFields2026 data — use `assert_official_unit_range`. The
    official format ships [0, 1] volumes and forbids rescaling intensity in training or
    evaluation; this transform both clips (discarding the top/bottom 0.5% of the
    histogram, which in MRI is signal) and remaps the range, so metrics computed on its
    output are not comparable to the challenge leaderboard. It is also per-volume, which
    on a cross-field task partially normalizes away the field-dependent intensity
    differences the model is supposed to learn.

    Kept for non-official data (external cohorts with raw scanner intensities) where a
    robust intensity normalization is genuinely needed.

    Percentile clip rather than raw min-max: MRI intensity has a long right tail (a
    handful of very bright voxels — vessels, motion/susceptibility artifacts) that would
    otherwise dominate the scale and compress the diagnostically-relevant anatomy into a
    narrow sub-range.
    """

    flat = image.reshape(-1)
    if flat.numel() > _QUANTILE_ELEMENT_LIMIT:
        flat = flat[:: flat.numel() // _QUANTILE_ELEMENT_LIMIT + 1]
    quantiles = torch.quantile(
        flat.float(),
        torch.tensor([lower_percentile / 100.0, upper_percentile / 100.0], dtype=flat.dtype),
    )
    lo, hi = quantiles[0], quantiles[1]
    if (hi - lo) <= eps:
        raise ValueError(
            f"normalize_percentile_clip_to_unit_range: degenerate image, "
            f"percentile range [{lo.item()}, {hi.item()}] is not wide enough to normalize."
        )
    clamped = image.clamp(min=lo, max=hi)
    return 2.0 * (clamped - lo) / (hi - lo) - 1.0


def random_crop(
    image: torch.Tensor,
    *,
    patch_size: Iterable[int],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random crop of `patch_size` over the trailing spatial dims.

    Needed for full-resolution 3D volumes (e.g. 364x436x364): decoding back toward full
    resolution with enough latent channels for the diffuser allocates tens of GB per
    intermediate activation, OOMing on essentially any GPU. Patch-based training (random
    crops, not the full volume) is the standard fix for 3D medical imaging at this
    scale — this is not optional once spatial_dims=3 is combined with a real volume's
    resolution, regardless of how much compute budget is available.
    """

    patch = tuple(int(p) for p in patch_size)
    spatial_shape = tuple(image.shape[-len(patch) :])
    starts = []
    for size, size_patch in zip(spatial_shape, patch):
        if size_patch > size:
            raise ValueError(f"patch_size {patch} exceeds image spatial shape {spatial_shape}.")
        starts.append(
            0
            if size_patch == size
            else int(torch.randint(0, size - size_patch + 1, (1,), generator=generator).item())
        )
    slices = tuple(slice(start, start + size_patch) for start, size_patch in zip(starts, patch))
    return image[(..., *slices)]


@dataclass(frozen=True, slots=True)
class StratifiedCropConfig:
    """Quotas for `stratified_crop`'s foreground / border / air strata.

    Uniform random crops are not viable here: on a 364x436x364 volume roughly 67-85% of
    random 64^3 crops contain <5% brain (measured by simulation over plausible brain
    extents), so most of the compute reconstructs air and the training loss is dominated
    by trivially-easy empty patches.

    Filtering to brain-only crops is the wrong correction. The three challenge metrics are
    computed over the *whole* volume, so every background voxel counts toward the score;
    a model that never saw air during training still has to tile air at inference and will
    emit whatever it extrapolates there. The brain/air boundary is also the highest-
    gradient region in the volume and the one SSIM is most sensitive to. So: bias toward
    foreground, but keep air and border deliberately represented.

    Fractions are relative weights over the three strata and are normalized; they are
    starting points to sweep, not derived constants. `min_foreground_fraction` is the
    floor that makes a crop count as foreground; `border` requires a crop to straddle the
    mask edge (some foreground AND some background).
    """

    foreground: float = 0.7
    border: float = 0.2
    air: float = 0.1
    min_foreground_fraction: float = 0.1
    max_attempts: int = 50

    def __post_init__(self) -> None:
        for name in ("foreground", "border", "air"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"StratifiedCropConfig.{name} must be non-negative, got {getattr(self, name)}.")
        if self.total_weight <= 0.0:
            raise ValueError("StratifiedCropConfig requires at least one stratum with positive weight.")
        if not 0.0 < float(self.min_foreground_fraction) <= 1.0:
            raise ValueError(
                f"min_foreground_fraction must be in (0, 1], got {self.min_foreground_fraction}."
            )
        if int(self.max_attempts) < 1:
            raise ValueError(f"max_attempts must be positive, got {self.max_attempts}.")

    @property
    def total_weight(self) -> float:
        return float(self.foreground) + float(self.border) + float(self.air)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "StratifiedCropConfig | None":
        """Build from a config mapping; `None`/empty means "keep uniform random cropping"."""

        if not data:
            return None
        defaults = cls()
        return cls(
            foreground=float(data.get("foreground", defaults.foreground)),
            border=float(data.get("border", defaults.border)),
            air=float(data.get("air", defaults.air)),
            min_foreground_fraction=float(
                data.get("min_foreground_fraction", defaults.min_foreground_fraction)
            ),
            max_attempts=int(data.get("max_attempts", defaults.max_attempts)),
        )


def stratified_crop(
    image: torch.Tensor,
    *,
    patch_size: Iterable[int],
    mask: torch.Tensor,
    config: StratifiedCropConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random crop drawn from one of three strata (foreground / border / air).

    `mask` is a binary foreground mask broadcastable to `image`'s trailing spatial dims
    (see `data/masks.py`). Rejection sampling: a stratum is chosen by weight, then crops
    are drawn until one satisfies that stratum, up to `config.max_attempts`. On exhaustion
    the last draw is returned rather than raising — a volume can legitimately have no
    qualifying crop for a stratum (e.g. `air` on a tightly-cropped volume), and failing a
    long training run over that would be worse than a slightly off-quota batch.
    """

    patch = tuple(int(p) for p in patch_size)
    spatial = tuple(int(d) for d in image.shape[-len(patch) :])
    mask_spatial = tuple(int(d) for d in mask.shape[-len(patch) :])
    if mask_spatial != spatial:
        raise ValueError(
            f"mask spatial shape {mask_spatial} does not match image spatial shape {spatial}."
        )

    stratum = _choose_stratum(config, generator)
    binary = (mask > 0.5)
    while binary.ndim > len(patch):
        binary = binary[0]

    voxels = 1
    for size in patch:
        voxels *= size
    floor = float(config.min_foreground_fraction)

    crop = None
    for _ in range(int(config.max_attempts)):
        starts = _random_starts(spatial, patch, generator)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, patch))
        fraction = float(binary[slices].sum()) / voxels
        crop = image[(..., *slices)]
        if stratum == "foreground" and fraction >= floor:
            return crop
        if stratum == "air" and fraction == 0.0:
            return crop
        if stratum == "border" and 0.0 < fraction < 1.0:
            return crop
    assert crop is not None  # max_attempts >= 1 is validated in StratifiedCropConfig
    return crop


def _choose_stratum(config: StratifiedCropConfig, generator: torch.Generator | None) -> str:
    weights = torch.tensor(
        [float(config.foreground), float(config.border), float(config.air)], dtype=torch.float32
    )
    index = int(torch.multinomial(weights / weights.sum(), 1, generator=generator).item())
    return ("foreground", "border", "air")[index]


def _random_starts(
    spatial: tuple[int, ...], patch: tuple[int, ...], generator: torch.Generator | None
) -> list[int]:
    starts: list[int] = []
    for size, size_patch in zip(spatial, patch):
        if size_patch > size:
            raise ValueError(f"patch_size {patch} exceeds image spatial shape {spatial}.")
        if size_patch == size:
            starts.append(0)
        else:
            starts.append(int(torch.randint(0, size - size_patch + 1, (1,), generator=generator).item()))
    return starts


def compose(transforms: Iterable[TensorTransform]) -> TensorTransform:
    ordered = tuple(transforms)

    def _apply(image: torch.Tensor) -> torch.Tensor:
        value = image
        for transform in ordered:
            value = transform(value)
        return value

    return _apply

