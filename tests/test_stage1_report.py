import builtins
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.contracts import RawBatch
from fieldbridge.data.datasets import collate_raw_batches
from fieldbridge.data.domains import Domain
from fieldbridge.evaluation.stage1_report import (
    _central_first_spatial_axis_slice,
    _hann_window_3d,
    _matplotlib_pyplot,
    run_stage1_eval,
    sliding_window_reconstruct,
)
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder


class _IdentityEncoder:
    """Duck-typed encoder whose latent mean IS the input tile (for blending-math tests)."""

    def encode_dist(self, x: torch.Tensor, domain: object) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros_like(x)


class _IdentityDecoder:
    def decode(self, z: torch.Tensor, domain: object) -> torch.Tensor:
        return z


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_sliding_window_blending_reconstructs_identity_exactly(overlap: float) -> None:
    # With identity encode/decode, the overlap + Hann weighting is a partition of unity:
    # the blended output must equal the input exactly (no seams introduced by the window).
    torch.manual_seed(0)
    image = torch.rand(1, 1, 40, 40, 40)

    recon = sliding_window_reconstruct(
        _IdentityEncoder(), _IdentityDecoder(), image, patch_size=(16, 16, 16), domain=None, overlap=overlap
    )

    assert recon.shape == image.shape
    assert torch.allclose(recon, image, atol=1e-5)


def test_sliding_window_rejects_out_of_range_overlap() -> None:
    image = torch.rand(1, 1, 32, 32, 32)
    with pytest.raises(ValueError):
        sliding_window_reconstruct(
            _IdentityEncoder(), _IdentityDecoder(), image, patch_size=(16, 16, 16), domain=None, overlap=1.0
        )


def test_hann_window_tapers_from_center_to_face() -> None:
    window = _hann_window_3d((16, 16, 16), torch.device("cpu"), torch.float32)
    assert window.shape == (16, 16, 16)
    # Center weight is (near) the max; the faces taper to the clamp floor.
    assert torch.isclose(window[8, 8, 8], window.max(), atol=1e-4)
    assert window[0, 8, 8] < window[8, 8, 8]


class _FullVolumeDataset(Dataset[RawBatch]):
    """Volumes larger than the patch, so the sliding window actually tiles (with edge overlap)."""

    def __init__(self, *, num_samples: int = 2, volume_shape: tuple[int, int, int, int] = (1, 40, 40, 40)) -> None:
        self.num_samples = num_samples
        self.volume_shape = volume_shape

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> RawBatch:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(self.volume_shape, generator=generator)
        domain = Domain(1.5, "T2w")
        return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"c{index}"})


def _loader() -> DataLoader[RawBatch]:
    return DataLoader(_FullVolumeDataset(), batch_size=1, shuffle=False, collate_fn=collate_raw_batches)


def test_sliding_window_reconstruct_preserves_shape() -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)
    image = torch.rand(1, 1, 40, 40, 40)

    recon = sliding_window_reconstruct(encoder, decoder, image, patch_size=(16, 16, 16), domain=None)

    assert recon.shape == image.shape
    assert torch.isfinite(recon).all()
    assert recon.min() >= 0.0 and recon.max() <= 1.0


class _MultiDomainDataset(Dataset[RawBatch]):
    """Volumes across several field strengths, with duplicates, to exercise per-domain dedup."""

    def __init__(self) -> None:
        self.fields = [1.5, 1.5, 3.0, 3.0, 7.0]  # 3 distinct domains, with repeats

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index: int) -> RawBatch:
        image = torch.rand(1, 24, 24, 24)
        domain = Domain(self.fields[index], "T2w")
        return RawBatch(image=image, source_domain=domain, target_domain=domain, metadata={"case_id": f"c{index}"})


def test_run_stage1_eval_per_domain_dedups_field_strength(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)
    loader = DataLoader(_MultiDomainDataset(), batch_size=1, shuffle=False, collate_fn=collate_raw_batches)

    payload = run_stage1_eval(
        encoder=encoder,
        decoder=decoder,
        loader=loader,
        patch_size=(16, 16, 16),
        out_dir=tmp_path,
        num_samples=10,
        device=torch.device("cpu"),
        lpips_num_slices=0,
        per_domain=True,
    )

    # 3 distinct field strengths -> exactly 3 samples despite 5 records.
    assert payload["num_samples"] == 3
    domains = {s["domain"] for s in payload["per_sample"]}
    assert domains == {"1.5T/T2w", "3T/T2w", "7T/T2w"}


def test_run_stage1_eval_writes_metrics_and_plots(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)

    payload = run_stage1_eval(
        encoder=encoder,
        decoder=decoder,
        loader=_loader(),
        patch_size=(16, 16, 16),
        out_dir=tmp_path,
        num_samples=2,
        device=torch.device("cpu"),
        lpips_num_slices=0,  # skips lpips net (LPIPS optional dep may be absent)
        loss_curve=[3.0, 2.0, 1.0],
    )

    assert payload["num_samples"] == 2
    assert set(payload["mean"]) >= {"nrmse", "ssim3d", "mse", "mae"}
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "diagnostics.png").exists()
    assert (tmp_path / "loss_curve.png").exists()
    written = json.loads((tmp_path / "metrics.json").read_text())
    assert len(written["per_sample"]) == 2


class _ConstantDecoder(torch.nn.Module):
    """Collapsed decoder: emits a constant, no structure (a model/training fault)."""

    def decode(self, z: torch.Tensor, domain: object) -> torch.Tensor:
        shape = (z.shape[0], 1, *(int(s) * 4 for s in z.shape[2:]))
        return torch.full(shape, 0.08)


class _SquashedDecoder(torch.nn.Module):
    """Perfect anatomy squashed into a narrow band that never reaches the 0 background.

    A pure calibration fault: structure is intact (correlation ~1) but the floor is wrong.
    """

    def __init__(self, volume: torch.Tensor) -> None:
        super().__init__()
        self._volume = volume

    def decode(self, z: torch.Tensor, domain: object) -> torch.Tensor:
        return self._volume * 0.25 + 0.4


def _single_volume_loader(volume: torch.Tensor) -> list[RawBatch]:
    domain = Domain(3.0, "T1w")
    return [
        collate_raw_batches(
            [RawBatch(image=volume[0], source_domain=domain, target_domain=domain, metadata={"case_id": "c0"})]
        )
    ]


def test_correlation_separates_constant_collapse_from_range_compression(tmp_path) -> None:
    # Both faults score SSIM3D ~ 0 and so are indistinguishable by the challenge metrics:
    # a recon whose DC level is wrong distorts SSIM's luminance term regardless of whether
    # structure is present. Correlation is the term that tells them apart.
    torch.manual_seed(0)
    coords = torch.linspace(-1, 1, 16)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    # Official [0, 1] contract: background is exactly 0, anatomy fills (0, 1].
    anatomy = (torch.sin(6 * x) * torch.cos(6 * y) + 1.0) / 2.0
    volume = torch.where((x**2 + y**2 + z**2).sqrt() < 0.7, anatomy, torch.zeros_like(x))
    volume = volume[None, None]

    encoder = KLVAEEncoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)

    def evaluate(decoder: object) -> dict:
        return run_stage1_eval(
            encoder=encoder,
            decoder=decoder,
            loader=_single_volume_loader(volume),
            patch_size=(16, 16, 16),
            out_dir=tmp_path,
            num_samples=1,
            device=torch.device("cpu"),
            lpips_num_slices=0,
            overlap=0.0,
        )["mean"]

    constant = evaluate(_ConstantDecoder())
    squashed = evaluate(_SquashedDecoder(volume))

    # SSIM cannot tell them apart: both sit low (~0.06 and ~0.14), so neither the value
    # nor the gap between them identifies which fault occurred. The threshold is looser
    # than on the old [-1, 1] contract, where the luminance-term sign flip pinned SSIM
    # harder; the point of the test is the *lack* of separation, not the exact value.
    assert abs(constant["ssim3d"]) < 0.2
    assert abs(squashed["ssim3d"]) < 0.2
    # Correlation does, decisively.
    assert abs(constant["correlation"]) < 0.05
    assert squashed["correlation"] > 0.95


def test_range_guard_rejects_denormalized_target(tmp_path) -> None:
    # A target left in raw scanner intensities must fail loudly rather than silently
    # producing meaningless metrics against a [0, 1] recon.
    raw_volume = torch.rand(1, 1, 16, 16, 16) * 800.0
    encoder = KLVAEEncoder(base_channels=4, latent_channels=4, spatial_dims=3, num_res_blocks=1)

    with pytest.raises(ValueError, match=r"\[0, 1\] contract"):
        run_stage1_eval(
            encoder=encoder,
            decoder=_ConstantDecoder(),
            loader=_single_volume_loader(raw_volume),
            patch_size=(16, 16, 16),
            out_dir=tmp_path,
            num_samples=1,
            device=torch.device("cpu"),
            lpips_num_slices=0,
            overlap=0.0,
        )


def test_missing_matplotlib_names_evaluation_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def missing_matplotlib(name, *args, **kwargs):
        if name == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_matplotlib)

    with pytest.raises(ImportError) as exc_info:
        _matplotlib_pyplot()

    message = str(exc_info.value)
    assert "'evaluation' extra" in message
    assert 'pip install -e ".[nifti,evaluation]"' in message


def test_diagnostic_slice_uses_first_raw_spatial_axis_without_plane_claim() -> None:
    volume = torch.empty(1, 1, 3, 4, 5)
    coordinates = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    for axis_index in range(3):
        volume[0, 0, axis_index] = coordinates + axis_index * 100

    selected = _central_first_spatial_axis_slice(volume)
    expected = np.rot90(volume[0, 0, 1].numpy())

    assert np.array_equal(selected, expected)
    assert not np.array_equal(selected, np.rot90(volume[0, 0, 0].numpy()))
    contract = _central_first_spatial_axis_slice.__doc__ or ""
    assert "first raw spatial axis" in contract
    assert "anatomical plane" in contract
    assert "axial" not in contract.lower()
