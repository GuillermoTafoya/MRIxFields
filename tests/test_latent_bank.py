from __future__ import annotations

import json

import pytest
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.latent_bank import (
    LatentBankConfig,
    _core_blocks,
    build_latent_bank,
    encode_latent,
)
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder

VOL = 16
FACTOR = 4


def _frozen_vae():
    encoder = KLVAEEncoder(in_channels=1, base_channels=8, latent_channels=2, spatial_dims=3)
    decoder = KLVAEDecoder(out_channels=1, base_channels=8, latent_channels=2, spatial_dims=3)
    encoder.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval()
    return encoder, decoder


def _volume_for(record: VolumeRecord) -> torch.Tensor:
    seed = abs(hash(record.case_id)) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 1, VOL, VOL, VOL, generator=generator)


def _records() -> dict[str, list[VolumeRecord]]:
    domains = [
        Domain(0.1, Contrast.T1W),
        Domain(3.0, Contrast.T2W),
        Domain(7.0, Contrast.T2_FLAIR),
    ]
    by_split: dict[str, list[VolumeRecord]] = {"train": [], "validation": [], "test": []}
    for split in by_split:
        for i, domain in enumerate(domains):
            by_split[split].append(
                VolumeRecord(
                    case_id=f"{split}_{i}",
                    image_path=f"synthetic/{split}_{i}.nii.gz",
                    domain=domain,
                    subject_id=f"s{split}{i}",
                )
            )
    return by_split


def test_core_blocks_partition_is_aligned_to_factor() -> None:
    assert _core_blocks(16, 8, 4) == [(0, 8), (8, 8)]
    assert _core_blocks(24, 8, 4) == [(0, 8), (8, 8), (16, 8)]
    # remainder stays a multiple of factor.
    assert _core_blocks(20, 8, 4) == [(0, 8), (8, 8), (16, 4)]


def test_core_blocks_rejects_unaligned_extent() -> None:
    with pytest.raises(ValueError, match="multiple of downsample factor"):
        _core_blocks(15, 8, 4)


def test_tiled_encode_equals_full_when_halo_covers_volume() -> None:
    encoder, _ = _frozen_vae()
    volume = torch.rand(1, 1, VOL, VOL, VOL, generator=torch.Generator().manual_seed(0))
    full, used_full = encode_latent(
        encoder, volume, Domain(3.0, Contrast.T2W),
        strategy="full", block_size=(8, 8, 8), halo=(16, 16, 16), precision="float32",
    )
    tiled, used_tiled = encode_latent(
        encoder, volume, Domain(3.0, Contrast.T2W),
        strategy="tiled", block_size=(8, 8, 8), halo=(16, 16, 16), precision="float32",
    )
    assert used_full == "full" and used_tiled == "tiled"
    assert full.shape == (1, 2, VOL // FACTOR, VOL // FACTOR, VOL // FACTOR)
    assert torch.allclose(full, tiled, atol=1e-5)


def test_build_latent_bank_writes_files_stats_and_roundtrip(tmp_path) -> None:
    encoder, decoder = _frozen_vae()
    config = LatentBankConfig(
        out_dir=tmp_path,
        strategy="tiled",
        store_dtype="float16",
        precision="float32",
        block_size=(8, 8, 8),
        halo=(4, 4, 4),
        roundtrip_samples=2,
    )
    manifest = build_latent_bank(
        encoder=encoder,
        decoder=decoder,
        records_by_split=_records(),
        config=config,
        device=torch.device("cpu"),
        checkpoint_sha256="deadbeef",
        git_commit="testcommit",
        vae_config_path="configs/experiment/stage1_vae_v2_fgw_freebits.yaml",
        volume_loader=_volume_for,
        log=False,
    )

    # per-record files exist with the expected latent shape.
    latent_file = tmp_path / "train" / "train_0.pt"
    assert latent_file.exists()
    payload = torch.load(latent_file, map_location="cpu")
    assert list(payload["latent"].shape) == [2, VOL // FACTOR, VOL // FACTOR, VOL // FACTOR]
    assert payload["latent"].dtype == torch.float16
    assert payload["vae_checkpoint_sha256"] == "deadbeef"

    # latent_stats over the train split.
    stats = json.loads((tmp_path / "latent_stats.json").read_text())
    assert stats["computed_over"] == "train"
    assert len(stats["per_channel_std"]) == 2
    assert len(stats["per_channel_mean"]) == 2

    # manifest + counts + roundtrip.
    assert manifest["counts"]["train"]["encoded"] == 3
    assert manifest["counts"]["test"]["total"] == 3
    assert manifest["roundtrip"]["mean_ssim3d"] is not None
    assert len(manifest["roundtrip"]["per_case"]) == 2
    bank_manifest = json.loads((tmp_path / "latent_bank_manifest.json").read_text())
    assert len(bank_manifest["records"]) == 9


def test_build_latent_bank_is_idempotent_on_rerun(tmp_path) -> None:
    encoder, decoder = _frozen_vae()
    config = LatentBankConfig(
        out_dir=tmp_path, block_size=(8, 8, 8), halo=(4, 4, 4), roundtrip_samples=1
    )
    common = dict(
        encoder=encoder, decoder=decoder, records_by_split=_records(),
        device=torch.device("cpu"), checkpoint_sha256="x", git_commit="c",
        vae_config_path="cfg", volume_loader=_volume_for, log=False,
    )
    first = build_latent_bank(config=config, **common)
    second = build_latent_bank(config=config, **common)
    assert first["counts"]["train"]["encoded"] == 3
    assert second["counts"]["train"]["encoded"] == 0
    assert second["counts"]["train"]["skipped"] == 3


def test_build_latent_bank_rejects_block_not_multiple_of_factor(tmp_path) -> None:
    encoder, decoder = _frozen_vae()
    config = LatentBankConfig(out_dir=tmp_path, block_size=(6, 8, 8), halo=(4, 4, 4))
    with pytest.raises(ValueError, match="multiples of the VAE downsample factor"):
        build_latent_bank(
            encoder=encoder, decoder=decoder, records_by_split=_records(),
            config=config, device=torch.device("cpu"), checkpoint_sha256="x",
            git_commit="c", vae_config_path="cfg", volume_loader=_volume_for, log=False,
        )


def test_build_latent_bank_requires_frozen_vae(tmp_path) -> None:
    encoder = KLVAEEncoder(in_channels=1, base_channels=8, latent_channels=2, spatial_dims=3)
    decoder = KLVAEDecoder(out_channels=1, base_channels=8, latent_channels=2, spatial_dims=3)
    # encoder NOT frozen → assert_frozen should fail.
    config = LatentBankConfig(out_dir=tmp_path, block_size=(8, 8, 8), halo=(4, 4, 4))
    with pytest.raises(RuntimeError, match="frozen"):
        build_latent_bank(
            encoder=encoder, decoder=decoder, records_by_split=_records(),
            config=config, device=torch.device("cpu"), checkpoint_sha256="x",
            git_commit="c", vae_config_path="cfg", volume_loader=_volume_for, log=False,
        )


def test_build_latent_bank_command_is_exposed_in_help() -> None:
    from fieldbridge.cli import build_parser

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["build-latent-bank", "--help"])
    assert exc_info.value.code == 0
