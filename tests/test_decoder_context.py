"""Synthetic coverage for native source context and historical decoder replay."""

from __future__ import annotations

import torch

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.evaluation.decoder_context import (
    HISTORICAL_DENORMALIZE,
    SOURCE_NATIVE_EXTERIOR,
    decoder_input_latent,
)
from fieldbridge.evaluation.low_memory_vae import enable_low_memory
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder


def test_decoder_context_preserves_transported_core_and_native_exterior():
    native = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 2, 4) / 11
    core = torch.zeros(1, 1, 2, 2, 4, dtype=torch.bool)
    core[..., 1:3] = True
    stats = FactoredLatentStats(
        mean=torch.tensor([1.25, -0.7]),
        std=torch.tensor([0.4, 2.0]),
        supported_count=torch.tensor([9, 11]),
        artifact_sha256="a" * 64,
    )
    start = stats.normalize(native, core)
    start_copy = start.clone()
    native_copy = native.clone()
    assert torch.count_nonzero(start.masked_select(~core)) == 0
    # The learned field is free to move the endpoint on the exterior too.
    for endpoint in (start, start + 0.5):
        before = endpoint.clone()
        expected = endpoint * stats.std[None, :, None, None, None]
        expected = expected + stats.mean[None, :, None, None, None]
        replay = decoder_input_latent(
            endpoint, stats=stats, native_source=native, valid_core=core,
            policy=HISTORICAL_DENORMALIZE,
        )
        current = decoder_input_latent(
            endpoint, stats=stats, native_source=native, valid_core=core,
            policy=SOURCE_NATIVE_EXTERIOR,
        )
        assert torch.equal(replay, expected)
        assert torch.equal(current, torch.where(core, expected, native))
        assert not torch.equal(current.masked_select(~core), replay.masked_select(~core))
        assert torch.equal(endpoint, before)
    assert torch.equal(start, start_copy)
    assert torch.equal(native, native_copy)


def _assert_archive_tolerance(actual, reference, *, bf16):
    assert actual.dtype == reference.dtype
    assert actual.shape == reference.shape
    delta = actual.double() - reference.double()
    rmse = delta.square().mean().sqrt()
    peak = delta.abs().max()
    if bf16:
        assert rmse <= 0.003 + 0.005 * reference.double().square().mean().sqrt()
        assert peak <= 0.04 + 0.02 * reference.double().abs().max()
    else:
        assert rmse <= 1e-4 + 1e-4 * reference.double().square().mean().sqrt()
        assert peak <= 1e-3 + 1e-3 * reference.double().abs().max()


@torch.inference_mode()
def test_low_memory_full_volume_decoder_preserves_groupnorm_and_bf16_result():
    # The unequal axes and local perturbation put spatial context across slab boundaries.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(39413)
        kwargs = dict(base_channels=4, latent_channels=2, num_res_blocks=1, spatial_dims=3)
        encoder = KLVAEEncoder(**kwargs).eval().requires_grad_(False)
        decoder = KLVAEDecoder(**kwargs).eval().requires_grad_(False)
        source = torch.rand(1, 1, 16, 20, 24)
    source[..., :3, :, :] = 0
    source[..., 6:13, 4:12, 7:17] *= 0.1
    domain = [Domain(7.0, Contrast.T1W)]
    mean, logvar = encoder.encode_dist(source, domain)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference = decoder.decode(mean, domain)
    assert reference.dtype == torch.bfloat16
    state = {key: value.clone() for key, value in decoder.state_dict().items()}
    enable_low_memory(encoder, decoder, slab_depth=3)
    offloaded_mean, offloaded_logvar = encoder.encode_dist(source, domain)
    _assert_archive_tolerance(offloaded_mean, mean, bf16=False)
    _assert_archive_tolerance(offloaded_logvar, logvar, bf16=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fixed = decoder.decode(mean, domain)
        roundtrip = decoder.decode(offloaded_mean, domain)
    _assert_archive_tolerance(fixed, reference, bf16=True)
    _assert_archive_tolerance(roundtrip, reference, bf16=True)
    for key, value in decoder.state_dict().items():
        assert torch.equal(value, state[key])
