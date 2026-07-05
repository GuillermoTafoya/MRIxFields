# Fase B — KL-VAE-GAN (Etapa 1, domain-agnostic)

> **SUPERSEDED (2026-07-02).** After a PI meeting, Etapa 1 was redesigned from this
> KL-VAE-GAN (patchGAN + L1/KL/adversarial/LPIPS) to a **VAE + conditional latent
> diffuser**, reference: Zhang et al., "Development-Driven Diffusion Model for
> Longitudinal Prediction of Fetal Brain MRI With Unpaired Data" (DDM, IEEE TMI, Sep
> 2025). Adversarial/PatchGAN training was dropped entirely. See
> `docs/ARCHITECTURE.md` §5–§7 and the `mrixfields-project` skill for the current
> design and implementation status. Kept below for historical reference only — do not
> implement against this file.
>
> **Two decisions in §3/§1 were later resolved the opposite way for the shipped VAE:**
> (1) the "2D estricto" rule (§3) was reversed — `KLVAEEncoder`/`Decoder` train on full
> 3D volumes (patch-cropped), and the SSIM/LPIPS losses were adapted to 3D (`ssim3d` via
> avg_pool3d; slice-averaged LPIPS) rather than kept 2D-only. (2) After a detour through
> `latent_channels=128`, the design returned to the **~4 latent channels** this plan
> originally specified (§1), at `/4` spatial ⇒ 16× compression.

Status (historical): **proposed, not yet implemented**. This was the detailed breakdown of
Fase B from the original development plan. Confirm the open decisions in §7 before coding
starts.

## 1. Goal

A shared, domain-agnostic KL-VAE-GAN: continuous latent, soft compression `f≈4`, ~4
latent channels, standardized latent (small KL → std≈1). This is the quality ceiling for
the whole pipeline — if it can't reconstruct 0.1T well, nothing in Etapa 2 fixes that.

Confirmed with Simón: encoder/decoder are **blind to (field, contrast)** at first —
no FiLM conditioning inside the VAE. A domain-conditioned variant is a later ablation,
not part of this phase.

## 2. Depends on (already implemented, Fase A)

- `training/losses.py`: `kl_divergence`, `reconstruction_mse`, `latent_l1`,
  `adversarial_hinge_loss_generator`/`_discriminator`, `lpips_loss` (optional `lpips`
  dependency).
- `training/checkpoints.py`: `save_checkpoint`/`load_checkpoint`, `checkpoint_filename`,
  overwrite protection, seed/config/git-hash metadata.
- `utils/seeding.seed_everything`, `training/batch.move_raw_batch`.
- `models/autoencoders/base.py`: `BaseEncoder`/`BaseDecoder` contracts.
- `models/factory.py`: registry to extend with `"kl_vae_gan"`.

## 3. Open decision to resolve first: 2D slices vs. the current dataset shape

The project constraint is **2D estricto** (train on slices, never 3D conv). But
`SyntheticVolumeDataset`/`ManifestVolumeDataset` currently produce tensors shaped
`(channels, depth, height, width)` — a 3D-volume convention that never mattered before
because `IdentityEncoder`/`IdentityDecoder` are shape-agnostic pass-throughs. Fase B's
`KLVAEEncoder` is the **first** component that actually needs a fixed tensor rank
(`Conv2d`, not `Conv3d`), so this gap becomes real.

**Recommendation**: don't touch `SyntheticVolumeDataset`'s existing shape convention
(it's tested and used by the translator smoke path). Instead:

- For **CPU sanity tests** in this phase, construct 2D dummy tensors directly
  (`torch.randn(B, 1, H, W)`), bypassing the dataset entirely — the VAE sanity test only
  needs encoder/decoder forward passes, not the full `RawBatch` machinery.
- For **real training** on real manifests, slice extraction belongs in the
  `image_loader` callable already injectable into `ManifestVolumeDataset` (e.g. a NIfTI
  loader that reads a volume and returns one chosen 2D slice as a tensor) — no change to
  `ManifestVolumeDataset` itself required.

Flag this to Simón before writing `KLVAEEncoder`/`KLVAEDecoder` — confirm the slice axis
convention (axial vs. the axis MRIxFields NIfTIs are natively oriented in) isn't already
decided elsewhere (e.g. in preprocessing scripts outside this repo).

## 4. Model

New file `src/fieldbridge/models/autoencoders/kl_vae_gan.py`:

```python
class KLVAEEncoder(BaseEncoder):
    def __init__(self, *, in_channels: int = 1, base_channels: int = 32, latent_channels: int = 4): ...
    def encode_dist(self, x: Tensor, domain=None) -> tuple[Tensor, Tensor]:  # (mean, logvar)
    def encode(self, x: Tensor, domain) -> Tensor:  # reparameterized sample

class KLVAEDecoder(BaseDecoder):
    def __init__(self, *, out_channels: int = 1, base_channels: int = 32, latent_channels: int = 4): ...
    def decode(self, z: Tensor, domain) -> Tensor
```

- Encoder: conv stem → 2 downsample blocks (stride-2 conv + norm + activation, total
  downsample ×4) → 1×1 conv to `2 * latent_channels`, split into `(mean, logvar)`.
  `domain` is accepted (contract compliance) and ignored, same pattern as
  `IdentityEncoder`.
- Decoder: mirrors the encoder. Prefer `nn.Upsample(scale_factor=2) + Conv2d` over
  `ConvTranspose2d` for the upsampling blocks — fewer checkerboard artifacts, cheap to
  get right the first time.
- `encode()` reparameterizes (`mean + eps * std`) for training; expose `encode_dist()`
  separately since `kl_divergence(mean, logvar)` needs both, not just the sample.

New file `src/fieldbridge/models/discriminators/patch_discriminator.py`:

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, *, in_channels: int = 1, base_channels: int = 32): ...
    def forward(self, x: Tensor) -> Tensor  # raw patch logits, no sigmoid (hinge loss expects logits)
```

Not registered in `models/factory.py` — the discriminator isn't part of the
encode/decode/translate inference contract; it's constructed directly by the stage 1
training entry point (§5).

Register in `models/factory.py`: `_ENCODERS["kl_vae_gan"] = KLVAEEncoder`,
`_DECODERS["kl_vae_gan"] = KLVAEDecoder`.

## 5. Training entry point — new dedicated module, not `train_loop.py`

`train_loop.py` assumes an encode → translate → decode pipeline with a `BaseTranslator`;
Etapa 1 has no translator and a different loss set (KL + adversarial + perceptual, no
transport-cost/cycle/identity). Reusing it would force awkward no-op translator plumbing.

New file `src/fieldbridge/training/stage1_vae.py`:

```python
@dataclass(frozen=True, slots=True)
class Stage1VAEConfig:
    steps: int
    batch_size: int
    seed: int
    lr: float
    loss_weights: dict[str, float]  # {"reconstruction": 1.0, "kl": 0.0, "adversarial": 0.0, "lpips": 0.0}
    warm_start_checkpoint: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 0
    resume_from: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage1VAEConfig": ...

def run_stage1_vae_train(config, *, encoder, decoder, discriminator, loader=None) -> Stage1VAETrainResult: ...
```

Composed loss per `training-conventions` (introduce weights in this order, default 0
except reconstruction): `reconstruction` → `+kl` → `+adversarial` → `+lpips`. Discriminator
gets its own optimizer and hinge loss step, alternating with the generator (encoder +
decoder) step — standard GAN training loop, nothing exotic.

Reuse `seed_everything`, `move_raw_batch`, `save_checkpoint`/`checkpoint_filename` exactly
as `train_loop.py` does. Same `_DEFAULTS`-instance pattern for `from_mapping` (see the bug
fixed in `TrainLoopConfig.from_mapping` — frozen+slots dataclasses can't use `cls.<field>`
as a default fallback).

Configs: `configs/model/kl_vae_gan.yaml` (`name: kl_vae_gan`, `base_channels`,
`latent_channels`), `configs/experiment/stage1_vae.yaml` (loss weights, steps, warm-start
path).

## 6. Warm-start from MAISI/Pinaya + 0.1T oversampling

- Warm-start: a `load_state_dict(..., strict=False)` helper tolerant of shape/key
  mismatches (MAISI/Pinaya checkpoints won't match this encoder/decoder 1:1) — log which
  keys were skipped rather than failing silently.
- Oversampling: real training draws from `ManifestVolumeDataset` with a
  `torch.utils.data.WeightedRandomSampler`. Add a small helper (new file
  `src/fieldbridge/data/sampling.py` or a function in `datasets.py` — decide based on size
  once written) `domain_oversampling_weights(records, *, boost_by_field: dict[float, float])`
  that returns per-record weights (e.g. `{0.1: 3.0}` to sample 0.1T three times as often).
- **Both of these run outside the repo** (GPU rental, real data) — the checked-in code
  only needs to make them possible, per `AGENTS.md`'s no-real-data/no-checkpoints rule.

## 7. Open decisions to confirm before coding

1. **2D slice axis and extraction** (§3) — where does slicing happen, which axis, is
   this already decided in an external preprocessing script?
2. **Discriminator architecture detail** — patch size / number of downsample layers
   not yet fixed; propose starting with a 3-layer PatchGAN (70×70 receptive field
   equivalent) unless there's a reason to go shallower for 0.1T's lower resolution
   content.
3. **Oversampling ratio for 0.1T** — no target ratio specified yet in the spine; needs a
   number before `domain_oversampling_weights` can ship with a sensible default.

## 8. Sanity tests (CPU, required before any real run)

- Forward `encode_dist()` on a dummy `(B, 1, H, W)` tensor → `(mean, logvar)` shapes
  match `(B, latent_channels, H/4, W/4)`, both finite.
- `encode()` → `decode()` round-trip preserves `(B, 1, H, W)` shape.
- `kl_divergence(mean, logvar)` on encoder output is finite and non-negative.
- `PatchDiscriminator` forward on a dummy image → finite logits, expected shape.
- Full `run_stage1_vae_train` smoke run (few steps, synthetic 2D tensors, all loss
  weights including adversarial) — loss finite, no NaN, mirrors the bar already applied
  to `train_loop.py` in Fase A.

## 9. Future ablation (not this phase)

A domain-conditioned variant of this VAE (FiLM from `DomainConditioner` inside
encoder/decoder) to compare against the blind version — this is the "continuous vs.
one-hot field embedding" ablation already in the project's evaluation ladder, just
applied one level earlier than originally scoped. Build only after the blind version is
validated and warm-started successfully.
