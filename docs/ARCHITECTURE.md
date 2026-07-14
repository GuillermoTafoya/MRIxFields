# FieldBridge — Architecture & Current State

Technical reference for the `fieldbridge` package as it exists after **Fase A**
(cross-cutting infra) and the **Etapa 1 v2 pivot** (VAE + conditional latent diffuser,
replacing the original KL-VAE-GAN plan — see `docs/plans/fase-b-vae.md` for the
superseded plan). For the full research roadmap (ablation ladder, losses, compute
budget) see the project's Claude Code skill `mrixfields-project`; this document covers
what is actually implemented and how the pieces fit together.

## 1. Purpose

`fieldbridge` is the implementation scaffold for MRIxFields Task 3: a single conditional
model that translates MRI volumes between 5 field strengths (0.1/1.5/3/5/7T) and 3
contrasts (T1w/T2w/T2-FLAIR), any combination to any combination, with shared parameters
(no per-field or per-pair subnetworks — a hard challenge requirement).

Target architecture: **Etapa 1** — a VAE (`KLVAEEncoder`/`KLVAEDecoder`) plus a
field-strength-conditioned latent diffuser (`DenoisingUNet`), the diffuser sitting
*between* encoder and decoder and frozen-VAE by default once trained — plus **Etapa 2**,
a conditional latent translator between different field strengths, built up through an
ablation ladder (StarGAN-v2 latent → OT-CFM → entropic-OT Schrödinger bridge → optional
adversarial refinement). The pivot only changed Etapa 1; Etapa 2's ladder plan is
unchanged.

## 2. Package layout

```
src/fieldbridge/
├── data/            # domain objects, records, manifests, sources, datasets, transforms, sampling
├── models/          # encoder/decoder/translator contracts, conditioning, factory
│   ├── autoencoders/ # identity, cnn_autoencoder, kl_vae (Etapa 1 VAE)
│   ├── translators/  # identity, conditional_cnn, conditional_unet, ot_cfm/sb stubs
│   └── diffusion/     # Etapa 1's conditional latent diffuser (timestep + field conditioning, schedule, UNet)
├── training/         # batch helpers, losses, checkpoints, warm-start, smoke/stage1/stage2/Etapa-2 train loops
├── evaluation/       # tensor metrics (3 official Task 3 metrics) + stage-1 VAE recon report
├── official/         # MRIxFields2026 challenge spec, submission build/validate
├── config/           # YAML load/merge helpers
└── cli.py            # `fieldbridge` entry point
```

Everything under `src/fieldbridge/` must keep running on CPU with synthetic data — no real
MRI data, checkpoints, or NIfTI files are committed to this repo (see `AGENTS.md`). Real
training happens outside the repo (rented GPU), driven by the same code.

## 3. Core data contracts (`data/contracts.py`, `data/domains.py`)

| Type | Fields | Notes |
|---|---|---|
| `Domain` | `field_strength_t`, `contrast` | Frozen dataclass. Validates against the 5 official field strengths and 3 contrasts. |
| `VolumeRecord` | `case_id`, `image_path`, `domain`, `subject_id`, `split`, `metadata` | One volume reference; storage-backend independent. |
| `RawBatch` | `image`, `source_domain`, `target_domain`, `metadata` | What datasets/dataloaders produce. |
| `LatentBatch` | `latent`, `source_domain`, `target_domain`, `metadata` | Post-encoder representation for the translator stage. |

### `Domain` encodings

- `field_encoding()` → `[log(field_strength_t), field_strength_t / 7.0]`, shape `(2,)`.
- `contrast_encoding()` → one-hot over `(T1w, T2w, T2-FLAIR)`, shape `(3,)`.
- `conditioning_vector()` → concatenation of the two, shape `(5,)`. This is a
  single-domain utility, distinct from `DomainConditioner` below (which conditions on a
  *pair*).

## 4. Conditioning (`models/conditioning.py`, `models/film.py`)

`DomainConditioner.forward(source_domains, target_domains)` takes a **pair** of domains
(not one) and returns a single `(batch, conditioning_dim)` vector combining:

- projected field features for source and target (shared `field_projection` weights),
- `log(f_target / f_source)` — computed inside the conditioner because it only makes
  sense for a pair, not a lone `Domain`,
- contrast embeddings for source and target (shared `nn.Embedding`).

This vector feeds `FiLMLayer` (`models/film.py`), which applies per-channel
`scale`/`shift` to a 2D feature map: `x * (1 + scale) + shift`. Every ladder translator
(StarGAN-v2, OT-CFM, SB) is expected to condition through FiLM/AdaGN layers built on
top of `DomainConditioner` + `FiLMLayer` — never a router that dispatches to
field/contrast-specific subnetworks (disqualifying under the challenge rules).

## 5. Models (`models/`)

| Component | Status |
|---|---|
| `autoencoders/{base,identity}.py` | `BaseEncoder`/`BaseDecoder` contracts + pass-through identity implementation (smoke tests only). |
| `autoencoders/kl_vae.py` | **Etapa 1's real VAE.** `KLVAEEncoder`/`KLVAEDecoder`, residual-block encoder/decoder (`num_res_blocks=2` per level, GroupNorm+SiLU pre-act, MONAI AutoencoderKL-style), `encode_dist()` → `(mean, logvar)`, `encode()` reparameterizes. Decoder upsamples with **nearest `Upsample` + conv** (the Odena et al. anti-checkerboard construction), *not* `ConvTranspose`. Blind to (field, contrast) — no FiLM inside the VAE; conditioning lives in the diffuser (§6). Decoder ends in unconditional `Tanh()` (bound to `[-1, 1]`, matching `normalize_percentile_clip_to_unit_range` and `lpips_loss`'s un-normalized-input assumption — do not make this activation optional). Supports `spatial_dims=2` (slices) or `spatial_dims=3` (full volumes) — 3D added as a deliberate, confirmed reversal of `fase-b-vae.md`'s original "2D estricto" rule for this component only, since the real manifest ships full NIfTI volumes and no slicing step exists in this pipeline. `latent_channels=4` default at `/4` spatial downsample ⇒ **16× compression** on a 64³ patch (16³×4 vs 64³) — the medical-3D consensus (SD kl-f8, Pinaya/MONAI use 3–4). The earlier `latent_channels=128` was a *2× expansion* (no bottleneck) that OOM'd eval and left nothing useful for the diffuser; reworked 2026-07-05. |
| `translators/base.py` | `BaseTranslator.forward(z, source_domain, target_domain, t=None)` — the contract every ladder translator implements. |
| `translators/identity.py` | Pass-through with an optional learnable scale (smoke tests). |
| `translators/conditional_cnn.py` | CPU-friendly conditional CNN baseline for `x_hat = G(x, source_domain, target_domain)` on 2D slices or 3D volumes. |
| `translators/conditional_unet.py` | Sharper deterministic U-Net baseline with conditioned decoder blocks and optional gated skips. |
| `translators/ot_cfm_stub.py`, `translators/sb_stub.py` | Intentional stubs — `raise NotImplementedError`. Real implementations replace these files in place (Fases D and E). |
| `models/factory.py` | Name-based registry: `build_encoder/decoder/translator("identity", **kwargs)`. Extended with `"stargan_v2_latent"`, `"ot_cfm"`, `"schrodinger_bridge"` as those stages land. |

No StarGAN-v2, OT-CFM, or SB model exists yet, not even as a partial stub beyond the two
`NotImplementedError` placeholders above.

### Conditional CNN field translator baseline

`ConditionalCNNFieldTranslator` is the first implemented any-to-any field/sequence
translation baseline. It encodes image content with source-domain conditioning, then
decodes with source-target conditioning from `DomainEmbedding` and FiLM GroupNorm blocks.
It supports same-domain reconstruction (`source_domain == target_domain`) and synthetic
cross-domain interface tests, but it makes no scientific translation claim yet.

This model is intentionally not a diffusion model, not a Schrodinger bridge, and not a
VAE. It exists to prove the shared-parameter contract and training interface before the
later ablation-ladder methods are implemented.

### Conditional U-Net field translator baseline

`ConditionalUNetFieldTranslator` keeps the same image-to-image call shape:
`model(x, source_domain, target_domain)`. It uses `DomainEmbedding` to build a
source-target conditioning vector and injects that vector into decoder blocks through
FiLM GroupNorm. Same-domain calls (`source_domain == target_domain`) are the initial
reconstruction path; cross-domain calls use the same shared parameters with different
conditioning.

The U-Net skips are configurable. The default `skip_mode="gated"` applies a channel-wise
sigmoid gate from the conditioning vector before concatenating skip features, preserving
high-resolution anatomy without making the skip path an unconditional source-domain copy.
`skip_mode="concat"` provides ordinary U-Net concatenation, and `skip_mode="none"` falls
back toward a bottleneck translator. This is still a deterministic baseline, not
diffusion, a Schrodinger bridge, adversarial training, or a VAE.

### Epoch pseudo-pair baseline

`train-pseudo-pairs` replaces the old eight-slice notebook-style overfit with an
epoch-based pseudo-pair pipeline around `ConditionalUNetFieldTranslator`. It is still a
deterministic synthetic-pretraining baseline: high-field T2-FLAIR target volumes are
degraded to synthetic 0.1T inputs, and the model learns to invert that synthetic
corruption. This is not evidence of learning the real low-field distribution.

The data path is volume-first. `build_volume_splits(...)` assigns retrospective volumes
to train/validation/test before slice expansion, audits case/path/subject leakage, and
persists the exact split JSON. Slice preprocessing follows the official released
`[0, 1]` intensity range: no per-slice z-score, optional model-boundary mapping to
`[-1, 1]`, axial slices in the configured range, and aspect-preserving fit/pad instead
of square stretching. Training uses dynamic degradation; validation/test use stable
per-item seeds. Prospective data must be evaluated at subject level, not by slice or
volume leakage.

Pseudo-pair commands consume the standard FieldBridge `Manifest` JSON/YAML schema, not
the MRIxFields audit JSONL: top-level `records`, each with `case_id`, `image_path`,
`domain.field_strength_t`, `domain.contrast`, and `subject_id`. `subject_id` is required
for this path so train/validation/test leakage can be rejected at subject level. Real
manifest files and any absolute Drive/local paths stay outside Git.

## 6. Diffusion (`models/diffusion/`) — Etapa 1's conditional latent diffuser

Sits between `KLVAEEncoder` and `KLVAEDecoder` (§5): `z ~ encoder.encode(x)` →
diffuser conditioned on the *source* domain's field strength/contrast → `z'` →
`decoder.decode(z')`. Reference: Zhang et al., "Development-Driven Diffusion Model for
Longitudinal Prediction of Fetal Brain MRI With Unpaired Data" (DDM, IEEE TMI, Sep
2025) — scaled down (`num_timesteps≈100` vs. the paper's 1000; 2D slices/patch-cropped
3D volumes vs. their full 3D + A100×96h). Etapa 2 (SB/OT-CFM ladder translating latents
*between* field strengths) is unchanged by this and sits after, not inside, this stage.

| Component | Purpose |
|---|---|
| `timestep_embedding.py` | `sinusoidal_timestep_embedding(timesteps, embedding_dim)` — standard fixed sin/cos DDPM embedding, no learnable params. |
| `field_conditioner.py` | `FieldStrengthConditioner` — projects a **single** `Domain.conditioning_vector()` (field + contrast, one volume, no source/target pair) into an embedding. Distinct from `models/conditioning.py`'s `DomainConditioner`, which conditions on a pair for Etapa 2. |
| `schedule.py` | `make_schedule(num_timesteps, beta_start, beta_end)` → `DiffusionSchedule`; `q_sample(z0, t, schedule)` — standard forward-noising DDPM math. |
| `denoising_unet.py` | `DenoisingUNet(z_t, t, domain) -> noise_pred`. Timestep + field-conditioning embeddings are summed, then fed through `FiLMLayer` per residual block (paper adds them into the residual stream directly; this project reuses FiLM instead since it's already implemented/tested and additive is FiLM's degenerate case). `num_levels` capped at 1–2 — a small denoiser sized for the VAE's latent grid, not a deep multi-scale U-Net. Supports `spatial_dims=2` or `3`, matching `KLVAEEncoder`/`KLVAEDecoder`. |

Trained by `training/stage2_diffuser.py` (§7) with a standard noise-prediction MSE loss
(not swappable for perceptual losses — there is no clean image mid-noising-process).

## 7. Training (`training/`)

| File | Purpose |
|---|---|
| `batch.py` | `move_raw_batch` / `move_latent_batch` — device transfer helpers. |
| `losses.py` | `reconstruction_mse`, `latent_l1`, `kl_divergence`, `transport_cost_loss`, `cycle_consistency_loss`, `identity_loss`, `adversarial_hinge_loss_generator/_discriminator`, `nrmse_loss`, `ssim_loss` (dispatches by rank: 4D→2D `ssim`, 5D→`ssim3d`), `lpips_loss` + `lpips_loss_3d` (slice-averaged 2D LPIPS for volumes), `build_lpips_net` (constructs LPIPS(vgg) with its stdout chatter redirected to stderr so `--json` output stays valid JSON), `synthseg_inloss_stub` (explicit `NotImplementedError`). `lpips_loss` needs the optional `lpips` dependency (fails explicitly if missing). |
| `checkpoints.py` | `save_checkpoint`/`load_checkpoint` with a size guardrail, **explicit overwrite protection** (`FileExistsError` unless `overwrite=True`), and run metadata (`seed`, `config`, `git_commit`) stored under `state["_meta"]`. `checkpoint_filename(stage, variant, step)` builds `{stage}_{variant}_{YYYYMMDD}_step{N}.pt` names. |
| `smoke_train.py` | Fixed CPU smoke test: identity encoder/decoder/translator, 2 steps, synthetic data. **Do not extend this — it's a stability tripwire, not a real trainer.** |
| `train_loop.py` | The real, reusable Etapa 2 training loop. Config-driven precision (`fp32`/`bf16` via `torch.autocast`), optional gradient checkpointing on the translator's forward, configurable loss weights (`reconstruction`, `transport_cost`, `cycle`, `identity` — default only `reconstruction=1.0`, rest `0.0`), resume-from-checkpoint (model + optimizer state, not dataloader position — see note in the module), and `assert_frozen(module)` to verify the Etapa 1 VAE is frozen before Etapa 2 training. |
| `pseudo_pair_epochs.py` | Real epoch-based pseudo-pair trainer for the deterministic conditional U-Net baseline: finite DataLoader epochs, AdamW, optional CUDA AMP, gradient clipping, scheduler, validation after every epoch, JSONL history, best/last checkpoints, and resume with optimizer/scheduler/global-step state. |
| `stage1_vae.py` | Etapa 1 VAE-only training (`KLVAEEncoder`/`KLVAEDecoder`, no diffuser, no translator). Loss composition is **all four terms active by default** (unlike Etapa 2's "0 except reconstruction" ladder convention) — `ssim`+`nrmse`+`lpips`+`kl` (default weights `1.0/1.0/1.0/1e-4`), relative weights swept experimentally rather than turned on term-by-term. The full recipe now runs on **3D volumes**: `ssim_loss` dispatches to `ssim3d` (avg_pool3d) and `lpips` uses `lpips_loss_3d` (slice-averaged, `lpips_num_slices` config). The LPIPS net is built once via `build_lpips_net`. Supports warm-start (`training/warm_start.py`) and resume-from-checkpoint. |
| `stage2_diffuser.py` | Etapa 1's conditional-diffuser training (`DenoisingUNet` on top of a trained `KLVAEEncoder`, §6). VAE frozen by default (`train_vae_jointly=False`, reuses `assert_frozen` from `train_loop.py`); standard DDPM noise-prediction MSE loss, `num_timesteps`/`beta_start`/`beta_end` config-driven. |
| `warm_start.py` | `load_state_dict_tolerant(module, state_dict)` — tolerant checkpoint loading for external (e.g. MAISI/Pinaya) warm-start weights: filters out shape-mismatched keys before calling `load_state_dict(strict=False)` (which alone still raises on shape mismatch) and logs skipped/missing/unexpected keys separately. |

`train_loop.py` assumes an encode → translate → decode pipeline (a `BaseTranslator`) and
is **not** used for Etapa 1 (no translator, different loss set) — `stage1_vae.py` and
`stage2_diffuser.py` are the dedicated Etapa 1 entry points instead.

### Any-to-any pair sampling (`data/datasets.py`, `data/sampling.py`)

- `random_any_to_any_selector(domains=ALL_DOMAINS, *, seed, allow_identity=True)` — a
  deterministic `TargetDomainSelector` (hash of `seed:case_id`, not global RNG state) for
  `ManifestVolumeDataset`.
- `SyntheticVolumeDataset(..., pair_sampling="random_any_to_any")` — same idea for the
  synthetic smoke dataset; default `pair_sampling="cycle"` preserves the original
  deterministic cycling behavior used by `smoke_train.py`.
- `ALL_DOMAINS` — the 15 domains (5 fields × 3 contrasts) used as the sampling pool.
- `data/sampling.py`'s `domain_oversampling_weights(records, *, boost_by_field)` — per-record
  weights for `torch.utils.data.WeightedRandomSampler` (e.g. `{0.1: 3.0}` to oversample
  0.1T 3x). No default map ships — the ratio is an experiment hyperparameter, not a guess.
- `data/pseudo_pairs.py`'s `PseudoPairSliceDataset` expands persisted volume splits into
  slices lazily with a tiny LRU volume cache and an injected loader. `make_field_balanced_sampler`
  uses inverse target-field frequency so 1.5T/3T/5T/7T examples contribute approximately
  equally within each epoch.
- `data/volume_splits.py` builds, saves, reloads, summarizes, and audits volume-disjoint
  splits before slice expansion.

### Transforms (`data/transforms.py`)

- `normalize_percentile_clip_to_unit_range(image, lower_percentile=0.5, upper_percentile=99.5)`
  — clips MRI's long-tailed intensity distribution then affine-maps to `[-1, 1]`, matching
  `KLVAEDecoder`'s `Tanh()` output and `lpips_loss`'s un-normalized-input assumption. Used
  by default for every manifest-backed loader (`cli.py`'s `_build_manifest_loader`).
- `random_crop(image, patch_size)` — random spatial-patch crop over the trailing dims.
  Required once `spatial_dims=3` is combined with full-resolution volumes (e.g.
  `364x436x364`): decoding a full 3D volume back toward full resolution OOMs on
  essentially any GPU, so real 3D training crops patches instead of feeding whole
  volumes. Wired in automatically when `data.patch_size` is set in the experiment config
  (`cli.py`'s `_manifest_transform`). Eval reconstructs full volumes tile-by-tile via
  `evaluation/stage1_report.py`'s sliding window (§8) rather than cropping.
- `compose(transforms)` — chains transforms in order.
- `data/preprocessing.py` owns the pseudo-pair slice path: official `[0, 1]` validation,
  uniform axial slice selection, model-boundary range mapping, and reversible fit/pad
  geometry metadata for visualization or volume reconstruction.

## 8. Evaluation (`evaluation/metrics.py`, `evaluation/stage1_report.py`)

The three official MRIxFields Task 3 metrics are implemented:

- `nrmse(prediction, target, data_range=1.0)` — RMSE normalized by intensity range.
- `ssim(prediction, target, data_range=1.0, window_size=7)` — 2D uniform-window SSIM
  (the *official* metric; raises `ValueError` on non-4D input). `ssim3d` (avg_pool3d) is
  the volumetric analogue used as a training-time loss term for `spatial_dims=3`.
- `lpips_metric(prediction, target, net=None)` — thin wrapper around
  `training.losses.lpips_loss` (2D; optional `lpips` dependency).

`mse`, `mae`, `psnr` remain available for quick debugging but are not official metrics.

### Stage-1 VAE reconstruction report (`evaluation/stage1_report.py`)

`run_stage1_eval(...)` (exposed as `fieldbridge eval-stage1-vae`, §10) evaluates a trained
VAE checkpoint and writes `metrics.json` + diagnostic PNGs. Deliberately **not** the
training forward pass:

- **Deterministic reconstruction** from the latent *mean* (`encode_dist(...)[0]`), no
  reparameterization sampling — sampling `mean + eps*sigma` is what made early notebook
  reconstructions look like noise.
- **Sliding-window tiling** (`sliding_window_reconstruct`) so a full volume is never
  decoded whole (the OOM/RAM blowup). `overlap` (default 0.5) + a separable **Hann weight
  window** blend tile faces — with stride == patch (overlap 0) each tile is encoded
  independently and the faces show up as a regular panel grid every `patch` voxels.
- `[-1, 1]` normalization identical to training; `--per-domain` samples one volume per
  distinct field strength (0.1T..7T); metrics are nRMSE / `ssim3d` / slice-LPIPS.

## 9. Official challenge layer (`official/`)

Complete and not touched by Fase A — this was already production-ready:

- `mrixfields2026.py` — official constants (fields, modalities, task pairs, submission
  shape/z-clip), filename parse/build, modality/field alias normalization.
- `submissions.py` — `expected_submission_entries`, `validate_submission_dir`,
  `validate_submission_zip`, `build_submission_zip`, `audit_prediction_manifest_rows`.
- `validation.py` — shape/dtype/intensity-range validators.

Reuse this layer as-is to package every ladder stage's predictions into a submission
zip; do not reimplement naming or validation logic elsewhere.

## 10. CLI (`cli.py`)

```powershell
fieldbridge smoke-train [--config PATH] [--steps N] [--batch-size N] [--seed N] [--json]
fieldbridge train        [--config PATH] [--steps N] [--batch-size N] [--seed N] [--manifest PATH] [--json]
fieldbridge train-pseudo-pairs [--config PATH] --manifest PATH [--epochs N] [--batch-size N] [--checkpoint-dir DIR] [--preflight] [--json]
fieldbridge eval-pseudo-pairs --checkpoint PATH --manifest PATH [--config PATH] [--split validation|test] [--json]
fieldbridge train-stage1-vae --manifest PATH [--config PATH] [--steps N] [--batch-size N] [--seed N] [--json]
fieldbridge eval-stage1-vae --checkpoint PATH --manifest PATH --out DIR [--config PATH] [--num-samples N] [--per-domain] [--overlap F] [--metrics-raw PATH]
fieldbridge train-stage2-diffuser --manifest PATH [--config PATH] [--steps N] [--batch-size N] [--seed N] [--json]
fieldbridge print-config --config PATH
fieldbridge audit-manifest MANIFEST [--strict-paths]
fieldbridge mrixfields2026-print-spec
fieldbridge mrixfields2026-audit-submission --root PATH --task {task1,task2,task3} [--allow-missing-seg] [--allow-extra-files] [--json]
fieldbridge mrixfields2026-zip-submission --submission-root PATH --task {task1,task2,task3} --out PATH.zip [--allow-missing-seg]
fieldbridge mrixfields2026-build-manifest --data-root PATH --out PATH.jsonl [--split NAME ...] [--inspect-payload] [--json]
fieldbridge mrixfields2026-audit-data (--manifest PATH | --data-root PATH) [--inspect-payload] [--json]
```

`train` reads `config["model"]["name"]` (default `"identity"`) and builds
encoder/decoder/translator via `models/factory.py`, passing the rest of the `model:`
section as translator kwargs. It does **not** replace `smoke-train`, which must keep
working unmodified per `AGENTS.md`.

`train-stage1-vae`/`train-stage2-diffuser` require `--manifest` (real NIfTI volumes via
the `nifti` extra) — there is no synthetic fallback for these two stages. Both build
`KLVAEEncoder`/`Decoder` via `models/factory.py`'s `"kl_vae"` key and wire the manifest
loader through `_manifest_transform` (percentile-clip normalize, plus `random_crop` when
`data.patch_size` is set — see §7's Transforms note). The training paths pass
`shuffle=True` + `num_workers` (from `training.num_workers`) to `_build_manifest_loader`;
non-training callers keep the deterministic manifest order.

`eval-stage1-vae` loads a checkpoint and writes `metrics.json` + diagnostic PNGs via
`evaluation/stage1_report.py` (§8) — deterministic recon, sliding-window blending
(`--overlap`), `--per-domain` field sampling, optional `--metrics-raw` to overlay the
training loss curve.

`eval-pseudo-pairs` reports degraded `x_low` versus `x_high` and predicted `x_pred`
versus `x_high`, with aggregate and per-target-field metrics plus improvement over the
degraded baseline. It also runs a target-conditioning audit by evaluating correct target
domains and intentionally permuted target domains. LPIPS is optional; when the dependency
or local weights are unavailable, the report marks LPIPS skipped instead of failing the
core package.

`train-pseudo-pairs --preflight` constructs/persists the split, audits leakage, builds the
datasets, loads one sample per non-empty split, and reports derived dataset lengths,
steps per epoch, configured preprocessing geometry/range, and sample intensity ranges
without running an optimizer step. This is the intended Colab check before GPU training
against a Drive-backed manifest.

## 11. Configuration schema

- `configs/data/*.yaml` — dataset config (`num_samples`, `volume_shape`,
  `source_domains`/`target_domains` as `{field_strength_t, contrast}` mappings, optional
  `patch_size` for 3D random-crop training).
- `configs/model/*.yaml` — `name` (factory key) + constructor kwargs for that model:
  `identity.yaml`, `kl_vae.yaml` (`base_channels`, `latent_channels`, `num_res_blocks`,
  `spatial_dims`), `conditional_cnn_translator.yaml`, `conditional_unet_translator.yaml`,
  `field_conditioned_unet.yaml` (the Etapa 1 `DenoisingUNet`).
- `configs/experiment/*.yaml` — top-level run config:
  - `smoke.yaml`/`autoencoder.yaml` consumed by `SmokeTrainConfig`/`TrainLoopConfig.from_mapping`:
    `seed`, `data:`, `model:`, `training:` (`steps`, `batch_size`, `lr`, and for
    `TrainLoopConfig` also `stage`, `precision`, `gradient_checkpointing`,
    `loss_weights`, `checkpoint_dir`, `checkpoint_every_steps`, `resume_from`).
  - `stage1_vae.yaml` consumed by `Stage1VAEConfig.from_mapping`: adds `loss_weights`
    (`ssim`/`nrmse`/`lpips`/`kl`), `ssim_window_size`, `lpips_num_slices`,
    `grad_clip_norm` (gradient-norm clip, default 1.0), `warm_start_checkpoint`,
    `checkpoint_at_end`, `checkpoint_max_bytes`, `log_every_steps`; `device: cuda`
    (fail-fast, no silent CPU fallback) and `training.num_workers` for the loader. The
    encoder also clamps `logvar` to `[-30, 20]` internally (KL/overflow guard).
  - `stage2_diffuser.yaml` consumed by `Stage2DiffuserConfig.from_mapping`: adds
    `num_timesteps`, `beta_start`, `beta_end`, `train_vae_jointly`, `vae_checkpoint`.
  - `pseudo_pair_t2flair_pilot.yaml` consumed by `train-pseudo-pairs`/`eval-pseudo-pairs`:
    volume counts per target field, `SlicePreprocessingSpec`, shared conditional U-Net
    kwargs, epoch count, batch size, AdamW/scheduler settings, checkpoint directory, and
    pseudo-pair loss weights.
  - `pseudo_pair_t2flair_micro.yaml` is the Colab preflight/micro-run variant: 2 train
    volumes per target field, 1 validation, 1 test, 8 slices/volume, `128x160` fit/pad,
    batch size 4, `num_workers=0`, and 1 epoch.

No magic numbers in code — every hyperparameter above is config-driven with an explicit
default in the corresponding dataclass.

## 12. Testing

Every new component carries a shape/no-NaN sanity test, per project convention — not
full coverage, but enough to not discover a shape or NaN bug after a long GPU run. Key
files:

- `test_domains.py`, `test_models.py` — encodings, conditioner, FiLM, factory.
- `test_datasets.py`, `test_sampling.py`, `test_transforms.py`, `test_sources.py` —
  synthetic dataset shapes, any-to-any sampler reproducibility, oversampling weights,
  percentile-clip/random-crop transforms, NIfTI loader.
- `test_losses.py`, `test_evaluation_metrics.py` — forward + backward sanity for every
  loss/metric, including the optional-dependency (`lpips`) failure path.
- `test_kl_vae.py` — `KLVAEEncoder`/`KLVAEDecoder` shape/round-trip for both
  `spatial_dims=2` and `3` (incl. residual-block 64³ forward/backward, no-NaN),
  3D SSIM/slice-LPIPS training paths, `kl_divergence` finiteness.
- `test_stage1_report.py` — sliding-window blending (identity encode/decode reconstructs
  exactly; Hann window taper), per-domain dedup, metrics/plot outputs.
- `test_diffusion.py` — timestep embedding, `FieldStrengthConditioner`,
  `DiffusionSchedule`/`q_sample`, `DenoisingUNet` forward, 2D and 3D.
- `test_warm_start.py` — `load_state_dict_tolerant` shape-mismatch/missing/unexpected
  key handling.
- `test_checkpoints.py` — round-trip, overwrite protection, naming convention.
- `test_train_loop.py` — finite losses, `assert_frozen`, checkpoint + resume (Etapa 2).
- `test_conditional_cnn_translator.py`, `test_conditional_unet_translator.py` — the two
  Etapa 2 baseline translators.
- `test_pseudo_pair_preprocessing.py`, `test_volume_splits.py`,
  `test_pseudo_pair_dataset.py`, `test_pseudo_pair_training.py`,
  `test_pseudo_pair_evaluation.py` — official `[0, 1]` slice preprocessing,
  volume-disjoint split persistence/leakage audits, lazy pseudo-pair slices, balanced
  sampling, epoch checkpoints/resume, and degraded/predicted evaluation reports.
- `test_cli_train.py` — the `train` command end-to-end on the default smoke config.
- `test_mrixfields2026_*.py` — the official challenge layer (spec, submission,
  validation, CLI, data manifest) — untouched, already exhaustive.

Run `pytest` (fast, CPU) and `fieldbridge smoke-train` before handing back any change that
touches package, CLI, data, model, or training code.

## 13. Status vs. the roadmap

| Stage | Status |
|---|---|
| Fase A — cross-cutting infra | **Done.** Field encoding, per-pair conditioner, FiLM, losses, metrics, checkpoint versioning, any-to-any sampler, real train loop, model factory, CLI `train`. |
| Etapa 1 — VAE (`KLVAEEncoder`/`Decoder`) + conditional latent diffuser (`DenoisingUNet`) | **Core implemented, not GPU-validated.** VAE reworked 2026-07-05: `latent_channels=4` (16× compression) + residual blocks, full 3D loss recipe (nRMSE+SSIM3D+slice-LPIPS+KL), `eval-stage1-vae` with deterministic recon + sliding-window blending. A 150-step validation run confirmed the pipeline runs end-to-end (checkpoint valid, LPIPS active) but is far too short to judge reconstruction quality. **Open before the real run:** the validation run measured ~37 s/step — at that rate an 8k-step run is ~82 GPU-h (over budget), so per-step cost (full-volume NIfTI load per step, batch size, GPU placement) must be profiled first. Real reconstruction/diffusion quality on real data (0.1T in particular) not yet confirmed. Supersedes the original Fase B KL-VAE-GAN plan (`docs/plans/fase-b-vae.md`, superseded). |
| Fase C — StarGAN-v2 latente (Etapa 2 ladder #1) | Not started. |
| Fase D — OT-CFM (Etapa 2 ladder #2) | Not started (`translators/ot_cfm_stub.py` is a placeholder). |
| Fase E — Entropic-OT bridge / SB (Etapa 2 ladder #3, primary) | Not started (`translators/sb_stub.py` is a placeholder). |
| Fase F — Adversarial refinement (Etapa 2 ladder #4, budget-gated) | Not started; do not build until C–E are quantified and extra budget is confirmed. |
