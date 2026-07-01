# CLB-Field — Architecture & Current State

Technical reference for the `clbfield` package as it exists after **Fase A** (cross-cutting
infra). For the full research roadmap (ablation ladder, losses, compute budget) see the
project's Claude Code skill `mrixfields-project`; this document covers what is actually
implemented and how the pieces fit together.

## 1. Purpose

`clbfield` is the implementation scaffold for MRIxFields Task 3: a single conditional
model that translates MRI volumes between 5 field strengths (0.1/1.5/3/5/7T) and 3
contrasts (T1w/T2w/T2-FLAIR), any combination to any combination, with shared parameters
(no per-field or per-pair subnetworks — a hard challenge requirement).

Target architecture: a frozen KL-VAE-GAN autoencoder (Etapa 1) + a conditional latent
translator (Etapa 2), the latter built up through an ablation ladder (StarGAN-v2 latent →
OT-CFM → entropic-OT Schrödinger bridge → optional adversarial refinement).

## 2. Package layout

```
src/clbfield/
├── data/            # domain objects, records, manifests, sources, datasets, transforms
├── models/          # encoder/decoder/translator contracts, conditioning, factory
├── training/         # batch helpers, losses, checkpoints, smoke + real train loops
├── evaluation/       # tensor metrics, including the 3 official Task 3 metrics
├── official/         # MRIxFields2026 challenge spec, submission build/validate
├── config/           # YAML load/merge helpers
└── cli.py            # `clbfield` entry point
```

Everything under `src/clbfield/` must keep running on CPU with synthetic data — no real
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
| `autoencoders/{base,identity}.py` | `BaseEncoder`/`BaseDecoder` contracts + pass-through identity implementation (smoke tests only). **No real VAE yet — that's Fase B.** |
| `translators/base.py` | `BaseTranslator.forward(z, source_domain, target_domain, t=None)` — the contract every ladder translator implements. |
| `translators/identity.py` | Pass-through with an optional learnable scale (smoke tests). |
| `translators/ot_cfm_stub.py`, `translators/sb_stub.py` | Intentional stubs — `raise NotImplementedError`. Real implementations replace these files in place (Fases D and E). |
| `models/factory.py` | Name-based registry: `build_encoder/decoder/translator("identity", **kwargs)`. Extended with `"stargan_v2_latent"`, `"ot_cfm"`, `"schrodinger_bridge"` as those stages land. |

No StarGAN-v2, OT-CFM, or SB model exists yet, not even as a partial stub beyond the two
`NotImplementedError` placeholders above.

## 6. Training (`training/`)

| File | Purpose |
|---|---|
| `batch.py` | `move_raw_batch` / `move_latent_batch` — device transfer helpers. |
| `losses.py` | `reconstruction_mse`, `latent_l1`, `kl_divergence`, `transport_cost_loss`, `cycle_consistency_loss`, `identity_loss`, `adversarial_hinge_loss_generator/_discriminator`, `lpips_loss` (optional `lpips` dependency, fails explicitly if missing), `synthseg_inloss_stub` (explicit `NotImplementedError` — depends on SynthSeg labels not yet confirmed available). |
| `checkpoints.py` | `save_checkpoint`/`load_checkpoint` with a size guardrail, **explicit overwrite protection** (`FileExistsError` unless `overwrite=True`), and run metadata (`seed`, `config`, `git_commit`) stored under `state["_meta"]`. `checkpoint_filename(stage, variant, step)` builds `{stage}_{variant}_{YYYYMMDD}_step{N}.pt` names. |
| `smoke_train.py` | Fixed CPU smoke test: identity encoder/decoder/translator, 2 steps, synthetic data. **Do not extend this — it's a stability tripwire, not a real trainer.** |
| `train_loop.py` | The real, reusable Etapa 2 training loop. Config-driven precision (`fp32`/`bf16` via `torch.autocast`), optional gradient checkpointing on the translator's forward, configurable loss weights (`reconstruction`, `transport_cost`, `cycle`, `identity` — default only `reconstruction=1.0`, rest `0.0`), resume-from-checkpoint (model + optimizer state, not dataloader position — see note in the module), and `assert_frozen(module)` to verify the Etapa 1 VAE is frozen before Etapa 2 training. |

`train_loop.py` assumes an encode → translate → decode pipeline (a `BaseTranslator`).
It is **not** meant for training the Etapa 1 autoencoder itself (no translator, different
losses) — Fase B introduces a dedicated module for that (see
`docs/plans/fase-b-vae.md`).

### Any-to-any pair sampling (`data/datasets.py`)

- `random_any_to_any_selector(domains=ALL_DOMAINS, *, seed, allow_identity=True)` — a
  deterministic `TargetDomainSelector` (hash of `seed:case_id`, not global RNG state) for
  `ManifestVolumeDataset`.
- `SyntheticVolumeDataset(..., pair_sampling="random_any_to_any")` — same idea for the
  synthetic smoke dataset; default `pair_sampling="cycle"` preserves the original
  deterministic cycling behavior used by `smoke_train.py`.
- `ALL_DOMAINS` — the 15 domains (5 fields × 3 contrasts) used as the sampling pool.

## 7. Evaluation (`evaluation/metrics.py`)

The three official MRIxFields Task 3 metrics are implemented:

- `nrmse(prediction, target, data_range=1.0)` — RMSE normalized by intensity range.
- `ssim(prediction, target, data_range=1.0, window_size=7)` — 2D uniform-window SSIM
  (this project is 2D-only; `ssim` raises `ValueError` on non-4D input).
- `lpips_metric(prediction, target, net=None)` — thin wrapper around
  `training.losses.lpips_loss`; same optional-dependency behavior.

`mse`, `mae`, `psnr` remain available for quick debugging but are not official metrics.

## 8. Official challenge layer (`official/`)

Complete and not touched by Fase A — this was already production-ready:

- `mrixfields2026.py` — official constants (fields, modalities, task pairs, submission
  shape/z-clip), filename parse/build, modality/field alias normalization.
- `submissions.py` — `expected_submission_entries`, `validate_submission_dir`,
  `validate_submission_zip`, `build_submission_zip`, `audit_prediction_manifest_rows`.
- `validation.py` — shape/dtype/intensity-range validators.

Reuse this layer as-is to package every ladder stage's predictions into a submission
zip; do not reimplement naming or validation logic elsewhere.

## 9. CLI (`cli.py`)

```powershell
clbfield smoke-train [--config PATH] [--steps N] [--batch-size N] [--seed N] [--json]
clbfield train        [--config PATH] [--steps N] [--batch-size N] [--seed N] [--json]
clbfield print-config --config PATH
clbfield audit-manifest MANIFEST [--strict-paths]
clbfield mrixfields2026-print-spec
clbfield mrixfields2026-audit-submission --root PATH --task {task1,task2,task3} [--allow-missing-seg] [--allow-extra-files] [--json]
clbfield mrixfields2026-zip-submission --submission-root PATH --task {task1,task2,task3} --out PATH.zip [--allow-missing-seg]
```

`train` reads `config["model"]["name"]` (default `"identity"`) and builds
encoder/decoder/translator via `models/factory.py`, passing the rest of the `model:`
section as translator kwargs. It does **not** replace `smoke-train`, which must keep
working unmodified per `AGENTS.md`.

## 10. Configuration schema

- `configs/data/*.yaml` — dataset config (`num_samples`, `volume_shape`,
  `source_domains`/`target_domains` as `{field_strength_t, contrast}` mappings).
- `configs/model/*.yaml` — `name` (factory key) + constructor kwargs for that model
  (e.g. `learnable_scale`, `initial_scale` for `"identity"`).
- `configs/experiment/*.yaml` — top-level run config consumed by `SmokeTrainConfig`/
  `TrainLoopConfig.from_mapping`: `seed`, `data:`, `model:`, `training:` (`steps`,
  `batch_size`, `lr`, and for `TrainLoopConfig` also `stage`, `precision`,
  `gradient_checkpointing`, `loss_weights`, `checkpoint_dir`, `checkpoint_every_steps`,
  `resume_from`).

No magic numbers in code — every hyperparameter above is config-driven with an explicit
default in the corresponding dataclass.

## 11. Testing

Every new component in Fase A carries a shape/no-NaN sanity test, per project
convention — not full coverage, but enough to not discover a shape or NaN bug after a
long GPU run. Key files:

- `test_domains.py`, `test_models.py` — encodings, conditioner, FiLM, factory.
- `test_datasets.py` — synthetic dataset shapes + any-to-any sampler reproducibility.
- `test_losses.py`, `test_evaluation_metrics.py` — forward + backward sanity for every
  loss/metric, including the optional-dependency (`lpips`) failure path.
- `test_checkpoints.py` — round-trip, overwrite protection, naming convention.
- `test_train_loop.py` — finite losses, `assert_frozen`, checkpoint + resume.
- `test_cli_train.py` — the `train` command end-to-end on the default smoke config.
- `test_mrixfields2026_*.py` — the official challenge layer (spec, submission,
  validation, CLI) — untouched, already exhaustive.

Run `pytest` (fast, CPU) and `clbfield smoke-train` before handing back any change that
touches package, CLI, data, model, or training code.

## 12. Status vs. the ablation ladder

| Stage | Status |
|---|---|
| Fase A — cross-cutting infra | **Done.** Field encoding, per-pair conditioner, FiLM, losses, metrics, checkpoint versioning, any-to-any sampler, real train loop, model factory, CLI `train`. |
| Fase B — KL-VAE-GAN (Etapa 1, domain-agnostic) | Not started. Plan: `docs/plans/fase-b-vae.md`. |
| Fase C — StarGAN-v2 latente (ladder #1) | Not started. |
| Fase D — OT-CFM (ladder #2) | Not started (`translators/ot_cfm_stub.py` is a placeholder). |
| Fase E — Entropic-OT bridge / SB (ladder #3, primary) | Not started (`translators/sb_stub.py` is a placeholder). |
| Fase F — Adversarial refinement (ladder #4, budget-gated) | Not started; do not build until C–E are quantified and extra budget is confirmed. |
