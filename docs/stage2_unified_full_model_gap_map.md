# Unified retrospective Stage-2 gap map

This document records the qualification of current `origin/main` at
`09605a0ce5a3c14d3e19ea7c719405d5cc5d35b3` and draft PR #37 at
`9441f74a280f9d52d5136d01c1aa65511632dd59`. PR #37 is reference material only.

## Compatible engineering reused

- One projection critic has a realism head, a shared 15-domain classifier, and target-domain
  projection conditioning. The unified implementation retains that structure.
- The proposed initial weights `adversarial=0.05`, `domain=0.1`, and `identity=0.1` are retained.
- Latent-space criticism is the primary path. Image-space criticism remains an explicit
  configuration ablation.
- Hinge adversarial objectives and explicit critic/generator optimizer state are retained.

## Conflicts corrected

| PR #37 / previous path | Unified resolution |
|---|---|
| Reads immutable legacy `latent-bank-v1` raw latents | Reads only `photometry-factored-latent-bank-v2`, `E(N_d(x_d))`; v1 is not mutated or relabelled |
| Does not carry canonical local-valid-core support | Validates packed support and uses supported cells for flow, identity, critic, anatomy, and graph terms |
| Partial resume (model/optimizer/sampler only) | Exact resume seals generator, critic, both optimizers/schedulers, scaler, Python/NumPy/Torch/CUDA RNG, sampler, cursor, config, bank, stats, and code identity |
| No anatomy or graph objectives | Adds source-supported low/mid-frequency anatomy/edge/gradient terms and differentiable direct-vs-composed field paths |
| Raw/image critic assumptions can exploit padding | Real and generated critic views use the identical frozen source-intersection-target support; masked values and the appended support channel follow one construction |
| No R-only bank contract | Manifest/sidecars are classified before tensor load; only R/train fits and R/validation evaluates; all P identities fail closed |
| Normalized latents passed directly to frozen decoder | Every anatomy, image-critic, validation, and evaluation decode receives `stats.denormalize(latent)`; the VAE remains frozen |
| No auxiliary-dominance check | The default 200-step full-objective pilot gates finite terms, per-term gradients, smoothed behavior, critic saturation, auxiliary/flow ratios, OOM, and measured train-plus-complete-validation runtime/cost; cadence, pilot-boundary, and terminal validation runs are projected exactly with the terminal counted once |
| No term-level diagnostics | Append-only JSONL records raw/weighted terms, per-term translator gradients, critic distributions/domain accuracies, graph paths, transitions, throughput, time, memory, and OOM hard stops |
| Training-loss-only checkpointing | A versioned, step/variant-independent validation plan freezes target, bridge time, and stochastic noise per case; a critic-independent rule drives sealed latest/best receipts and evaluation resolves the selected best checkpoint |
| Unspecified paired and baseline manifests | Strict feasibility excludes only positively classified P identities and raises on malformed evidence; deterministic producer contracts cover complete materialized arrays/Stage-1 ceilings and existing Gate-0.1/SB-v2 artifacts, and a readiness receipt blocks 100k training until the complete path exists |

## Scientific boundary

The user explicitly authorized strongest-model-first engineering followed by backward
ablations, superseding the earlier staged implementation order. This does not declare a
scientific promotion, learned disentanglement, or authorization to use the structural
descriptor for coupling. Descriptor coupling remains false. The full model and each backward
ablation must be compared on the complete sealed retrospective validation inventory before a
scientific conclusion is made.

Anatomy loss compares only smoothed low/mid-frequency canonical reconstructions and their
smoothed edges/gradients inside conservative source support. It contains no raw high-pass
equality term. GroupNorm's global statistical dependence remains part of bank provenance;
local-valid-core support is anatomical spatial validity, not complete computational
independence.

## Remaining external operator inputs

The repository contains no private data or model artifacts. A run therefore supplies the
sealed split, retrospective root, frozen Variant-A artifact/qualification, and frozen VAE config
and checkpoint. The notebook first audits whether genuine R/validation pairs exist. Only when
they do, the operator additionally supplies complete deterministic-producer artifacts for
materialized R/validation arrays with Stage-1 ceilings and existing Gate 0.1
calibrated-identity/original-SB-v2 predictions. Production commands build the final manifests and
seal long-run readiness; no operator-curated paired subset is accepted. Without that receipt the
100k authorization is never consulted. No endpoint or prospective data is committed.
