# Stage 2 — v1 Schrödinger-bridge OT coupling

**Date:** 2026-07-27  
**Git commit(s):** substrate [`403204e`](https://github.com/GuillermoTafoya/MRIxFields/commit/403204e5543d4fe470e5aad7cd7c1b50209e69ef), field-only coupling [`eacb8f4`](https://github.com/GuillermoTafoya/MRIxFields/commit/eacb8f44e4eeee7973b2318682cae05928f38755)  
**Config:** [`configs/experiment/stage2_transport_sb_v1.yaml`](../../configs/experiment/stage2_transport_sb_v1.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage2_transport_sb_v1/checkpoints/`; latent bank path supplied externally by the A100 notebook

## Hypothesis / what this run changed

Relative to v1 OT-CFM, this primary rung changed only the path objective: `bridge: schrodinger` with Brownian volatility `sigma: 0.1`, regressing `(z1-z_t)/(1-t)` instead of constant velocity `z1-z0`. Architecture, minibatch-OT coupling, same-contrast cross-field sampling, pooled descriptors, optimizer, and 20,000-step horizon were held fixed ([FM config](../../configs/experiment/stage2_transport_fm_v1.yaml), [SB config](../../configs/experiment/stage2_transport_sb_v1.yaml)).

## Why (decision rationale)

The experiment was declared as the primary rung of the transport ladder to compare deterministic OT-CFM with simulation-free Brownian bridge matching using the same network and endpoints ([config header](../../configs/experiment/stage2_transport_sb_v1.yaml), [`403204e`](https://github.com/GuillermoTafoya/MRIxFields/commit/403204e5543d4fe470e5aad7cd7c1b50209e69ef)).

## What was measured

No SB-specific history, checkpoint, traveller-gate JSON, or board metrics artifact was found. The subject-0009 numbers later used to motivate v2 are not identified as SB results in the commit body and therefore are not assigned to this run. No official/range nRMSE, SSIM, LPIPS, wall-clock, or selected epoch can be reported.

## Outcome

Configured primary rung; execution and promotion are unverified.

## Learnings

No SB-specific empirical conclusion is supported. The config does preserve a clean FM-versus-SB single-variable comparison at v1.

## Open questions / follow-ups

Whether the stochastic bridge outperforms OT-CFM was not resolved by an artifact. The same comparison was carried forward to v2, where it again remained planned rather than evidenced in this workspace ([v2 SB](stage2_v2_sb_nn_leakage_fix.md)).
