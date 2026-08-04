# Stage 2 — v2 Schrödinger bridge NN coupling and leakage fix

**Date:** 2026-07-29  
**Git commit(s):** v2 bundle [`816d6f3`](https://github.com/GuillermoTafoya/MRIxFields/commit/816d6f370c2456dff67c36d9129b26fe7b93b835), notebook [`3aeb508`](https://github.com/GuillermoTafoya/MRIxFields/commit/3aeb5081050d6d5c44b66b4a8e9b0824c517ab98), leakage fix [`34496bb`](https://github.com/GuillermoTafoya/MRIxFields/commit/34496bb6f43c1957d4f9ced527bbc9be662b7d74)  
**Config:** [`configs/experiment/stage2_transport_sb_v2.yaml`](https://github.com/GuillermoTafoya/MRIxFields/blob/34496bb6f43c1957d4f9ced527bbc9be662b7d74/configs/experiment/stage2_transport_sb_v2.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage2_transport_sb_v2/checkpoints/`; shares the full-volume Run-C bank and NN descriptor cache with v2 FM

## Hypothesis / what this run changed

Relative to v2 FM, this primary rung changes only `bridge: schrodinger` and uses `sigma: 0.1`; architecture, NN coupling, five-candidate sampling, zero initialization, regularizer weights, explicit `paired_fraction: 0.0`, resplit, and decode contract are held fixed. Relative to v1 SB, however, it is the same multi-component v2/leakage bundle documented for [v2 FM](stage2_v2_fm_nn_leakage_fix.md) ([v2 configs](https://github.com/GuillermoTafoya/MRIxFields/tree/34496bb6f43c1957d4f9ced527bbc9be662b7d74/configs/experiment)).

## Why (decision rationale)

The v2 bundle was motivated by the v1 0009 photometry-only diagnosis and then hardened after the NN leakage audit. SB remains the primary rung so that its Brownian drift objective can be compared with deterministic FM under identical coupling and safeguards ([config header](https://github.com/GuillermoTafoya/MRIxFields/blob/34496bb6f43c1957d4f9ced527bbc9be662b7d74/configs/experiment/stage2_transport_sb_v2.yaml)).

## What was measured

No completed v2 SB history, checkpoint, sanity output, or traveller-gate metrics JSON was found. No official/range nRMSE, SSIM, LPIPS, wall-clock, or selected step can be reported. The notebook schedules SB after FM so a failed session still leaves one completed model, but this is an execution plan, not evidence of completion ([v2 notebook](https://github.com/GuillermoTafoya/MRIxFields/blob/34496bb6f43c1957d4f9ced527bbc9be662b7d74/notebooks/stage2_v2_nn_coupling_fm_sb_gate_A100.ipynb)).

## Outcome

Configured primary rung and leakage-hardened; execution and promotion are unverified.

## Learnings

No SB-specific empirical conclusion is available. The design preserves a clean v2 FM-versus-SB comparison while inheriting the cohort-safe resplit and explicit training-subject gate semantics.

## Open questions / follow-ups

The FM-versus-SB comparison and the held-out 0006 promotion gate remain unresolved. Subject 0009 must stay frozen until the 0006 development gate justifies spending it again.
