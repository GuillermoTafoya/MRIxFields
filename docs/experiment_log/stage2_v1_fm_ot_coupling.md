# Stage 2 — v1 OT-CFM coupling

**Date:** 2026-07-29  
**Git commit(s):** substrate [`403204e`](https://github.com/GuillermoTafoya/MRIxFields/commit/403204e5543d4fe470e5aad7cd7c1b50209e69ef), field-only coupling [`eacb8f4`](https://github.com/GuillermoTafoya/MRIxFields/commit/eacb8f44e4eeee7973b2318682cae05928f38755), gate [`f9527e8`](https://github.com/GuillermoTafoya/MRIxFields/commit/f9527e891f088c53ca1e658a89cc539cbdb670cf), traveller filter [`c4b9c39`](https://github.com/GuillermoTafoya/MRIxFields/commit/c4b9c399baef588d9547e89100de5036e1ccfcdb)  
**Config:** [`configs/experiment/stage2_transport_fm_v1.yaml`](../../configs/experiment/stage2_transport_fm_v1.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage2_transport_fm_v1/checkpoints/`; full-volume bank and 4,000-step gate paths are Drive-defined in [`notebooks/TAFOYAVERSIONstage2_runC_full_bank_gate_A100.ipynb`](../../notebooks/TAFOYAVERSIONstage2_runC_full_bank_gate_A100.ipynb)

## Hypothesis / what this run changed

The v1 FM run used one shared FiLM-conditioned velocity network, deterministic OT-CFM (`bridge: ot_cfm`), exact minibatch OT over batch 8, same-contrast cross-field sampling, and pooled `4³` descriptors. Non-flow loss weights were all zero. Relative to the initial transport substrate, same-contrast/cross-field sampling and pooled OT were a bundle added to stop training cross-contrast transitions and reduce high-dimensional distance concentration ([`403204e`](https://github.com/GuillermoTafoya/MRIxFields/commit/403204e5543d4fe470e5aad7cd7c1b50209e69ef), [`eacb8f4`](https://github.com/GuillermoTafoya/MRIxFields/commit/eacb8f44e4eeee7973b2318682cae05928f38755)).

## Why (decision rationale)

Task 3 changes field strength while preserving contrast. The prior sampler drew arbitrary contrast/field pairs, and full-latent L2 at roughly one million dimensions was reported to concentrate toward random assignments. The 0009 gate was added because unpaired OT-CFM training loss has an irreducible floor and was not considered a quality signal ([`eacb8f4`](https://github.com/GuillermoTafoya/MRIxFields/commit/eacb8f44e4eeee7973b2318682cae05928f38755), [`f9527e8`](https://github.com/GuillermoTafoya/MRIxFields/commit/f9527e891f088c53ca1e658a89cc539cbdb670cf)).

## What was measured

The frozen subject-0009 gate covered 60 ordered field pairs. The later v2 commit records official Task-3 nRMSE `0.977` for identity versus `0.694` for transport, and SSIM `0.8730` versus `0.8755`, with a VAE ceiling of `0.9573`. Thirteen pairs had identity nRMSE greater than 1 and mean transport gain `-1.41`; on the remaining 47, transport lost `+0.029` nRMSE and `-0.005` SSIM. Correlation between identity nRMSE and improvement was `-0.897`. Up-field pairs changed nRMSE `1.534 -> 0.875` and SSIM `0.8608 -> 0.8618`; T2w won only `5/20` nRMSE and `2/20` SSIM pairs ([`816d6f3`](https://github.com/GuillermoTafoya/MRIxFields/commit/816d6f370c2456dff67c36d9129b26fe7b93b835)).

All gate numbers above are **[claimed, unverified in artifacts]**: no gate metrics JSON is checked into the repository/workspace. The nRMSE is explicitly official `||p-t|| / ||t||`. LPIPS, wall-clock, and a final selected training checkpoint are not reported.

## Outcome

Not promoted. The gate was frozen as spent held-out evidence and diagnosed as a photometric/intensity-rescaling result rather than structural transport ([`816d6f3`](https://github.com/GuillermoTafoya/MRIxFields/commit/816d6f370c2456dff67c36d9129b26fe7b93b835)).

## Learnings

Aggregate nRMSE concealed the failure mechanism: nearly all apparent benefit came from cases where identity had catastrophic intensity mismatch, while ordinary pairs and T2w regressed and SSIM stayed essentially flat. Minibatch OT over eight examples drawn from pools of 37–191 was judged much closer to random anatomical pairing than global nearest-neighbour matching. The original gate also inherited an encode halo into tiled decode; the later commit estimates a roughly `0.04` official-nRMSE cost on all three comparison columns **[claimed, unverified in artifacts]**.

## Open questions / follow-ups

The v2 bundle changed global coupling, initialization, regularization, and decode correctness together, so it cannot isolate which correction matters. See [Stage-2 v2 FM](stage2_v2_fm_nn_leakage_fix.md).
