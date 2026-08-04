# Stage 2 — v2 OT-CFM NN coupling and leakage fix

**Date:** 2026-07-29  
**Git commit(s):** v2 bundle [`816d6f3`](https://github.com/GuillermoTafoya/MRIxFields/commit/816d6f370c2456dff67c36d9129b26fe7b93b835), notebook [`3aeb508`](https://github.com/GuillermoTafoya/MRIxFields/commit/3aeb5081050d6d5c44b66b4a8e9b0824c517ab98), leakage fix [`34496bb`](https://github.com/GuillermoTafoya/MRIxFields/commit/34496bb6f43c1957d4f9ced527bbc9be662b7d74), repin [`9153361`](https://github.com/GuillermoTafoya/MRIxFields/commit/9153361e8b4e71801e18208e5b8897d0f80c4aac)  
**Config:** [`configs/experiment/stage2_transport_fm_v2.yaml`](https://github.com/GuillermoTafoya/MRIxFields/blob/34496bb6f43c1957d4f9ced527bbc9be662b7d74/configs/experiment/stage2_transport_fm_v2.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage2_transport_fm_v2/checkpoints/`; reuses the verified full-volume Run-C latent bank; descriptor cache is stored beside the bank when `descriptor_cache: null`

## Hypothesis / what this run changed

This is a bundle, not an ablation. Versus v1 FM it changed minibatch OT to global NN coupling over standardized pooled descriptors (`nn_candidates: 5`), zero-initialized the velocity output, enabled transport-cost and identity losses at `0.1` each, and corrected decode to full volume with tiled fallback. The later leakage fix additionally made paired supervision explicit (`paired_fraction: 0.0` by default), applied a split override to the prebuilt bank, and resplit travellers to 0007 train / 0006 validation / 0009 test ([v2 config](https://github.com/GuillermoTafoya/MRIxFields/blob/34496bb6f43c1957d4f9ced527bbc9be662b7d74/configs/experiment/stage2_transport_fm_v2.yaml), [`34496bb`](https://github.com/GuillermoTafoya/MRIxFields/commit/34496bb6f43c1957d4f9ced527bbc9be662b7d74)).

## Why (decision rationale)

The v1 0009 gate showed official nRMSE improvement concentrated in 13 catastrophic identity pairs, flat SSIM, and T2w regression; see [v1 FM](stage2_v1_fm_ot_coupling.md). Global NN was intended to reduce anatomical mismatch from batch-8 OT. Zero initialization made the initial model exactly identity. The leakage audit then found that a traveller in the NN training pool retrieved its own target-field volume and received implicit paired supervision at an unknown rate. It also found that resplitting was previously ignored because bank manifests baked in membership, and that bare numeric IDs could collide across prospective/retrospective cohorts ([`34496bb`](https://github.com/GuillermoTafoya/MRIxFields/commit/34496bb6f43c1957d4f9ced527bbc9be662b7d74)).

## What was measured

No completed v2 FM history, checkpoint, or gate metrics JSON was found. The configured 20,000 steps, `0.1` regularizer weights, and five NN candidates are settings, not measured outcomes.

The notebook specifies a 200-step sanity that must show decreasing loss and expose flow/transport-cost/identity ratios before the long run. Promotion is based on official metrics for held-out validation traveller 0006, with training traveller 0007 reported only as an optimistic upper bound; 0009 remains untouched. These are planned checks, not recorded results ([v2 notebook at the leakage-fix commit](https://github.com/GuillermoTafoya/MRIxFields/blob/34496bb6f43c1957d4f9ced527bbc9be662b7d74/notebooks/stage2_v2_nn_coupling_fm_sb_gate_A100.ipynb)).

## Outcome

Configured and leakage-hardened, but completion/promotion is unverified.

## Learnings

Nearest-neighbour coupling can silently become paired memorization when the same traveller appears across fields in the training pool. Split membership must be cohort-scoped and supplied independently of a frozen bank manifest after resplitting. Training-subject evaluation is now explicitly labeled and blocked by default, preventing an optimistic training read from being mistaken for a promotion signal.

## Open questions / follow-ups

It remains unknown whether v2 improves held-out 0006 SSIM on non-catastrophic pairs or avoids T2w regression. Because coupling, initialization, two loss terms, decode, and leakage handling changed together, a positive result would belong to the bundle. The config identifies `paired_fraction` as a future explicit ablation, but no such run artifact exists.
