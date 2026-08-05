# Stage 1 — v3 joint-domain free-bits arm

**Date:** 2026-07-23  
**Git commit(s):** [`e74d9a9`](https://github.com/GuillermoTafoya/MRIxFields/commit/e74d9a95b31e3e94e47c83f3b4b355c612f3de7b), contract hardening in [`df7fcae`](https://github.com/GuillermoTafoya/MRIxFields/commit/df7fcaefc010b7471d445af5dd5463e004388671)  
**Config:** [`configs/experiment/stage1_vae_v3_joint_domain_freebits.yaml`](../../configs/experiment/stage1_vae_v3_joint_domain_freebits.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage1_vae_v3_joint_domain_freebits/checkpoints/`

## Hypothesis / what this run changed

This declared arm replaced Run C's field-only balance with joint field/contrast/subject balance; used an unbounded decoder output (`output_activation: none`); changed the loss to masked L1 `1.0`, background `0.1`, SSIM `0.25`, nRMSE `0.0`, LPIPS `0.0`, and KL `0.0001`; reduced free bits from `0.5` to `0.01`; and added a 10-epoch KL warm-up plus explicit latent-activity/promotion gates. This is a broad development bundle, not an ablation ([config](../../configs/experiment/stage1_vae_v3_joint_domain_freebits.yaml)).

## Why (decision rationale)

The config makes the scientific intent explicit: require all validation domains, at least three active channels, and activity under both standard-deviation and KL/input-dependence rules. No fuller rationale appears in the commit body, so no additional causal motivation is asserted.

## What was measured

No history, checkpoint, evaluation JSON, or output directory for this arm was found. The config declares a maximum of 40 epochs, but that is a planned horizon, not a measured duration. No nRMSE definition or result can be assigned; the configured nRMSE loss weight is `0.0`.

## Outcome

Declared development arm; execution and promotion are unverified.

## Learnings

No empirical learning is supportable. Methodologically, the config shows a shift toward domain-complete validation and explicit latent-health promotion criteria.

## Open questions / follow-ups

Whether the altered reconstruction recipe, joint-domain balance, or KL warm-up improves any metric remains open. Because all changed together, even a later result would be attributable only to the bundle unless isolated.
