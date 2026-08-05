# Stage 1 — v3 target-decoder FiLM arm

**Date:** 2026-07-23  
**Git commit(s):** [`e74d9a9`](https://github.com/GuillermoTafoya/MRIxFields/commit/e74d9a95b31e3e94e47c83f3b4b355c612f3de7b)  
**Config:** [`configs/experiment/stage1_vae_v3_target_decoder_film.yaml`](../../configs/experiment/stage1_vae_v3_target_decoder_film.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage1_vae_v3_target_decoder_film/checkpoints/`

## Hypothesis / what this run changed

Relative to the v3 joint-domain arm, this experimental arm changed the shared decoder from unconditioned source reconstruction (`domain_conditioning_dim: 0`, `decoder_domain: source`) to a low-capacity target-domain FiLM decoder (`domain_conditioning_dim: 16`, `decoder_domain: target`). The remaining declared schedule, loss, balance, KL, and promotion settings match the sibling config, making this a one-config-variable architecture comparison ([joint config](../../configs/experiment/stage1_vae_v3_joint_domain_freebits.yaml), [FiLM config](../../configs/experiment/stage1_vae_v3_target_decoder_film.yaml)).

## Why (decision rationale)

The config labels this an isolated experimental arm rather than a replacement for the unconditioned v3 arm. The commit body contains no measured precursor or additional rationale, so none is inferred.

## What was measured

No history, checkpoint, metrics JSON, or output directory for this arm was found. The 40-epoch setting is a planned maximum, not an observed run length. nRMSE loss was configured off (`0.0`), and there are no official or range-normalized evaluation results.

## Outcome

Declared experimental arm; execution and promotion are unverified.

## Learnings

No empirical conclusion can be drawn. The config does establish the intended clean comparison: target-domain FiLM conditioning is the only declared difference from its v3 joint-domain sibling.

## Open questions / follow-ups

The comparison is unresolved because neither arm has a checked-in run artifact. A future result must preserve the shared settings to remain an interpretable conditioning ablation.
