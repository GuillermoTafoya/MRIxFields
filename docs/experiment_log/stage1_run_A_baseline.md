# Stage 1 — Run A baseline

**Date:** 2026-07-21  
**Git commit(s):** [`c7c8f7a`](https://github.com/GuillermoTafoya/MRIxFields/commit/c7c8f7a257a3e6fbb0cc35d7b2d67a146bc1a711), building on the Stage-1 contract in [`539ffec`](https://github.com/GuillermoTafoya/MRIxFields/commit/539ffecdd7bbd6872971df059d28aa3fcd6494f1)  
**Config:** [`configs/experiment/stage1_vae.yaml`](../../configs/experiment/stage1_vae.yaml)  
**Checkpoint / bank / output dir:** `outputs/stage1_vae/checkpoints/`; archived as `../run_archive/A_stage1_vae_ep11_20260721/` (archive origin and mapping: [`run_archive/README.md`](../../../run_archive/README.md))

## Hypothesis / what this run changed

Run A was the first archived execution of the 3D, four-channel KL-VAE under the official `[0,1]` data contract. Relative to the earlier unstable development path, the lineage had already added log-variance clamping, gradient clipping, stratified 64³ patch sampling, per-epoch validation, and background-reader overlap. This is a baseline bundle, not an ablation; its outcome cannot be assigned to one component ([`stage1_vae.yaml`](../../configs/experiment/stage1_vae.yaml), commits [`9c92f07`](https://github.com/GuillermoTafoya/MRIxFields/commit/9c92f079f21dfc453080589de9e84614d3e07a09), [`539ffec`](https://github.com/GuillermoTafoya/MRIxFields/commit/539ffecdd7bbd6872971df059d28aa3fcd6494f1), and [`c7c8f7a`](https://github.com/GuillermoTafoya/MRIxFields/commit/c7c8f7a257a3e6fbb0cc35d7b2d67a146bc1a711)).

## Why (decision rationale)

The preceding 20-step validation run had diverged from loss `3 -> 7 -> 37 -> 203`; the commit attributed the unbounded component to KL/log-variance growth and introduced `logvar ∈ [-30,20]` plus gradient clipping at `1.0` ([`9c92f07`](https://github.com/GuillermoTafoya/MRIxFields/commit/9c92f079f21dfc453080589de9e84614d3e07a09)). Uniform crops had also landed on near-empty air more than two-thirds of the time in a simulation, motivating the configured `0.7/0.2/0.1` foreground/border/air strata ([`stage1_vae.yaml`](../../configs/experiment/stage1_vae.yaml)).

## What was measured

The archive history contains 11 epochs. Best validation total was `0.6102030776` at epoch 7; at that epoch validation range-normalized nRMSE was `0.0632810060` and SSIM3D was `0.8002814966`. The last recorded wall time was `16,871.225 s` (`4.686 h`) at epoch 11 ([`history.jsonl`](../../../run_archive/A_stage1_vae_ep11_20260721/history.jsonl)).

The archived evaluation covers only four T1w cases. Its means are range-normalized nRMSE `0.0343788955`, SSIM3D `0.8194357008`, and LPIPS `0.0827978421` ([`metrics.json`](../../../run_archive/A_stage1_vae_ep11_20260721/eval/metrics.json)). This nRMSE is `RMSE / data_range`, not official Task-3 `||p-t|| / ||t||`; no official nRMSE is present for Run A.

## Outcome

Superseded. Training stopped at epoch 11 with best-by-validation at epoch 7; the archive labels this a short run and Run B replaced it with a 75-epoch cosine schedule and richer evaluation ([`run_archive/README.md`](../../../run_archive/README.md)).

## Learnings

The run established a finite, trainable baseline, but its four-case, T1w-only evaluation was too narrow for cross-domain model selection. Its 5T T1w reconstruction reached only `0.52635` maximum intensity versus target maximum `1.0`, an early instance of the high-field dynamic-range problem later quantified in Run B ([`metrics.json`](../../../run_archive/A_stage1_vae_ep11_20260721/eval/metrics.json)).

## Open questions / follow-ups

The archive did not contain latent-channel activity statistics for Run A. Run B added per-epoch latent diagnostics and a 21-sample domain-balanced audit; Run C then directly addressed the collapse and dynamic-range defects ([Run B](stage1_run_B_75ep_cosine.md), [Run C](stage1_run_C_fgw_freebits.md)).
