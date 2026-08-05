# Stage 1 — Run D official nRMSE

**Date:** 2026-07-29  
**Git commit(s):** [`a92f0ca`](https://github.com/GuillermoTafoya/MRIxFields/commit/a92f0ca650efb6fa56c5dc2d6202b6bd6f1a3e94), rationale corrected in [`162dbeb`](https://github.com/GuillermoTafoya/MRIxFields/commit/162dbeba0af694eee7d2252c02fc4047857b1e5f)  
**Config:** [`configs/experiment/stage1_vae_v3_official_nrmse.yaml`](https://github.com/GuillermoTafoya/MRIxFields/blob/162dbeba0af694eee7d2252c02fc4047857b1e5f/configs/experiment/stage1_vae_v3_official_nrmse.yaml)  
**Checkpoint / bank / output dir:** configured as `outputs/stage1_vae_v3_official_nrmse/checkpoints/`; warm start `outputs/stage1_vae_v2_fgw_freebits/checkpoints/vae_kl_vae_best.pt`

## Hypothesis / what this run changed

Run D was designed as a one-variable follow-up to Run C: keep the v2 bundle fixed but change `training.nrmse_mode` from implicit `range` to `official`, using `||p-t|| / ||t||` with `nrmse_rms_floor: 0.02`. It warm-started weights only from Run C epoch 15 and used a fresh optimizer/cosine curve. The isolated checkpoint path and 25-epoch (`79,500`-step) horizon are operational changes required by the new objective, not separate scientific ablations ([config header](https://github.com/GuillermoTafoya/MRIxFields/blob/162dbeba0af694eee7d2252c02fc4047857b1e5f/configs/experiment/stage1_vae_v3_official_nrmse.yaml)).

## Why (decision rationale)

Run C optimized `RMSE / data_range`, whereas Task 3 scores `||p-t|| / ||t||`. The commit reports a `5–70x` disagreement and, for T1w 7T, `0.0066` range-normalized versus `0.4449` official; it also reports a frozen-VAE traveller ceiling of official nRMSE `0.285` and SSIM `0.957`. These probe values are **[claimed, unverified in artifacts]** because the repository has no corresponding probe metrics file ([`a92f0ca`](https://github.com/GuillermoTafoya/MRIxFields/commit/a92f0ca650efb6fa56c5dc2d6202b6bd6f1a3e94)).

## What was measured

A 100-real-step weight probe measured official nRMSE terms `0.076–0.155`, SSIM loss `0.028–0.19`, and LPIPS `0.056–0.12`; consequently the nRMSE weight remained `1.0`. These values are present in the corrected config header and commit body but no standalone probe log exists, so they are **[claimed, unverified in artifacts]** ([`162dbeb`](https://github.com/GuillermoTafoya/MRIxFields/commit/162dbeba0af694eee7d2252c02fc4047857b1e5f)).

No completed Run-D history, checkpoint, or evaluation metrics artifact was found in `run_archive/` or the workspace. Therefore no Run-D official nRMSE, SSIM, LPIPS, wall-clock, or selected epoch can be reported.

## Outcome

Configured and probe-checked, but completion/promotion is unverified. It must not be described as a completed or successful run from the available artifacts.

## Learnings

The probe corrected an initially wrong scale rationale: official nRMSE is not about ten times larger on the foreground-enriched 64³ training patches. Patch composition changes `||t||`, so volume-level metric scale cannot be transferred directly to loss-weight selection.

## Open questions / follow-ups

The central question—whether optimizing official nRMSE improves official full-volume nRMSE without reducing SSIM/LPIPS—remains unanswered until a Run-D history and official evaluation artifact are archived.
