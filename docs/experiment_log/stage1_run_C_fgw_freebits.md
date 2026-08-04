# Stage 1 — Run C foreground-weighted free-bits bundle

**Date:** 2026-07-22  
**Git commit(s):** [`92f12f1`](https://github.com/GuillermoTafoya/MRIxFields/commit/92f12f175fa30168c7478bb918c71a72493efe38)  
**Config:** [`configs/experiment/stage1_vae_v2_fgw_freebits.yaml`](../../configs/experiment/stage1_vae_v2_fgw_freebits.yaml)  
**Checkpoint / bank / output dir:** `outputs/stage1_vae_v2_fgw_freebits/checkpoints/`; archived as `../run_archive/C_stage1_vae_v2_fgw_freebits_ep15_done/`

## Hypothesis / what this run changed

Run C was explicitly a bundle, not an ablation. Versus Run B it enabled foreground-weighted L1 (`foreground_weight: 4.0`), inverse-frequency field-balanced training, per-channel free bits (`0.5` nats/latent-element), and validation early stopping with patience 7. Model, data, and cosine schedule were otherwise held fixed ([config diff and header](../../configs/experiment/stage1_vae_v2_fgw_freebits.yaml), [`92f12f1`](https://github.com/GuillermoTafoya/MRIxFields/commit/92f12f175fa30168c7478bb918c71a72493efe38)).

## Why (decision rationale)

The bundle targeted Run B's three measured defects: 27 epochs of validation overfitting after epoch 9, clipped T1w intensity at 5T/7T, and only `1/4` active latent channels. The config records field counts of `5T=220`, `0.1T=240`, and `1.5T=441` volumes as the imbalance motivating inverse-frequency sampling ([config header](../../configs/experiment/stage1_vae_v2_fgw_freebits.yaml)).

## What was measured

The full history contains 22 epochs and selects epoch 15 with validation total `1.0793057751`, range-normalized validation nRMSE `0.0361581135`, and SSIM3D `0.8977943446`. The final record at epoch 22 has elapsed time `17,608.909 s` (`4.891 h`) and validation total `1.7870821902`; all four latent channels remained active ([`history_full.jsonl`](../../../run_archive/C_stage1_vae_v2_fgw_freebits_ep15_done/history_full.jsonl)).

The 21-sample evaluation reports mean range-normalized nRMSE `0.0299075173`, SSIM3D `0.9724663269`, and LPIPS `0.0349563222`, with four active and zero dead latent units ([`metrics.json`](../../../run_archive/C_stage1_vae_v2_fgw_freebits_ep15_done/eval/metrics.json)). These archived nRMSE values are `RMSE / data_range`, not official Task-3 nRMSE.

The later Run-D commit reports that the same epoch-15 reconstruction measured `0.0066` range-normalized versus `0.4449` official nRMSE for T1w 7T, and a traveller ceiling of official nRMSE `0.285` and SSIM `0.957`; these values are **[claimed, unverified in artifacts]** because no official-metric JSON for that probe is present ([`a92f0ca`](https://github.com/GuillermoTafoya/MRIxFields/commit/a92f0ca650efb6fa56c5dc2d6202b6bd6f1a3e94)).

## Outcome

Promoted as the frozen Stage-1 VAE for Stage 2. Epoch 15 was archived as the best checkpoint, and the Stage-2 bank/notebooks identify Run C's VAE as the frozen source model ([archive checkpoint](../../../run_archive/C_stage1_vae_v2_fgw_freebits_ep15_done/checkpoints/vae_kl_vae_best_ep15.pt), [`403204e`](https://github.com/GuillermoTafoya/MRIxFields/commit/403204e5543d4fe470e5aad7cd7c1b50209e69ef)).

## Learnings

The bundle coincided with restoration of all four active channels and materially better archived range-nRMSE/SSIM/LPIPS than Run B, but its bundled design prevents assigning the gain to free bits, foreground weighting, field balance, or validation stopping individually. More importantly, the strong range-normalized reconstruction score concealed large relative error under the official dark-volume-sensitive nRMSE.

## Open questions / follow-ups

The unisolated bundle effects remain open. Run D changed only the nRMSE definition while warm-starting from Run C, making that objective change directly attributable if a completed Run-D artifact becomes available ([Run D](stage1_run_D_official_nrmse.md)).
