# Stage 1 — Run B 75-epoch cosine

**Date:** 2026-07-22  
**Git commit(s):** [`c2abe7f`](https://github.com/GuillermoTafoya/MRIxFields/commit/c2abe7f2747b66359efc52cbd65472e9392b068e)  
**Config:** [`configs/experiment/stage1_vae_75ep_cosine.yaml`](../../configs/experiment/stage1_vae_75ep_cosine.yaml)  
**Checkpoint / bank / output dir:** `outputs/stage1_vae_75ep_cosine/checkpoints/`; archived as `../run_archive/B_stage1_vae_75ep_cosine_ep37oom_20260722/`

## Hypothesis / what this run changed

Run B kept Run A's model, data, and loss mathematics fixed and changed instrumentation/scheduling: training-EMA early stopping off, a 75-epoch cosine schedule with 500-step warm-up and `0.05` minimum LR factor, validation each epoch, latent-collapse statistics, and reconstruction panels every five epochs. Foreground-weighted L1 was wired but disabled. This is an instrumentation/schedule bundle, not a loss ablation ([config header](../../configs/experiment/stage1_vae_75ep_cosine.yaml)).

## Why (decision rationale)

Run A had stopped after 11 epochs under a training-loss EMA even though that signal was documented as a GPU-saving plateau detector rather than a precise convergence criterion. Run B was intended to complete the schedule and select strictly by validation loss ([config header](../../configs/experiment/stage1_vae_75ep_cosine.yaml)).

## What was measured

The 36-record history shows best validation total `0.6439787268` at epoch 9, with range-normalized validation nRMSE `0.0744704976` and SSIM3D `0.7975811438`. At epoch 36, validation total had risen to `1.5865282624` while training continued to improve; elapsed time was `55,063.830 s` (`15.296 h`) ([`history.jsonl`](../../../run_archive/B_stage1_vae_75ep_cosine_ep37oom_20260722/history.jsonl)).

The 21-sample audit reports mean range-normalized nRMSE `0.0511459630`, SSIM3D `0.9214972172`, and LPIPS `0.0702171785`. It also reports one active and three dead latent units; per-channel mean KL was `[0.0013058, 0.0043431, 0.3796464, 0.0074994]` ([`metrics.json`](../../../run_archive/B_stage1_vae_75ep_cosine_ep37oom_20260722/eval/metrics.json)). These nRMSE values are `RMSE / data_range`, not official Task-3 nRMSE.

## Outcome

Diverged after its best checkpoint and was not promoted. The run crashed with CUDA OOM at epoch 37 rather than ending by early stopping; epoch 9 remained best-by-validation ([`run_archive/README.md`](../../../run_archive/README.md)).

## Learnings

Three mechanisms were recorded: validation overfitting after epoch 9 while training loss fell; high-field T1w range collapse (reconstruction maxima `0.61765` at 5T and `0.71588` at 7T versus target `1.0`); and posterior collapse to one of four active channels ([`metrics.json`](../../../run_archive/B_stage1_vae_75ep_cosine_ep37oom_20260722/eval/metrics.json), [`run_archive/README.md`](../../../run_archive/README.md)). The run demonstrated why a longer schedule alone was not the next lever.

## Open questions / follow-ups

Because Run C changed foreground weighting, field balance, free bits, and early stopping together, it tested a corrective bundle rather than isolating which defect mattered most. That bundle is documented in [Run C](stage1_run_C_fgw_freebits.md).
