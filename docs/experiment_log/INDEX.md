# MRIxFields Task-3 experiment log

Chronological paper-writing entry point. “Unverified” means a config/notebook exists but no completed run artifact was found; commit-only measurements are labeled inside the run file.

- 2026-07-21 — [Stage 1 Run A baseline](stage1_run_A_baseline.md): superseded after 11 epochs; best validation epoch 7.
- 2026-07-22 — [Stage 1 Run B 75-epoch cosine](stage1_run_B_75ep_cosine.md): not promoted; best epoch 9, validation overfit, then CUDA OOM at epoch 37.
- 2026-07-22 — [Stage 1 Run C foreground-weighted free-bits bundle](stage1_run_C_fgw_freebits.md): promoted as the frozen Stage-1 VAE; best epoch 15.
- 2026-07-23 — [Stage 1 v3 joint-domain free-bits arm](stage1_v3_joint_domain_freebits.md): declared; execution unverified.
- 2026-07-23 — [Stage 1 v3 target-decoder FiLM arm](stage1_v3_target_decoder_film.md): declared experimental arm; execution unverified.
- 2026-07-27 — [Stage 2 v1 Schrödinger-bridge OT coupling](stage2_v1_sb_ot_coupling.md): configured primary rung; execution unverified.
- 2026-07-29 — [Stage 1 Run D official nRMSE](stage1_run_D_official_nrmse.md): configured and probe-checked; completed run/promotion unverified.
- 2026-07-29 — [Stage 2 v1 OT-CFM coupling](stage2_v1_fm_ot_coupling.md): not promoted after the frozen 0009 gate diagnosed photometric rather than structural gains; gate values are commit-only.
- 2026-07-29 — [Stage 2 v2 OT-CFM NN coupling + leakage fix](stage2_v2_fm_nn_leakage_fix.md): configured and leakage-hardened; execution/promotion unverified.
- 2026-07-29 — [Stage 2 v2 Schrödinger bridge NN coupling + leakage fix](stage2_v2_sb_nn_leakage_fix.md): configured primary rung; execution/promotion unverified.
- 2026-08-03 — [Stage 2 Gate 0 diagnostic](stage2_gate0_diagnostic.md): steps 2 and 5 executed on CPU; the closed-form latent affine explains ~0-2.5% of the cross-field displacement and what it leaves transfers between travellers at 0.5-1.4% (n=2). Verdict `gate0_close_generative_branch`; steps 1/3/4 pending on Colab.

## Directly comparable deltas

| Lineage | Single comparable delta | Recorded outcome |
|---|---|---|
| Stage 1 A → B | Schedule/instrumentation bundle; model/data/loss unchanged | B improved evaluation coverage but overfit after epoch 9 and collapsed to 1/4 active latent channels; not a single-component ablation. |
| Stage 1 B → C | Foreground-weighted L1 + field balance + free bits + validation early stop | Range-nRMSE `0.05115 → 0.02991`, SSIM3D `0.92150 → 0.97247`, LPIPS `0.07022 → 0.03496`; bundle attribution only. |
| Stage 1 C → D | nRMSE objective `range → official` (weights warm-started from C) | No completed Run-D artifact; outcome unresolved. |
| Stage 1 v3 joint → v3 FiLM | Decoder conditioning `none/source → 16-D target FiLM` | Neither execution is evidenced; outcome unresolved. |
| Stage 2 v1 FM → v1 SB | Deterministic OT-CFM → Brownian Schrödinger bridge (`sigma=0.1`) | No SB metrics artifact; outcome unresolved. |
| Stage 2 v1 FM → v2 FM | NN coupling + identity start + two regularizers + correct decode + leakage-safe resplit | Bundle, not an ablation; no completed v2 artifact. |
| Stage 2 v2 FM → v2 SB | Deterministic OT-CFM → Brownian Schrödinger bridge (`sigma=0.1`) | No completed metrics artifacts; outcome unresolved. |
