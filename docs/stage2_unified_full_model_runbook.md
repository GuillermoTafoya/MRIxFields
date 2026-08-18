# Unified retrospective Stage-2 runbook

This runbook extends the merged Variant-A and canonical-artifact runbooks. Their artifact
contracts and fail-closed preflights remain authoritative. All paths below are external to the
repository. Use the complete R/train and R/validation inventories; never point these commands at
prospective records.

## Ordered execution

1. Apply the reviewed Windows-root-to-Colab-root remap without changing the immutable split.
   Recompute the merged notebook's exact inventory arithmetic and verify the four sealed input
   hashes before loading any array.
2. Run `fit-stage2-photometry`, build the Gate 0.1 continuity reference, and run
   `audit-stage2-photometry`. Stop only if the official qualification result has
   `canonical_latent_bank_authorized: false` or an integrity check fails.
3. Run the canonical artifact storage/filesystem preflight, streamed build, and full audit:

```bash
fieldbridge preflight-photometry-factored-latent-bank \
  --config configs/experiment/stage2_canonical_artifacts_v2.yaml \
  --split-json "$OPERATIONAL_SPLIT" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --qualification "$QUALIFICATION_JSON" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --out-dir "$BANK_DIR" --device cuda

fieldbridge build-photometry-factored-latent-bank \
  --config configs/experiment/stage2_canonical_artifacts_v2.yaml \
  --split-json "$OPERATIONAL_SPLIT" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --qualification "$QUALIFICATION_JSON" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --out-dir "$BANK_DIR" --device cuda --resume --log-every 1

fieldbridge audit-photometry-factored-latent-bank \
  --config configs/experiment/stage2_canonical_artifacts_v2.yaml \
  --split-json "$OPERATIONAL_SPLIT" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --qualification "$QUALIFICATION_JSON" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --bank-dir "$BANK_DIR" --device cuda --log-every 1
```

If hard-link atomic publication is unsupported on the external filesystem, the preflight must
stop. Build on local scratch and hash-verify archival; do not introduce an overwrite fallback.

4. Run the short full-model sanity phase. It uses the same full objective and initial weights as
   the long run, with only the step count changed:

```bash
fieldbridge train-stage2-unified \
  --config configs/experiment/stage2_unified_full_retrospective_v1.yaml \
  --bank-dir "$BANK_DIR" --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$SANITY_CHECKPOINTS" --history-jsonl "$SANITY_HISTORY" \
  --steps 20 --sanity-steps 20 --device cuda
```

The command hard-stops on non-finite terms, OOM, missing 15-domain R/train coverage, empty
support, frozen-VAE mismatch, or weighted auxiliary domination. It never falls back to a smaller
model or tiled arithmetic.

5. With the notebook's single long-run authorization flag, train/resume the full model first,
   then the backward ablations. Resume from the greatest step-numbered exact checkpoint in the
   same variant directory. Each variant has its own external YAML, checkpoint directory, and
   append-only JSONL history. The backward order is `full`, `no_graph`, `no_anatomy_graph`,
   `no_adversarial_domain`, `sb_identity_only`, `sb_only`.

```bash
fieldbridge train-stage2-unified \
  --config "$VARIANT_CONFIG" --bank-dir "$BANK_DIR" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$CHECKPOINT_DIR" --history-jsonl "$HISTORY_JSONL" \
  --device cuda [--resume-from "$LATEST_EXACT_CHECKPOINT"]
```

6. Evaluate the full model on the complete paired R/validation manifest. The separately sealed
   baseline-prediction manifest supplies per-case Gate 0.1 calibrated identity and original
   SB-v2 arrays for same-case metrics/montages; it cannot select cases or affect model arithmetic.

```bash
fieldbridge eval-stage2-unified \
  --config configs/experiment/stage2_unified_full_retrospective_v1.yaml \
  --bank-dir "$BANK_DIR" --checkpoint "$FULL_CHECKPOINT" \
  --sb-only-checkpoint "$SB_ONLY_CHECKPOINT" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --paired-manifest "$PAIRED_R_VALIDATION_MANIFEST" \
  --baseline-predictions "$BASELINE_PREDICTIONS_MANIFEST" \
  --out "$EVALUATION_DIR" --device cuda --integration-steps 20 --solver heun --resume
```

The result includes official nRMSE/SSIM/LPIPS overall, by contrast, by directed field pair, and
by raw-identity ordinary/catastrophic stratum; requested-versus-wrong-target controls; graph,
anatomy/edge/gradient, and background diagnostics; and deterministic montages. It is
retrospective evidence only and makes no learned-disentanglement or promotion claim.
