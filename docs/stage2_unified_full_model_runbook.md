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

4. Quantify residual domain separability on the normalized factored latents and audit paired
   evaluation feasibility before asking an operator for a paired manifest:

```bash
fieldbridge preflight-stage2-factored-domain-separability \
  --bank-dir "$BANK_DIR" --out "$DOMAIN_SEPARABILITY_JSON"

fieldbridge audit-stage2-retrospective-pair-feasibility \
  --split-json "$OPERATIONAL_SPLIT" --out "$PAIR_FEASIBILITY_JSON"
```

The classifier fits deterministic domain centroids only on supported R/train cells and scores
the subject-disjoint complete R/validation inventory, reporting a 15x15 confusion matrix and
per-domain accuracy. This evidence measures residual predictability; it does not prove legitimate
target-domain control or learned disentanglement.

The feasibility audit independently reconciles every cohort identity before source-file access,
opens no array, and excludes only records positively classified as P. Missing, malformed, or
conflicting R/P evidence is an integrity error; it cannot coexist with
`complete_inventory_no_selection: true`. The audit then
seals every same-subject/same-contrast/cross-field directed R/validation edge. If it reports no
edges, paired R/validation evaluation is impossible and unrelated subjects must not be paired.
A P-traveller protocol would require a separately versioned explicit authorization. If edges do
exist, build rather than hand-author the two evaluation manifests:

```bash
fieldbridge build-stage2-retrospective-paired-manifest \
  --feasibility "$PAIR_FEASIBILITY_JSON" \
  --materialized-arrays "$SEALED_R_VALIDATION_ARRAYS_AND_STAGE1_CEILINGS" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --authorization-reference "complete-R-validation-feasibility:<result-sha256>" \
  --out "$PAIRED_R_VALIDATION_MANIFEST"

fieldbridge build-stage2-unified-baseline-manifest \
  --paired-manifest "$PAIRED_R_VALIDATION_MANIFEST" \
  --source-artifact "$SEALED_EXISTING_GATE01_AND_SBV2_PREDICTIONS" \
  --out "$BASELINE_PREDICTIONS_MANIFEST"
```

The materialized-array export must use `stage2-materialized-r-validation-arrays-v2`, include one
frozen Stage-1 reconstruction for every target, and seal deterministic producer provenance under
`stage2-r-validation-array-and-stage1-ceiling-producer-v1`. The baseline source must use
`stage2-existing-gate01-sbv2-baseline-source-v2` with producer contract
`stage2-gate01-sbv2-baseline-export-producer-v1`. Both producers seal source-code/config hashes,
full-volume arithmetic, and complete-inventory/no-selection status. Missing inputs fail with exact
operator instructions; no curated subset is accepted.

Before a 100k-step authorization can be consulted, seal the complete path:

```bash
fieldbridge seal-stage2-long-run-evaluation-readiness \
  --feasibility "$PAIR_FEASIBILITY_JSON" \
  --materialized-arrays "$SEALED_R_VALIDATION_ARRAYS_AND_STAGE1_CEILINGS" \
  --paired-manifest "$PAIRED_R_VALIDATION_MANIFEST" \
  --baseline-source "$SEALED_EXISTING_GATE01_AND_SBV2_PREDICTIONS" \
  --baseline-predictions "$BASELINE_PREDICTIONS_MANIFEST" \
  --out "$LONG_RUN_EVALUATION_READINESS_JSON"
```

This command has no prospective fallback. If genuine retrospective pairs or any deterministic
producer/output identity is absent, long training is blocked. A prospective protocol would need
a separately versioned reviewed implementation and explicit authorization.

5. Run the 200-step full-objective pilot. It uses the same six objectives and initial weights as
   the strongest model:

```bash
fieldbridge train-stage2-unified \
  --config configs/experiment/stage2_unified_full_retrospective_v3.yaml \
  --bank-dir "$BANK_DIR" --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$PILOT_CHECKPOINTS" --history-jsonl "$PILOT_HISTORY" \
  --steps 200 --pilot-steps 200 --device cuda
```

The pilot logs raw/weighted terms, each term's translator-gradient norm, generator/critic norms,
critic score means/std/quantiles/separation/saturation, real/generated domain accuracy,
first/last/smoothed loss behavior, auxiliary/flow ratios, throughput, peak memory, and projected
100k-step time. Cost is reported only when the operator supplies an hourly GPU rate. It hard-stops
on nonfinite values, missing gradients, discriminator saturation, uncontrolled auxiliary
dominance/loss growth, or OOM. Frozen decoder parameters remain unchanged and every decoder call
receives denormalized latent coordinates.

Before any validation array is loaded, `stage2-unified-validation-plan-v1` freezes the complete
R/validation source inventory, a subject-excluded same-contrast independent target, bridge time,
and stochastic-noise seed for every case. These draws depend only on validation seed `20260818`
and case identity—never training step, model, variant, checkpoint, or training RNG. The plan hash
is part of every validation event, checkpoint, run fingerprint, and selection receipt, so the
full model and every ablation must use the identical plan.

This is not paired endpoint evaluation. Immutable step checkpoints and selection receipts seal
latest/best using the fixed critic-independent rule:

`val_sb + 0.1*val_identity + 0.02*val_anatomy + 0.01*val_graph`.

The jointly trained critic's realism scores and real/generated domain accuracies remain diagnostic
only and never enter checkpoint ranking. The fixed rule does not inherit an ablation's training
weights, so it remains defined for SB-only.

6. The notebook's first long-run flag trains/resumes **only** the strongest full model. Review its
   pilot and evaluation before setting the separate backward-ablation authorization flag. Resume
   from the greatest step-numbered exact checkpoint in the same variant directory. Each variant
   has its own resolved config, checkpoint directory, append-only JSONL history, and selection
   receipts. A completed receipt seals hashes for both selected best and final. Evaluation verifies
   the latest receipt and loads `best_checkpoint`; final is retained only as a separately labelled
   diagnostic. The later backward order is `no_graph`, `no_anatomy_graph`,
   `no_adversarial_domain`, `sb_identity_only`, `sb_only`.

```bash
fieldbridge train-stage2-unified \
  --config "$VARIANT_CONFIG" --bank-dir "$BANK_DIR" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$CHECKPOINT_DIR" --history-jsonl "$HISTORY_JSONL" \
  --device cuda [--resume-from "$LATEST_EXACT_CHECKPOINT"]
```

7. If and only if feasibility found genuine R/validation pairs, evaluate the strongest model and
   later every trained ablation. The built baseline manifest supplies same-case Gate 0.1 calibrated
   identity and original SB-v2 arrays; it cannot select cases or affect model arithmetic.

```bash
fieldbridge eval-stage2-unified \
  --config configs/experiment/stage2_unified_full_retrospective_v3.yaml \
  --bank-dir "$BANK_DIR" --selection-receipt "$LATEST_FULL_SELECTION_RECEIPT" \
  --sb-only-checkpoint "$SB_ONLY_CHECKPOINT" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --paired-manifest "$PAIRED_R_VALIDATION_MANIFEST" \
  --baseline-predictions "$BASELINE_PREDICTIONS_MANIFEST" \
  --out "$EVALUATION_DIR" --device cuda --integration-steps 20 --solver heun --resume
```

Pass every additionally trained variant as repeated `--ablation-checkpoint NAME=PATH`; the command
records the exact set and evaluates all of them. Wrong-condition predictions are rendered through
the same requested-target photometry map for the mechanistic comparison; condition-native renders
are separately labelled diagnostics. Results include nRMSE/SSIM/LPIPS overall and by source
domain, target domain, contrast, directed pair, and ordinary/catastrophic stratum; all valid graph
intermediates; anatomy/edge/gradient preservation; raw pre-mask decoder background leakage; and
montages containing raw identity, calibrated identity, original SB-v2, factored SB-only when
trained, full prediction, Stage-1 ceiling, and error. This is retrospective evidence only and
makes no learned-disentanglement or promotion claim.

Spectral normalization and lazy R1 remain disabled in the primary configuration. Critic score
distributions, domain accuracy, and saturation evidence must be reviewed before either is added as
a separately configured and tested stability ablation.
