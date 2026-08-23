# Unified retrospective Stage-2 runbook

This runbook extends the merged Variant-A and canonical-artifact runbooks. Their artifact
contracts and fail-closed preflights remain authoritative. All paths below are external to the
repository. Use the complete R/train and R/validation inventories; never point these commands at
prospective records.

The operational namespace and configuration contract are v7. Earlier unified configuration and
resume plans are retained only as historical review material and are rejected by the v7 loader.

## Ordered execution

1. Apply the reviewed Windows-root-to-Colab-root remap without changing the immutable split.
   Recompute the merged notebook's exact inventory arithmetic and verify the four sealed input
   hashes before loading any array.
2. Run `fit-stage2-photometry`, build the Gate 0.1 continuity reference, and run
   `audit-stage2-photometry`. Stop only if the official qualification result has
   `canonical_latent_bank_authorized: false` or an integrity check fails.
3. Set `BANK_DIR` to local scratch (for Colab v7,
   `/content/stage2_unified_v7_scratch/photometry_factored_latent_bank_v2_working`),
   never to the mounted Drive. Run the canonical artifact storage/filesystem preflight, streamed
   build, and full audit against that local path:

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

After the local audit passes, compute a deterministic tree identity over every relative path,
byte count, and file SHA-256. Copy to a unique partial archive attempt on Drive, verify the same
tree identity, and publish by no-clobber rename. If publication or Drive remount/retry fails, stop
with the local bank intact; do not introduce an overwrite fallback. On a later runtime, copy the
immutable archive back to local scratch and require byte-identical tree identity before any
training read.

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
A P endpoint is never a substitute for R-only validation or checkpoint selection. If genuine R edges do
exist, import the reviewed producer-export archive and let the repository construct every
evaluator manifest:

```bash
fieldbridge import-stage2-retrospective-paired-evaluation \
  --feasibility "$PAIR_FEASIBILITY_JSON" \
  --archive-root "$REVIEWED_R_PAIRED_PRODUCER_EXPORT_ARCHIVE" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --authorization-reference "complete-R-validation-feasibility:<result-sha256>" \
  --out-dir "$EVALUATION_INPUT_DIR"
```

The materialized-array export must use `stage2-materialized-r-validation-arrays-v2`, include one
frozen Stage-1 reconstruction for every target, and seal deterministic producer provenance under
`stage2-r-validation-array-and-stage1-ceiling-producer-v1`. The baseline source must use
`stage2-existing-gate01-sbv2-baseline-source-v2` with producer contract
`stage2-gate01-sbv2-baseline-export-producer-v1`. Both producers seal source-code/config hashes,
full-volume arithmetic, and complete-inventory/no-selection status. Missing inputs fail with exact
operator instructions; no curated subset is accepted. The importer requires exactly one export
of each contract and creates the paired manifest, baseline manifest, and R-path readiness receipt
with atomic no-clobber writes.

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

When the feasibility audit proves that no genuine R pair exists, import the already sealed
`Gate01Private_8012a3f` graph instead. Its forward-versioned data role is exactly
`development_validation_P0006_evaluation_only`. P:0006 was already inspected during method
development, so this protocol supports development/model assessment only and cannot support
population or generalization claims. It is not a training or model-selection input:

```bash
fieldbridge import-stage2-gate01-p0006-evaluation \
  --archive-root "$GATE01_PRIVATE_8012A3F" \
  --expected-gate01-result-sha256 \
    454747cd3e4b1376855915244a7c40fe281b758150e86f584fbea96f94d531f5 \
  --bank-dir "$BANK_DIR" \
  --validation-plan "$FROZEN_VALIDATION_PLAN_V2" \
  --out "$P0006_EVALUATION_PROTOCOL"

fieldbridge seal-stage2-long-run-evaluation-readiness \
  --feasibility "$PAIR_FEASIBILITY_JSON" \
  --p0006-evaluation-protocol "$P0006_EVALUATION_PROTOCOL" \
  --out "$LONG_RUN_EVALUATION_READINESS_JSON"
```

The importer verifies the detached archive inventory, exact Gate 0.1 result hash, independent
protocol lock, calibrator, all 15 acquisition nodes, all 60 directions, all 180 wrong-target
references, Stage-1 ceilings, and the producer's actual `path_used=[full]`. It loads and hashes
every private array only for this import validation. It rejects any traveller other than P:0006,
proves the factored train/validation bank contains only R, and proves the frozen unpaired
validation plan contains no P endpoint. Original SB-v2 arrays are imported; calibrated identity
is deterministically re-derived from the existing raw identity, source support, and frozen Gate
0.1 calibrator and its tensor identity is sealed. No baseline model is rerun.

Readiness v3 accepts exactly one feasible route: complete genuine R/validation pairs when they
exist, otherwise the sealed P:0006 development-validation evaluation-only protocol. It records
that P:0009 remains frozen and unused for possible later confirmation; this PR does not execute
P:0009. Absence of both routes hard-stops before the long-run authorization flag.

5. Run the dedicated one-step A100 gate, then the 20-step recovery probe, and only then the
   200-step full-objective pilot. All three use the same six objectives and initial weights as
   the strongest model. The notebook derives three separately sealed resolved configurations
   and exact-resume directories from the primary v7 file:

```bash
fieldbridge train-stage2-unified \
  --qualification-only \
  --config "$A100_GATE_RESOLVED_CONFIG" \
  --bank-dir "$BANK_DIR" --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$A100_GATE_CHECKPOINTS" --history-jsonl "$A100_GATE_HISTORY" \
  --device cuda

fieldbridge train-stage2-unified \
  --config "$PILOT_20_RESOLVED_CONFIG" \
  --bank-dir "$BANK_DIR" --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$PILOT_20_CHECKPOINTS" --history-jsonl "$PILOT_20_HISTORY" \
  --device cuda [--resume-from "$LATEST_PILOT_20_CHECKPOINT"]

fieldbridge train-stage2-unified \
  --config "$PILOT_200_RESOLVED_CONFIG" \
  --bank-dir "$BANK_DIR" --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --checkpoint-dir "$PILOT_CHECKPOINTS" --history-jsonl "$PILOT_HISTORY" \
  --device cuda [--resume-from "$LATEST_PILOT_200_CHECKPOINT"]
```

The v7 primary profile is fixed to BF16, batch size one, four-step Heun transport, and the same
six weights. Its decoder contract checkpoints up1 and up2 as separate complete-volume regions.
Each residual block has two separate complete-volume GroupNorm -> SiLU -> Conv3d regions.
Residual skips remain outside replay and execute after the second branch exactly as in the
frozen decoder. There is no outer whole-decoder checkpoint, crop, tile, normalization
approximation, resolution reduction, integration-step reduction, disabled full-model loss, or
allocator fallback. Source-image anatomy decoding is under no_grad and uses the ordinary
decoder; only generated, gradient-bearing decoding takes the fine-grained path.

The generator ports the reviewed term-wise v6 lifetime exactly. It freezes the selected batch,
bridge time/noise, intermediate domains, and a differentiable-forward RNG replay seed once per
step. For each enabled term in the sealed order sb, identity, anatomy, graph, adversarial,
domain, it constructs only that term's graph under `save_on_cpu`, routes every differentiable
translator invocation through non-reentrant RNG-preserving checkpointing, and immediately calls
backward without retaining the
graph, releases the graph, and only then constructs the next term. The pilot gradient probe does
not reconstruct a joint six-term graph. Accumulated weighted gradients receive exactly one
generator optimizer update. During the configured pilot steps only, parameter hooks measure each
term's translator gradient from that same backward. After the pilot,
`qualify_term_gradients=false` bypasses all six hook sets and their GPU-to-CPU norm
synchronizations while preserving the identical weighted backward and optimizer update.

The first step emits `stage2-unified-anatomy-memory-qualification-v1` with allocated and reserved
CUDA peaks for anatomy forward/backward and the whole step, the checkpoint-granularity hash, and
unchanged before/after frozen-decoder state hashes. It also emits
`stage2-unified-a100-one-step-memory-gate-v1`. The gate requires an NVIDIA A100 and peak allocated
memory no greater than 72 GiB (77,309,411,328 bytes); failure stops before the 20- or 200-step
command is launched. The first command uses
`stage2-unified-a100-qualification-only-exit-v1`: it atomically writes
`stage2_unified_a100_qualification_only_receipt_v1.json` and exits before complete 60-cell
validation or checkpoint publication. The notebook verifies this receipt before launching the
20-step probe.

Each Colab command is an attempt. Stdout is streamed visibly into an ephemeral
`/content/stage2_unified_v7_scratch` log; no Drive log is held open while the subprocess runs.
After completion, the local log is copied once into a unique external attempt path and sealed
with a SHA-256 receipt. Each scientific run also uses
`scientific_attempts/attempt-NNNN/{history.jsonl,checkpoints/}`. A rerun resumes the greatest
immutable checkpoint in the latest attempt. If the latest attempt is nonempty but has no
checkpoint (including OOM or interruption before step 10/20), it is preserved and a new attempt
namespace is selected, avoiding the nonempty-history/no-checkpoint collision. Critical
exact-resume checkpoints and their history prefix remain in the variant's durable external
attempt directory; temporary files, package caches, command logs, and all bank training reads use
local scratch. Existing attempts and logs are never appended across namespaces or overwritten.

The pilot logs raw/weighted terms, each term's translator-gradient norm, generator/critic norms,
critic score means/std/quantiles/separation/saturation, real/generated domain accuracy,
first/last/smoothed loss behavior and auxiliary/flow ratios. Runtime projection v1 measures and
reports the mean training-step seconds and one complete 60-cell validation separately. It computes
projected training seconds, the exact union of cadence, pilot-boundary, and terminal validation
runs (with the terminal counted exactly once), projected validation seconds, total hours, total
GPU cost at the operator-supplied rate, and peak memory across both phases. The long-run flag is
shown only alongside this training-plus-validation total; a training-only projection is not an
authorization estimate. The pilot hard-stops on nonfinite values, missing gradients,
discriminator saturation, uncontrolled auxiliary dominance/loss growth, or training/validation
OOM. Frozen decoder parameters remain unchanged and every decoder call receives denormalized
latent coordinates.

Before any validation array is loaded, `stage2-unified-validation-plan-v2` freezes the complete
R/validation source inventory. Every source is assigned one subject-excluded target at each of
the four other fields, so all 60 contrast x directed-field cells must be represented. Target
identity, bridge time, and stochastic-noise seed are edge-specific functions of seed `20260818`
and case identity--never training step, model, variant, checkpoint, or training RNG. The plan hash
is part of every validation event, checkpoint, run fingerprint, and selection receipt, so the
full model and every ablation use byte-identical draws. The plan records per-cell counts and
fails closed if any required cell has no subject-excluded target.

This is not paired endpoint evaluation. Immutable step checkpoints and selection receipts seal
latest/best using the fixed critic-independent rule:

`val_sb + 0.1*val_identity + 0.02*val_anatomy + 0.01*val_graph`.

Each `val_*` is an equal-weighted macro mean over the 60 directed-domain cells: records are
averaged within a cell first, then every cell receives one equal vote. Record-weighted means are
reported only as diagnostics, so a common domain cannot dominate selection.

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

7. Evaluate selected-best, never merely final. Use the genuine R manifests when feasibility
   succeeds; otherwise use the sealed P:0006 development-validation protocol. P:0006 is encoded at evaluation time as
   `E(N_d(x))` through the frozen posterior-mean full-volume encoder and normalized with R/train
   factored-bank statistics. It never enters that bank, the pilot, validation, selection, or
   checkpoints.

```bash
fieldbridge eval-stage2-unified \
  --config configs/experiment/stage2_unified_full_retrospective_v7.yaml \
  --bank-dir "$BANK_DIR" --selection-receipt "$LATEST_FULL_SELECTION_RECEIPT" \
  --sb-only-checkpoint "$SB_ONLY_CHECKPOINT" \
  --vae-config "$VAE_CONFIG" --vae-checkpoint "$VAE_CHECKPOINT" \
  --photometry-artifact "$PHOTOMETRY_ARTIFACT" \
  --paired-manifest "$PAIRED_R_VALIDATION_MANIFEST" \
  --baseline-predictions "$BASELINE_PREDICTIONS_MANIFEST" \
  --out "$EVALUATION_DIR" --device cuda --integration-steps 20 --solver heun --resume
```

For the P:0006 branch, replace `--paired-manifest` and `--baseline-predictions` with:

```bash
--p0006-evaluation-protocol "$P0006_EVALUATION_PROTOCOL"
```

Pass every additionally trained variant as repeated `--ablation-checkpoint NAME=PATH`; the command
records the exact set and evaluates all of them. Wrong-condition predictions are rendered through
the same requested-target photometry map for the mechanistic comparison; condition-native renders
are separately labelled diagnostics. Results include nRMSE/SSIM/LPIPS overall and by source
domain, target domain, contrast, directed pair, and ordinary/catastrophic stratum; all valid graph
intermediates; anatomy/edge/gradient preservation; raw pre-mask decoder background leakage; and
montages containing raw identity, calibrated identity, original SB-v2, factored SB-only when
trained, full prediction, Stage-1 ceiling, and error. This makes no learned-disentanglement or
promotion claim. The P:0006 branch is labelled
`development_validation_P0006_evaluation_only`: it supports development/model assessment only
and cannot support population or generalization claims. It is separate from the frozen unpaired
R/validation plan and never enters training or checkpoint selection. P:0009 stays frozen and
unused for possible later confirmation.

Spectral normalization and lazy R1 remain disabled in the primary configuration. Critic score
distributions, domain accuracy, and saturation evidence must be reviewed before either is added as
a separately configured and tested stability ablation.
