# Evaluation Protocol

This document defines the evidence required to move FieldBridge experiments from
engineering checks to scientific and challenge-level claims. It applies to both the
deterministic pseudo-pair baseline and the volumetric Stage-1 stack.

## Evidence Gates

### 1. Engineering Gate

Purpose: prove that code paths and contracts execute as intended.

Required evidence:

- Synthetic CPU tests, focused regression tests, and `fieldbridge smoke-train` pass.
- Manifest validation, split construction, leakage audit, preprocessing, checkpoint
  loading, and resume paths execute without errors.
- Seeds, resolved config, Git commit, split fingerprint, and checkpoint version are
  recorded.
- Optional metrics report an explicit skip reason when dependencies are unavailable.

Passing this gate supports an engineering-readiness statement only. It does not support
a claim about MRI restoration or field translation quality.

### 2. Exploratory Scientific Gate

Purpose: determine whether a method is worth a controlled promotion run.

Required evidence:

- Evaluation uses external data with held-out subjects.
- Subject and volume assignment is completed before slice or patch expansion.
- The degraded input is evaluated against the same target as the prediction.
- Results include per-volume, per-target-field, and macro-per-field summaries.
- Target-conditioning counterfactuals are reported with effect sizes.
- Failures, missing fields, skipped metrics, and excluded volumes are disclosed.

Exploratory results are directional. Thresholds applied after seeing the results cannot
be presented as predeclared promotion criteria.

### 3. Promotion Gate

Purpose: decide whether an experiment can become a supported baseline.

Required evidence:

- The run uses the current checkpoint version and a frozen split fingerprint.
- Promotion thresholds and failure conditions are written before the run.
- Results are stable across the prescribed seeds or repeats.
- Improvements over the degraded-input baseline are reported per field and as a macro
  average, not only as a pooled slice average.
- Correct conditioning is compared with every supported wrong target and with a
  deterministic permutation. Absolute and relative effects, best-target fractions, and
  margins are reported.
- No unsupported claim is inferred from a tiny positive floating-point delta.

A failed promotion gate remains useful negative evidence and must not be hidden by
changing degradation strength or evaluation scope after the fact.

### 4. Final-Volume Gate

Purpose: support complete-volume or challenge-level claims.

Required evidence:

- Complete held-out NIfTI volumes are reconstructed and evaluated.
- Output geometry, shape, affine/orientation metadata, and volume identity are preserved
  or explicitly restored.
- Metrics are computed per complete volume and then aggregated by field.
- Challenge-required metrics and validation are run on the reconstructed outputs.
- Anatomical consistency metrics are included when their validated tooling and protocol
  are available.

Slice-only metrics, selected panels, and patch-only reconstructions cannot pass this
gate.

## Split And Aggregation Contract

The evaluation unit is the held-out subject. All volumes belonging to one subject stay
in one split. Volume IDs and paths must also be disjoint. Splitting happens before any
slice or patch expansion, and the persisted split fingerprint is part of run identity.

For each complete held-out volume, compute each metric once from the reconstructed
volume and its target. Then:

1. Summarize the distribution of per-volume values within each target field.
2. Compute one mean per target field.
3. Compute the macro result as the unweighted mean of the available field means.
4. Report subject and volume counts, exclusions, and missing fields beside the metric.

Slice-level summaries may be retained for diagnostics, but they do not replace
macro-per-volume reporting and must not give larger volumes more weight merely because
they contribute more slices.

A selected-slice experiment may also average its selected-slice metrics once within each
volume, then average volumes within each field, then macro-average fields. This is an
exploratory sampled-slice/per-volume diagnostic and must be labeled
`complete_volume: false`. It is not full-volume nRMSE or SSIM and does not satisfy the
final-volume gate.

### Prospective Paired Selected-Slice Development Audit

The Track-A prospective paired zero-shot audit is a descriptive development diagnostic,
not a promotion or confirmation run. It uses real paired T2-FLAIR acquisitions for the
predeclared cases and indices in
`configs/experiment/prospective_paired_zero_shot_v1.yaml`, reports
`evidence_scope: prospective_paired_selected_slice_development`, and always reports
`complete_volume: false`.

Each source/target pair must have exactly equal shape, affine, orientation codes, voxel
sizes, and resulting `SliceGeometry`. Acquisitions stay in the released `[0, 1]` scale;
they are never normalized independently. `preprocess_volume_slice` applies the exact
historical `fit_pad` contract, so native aspect ratio is preserved rather than directly
distorted to the model canvas.

For every actual target, the audit compares the real 0.1T source, the correctly
conditioned frozen prediction, and all wrong requested-target predictions against that
same actual target. Foreground is the nonzero actual-target support inside the unpadded
frame; outside-mask error is absolute prediction-to-target error on the remainder of
that frame. Signed foreground bias is `mean(candidate - target)` on foreground, and
residual magnitude is `mean(abs(prediction - source))` on foreground. Conditioning
margins are oriented so positive means the correct condition is better; signed bias is
compared by absolute magnitude. Error-improvement maps are exactly
`abs(source-target) - abs(prediction-target)`, so positive values mean improvement.

Slices are weighted equally within case, cases equally within target field, and target
fields equally in the macro summary. No scientific pass/fail threshold may be created
after observing these cases.

Checkpoint provenance validation compares the historical training YAML only after
normalization through `PseudoPairEpochConfig.from_mapping().to_dict()`, the same contract
used when the trainer saved `pseudo_pair_config` and `_meta.config`. Those two checkpoint
copies must agree exactly. `num_workers` is a DataLoader runtime field and is not part of
the legacy `PseudoPairEpochConfig` checkpoint schema: the historical YAML and launcher
predeclare `num_workers=0`, but the checkpoint alone cannot independently attest it.
Unknown historical training keys are errors rather than silently ignored fields; only
explicit runtime/path overrides receive separate handling.

### Prospective Paired LOSO Residual Feasibility v1

The next Track-A experiment uses three subject-first folds: train 0007+0009/evaluate
0006, train 0006+0009/evaluate 0007, and train 0006+0007/evaluate 0009. It uses only real
paired T2-FLAIR examples with real 0.1T source and actual 1.5T, 3T, 5T, and 7T targets.
No synthetic example enters paired training. Private manifests, NIfTIs, checkpoints,
run directories, and reconstructed images remain external.

Four frozen endpoint arms are compared: unchanged real 0.1T; per-target-field affine
scale+bias fitted only on that fold's two training cases; an identity-initialized
`ConditionalResidualUNetFieldTranslator`; and the identical model initialized from the
frozen synthetic residual checkpoint. Neural arms share optimizer, loss, sampling,
preprocessing, epochs, steps, and endpoint. There is no validation loader, early
stopping, or held-out checkpoint selection; each fold is evaluated exactly once after
its fixed endpoint.

Training expands subjects only after the fold is fixed and uses every slice in the
predeclared brain-support interval `[72, 292)`. Selected-slice evaluation retains
`[72, 103, 135, 166, 197, 228, 260, 291]`. Complete-volume evaluation additionally
processes every z index through the same fit-pad/model-range contract, records model-grid
and inverse-restored native-grid metrics separately, and reports `complete_volume: true`
only if slice coverage, paired geometry, output shape, and every inverse geometry check
pass.

The following relative viability rules are frozen before execution for each neural arm:

- macro nRMSE must beat the real-0.1T source and macro SSIM must not decrease;
- nRMSE improvement must occur in at least three of four fields and two of three held-out
  cases;
- correct conditioning must have the lowest nRMSE in at least nine of twelve held-out
  case-field units and a positive mean margin in every field;
- no field may regress from source by more than `0.005` absolute nRMSE;
- the neural arm must beat the train-fold-only affine baseline in macro nRMSE without
  reducing macro SSIM;
- synthetic initialization is retained only if it beats identity initialization in
  macro nRMSE without reducing SSIM or mean conditioning margin.

Slices are averaged within held-out case-field, case/fold units are averaged within
field, and fields are macro-averaged equally. These are feasibility rules on observed
development subjects, not confirmatory claims.

## Baselines And Metrics

### Stage-1 Domain-Balanced Full-Volume Audit v1

The Track-B Stage-1 audit freezes four test-split volumes in each of all 15 canonical
field/contrast domains before checkpoint evaluation. Selection is stable under record
ordering, tied to the subject-held-out split fingerprint and an all-record fingerprint,
and fails closed on leakage or incomplete domain coverage. Complete native tensors are
reconstructed with the existing fixed posterior-mean sliding-window path under inference
mode. Metrics use the raw decoder output only for range diagnostics and the `[0,1]`-clamped
output for reconstruction scores.

The primary result averages complete volumes equally within domain and all 15 domain
means equally. Pooled micro metrics are secondary. Exact metric formulas, recovery rules,
artifacts, and the two-step server workflow are frozen in
`docs/STAGE1_FULL_VOLUME_AUDIT.md`. The audit reports evidence and checkpoint comparisons;
it has no automatic scientific viability gate.

Every pseudo-pair report includes two comparisons against the unmodified high-field
target:

- degraded synthetic source `x_low` versus target `x_high`;
- model prediction `x_pred` versus target `x_high`.

At minimum report nRMSE, SSIM, and LPIPS when LPIPS is available. nRMSE and SSIM use the
released `[0, 1]` data range after reversible model-boundary normalization. LPIPS uses
its expected image range and grayscale channel handling. If LPIPS or its local weights
are unavailable, record that it was skipped; do not download weights during core tests.

The deterministic pseudo-pair report also retains masked MAE, gradient MAE,
outside-support mean absolute intensity, and correlation as diagnostic measures.
Masks weight losses and metrics; they must not erase valid dark anatomy.

## Conditioning Counterfactuals

Evaluate the prediction with:

- the correct target field;
- every other supported target field;
- a deterministic permutation of target labels.

Report absolute and relative correct-versus-counterfactual changes, the fraction of
samples or volumes for which the correct target has the best nRMSE, the mean and median
margin versus the best wrong target, and aggregation by true target field. The report
contains effect sizes, not an automatic claim of meaningful conditioning.

## Eventual Anatomical Evaluation

Once validated anatomical tooling is available, the final-volume gate should add
predeclared measures such as segmentation overlap, structure-volume error, boundary
distance, topology failures, and clinically relevant regional consistency. These
metrics require complete volumes and documented segmentation provenance. They are not
retroactively inferred from image-quality metrics.

## Allowed Current Claim

The current deterministic pseudo-pair experiment may claim only restoration of a known
synthetic degradation applied to retrospective high-field T2-FLAIR data.

It does not demonstrate learned translation from the real 0.1T distribution, real
any-to-any field translation, or complete-volume challenge performance. Challenge-level
evidence requires reconstructed complete NIfTI volumes evaluated under the final-volume
gate.
