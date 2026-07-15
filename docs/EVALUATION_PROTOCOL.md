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

## Baselines And Metrics

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
