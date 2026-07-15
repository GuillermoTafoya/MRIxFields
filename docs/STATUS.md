# Research Status

Snapshot date: 2026-07-14

Repository baseline: `origin/main` at
`8631962f96ea07d1dfe51bbfa486ddac266cb828`.

The shared north star is one shared-parameter conditional model that emits complete 3D
MRI volumes. The current tracks test different risks and are intentionally separate.

## Track A: Deterministic Pseudo-Pair Baseline

Status: corrected implementation available; micro-v2 produced valid negative
development evidence; duration-only probe pending.

Track A is the 2D axial `ConditionalUNetFieldTranslator` experiment for T2-FLAIR. It
uses retrospective high-field targets, synthetic 0.1T degradation, volume/subject-first
splits, `volume[:, :, :, z]` slice extraction, target-aware background loss, and
checkpoint version 2.

This track is the fastest falsifiable field-conditioning experiment. Its allowed claim
is synthetic-degradation restoration only. It does not establish translation from real
0.1T acquisitions or complete-volume performance.

### Micro-v2 Development Result

The following is user-supplied evidence from a private Colab/Drive run. Repository
maintainers and coding agents do not have access to those private artifacts and have not
independently verified the split file, checkpoint, logs, or metrics.

- Code commit: `8631962f96ea07d1dfe51bbfa486ddac266cb828`.
- Split fingerprint:
  `17f00411ab04331fa0380526b2d8f0cd0173e4ff6f8978f72c61053fa7385dbe`.
- Checkpoint metadata: pseudo-pair pipeline v2, epoch 2, global step 32.
- Runtime report: CUDA and AMP active; engineering gate passed.
- Scientific gate: failed.
- Evidence scope: sampled-slice/per-volume exploratory; `complete_volume: false`.

| Metric | Degraded input | Prediction |
| --- | ---: | ---: |
| Macro nRMSE | 0.054048 | 0.104325 |
| Macro SSIM | 0.874324 | 0.167202 |

No target field improved nRMSE (`0/4`). Correct conditioning had the best mean
selected-slice nRMSE for `1/4` volumes. The mean margin versus the best wrong target was
`-0.00113659`; relative correct-versus-wrong and correct-versus-permuted nRMSE effects
were `0.005750` and `0.002822`, respectively.

This is valid negative evidence for the predeclared micro-v2 viability gates: at the
declared endpoint, prediction was worse than the degraded-input baseline and target
conditioning was not materially discriminative. It is not evidence about complete
volumes, real 0.1T acquisitions, a confirmatory held-out split, or the eventual 3D
translation contract.

The observed split is now development evidence because its test summaries have been
examined. Reusing it for the predeclared duration probe controls the split while testing
one variable, but no result from that reuse is confirmatory evidence.

## Track B: Volumetric Stage-1 Stack

Status: infrastructure implemented; scientific and challenge evidence pending.

Track B contains the 3D patch-based KL-VAE, resumable patch bank, sliding-window
full-volume reconstruction, and conditional latent diffuser. It addresses volumetric
representation and reconstruction constraints, not the paired synthetic-restoration
question owned by Track A.

The Stage-1 patch bank is not a pseudo-pair dataset. Its current provenance contract does
not establish paired targets, degradation parameters, or pseudo-pair slice geometry.

## Invalid Evidence

Commit `be8b4792344ac4f8b73112cd9ff6db4298af6b13` predates the pseudo-pair axis and
target-aware background-loss corrections. Checkpoints and reported results produced by
that implementation are pseudo-pair v1 artifacts and are invalid as scientific
evidence. They must not be resumed or compared as if they were v2 results.

## Current Blockers

- Track A micro-v2 failed its predeclared sampled-slice/per-volume scientific gate.
- It is not yet known whether the negative result is a short-duration optimization
  failure or a limitation of the deterministic conditional baseline.
- The development split has been observed and cannot support a confirmatory claim.
- Track A does not yet reconstruct and evaluate complete held-out NIfTI volumes.
- Track B still needs controlled real-data reconstruction evidence and runtime/resource
  profiling before promotion.
- A shared 3D paired/degradation provenance contract does not yet exist between tracks.
- Eventual anatomical metrics require a validated toolchain and documented protocol.

## Next Decision Gate

Run one fresh-initialization, 10-epoch duration probe with the same model, seed,
degradation, preprocessing, target fields, losses, and development split after the
launcher matches its supplied fingerprint. Evaluate the test split once at epoch 10,
record GPU telemetry and throughput, and apply the already frozen micro-v2 viability
thresholds without post-hoc changes.

The scaled pilot remains blocked. A failed duration probe is further valid negative
development evidence. A passing probe may justify designing a new confirmatory run, but
cannot itself satisfy promotion or final-volume gates because it reuses the observed
split and still evaluates selected slices only.

Track B proceeds independently through its volumetric reconstruction gate.

## Artifact Location Policy

Real manifests, MRI data, checkpoints, split/run outputs, credentials, and
machine-specific absolute paths remain outside Git. This status document records
identities and decisions, not private storage locations.
