# Research Status

Snapshot date: 2026-07-14

Repository baseline: `origin/main` at
`b7986e8a498a9a57ce36ebdbd04b9421f39b57a0`.

The shared north star is one shared-parameter conditional model that emits complete 3D
MRI volumes. The current tracks test different risks and are intentionally separate.

## Track A: Deterministic Pseudo-Pair Baseline

Status: corrected implementation available; fresh v2 run pending.

Track A is the 2D axial `ConditionalUNetFieldTranslator` experiment for T2-FLAIR. It
uses retrospective high-field targets, synthetic 0.1T degradation, volume/subject-first
splits, `volume[:, :, :, z]` slice extraction, target-aware background loss, and
checkpoint version 2.

This track is the fastest falsifiable field-conditioning experiment. Its allowed claim
is synthetic-degradation restoration only. It does not establish translation from real
0.1T acquisitions or complete-volume performance.

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

- Track A has no fresh real-manifest v2 micro-run after the axis and loss corrections.
- Conditioning effect thresholds have not yet been predeclared for a promotion run.
- Track A does not yet reconstruct and evaluate complete held-out NIfTI volumes.
- Track B still needs controlled real-data reconstruction evidence and runtime/resource
  profiling before promotion.
- A shared 3D paired/degradation provenance contract does not yet exist between tracks.
- Eventual anatomical metrics require a validated toolchain and documented protocol.

## Next Decision Gate

Run a fresh Track A v2 micro-pilot from a recorded commit and split fingerprint. Evaluate
held-out subjects against the degraded-input baseline and all target-conditioning
counterfactuals using predeclared thresholds.

If the v2 run shows a material, field-consistent conditioning effect and restoration
improvement, define a full-volume Track A reconstruction/evaluation step. If it does not,
retain Track A as a synthetic-restoration control and do not force a bridge into Track B.

Track B proceeds independently through its volumetric reconstruction gate.

## Artifact Location Policy

Real manifests, MRI data, checkpoints, split/run outputs, credentials, and
machine-specific absolute paths remain outside Git. This status document records
identities and decisions, not private storage locations.
