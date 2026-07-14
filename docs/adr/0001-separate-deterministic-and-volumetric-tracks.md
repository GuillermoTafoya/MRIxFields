# ADR 0001: Separate Deterministic And Volumetric Research Tracks

- Status: Accepted
- Date: 2026-07-14
- Decision owners: FieldBridge research maintainers

## Context

FieldBridge currently has two useful but scientifically different implementation tracks.

Track A is a deterministic 2D pseudo-pair baseline. It degrades retrospective high-field
T2-FLAIR volumes into synthetic 0.1T slices and tests whether one shared conditional
U-Net can restore them while responding materially to the requested target field.

Track B is a volumetric Stage-1 stack. It uses 3D patches to train a KL-VAE, supports a
resumable patch bank and sliding-window reconstruction, and includes a conditional latent
diffuser.

Both tracks contribute to the long-term goal of a shared-parameter conditional model
that emits complete 3D MRI volumes. Their current data provenance, objectives, geometry,
and evidence levels are not interchangeable.

## Decision

Keep Track A and Track B separate for now.

Share stable contracts for data identity, raw tensor order, intensity ranges, subject and
volume splits, leakage audits, evaluation aggregation, run provenance, and checkpoint
versioning. Do not share a dataset or artifact merely because both tracks consume MRI
volumes.

Track A owns pseudo-pair degradation, paired target identity, 2D axial slice geometry,
dynamic training degradation, deterministic evaluation seeds, and conditioning
counterfactuals.

Track B owns 3D patch extraction, patch-bank resumability, KL-VAE reconstruction,
conditional latent diffusion, and sliding-window complete-volume reconstruction.

## Patch-Bank Boundary

Do not reuse the Stage-1 patch bank as the pseudo-pair loader unless a new versioned
provenance contract records, for every item:

- subject, source volume, and paired target volume identity;
- source and target domains;
- degradation implementation version, seed, and sampled parameters;
- raw-axis and crop/slice geometry needed for reconstruction;
- storage and model intensity ranges plus reversible transforms;
- split fingerprint established before patch or slice expansion;
- deterministic item identity and resume position;
- checkpoint/config compatibility information.

Without these fields, patch reuse would make pairing, leakage, and reproducibility
ambiguous.

## Shared Contracts

The tracks may share:

- manifest and `VolumeRecord` identity semantics;
- raw NIfTI order `(C, X, Y, Z)` without inferred anatomical labels;
- official `[0, 1]` storage and explicit model-boundary normalization;
- subject/volume-first split and leakage rules;
- run identity fields: commit, resolved config, split fingerprint, seed, checkpoint
  version;
- metric implementations whose data range and aggregation unit are explicit;
- final complete-volume NIfTI reconstruction and validation requirements.

They should retain separate training datasets, samplers, checkpoint namespaces, and
scientific claims until a bridge satisfies the criteria below.

## Criteria For A Future Bridge

A bridge between tracks requires all of the following:

1. A concrete experiment that needs 3D pseudo-pairs or conditional latent translation,
   with an allowed claim defined before implementation.
2. A versioned paired/degradation provenance schema covering every generated item.
3. Tests proving subject/volume splits precede expansion and reject identity/path/subject
   leakage.
4. Reversible range and geometry transforms from raw volume through model input and back
   to complete output volume.
5. Deterministic validation/test generation and resumable training identity.
6. Explicit checkpoint migration or rejection rules across both tracks.
7. Macro-per-volume, per-field, and conditioning-counterfactual evaluation.
8. Complete NIfTI reconstruction that preserves or restores geometry and orientation
   metadata.
9. Evidence that sharing removes real duplication or enables a scientific test without
   weakening either track's provenance contract.

Meeting only a tensor-shape or I/O compatibility condition is insufficient.

## Consequences

The repository carries two scoped pipelines for now. Some implementation duplication may
remain where provenance differs. This cost is accepted because premature unification
would blur scientific claims and make split, degradation, and resume identities harder
to audit.

Future work can still converge on the shared north star, but convergence must happen
through explicit versioned contracts and full-volume evidence rather than implicit reuse.
