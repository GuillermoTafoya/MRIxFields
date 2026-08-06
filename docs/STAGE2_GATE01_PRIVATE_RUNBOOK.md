# Stage 2 Gate 0.1 private execution runbook

This runbook is the only scientific-mode path for Gate 0.1. It deliberately keeps the
training-derived calibrator, protocol lock, selection, arrays, manifests, results, and
figures outside the Git checkout. The repository commands below do not train SB-v2 or
produce predictions. No private execution was performed while implementing this path.

## 1. Preconditions and external paths

Run from a reviewed, clean checkout of the exact Gate 0.1 commit. Set each placeholder
to an existing location on the private system; `GATE01_EXTERNAL` must not be inside the
repository.

```powershell
$Gate01Repo = (Get-Location).Path
$Gate01External = (Resolve-Path $env:GATE01_EXTERNAL).Path
$Gate01Split = Join-Path $Gate01External "frozen-resplit.json"
$Gate01Calibrator = Join-Path $Gate01External "gate01-target-calibrator.json"
$Gate01Selection = Join-Path $Gate01External "gate01-selection-descriptors.json"
$Gate01ProtocolSpec = Join-Path $Gate01External "gate01-protocol-spec.json"
$Gate01ProtocolLock = Join-Path $Gate01External "gate01-protocol-lock.json"
$Gate01BuildPlan = Join-Path $Gate01External "gate01-private-build-plan.json"
$Gate01Manifest = Join-Path $Gate01External "gate01-private-manifest.json"
$Gate01BuildState = Join-Path $Gate01External "gate01-private-build-state.json"
$Gate01Results = Join-Path $Gate01External "gate01-results.json"
$Gate01Report = Join-Path $Gate01External "gate01-report.md"
$Gate01Contract = Join-Path $Gate01External "gate01-result-contract.json"
$Gate01Montages = Join-Path $Gate01External "gate01-montages"

git status --short
git rev-parse HEAD
git diff --quiet
git diff --cached --quiet
```

Stop if either diff command is nonzero, if `git status --short` prints anything, or if
HEAD is not the reviewed Gate 0.1 commit. Do not copy any external file into the checkout.

## 2. Fit and seal the training-derived calibrator

The split must have fingerprint
`92187cf5f08ba00c446c08151f0658534efffa917569106a73062fdc70bcaf5f`.
The fit streams retrospective training volumes, retains only per-volume quantiles and
content identities, requires all 15 domains, freezes support threshold `0.0`, and rejects
a dirty checkout.

```powershell
fieldbridge fit-gate01-target-calibrator `
  --split-json $Gate01Split `
  --out $Gate01Calibrator `
  --training-cohort-identity $env:GATE01_TRAINING_COHORT_IDENTITY `
  --num-quantiles 256 `
  --mask-threshold 0.0 `
  --low-probability 0.01 `
  --high-probability 0.99 `
  --device cpu
```

Archive the command JSON output with the external run record. It reports both the
calibrator artifact SHA-256 and aggregate template SHA-256. Do not hand-edit the
calibrator.

## 3. Freeze selection independently of predictions

Before reading or building a prediction manifest, create `$Gate01Selection` in the
external protocol area as a JSON list of exactly 60 sanitized descriptors:

```json
[
  {
    "case_identity_sha256": "<sha256-of-external-case-id>",
    "traveller_identity_sha256": "<frozen-traveller-sha256>",
    "contrast": "T1w",
    "source_field_t": 0.1,
    "target_field_t": 1.5
  }
]
```

The complete file must contain each of the 20 nonidentity directed field pairs exactly
once for each of `T1w`, `T2w`, and `T2-FLAIR`, with one traveller digest throughout.
Compute its canonical fingerprint without consulting the prediction manifest:

```powershell
$Gate01SelectionFingerprint = (fieldbridge fingerprint-gate01-selection `
  --selection $Gate01Selection).Trim()
```

## 4. Create the independent protocol lock

Create `$Gate01ProtocolSpec` outside Git. It must contain exactly these fields; replace
only the four marked digests with the externally frozen values. The montage object must
be copied exactly.

```json
{
  "traveller_identity_sha256": "<frozen-traveller-sha256>",
  "selection_fingerprint_sha256": "<Gate01SelectionFingerprint>",
  "split_fingerprint": "92187cf5f08ba00c446c08151f0658534efffa917569106a73062fdc70bcaf5f",
  "support_threshold": 0.0,
  "calibrator_artifact_sha256": "<artifact_sha256-from-calibrator>",
  "calibrator_template_sha256": "<template_sha256-from-calibrator>",
  "artifact_provenance": {
    "stage1_run_c_checkpoint_sha256": "74132b9c514bb91b86d8eb43c63542780bce11304e31e67d3bf75c90ff5d4d79",
    "full_latent_bank_build_commit": "c4b9c399baef588d9547e89100de5036e1ccfcdb",
    "gate0_diagnostic_commit": "d3476b900866019b428d52d01a6d5b26b93ca65d",
    "sb_v2_checkpoint_sha256": "39c71b5dae702a68d9518376c2d25c13605abd985a5e74e8fb1b4c58d17a1108",
    "resplit_fingerprint": "92187cf5f08ba00c446c08151f0658534efffa917569106a73062fdc70bcaf5f"
  },
  "official_metrics": ["nrmse", "ssim", "lpips"],
  "montage_specification": {
    "version": "gate01-montage-v1",
    "selection_frozen_before_private_run": true,
    "selection_basis": "contrast and directed field pair only; never metric rank",
    "tensor_axis_convention": "volume[..., z]; no anatomical plane name is inferred without affine/orientation",
    "relative_slice_positions": [0.35, 0.5, 0.65],
    "display_order": ["target", "raw_identity", "calibrated_identity", "raw_sb_v2", "calibrated_sb_v2", "stage1_reconstruction_ceiling"],
    "directed_pairs_per_contrast": [
      {"source_field_t": 0.1, "target_field_t": 7.0},
      {"source_field_t": 7.0, "target_field_t": 0.1},
      {"source_field_t": 1.5, "target_field_t": 3.0},
      {"source_field_t": 3.0, "target_field_t": 1.5}
    ],
    "contrasts": ["T1w", "T2w", "T2-FLAIR"],
    "rendering": {
      "shared_display_range_within_pair": true,
      "interpolation": "none",
      "crop": "none unless a separately frozen source-derived crop is supplied"
    }
  }
}
```

Seal it atomically. This command reads no prediction manifest:

```powershell
fieldbridge lock-gate01-protocol `
  --spec $Gate01ProtocolSpec `
  --out $Gate01ProtocolLock
```

Any later change to selection, traveller, split, threshold, calibrator, checkpoint/build
identity, official metrics, or montage requires a new reviewed protocol lock.

## 5. Produce frozen arrays and the external build plan

Use the already reviewed Stage-1/SB-v2 private inference process to produce arrays
outside Git. Do not retrain. The external build plan is not the protocol lock and cannot
change its expected values. It has contract version
`stage2-gate01-private-build-plan-v1`, execution mode `scientific`, the locked selection,
split, artifact and support contracts, and `evidence_scope` containing
`"evidence_kind": "private"` and `"private_data_run": true`.

Each of its 60 cases must contain `case_id`, the traveller digest, source/target domains,
and these references:

```json
{
  "path": "relative/or/absolute/external-volume.npy",
  "expected_sha256": "<canonical-loaded-array-sha256>"
}
```

Required roles are `source_image`, `source_support_mask`, `target`, `raw_identity`,
`raw_sb_v2`, and `stage1_reconstruction_ceiling`. `wrong_target_sb_v2` must contain the
three other non-source, non-requested sibling fields, using the same files and canonical
hashes as those siblings' requested `raw_sb_v2` predictions. The 15 acquisition nodes
must reuse exact source/target files and hashes; their Stage-1 source identity/target
ceiling roles must likewise reuse exact files and hashes. Support masks are boolean
`.npy` arrays equal to `abs(source_image) > 0.0`.

Canonical hashes are over the loaded shape, canonical dtype, and C-order array bytes—not
container bytes. Use `fieldbridge.evaluation.stage2_gate01.canonical_loaded_array_sha256`
in the external producer. Never derive an expected hash by reading the manifest under
validation.

## 6. Verify and resumably build the manifest

First attempt:

```powershell
fieldbridge build-gate01-private-manifest `
  --plan $Gate01BuildPlan `
  --protocol-lock $Gate01ProtocolLock `
  --calibrator $Gate01Calibrator `
  --out $Gate01Manifest `
  --state $Gate01BuildState
```

After an interruption, rerun the same command with `--resume`:

```powershell
fieldbridge build-gate01-private-manifest `
  --plan $Gate01BuildPlan `
  --protocol-lock $Gate01ProtocolLock `
  --calibrator $Gate01Calibrator `
  --out $Gate01Manifest `
  --state $Gate01BuildState `
  --resume
```

The builder verifies every canonical array identity, the complete 15-node/60-direction
graph, all 180 sibling wrong-target references, protocol lock, calibrator identities,
and support threshold. State and final manifest writes are atomic. Resume is accepted
only for the same plan, lock, and calibrator. The builder performs no inference.

## 7. Execute the official scientific diagnostic

Recheck the checkout immediately before execution:

```powershell
git status --short
git rev-parse HEAD
```

Stop unless it is still clean and at the reviewed commit. Then run all official metrics
together and render the frozen montage:

```powershell
fieldbridge gate01-equal-photometry `
  --manifest $Gate01Manifest `
  --calibrator $Gate01Calibrator `
  --protocol-lock $Gate01ProtocolLock `
  --metrics nrmse ssim lpips `
  --device cuda `
  --out $Gate01Results `
  --markdown-out $Gate01Report `
  --contract-out $Gate01Contract `
  --montage-dir $Gate01Montages
```

Archive the manifest, calibrator, protocol lock, build state, JSON result, Markdown report,
contract JSON, montage PNGs/manifest, command output, and environment record in the
external protocol area. Confirm `scientific_status.eligible_for_scientific_conclusions`
is `true`; otherwise the run is development-only. Promotion remains unset until the
private result receives scientific review.
