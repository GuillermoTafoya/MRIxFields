# Stage 2 Gate 0.1 private execution runbook

This is the required scientific-mode order for Gate 0.1. The calibrator, selection,
protocol lock, producer specification/state, arrays, build plan, manifest, results, and
figures remain outside Git. These commands must be run only after the reviewed change is
merged and its final commit is known. No private execution was performed for this PR.

## 1. Establish external paths and the final reviewed checkout

Run from the repository root. `GATE01_EXTERNAL` and every referenced input/output must
resolve outside the repository.

```powershell
$Gate01Repo = (Get-Location).Path
$Gate01External = (Resolve-Path $env:GATE01_EXTERNAL).Path
$Gate01EvaluationSplit = Join-Path $Gate01External "frozen-scientific-resplit.json"
$Gate01BankSourceSplit = Join-Path $Gate01External "original-split-v3.json"
$Gate01Bank = Join-Path $Gate01External "full-latent-bank"
$Gate01Stage1Config = Join-Path $Gate01External "stage1-run-c.yaml"
$Gate01Stage1Checkpoint = Join-Path $Gate01External "stage1-run-c.pt"
$Gate01SbConfig = Join-Path $Gate01External "sb-v2.yaml"
$Gate01SbCheckpoint = Join-Path $Gate01External "sb-v2.pt"
$Gate01Calibrator = Join-Path $Gate01External "gate01-target-calibrator.json"
$Gate01Selection = Join-Path $Gate01External "gate01-prospective-selection.json"
$Gate01ProtocolSpec = Join-Path $Gate01External "gate01-protocol-spec.json"
$Gate01ProtocolLock = Join-Path $Gate01External "gate01-protocol-lock.json"
$Gate01ProducerSpec = Join-Path $Gate01External "gate01-producer-spec.json"
$Gate01ProducerOutput = Join-Path $Gate01External "producer-output"
$Gate01ProducerState = Join-Path $Gate01External "producer-state"
$Gate01ProducerStateFile = Join-Path $Gate01ProducerState "producer-state.json"
$Gate01BuildPlan = Join-Path $Gate01ProducerOutput "private-build-plan.json"
$Gate01Manifest = Join-Path $Gate01External "gate01-private-manifest.json"
$Gate01BuildState = Join-Path $Gate01External "gate01-private-build-state.json"
$Gate01Results = Join-Path $Gate01External "gate01-results.json"
$Gate01Report = Join-Path $Gate01External "gate01-report.md"
$Gate01Contract = Join-Path $Gate01External "gate01-result-contract.json"
$Gate01Montages = Join-Path $Gate01External "gate01-montages"
$Gate01Archive = Join-Path $Gate01External "archive"

$Gate01ReviewedCommit = (git rev-parse HEAD).Trim()
git status --short
git diff --quiet
git diff --cached --quiet
```

Stop unless status is empty and both diff commands return zero. `$Gate01ReviewedCommit`
must be the final reviewed/merged Gate 0.1 commit, not a transient PR head. Never copy an
external artifact into the checkout.

## 2. Fit the training-derived calibrator

The scientific evaluation-resplit fingerprint is frozen at
`92187cf5f08ba00c446c08151f0658534efffa917569106a73062fdc70bcaf5f`.
The fit streams retrospective training volumes and retains per-volume quantiles/content
identities only. It requires all 15 domains, a clean checkout, and support threshold 0.0.

```powershell
fieldbridge fit-gate01-target-calibrator `
  --split-json $Gate01EvaluationSplit `
  --out $Gate01Calibrator `
  --training-cohort-identity $env:GATE01_TRAINING_COHORT_IDENTITY `
  --num-quantiles 256 `
  --mask-threshold 0.0 `
  --low-probability 0.01 `
  --high-probability 0.99 `
  --device cpu
```

Record the printed calibrator artifact and template SHA-256 values. Do not hand-edit it.

## 3. Resolve and freeze the 15 acquisitions

This command reads the external scientific evaluation resplit, resolves one prospective
validation traveller, checks
one acquisition in every one of the 15 domains, and writes only case/traveller hashes.
It automatically freezes the 60-direction selection fingerprint; the operator does not
author 60 cases.

```powershell
fieldbridge prepare-gate01-private-selection `
  --split-json $Gate01EvaluationSplit `
  --traveller-subject-id $env:GATE01_TRAVELLER_SUBJECT_ID `
  --out $Gate01Selection
```

Verify that the selection contains 15 acquisition identities and no raw case or traveller
identifier. Scientific selection rejects prospective train and test travellers. Preserve
the protocol roles: development evidence remains development-only, the locked traveller is
validation evidence, and the untouched test traveller remains unselected.

## 4. Independently freeze evaluation code and protocol

The external protocol specification is created only after the final reviewed commit is
known. Its `evaluation_git_commit` is `$Gate01ReviewedCommit`. Its
`evaluation_module_sha256` object must contain exactly this reviewed set, with values
computed once from that clean commit and then frozen outside Git:

```powershell
$Gate01ScientificModules = @(
  "src/fieldbridge/cli.py",
  "src/fieldbridge/config/__init__.py",
  "src/fieldbridge/data/contracts.py",
  "src/fieldbridge/data/domains.py",
  "src/fieldbridge/data/latent_bank.py",
  "src/fieldbridge/data/latent_bank_dataset.py",
  "src/fieldbridge/data/manifests.py",
  "src/fieldbridge/data/sources.py",
  "src/fieldbridge/data/transforms.py",
  "src/fieldbridge/data/vae_splits.py",
  "src/fieldbridge/evaluation/metrics.py",
  "src/fieldbridge/evaluation/mrixfields2026_official.py",
  "src/fieldbridge/evaluation/stage2_gate01.py",
  "src/fieldbridge/evaluation/stage2_gate01_builder.py",
  "src/fieldbridge/evaluation/stage2_gate01_calibration.py",
  "src/fieldbridge/evaluation/stage2_gate01_montage.py",
  "src/fieldbridge/evaluation/stage2_gate01_producer.py",
  "src/fieldbridge/evaluation/stage2_gate01_protocol.py",
  "src/fieldbridge/evaluation/stage2_transport_eval.py",
  "src/fieldbridge/official/mrixfields2026.py",
  "src/fieldbridge/models/autoencoders/base.py",
  "src/fieldbridge/models/autoencoders/kl_vae.py",
  "src/fieldbridge/models/conditioning.py",
  "src/fieldbridge/models/diffusion/field_conditioner.py",
  "src/fieldbridge/models/diffusion/timestep_embedding.py",
  "src/fieldbridge/models/factory.py",
  "src/fieldbridge/models/film.py",
  "src/fieldbridge/models/translators/base.py",
  "src/fieldbridge/models/translators/conditional_unet.py",
  "src/fieldbridge/models/translators/flow_transport.py",
  "src/fieldbridge/training/checkpoints.py"
)
$Gate01CurrentModuleSha256 = [ordered]@{}
$Gate01ScientificModules | ForEach-Object {
  $Gate01CurrentModuleSha256[$_] = (
    Get-FileHash -Algorithm SHA256 $_
  ).Hash.ToLower()
}
```

The separately reviewed module comparison evidence is an object, never a list of
`{module, sha256}` rows. Its exact shape is:

```json
{
  "changed_modules": ["src/fieldbridge/models/translators/flow_transport.py"],
  "evaluation_git_commit": "<current-lowercase-40-hex-commit>",
  "evaluation_module_sha256": {
    "<each-of-the-exact-31-scientific-module-paths>": "<lowercase-sha256>"
  },
  "previous_evaluation_git_commit": "<distinct-previous-lowercase-40-hex-commit>",
  "previous_evaluation_module_sha256": {
    "<the-same-exact-31-scientific-module-paths>": "<lowercase-sha256>"
  }
}
```

Both maps must contain exactly the same 31 keys listed above. `changed_modules` must be
the unique, canonical module paths whose current and previous digests differ; for the
reviewed `8012a3f` comparison it is exactly
`src/fieldbridge/models/translators/flow_transport.py`. The current commit and map are
authoritative and must exactly match the protocol lock. The distinct previous commit and
map are provenance only and can never authorize evaluation. An old module list cannot
substitute for this five-field object. Write the object only from both independently
reviewed maps and their commits to
`gate01-reviewed-module-sha256-8012a3f.json`; do not derive the previous map from the
current checkout or from runtime evaluator output.

Transcribe the current reviewed values into the exact module-keyed map in
`$Gate01ProtocolSpec`; do not source expected values from a prediction manifest or runtime
evaluator output. The specification contains exactly:

- the hash-only traveller and 60-direction fingerprint from `$Gate01Selection`;
- `evaluation_split_file_sha256` plus `split_fingerprint` for the scientific
  evaluation resplit;
- `bank_source_split_file_sha256` for the independently supplied original
  bank-source split, frozen at
  `f6a19d7a31c4c3bb73edd92088ea078192e88ee4b276309bad81c548ab7f94d5`
  and `bank_source_split_fingerprint` frozen at
  `9d50db941d57af3333b10fed7f25262bae7d1e3346dc37b7c38327de50d9e534`;
- the support threshold `0.0`;
- calibrator artifact/template SHA-256;
- `evaluation_git_commit` and the exact 31-entry `evaluation_module_sha256` map above;
- the frozen Stage-1, latent-bank build, Gate-0, SB-v2, and resplit identities;
- `official_metrics` equal to `['nrmse', 'ssim', 'lpips']` in that order;
- the unchanged object returned by
  `fieldbridge.evaluation.stage2_gate01.fixed_montage_specifications()`.

Seal the independently reviewed specification:

```powershell
fieldbridge lock-gate01-protocol `
  --spec $Gate01ProtocolSpec `
  --out $Gate01ProtocolLock
```

Scientific runtime fails unless the checkout is clean, HEAD equals the externally locked
commit, and its exact runtime module map equals the externally locked map. The repository
does not hardcode a PR head or predict the future merge commit.

## 5. Seal the deterministic producer inputs

The command below verifies and pins the selection, both split files, every latent-bank record,
latent statistics/manifest/build commit, both configs/checkpoints, solver, step count,
full-volume decode path, deterministic seed, and protocol-lock identity. It also loads each
of the 15 selected source acquisitions once at a time and seals its canonical loaded-array
SHA-256 and exact `(1, 1, X, Y, Z)` shape in the external producer specification. The
frozen configuration file hashes are:

- Stage-1 Run C: `55921ffc53bac074883b66d368051589dd3cc3f2ce5c8e2cc1d304be4245888f`
- SB-v2: `d66a197533a9fa574146cfa21ac7c53c8471071c2b337325cf871c65588ff1aa`

```powershell
fieldbridge lock-gate01-producer-spec `
  --selection $Gate01Selection `
  --split-json $Gate01EvaluationSplit `
  --bank-source-split-json $Gate01BankSourceSplit `
  --bank-dir $Gate01Bank `
  --stage1-config $Gate01Stage1Config `
  --stage1-checkpoint $Gate01Stage1Checkpoint `
  --sb-v2-config $Gate01SbConfig `
  --sb-v2-checkpoint $Gate01SbCheckpoint `
  --protocol-lock $Gate01ProtocolLock `
  --solver heun `
  --n-steps 20 `
  --decode-block-size 128 128 128 `
  --decode-halo 16 16 16 `
  --decode-precision bfloat16 `
  --deterministic-seed 0 `
  --out $Gate01ProducerSpec
```

Use the reviewed frozen SB-v2 sampler/decode values. A changed config, checkpoint, bank,
selection, either split, lock, solver, step count, or decode value requires a new reviewed spec.
The bank-source split is the split that labeled payload storage when the frozen full bank
was built; the evaluation resplit governs scientific eligibility. Their labels need not
match. The complete bank manifest and every payload must agree with the bank-source split,
while the selected prospective traveller must be validation in the evaluation resplit.
The emitted decode specification must contain `"strategy": "full"`. The block/halo
arguments remain required only by the shared decode configuration schema; Gate 0.1 never
invokes the tiled decoder with them.

## 6. Produce the external 15/60/180 artifact graph

First attempt:

```powershell
fieldbridge produce-gate01-private-artifacts `
  --spec $Gate01ProducerSpec `
  --selection $Gate01Selection `
  --split-json $Gate01EvaluationSplit `
  --bank-source-split-json $Gate01BankSourceSplit `
  --bank-dir $Gate01Bank `
  --stage1-config $Gate01Stage1Config `
  --stage1-checkpoint $Gate01Stage1Checkpoint `
  --sb-v2-config $Gate01SbConfig `
  --sb-v2-checkpoint $Gate01SbCheckpoint `
  --protocol-lock $Gate01ProtocolLock `
  --output-dir $Gate01ProducerOutput `
  --state-dir $Gate01ProducerState `
  --device cuda
```

After interruption, run the identical command with `--resume`. Resume re-hashes every
completed array, reloads and revalidates all 15 sealed source hashes/shapes, and rejects
stale state, source or output mutation, missing or unexpected paths. It never repeats
verified inference:

```powershell
fieldbridge produce-gate01-private-artifacts `
  --spec $Gate01ProducerSpec `
  --selection $Gate01Selection `
  --split-json $Gate01EvaluationSplit `
  --bank-source-split-json $Gate01BankSourceSplit `
  --bank-dir $Gate01Bank `
  --stage1-config $Gate01Stage1Config `
  --stage1-checkpoint $Gate01Stage1Checkpoint `
  --sb-v2-config $Gate01SbConfig `
  --sb-v2-checkpoint $Gate01SbCheckpoint `
  --protocol-lock $Gate01ProtocolLock `
  --output-dir $Gate01ProducerOutput `
  --state-dir $Gate01ProducerState `
  --device cuda `
  --resume
```

Verify the command reports 15 acquisitions, 15 Stage-1 inferences, 60 SB-v2 inferences,
60 directions, and 180 wrong-target references. The wrong-target entries are references
to the 60 sibling predictions; no extra inference occurs. The producer writes 15 masks as
`abs(source) > 0.0`, canonical loaded-array hashes, and `$Gate01BuildPlan` atomically.
It fails before inference unless the bank manifest and all selected latent payloads prove
full encoding and match the frozen bank-source case/domain/storage-split, factor 4,
Stage-1 checkpoint,
and bank-build commit. Decoding is one strict full-volume forward per inference: an OOM is
a hard failure, tiled fallback is prohibited, and the completed producer state/result must
record `decode_strategy="full"` and `path_used=["full"]`.

## 7. Independently verify and build the manifest

```powershell
fieldbridge build-gate01-private-manifest `
  --plan $Gate01BuildPlan `
  --producer-spec $Gate01ProducerSpec `
  --producer-state $Gate01ProducerStateFile `
  --protocol-lock $Gate01ProtocolLock `
  --calibrator $Gate01Calibrator `
  --out $Gate01Manifest `
  --state $Gate01BuildState
```

After interruption, repeat it exactly with `--resume`:

```powershell
fieldbridge build-gate01-private-manifest `
  --plan $Gate01BuildPlan `
  --producer-spec $Gate01ProducerSpec `
  --producer-state $Gate01ProducerStateFile `
  --protocol-lock $Gate01ProtocolLock `
  --calibrator $Gate01Calibrator `
  --out $Gate01Manifest `
  --state $Gate01BuildState `
  --resume
```

This second-stage builder performs no inference. It rejects every legacy build plan and
independently verifies the sealed producer-spec artifact hash, the complete v3 producer
state, the state's exact build-plan hash, the lock and all configuration/checkpoint/full-bank/
source-array/selected-payload identities, including both sealed split roles. It requires
`decode_strategy="full"` and
`path_used=["full"]`, reloads and hashes every array, verifies the exact
15-node/60-direction/180-sibling graph, calibrator and threshold, then atomically writes
the scientific manifest. The manifest embeds a verified receipt containing the spec-file,
state-file and build-plan hashes plus the complete full-decode proof; the evaluator carries
that receipt unchanged into its result contract and Markdown report.

## 8. Execute the official diagnostic

Immediately recheck the checkout and locked commit:

```powershell
git status --short
if ((git rev-parse HEAD).Trim() -ne $Gate01ReviewedCommit) { throw "Gate 0.1 HEAD changed" }
```

Then run all three official metrics and the frozen renderer together:

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

The wrong-target report contains two distinct endpoints: common requested-domain
calibration (the equal-photometry mechanistic comparison) and condition-native
calibration (an endpoint diagnostic that changes both condition and template and therefore
does not isolate target control).

## 9. Verify and archive outside Git

Require `scientific_status.eligible_for_scientific_conclusions=true` and
`promotion_decision=unset_pending_scientific_review`. Then create a detached hash inventory
and archive only in the external area:

```powershell
$Gate01ResultContract = Get-Content $Gate01Contract -Raw | ConvertFrom-Json
$Gate01Receipt = $Gate01ResultContract.producer_receipt
$Gate01Splits = $Gate01ResultContract.split_provenance
if ($Gate01Splits.evaluation.file_sha256 -ne (Get-FileHash -Algorithm SHA256 $Gate01EvaluationSplit).Hash.ToLowerInvariant()) { throw "Evaluation resplit differs from result contract" }
if ($Gate01Splits.bank_storage.file_sha256 -ne (Get-FileHash -Algorithm SHA256 $Gate01BankSourceSplit).Hash.ToLowerInvariant()) { throw "Bank-source split differs from result contract" }
if ($Gate01Receipt.producer_spec_file_sha256 -ne (Get-FileHash -Algorithm SHA256 $Gate01ProducerSpec).Hash.ToLowerInvariant()) { throw "Archived producer spec differs from evaluated receipt" }
if ($Gate01Receipt.producer_state_file_sha256 -ne (Get-FileHash -Algorithm SHA256 $Gate01ProducerStateFile).Hash.ToLowerInvariant()) { throw "Archived producer state differs from evaluated receipt" }
if ($Gate01Receipt.build_plan_sha256 -ne (Get-FileHash -Algorithm SHA256 $Gate01BuildPlan).Hash.ToLowerInvariant()) { throw "Archived build plan differs from evaluated receipt" }
if ($Gate01Receipt.producer_receipt.decode_strategy -ne "full") { throw "Producer receipt is not full-decode sealed" }
if (($Gate01Receipt.producer_receipt.path_used -join ",") -ne "full") { throw "Producer receipt used a non-full path" }
New-Item -ItemType Directory -Force $Gate01Archive | Out-Null
$Gate01ArchiveInputs = @(
  $Gate01EvaluationSplit, $Gate01BankSourceSplit,
  $Gate01Calibrator, $Gate01Selection, $Gate01ProtocolSpec, $Gate01ProtocolLock,
  $Gate01ProducerSpec, $Gate01ProducerStateFile, $Gate01BuildPlan,
  $Gate01Manifest, $Gate01BuildState,
  $Gate01Results, $Gate01Report, $Gate01Contract
)
$Gate01ArchiveInputs | ForEach-Object {
  Get-FileHash -Algorithm SHA256 $_
} | Export-Csv -NoTypeInformation (Join-Path $Gate01Archive "sha256-inventory.csv")
Copy-Item $Gate01ArchiveInputs -Destination $Gate01Archive
Copy-Item $Gate01ProducerOutput -Destination $Gate01Archive -Recurse
Copy-Item $Gate01ProducerState -Destination $Gate01Archive -Recurse
Copy-Item $Gate01Montages -Destination $Gate01Archive -Recurse
git status --short
```

Stop if Git status is not empty. Scientific promotion remains unset until the archived
private result receives scientific review.
