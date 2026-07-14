# AGENTS.md

Repository guidance for Codex and other coding agents working on FieldBridge.

## Project Purpose

FieldBridge is a research scaffold for polymorphic MRI field and contrast translation.
Repository tests and fixtures are synthetic-only and CPU-friendly. Real-data execution
is supported through external manifests consumed by the same package contracts. Real
MRI data and generated research artifacts must never be committed to this repository.

## Repo Layout

- `src/fieldbridge/`: installable Python package.
- `src/fieldbridge/data/`: domain objects, records, manifests, data sources, datasets,
  and transforms.
- `src/fieldbridge/models/`: encoder, decoder, translator, and conditioning contracts,
  identity implementations, and research stubs.
- `src/fieldbridge/training/`: batch helpers, losses, synthetic smoke training, and
  checkpoint helpers.
- `src/fieldbridge/evaluation/`: lightweight tensor metrics.
- `src/fieldbridge/config/`: YAML config loading helpers.
- `configs/`: checked-in synthetic/model/experiment YAML examples.
- `docs/`: architecture reference, evaluation protocol, current research status,
  architecture decision records, and per-phase implementation plans.
- `notebooks/`: lightweight bootstrap scripts only. Do not add executed notebooks
  or outputs.
- `tests/`: pytest coverage for domains, datasets, model interfaces, and smoke
  training.
- `.github/workflows/ci.yml`: CPU CI for install, tests, and smoke CLI.

## How To Run

Use PowerShell on Windows unless another shell is explicitly requested.

```powershell
python -m pip install -e ".[dev]"
pytest
fieldbridge smoke-train
fieldbridge print-config --config configs/experiment/smoke.yaml
```

The package is designed so the default smoke path runs on CPU and uses synthetic
tensors only.

## Quality Commands

Run the narrowest useful checks while editing, then run the full acceptance checks
before handing work back.

```powershell
pytest
fieldbridge smoke-train
```

Optional quality extras are declared separately:

```powershell
python -m pip install -e ".[dev,quality]"
ruff check .
mypy
```

Do not install or download dependencies unless the user explicitly approves it.

## Engineering Conventions

- Keep contracts storage-backend-independent. Datasets should consume records and
  injected loaders rather than assuming a local filesystem or cloud provider.
- Keep MONAI compatibility optional. Core smoke tests should not require MONAI.
- Prefer typed dataclasses, protocols, and small modules over broad framework code.
- Keep identity models and synthetic datasets simple, deterministic, and useful for
  interface tests.
- Stubs for cloud or research methods should fail explicitly with
  `NotImplementedError` until real behavior is added.
- Preserve the CLI commands `smoke-train`, `print-config`, and `audit-manifest`.
- Keep changes scoped. Avoid unrelated refactors when making feature or bug fixes.

## Scientific And Data Invariants

- Repository tests and fixtures must remain synthetic-only. Real-data execution is
  supported only through external manifests that are not committed to the repository.
- The raw NIfTI loader tensor order is `(C, X, Y, Z)`.
- Do not infer anatomical plane names from tensor axes without orientation or affine
  metadata.
- The pseudo-pair axial convention is `volume[:, :, :, z]`.
- Subject and volume splitting must happen before slice or patch expansion.
- Real data, checkpoints, run outputs, and machine-specific absolute paths stay outside
  the repository.
- Every run must record the Git commit, resolved config, split
  fingerprint, seed, and checkpoint version.
- Pseudo-pair v1 checkpoints predate the axis and background-loss corrections and must
  not be used as scientific evidence.
- Slice metrics alone do not support complete-volume or challenge-level translation
  claims.

## Data And Artifact Rules

Do not add or generate any of the following:

- Real data, patient data, private data, or proprietary data.
- Credentials, API keys, tokens, passwords, or service account files.
- NIfTI, DICOM, MGZ/MGH, NRRD, zip, tar, 7z, checkpoint, tensor, or model artifact
  files.
- Large outputs, run directories, wandb logs, notebook outputs, or downloaded assets.

If a change needs sample inputs, use tiny synthetic tensors or small inline manifest
fixtures in tests.

## Branch And PR Expectations

- Keep feature work on a branch such as `scaffold/base-architecture`; do not land
  directly on `main` unless the user explicitly asks.
- Keep `main` as the default branch.
- Before opening or updating a PR, verify that the feature branch is based on `main`
  and GitHub can compare it.
- PR descriptions should summarize behavior changes, tests run, and any intentionally
  unimplemented stubs.

## Definition Of Done

Work is ready to hand back when:

- The requested files or behavior are implemented.
- `pytest` passes, or any inability to run it is clearly reported.
- `fieldbridge smoke-train` still runs for changes touching package, CLI, data, model,
  or training code.
- No forbidden data, secrets, checkpoints, archives, or generated artifacts were added.
- The worktree status is understood and reported.
