# AGENTS.md

Repository guidance for Codex and other coding agents working on CLB-Field.

## Project Purpose

CLB-Field is a research scaffold for polymorphic MRI field and contrast translation.
The current codebase is intentionally synthetic-data-only and CPU-friendly. It defines
contracts, interfaces, stubs, configs, tests, and a smoke training path, but it must
not include real MRI data or generated research artifacts.

## Repo Layout

- `src/clbfield/`: installable Python package.
- `src/clbfield/data/`: domain objects, records, manifests, data sources, datasets,
  and transforms.
- `src/clbfield/models/`: encoder, decoder, translator, and conditioning contracts,
  identity implementations, and research stubs.
- `src/clbfield/training/`: batch helpers, losses, synthetic smoke training, and
  checkpoint helpers.
- `src/clbfield/evaluation/`: lightweight tensor metrics.
- `src/clbfield/config/`: YAML config loading helpers.
- `configs/`: checked-in synthetic/model/experiment YAML examples.
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
clbfield smoke-train
clbfield print-config --config configs/experiment/smoke.yaml
```

The package is designed so the default smoke path runs on CPU and uses synthetic
tensors only.

## Quality Commands

Run the narrowest useful checks while editing, then run the full acceptance checks
before handing work back.

```powershell
pytest
clbfield smoke-train
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
- `clbfield smoke-train` still runs for changes touching package, CLI, data, model,
  or training code.
- No forbidden data, secrets, checkpoints, archives, or generated artifacts were added.
- The worktree status is understood and reported.
