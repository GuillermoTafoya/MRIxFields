# CLB-Field

Initial research scaffold for CLB-Field, a polymorphic PyTorch/MONAI-compatible
framework for MRI field and contrast translation.

This repository intentionally contains no real MRI data, credentials, model
checkpoints, NIfTI files, archives, or generated artifacts. The included smoke
path uses synthetic tensors only.

## Quickstart

```powershell
pip install -e ".[dev]"
pytest
clbfield smoke-train
```

## CLI

```powershell
clbfield smoke-train
clbfield print-config --config configs/experiment/smoke.yaml
clbfield audit-manifest path/to/manifest.yaml
```

