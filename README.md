# FieldBridge

Initial research scaffold for FieldBridge, a polymorphic PyTorch/MONAI-compatible
framework for MRI field and contrast translation.

This repository intentionally contains no real MRI data, credentials, model
checkpoints, NIfTI files, archives, or generated artifacts. The included smoke
path uses synthetic tensors only.

## Quickstart

```powershell
pip install -e ".[dev]"
pytest
fieldbridge smoke-train
```

## CLI

```powershell
fieldbridge smoke-train
fieldbridge print-config --config configs/experiment/smoke.yaml
fieldbridge audit-manifest path/to/manifest.yaml
```

