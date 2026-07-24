# MRIxFields2026 Task-3 metric contracts

MRIxFields uses three distinct metric contracts. They must not be substituted for one
another, even when their display names are similar.

| Contract | Purpose | Implementation |
|---|---|---|
| Published Task 3 | Challenge parity | `evaluation/mrixfields2026_official.py` |
| Stage-1 full-volume metrics v1 | Reproduce the completed 60-volume audit | `stage1_full_volume_ssim3d_v1` plus the frozen audit module |
| Stage-1 training proxy | Differentiable, autocast-safe optimization | `training/ssim.py` |

## Published Task-3 adapter

The adapter is pinned to the public `MRIxFields/MRIxFields2026` repository:

- repository commit: `5d55309253951d9dfb7847856f4f46893a44d63b`;
- `Evaluation/evaluate.py` blob:
  `4e21a48b097ef274f9fceeef536f9790eb451385`;
- `Evaluation/README.md` blob:
  `7d3a73a38990450c58b9d05a89b82acb0b73e638`.

`load_official_nifti` performs `nib.load`, `nib.as_closest_canonical`, then
`get_fdata(dtype=np.float32)`. The official metrics then reproduce the published
conversions and reductions:

- nRMSE casts to float64 and returns
  `norm(prediction-target) / norm(target)` over the unmasked full volume, with zero when
  the target norm is at most `1e-10`;
- SSIM casts to float64, uses the global target range, computes scikit-image SSIM on
  slices along axis 2, omits constant target slices, and averages the remaining slices;
- LPIPS maps `[0,1]` to `[-1,1]`, repeats each axial slice to three channels, evaluates
  `lpips.LPIPS(net="alex")`, applies the published target-slice filter, and averages.

Use `evaluate_official_task3_pair(prediction_path, target_path)` for file-based parity.
This requires the optional dependencies:

```powershell
python -m pip install -e ".[official-evaluation]"
```

No official-evaluation dependency is needed for core synthetic tests.

## Frozen audit v1

`stage1-full-volume-metrics-v1` predates the published evaluator and remains immutable.
Its `ssim3d` result is the zero-padded `avg_pool3d` Torch calculation used at commit
`be60d75`. The audit imports `stage1_full_volume_ssim3d_v1` explicitly, so changes to
generic tensor helpers or training proxies cannot silently change completed-audit
semantics.

Its foreground range-normalized RMSE, histogram, quantile, tail, bias, gradient, and
background diagnostics remain useful for Stage-1 candidate selection, but they are not
challenge-score aliases.

## Training proxy

`stable_training_ssim` and `stable_training_ssim3d` compute moments in float32 outside
autocast, project invalid variances and covariance, and bound similarity to `[-1,1]`.
`ssim_loss` validates finiteness and nonnegativity before returning `1-similarity`.
These properties fix the observed negative-loss failure while making no claim of exact
scikit-image parity.
