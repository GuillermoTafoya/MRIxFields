# Stage-1 Domain-Balanced Full-Volume Audit v1

This is an evaluation-only Track-B contract for frozen Stage-1 KL-VAE checkpoints. It
does not train or modify a model, start Stage 2, or declare scientific viability. Real
split files, selections, NIfTIs, checkpoints, and outputs stay outside Git.

Do not launch this audit while Stage-1 training is using the same GPU. Wait for training
to finish, then use the exact audit commit reported by the draft PR.

## Frozen selection

`select-stage1-vae-audit` reads the existing subject-held-out VAE split JSON, reruns its
case/path/subject leakage audit, and uses only `test`. The canonical domain order is the
repository's five field strengths crossed with its three contrasts. For every domain,
the v1 selector orders eligible records by:

```text
SHA256(algorithm_version | seed | split_fingerprint | stable_record_id | domain)
```

It retains the first four, for exactly 60 volumes. Sorting happens after hashing, so the
result is independent of input record order. The official adapter maps `sample_id` to
`case_id`; that value is the stable volume identity. Duplicate identities, split
leakage, a missing domain, or fewer than four eligible test records are fatal.

The private selection contains record identities and paths required to rerun the audit
and must stay private. The sanitized companion contains only fixed `domain-NN` and
`domain-NN-case-NN` slots. If the private selection already exists, a different split
fingerprint, all-record fingerprint, seed, algorithm version, or selected record is an
error. It is never silently regenerated.

## Reconstruction and recovery

`audit-stage1-vae` reuses `sliding_window_reconstruct`. It traverses windows in ascending
raw tensor-axis order, edge-clamps the final start on each axis, decodes `z = mu`, and
normalizes a separable 3D Hann-weighted sum. There are no crops, selected slices,
augmentations, or posterior samples. Encoder and decoder are in eval mode and execution
uses `torch.inference_mode()` with deterministic PyTorch algorithms.

The input must be finite and in the project `[0,1]` range. Raw decoder output is retained
for range diagnostics, then clamped to `[0,1]` for the frozen audit metrics.
`float32` is the primary default. `--precision amp-bfloat16` is explicit, CUDA-only, and
part of the recovery fingerprint.

Each anonymous volume result is atomically saved with the checkpoint/audit fingerprint.
With `--resume`, validated results and completed exemplar panels are reused. A changed
checkpoint hash, config hash, split/selection fingerprint, commit, device, precision,
patch size, overlap, threshold, or metric contract fails closed. Without `--resume`, an
existing checkpoint result directory is refused.

## Metric contract

Let `t` be the target, `r_raw` the raw reconstruction, `r=clamp(r_raw,0,1)`, and
`F={i:t_i>foreground_threshold}`. The frozen threshold is read from the experiment data
config (currently zero). Empty foreground is fatal.

- Foreground MAE: `mean_F |r-t|`.
- Foreground range-normalized RMSE: `sqrt(mean_F (r-t)^2) / 1`; this historical audit
  diagnostic is not the published Task-3 L2-ratio nRMSE.
- SSIM3D: `stage1_full_volume_ssim3d_v1`, the exact repository zero-padded
  uniform-window volumetric calculation used at audit commit `be60d75`, on complete
  clamped tensors with `data_range=1`. It is not the published slice-wise Task-3 SSIM.
- Correlation: Pearson correlation over foreground voxels. Equal constant vectors report
  one; other constant cases report zero with an explicit status.
- Gradient MAE: finite forward differences of `r` and `t`, masked only where both adjacent
  voxels are foreground, averaged equally over available raw x/y/z tensor axes.
- Background leakage: `mean_not-F |r|`. If no background exists, the value is null with
  `not_available_no_background` status.
- Signed foreground bias: `mean_F(r-t)`.
- Prediction-minus-source residual magnitude: `mean_F |r-t|`. Stage-1 is an
  auto-reconstruction audit, so its source and target are the same native volume.
- Foreground quantiles: linear `q01,q05,q50,q95,q99` for target and reconstruction over
  the target foreground, plus signed `q_r-q_t` and absolute quantile errors.
- High-intensity tail: target-foreground voxels satisfying `t >= target_q99`; report
  `mean |r-t|` and `mean(r-t)` on that fixed target-defined set.
- Foreground histogram distance: 256 fixed equal bins over `[0,1]`, normalized separately,
  then `sum_k |CDF_r(k)-CDF_t(k)| / 256`. This is the deterministic fixed-bin 1D
  Wasserstein/CDF approximation.
- Raw range diagnostics: raw min/max and fractions below zero and above one before clamp.
- Latent diagnostics: every deterministic reconstruction tile contributes posterior mean
  and log-variance to the existing per-channel accumulator. Reports global posterior-mean
  mean/std, mean per-channel KL, per-channel KL/std, and active-channel count using the
  existing `latent_active_kl_threshold`. Because overlapping tiles repeat overlap voxels,
  these are explicitly tile-weighted diagnostics, not a reconstructed latent volume.

All non-finite inputs/outputs are fatal. Unavailable values are null with a status; the
audit does not silently serialize NaN.

## Aggregation and artifacts

Every metric is computed once per complete selected volume. Volumes are weighted equally
within each of the 15 domain means, and the 15 means are weighted equally in the primary
macro. A pooled per-volume micro mean is labeled secondary. A domain with more records can
never dominate the primary macro.

The audit root contains `audit_contract.json`, `selection_fingerprint.json`,
`run_progress_sanitized.json`, `checkpoint_comparison.json`, `report.md`, and
`sanitized_handoff.json`. Each `checkpoints/checkpoint-NN/` directory contains
`per_volume_metrics.jsonl`, `per_domain_metrics.json`, `macro_metrics.json`, `report.md`,
and one deterministic target/reconstruction/absolute-error/foreground-histogram panel
for the predeclared first selected case in every domain. Paths in handoff files are
relative sanitized artifact names. Independent compatible checkpoint results can be
compared without recomputing validated volumes.

## Windows server workflow (run only after training finishes)

From PowerShell, replace placeholders with private paths. Do not execute this while the
training process owns the GPU.

```powershell
git fetch origin
git switch --detach <EXACT_AUDIT_COMMIT>
python -m pip install -e ".[nifti,evaluation]"

fieldbridge select-stage1-vae-audit `
  --split-json <PRIVATE_VAE_SPLIT_JSON> `
  --private-out <PRIVATE_AUDIT_ROOT>\selection_private.json `
  --sanitized-out <PRIVATE_AUDIT_ROOT>\selection_sanitized.json `
  --seed 13

fieldbridge audit-stage1-vae `
  --split-json <PRIVATE_VAE_SPLIT_JSON> `
  --selection <PRIVATE_AUDIT_ROOT>\selection_private.json `
  --config configs\experiment\stage1_vae.yaml `
  --checkpoint "epoch-7-best=<PRIVATE_CHECKPOINT>" `
  --checkpoint "epoch-9-step-28620-best=<PRIVATE_CHECKPOINT>" `
  --checkpoint "step-60000=<PRIVATE_CHECKPOINT>" `
  --checkpoint "epoch-75-final=<PRIVATE_CHECKPOINT>" `
  --checkpoint "later-best=<PRIVATE_CHECKPOINT>" `
  --out <PRIVATE_AUDIT_ROOT>\results `
  --device cuda `
  --precision float32 `
  --overlap 0.5
```

After an interruption, rerun the exact command with `--resume`. Keep checkpoint ordering
and labels stable because `checkpoint-NN` is the anonymous comparison slot.
