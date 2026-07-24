# Stage-1 v3 development bundle

This additive bundle responds to the frozen 60-volume audit implemented at `be60d75`
and evaluated over four immutable checkpoints. It does not alter the legacy, cosine,
or v2 experiment YAML files and does not treat any private-data result as repository
evidence.

## Repository and v2 audit

The implementation reviewed from `origin/main` was v2 commit `92f12f1`, merged as
`7171035`. Its sampler equalizes five field strengths only, samples volumes with
replacement, and does not balance complete field-by-contrast domains or subjects.
Its free-bits term floors each channel's batch-and-spatial mean KL in
nats/latent-element, rescales by latent spatial elements, and sums channels. It has LR
warm-up but no KL warm-up. Active channels are defined only by per-channel KL greater
than `0.01`.

Validation in v2 averages patch batches from the natural validation distribution.
Selection and validation early stopping use weighted patch total loss. Resume restores
model, optimizer, step, and optionally training-EMA state, but not RNG, sampler,
validation early-stop, best-validation, or latent-health state.

The observed negative SSIM loss was not a harmless logging issue. The former SSIM
computed `E[x^2] - E[x]^2` in bf16 autocast on unbounded decoder predictions.
Cancellation could produce invalid negative variances and covariance/denominator
combinations, allowing similarity above 1 and therefore negative `1 - SSIM`.

## Sampling

`data.joint_domain_balance` builds one deterministic global schedule per pass:

1. allocate draws equally across all 15 field-by-contrast domains;
2. allocate each domain's draws equally across subjects;
3. rotate across each subject's volumes;
4. rotate integer remainders between passes; and
5. interleave domains before sharding schedule positions across loader workers.

Missing any of the 15 domains is a fatal configuration error. History records expected
and observed exposure by domain and by subject within domain. Small domains still repeat
records by necessity, but the repetition is explicit and cannot silently concentrate on
one subject or volume.

## Loss and numerical contract

The differentiable training SSIM now has a deliberately separate name and module.
Its moments are evaluated in float32 outside autocast. Variances are projected
nonnegative and covariance is projected onto its Cauchy-Schwarz bound before the
luminance and structure terms are formed. Similarity is documented and checked in
`[-1, 1]`; loss is finite and nonnegative. It is a training proxy, not the frozen
Stage-1 audit SSIM3D and not the published Task-3 scikit-image metric.

The completed audit remains on `stage1-full-volume-metrics-v1`, whose SSIM3D arithmetic
is frozen explicitly as `stage1_full_volume_ssim3d_v1` at the `be60d75` behavior. The
source-pinned official Task-3 adapter is documented separately in
`MRIXFIELDS2026_TASK3_METRICS.md`.

New arms use target-derived foreground (`target > 0`) for masked L1 and a separate mean
absolute prediction penalty on exact-zero background. All raw and weighted loss
contributions are logged. Histogram distance, signed foreground bias, high-intensity
tail error, and quantiles remain diagnostics; no unsupported differentiable histogram
loss is introduced.

## Controlled arms

- `stage1_ae_v3_joint_domain.yaml`: deterministic latent mean, no sampling or KL.
  Activity is monitored by per-channel latent standard deviation.
- `stage1_vae_v3_joint_domain_freebits.yaml`: stochastic KL-VAE, raw and effective
  per-channel free-bits KL, and a ten-epoch linear KL warm-up resolved from the split's
  `steps_per_epoch`.
- `stage1_vae_v3_target_decoder_film.yaml`: separate experimental arm with one shared
  16-dimensional target-domain conditioner and FiLM in the decoder. It has no routers
  or domain-specific subnetworks and writes to an isolated checkpoint directory.

The decoder experiment is interface-compatible with the Stage-2 contract:
`Decoder.decode(z, domain)` already accepts the decode domain, while latent transport
continues to accept source and target domains separately. It changes where calibration
can be represented, so it must be compared with the unconditioned arm and must not be
silently substituted.

## Validation and checkpoints

Validation first averages patches within each volume, then volumes within each domain,
then all available domains equally. The efficient SSIM value is explicitly named
`ssim3d_proxy`; it is not presented as the frozen complete-volume audit.

Candidate selection uses metric-specific bests and a Pareto frontier over masked nRMSE,
masked MAE, SSIM proxy, histogram distance, absolute signed foreground bias,
high-intensity-tail error, background leakage, and latent utilization. It does not
collapse those quantities into guessed scalar weights. Promotion requires at least
three of four active channels.

Epoch-boundary `vae_stage1_latest_recoverable.pt` stores model, optimizer, explicit
scheduler position, explicit absent bf16 scaler state, Python/CPU/CUDA RNG, sampler
pass, epoch/global step, training and validation early-stop state, candidate frontier,
and latent health. Metric-specific and latest-Pareto checkpoints are stored separately.
Mid-epoch step checkpoints are marked nonrecoverable for exact sampler replay.

The expensive frozen 60-volume audit remains an offline promotion gate. Patch proxies
can nominate candidates; they cannot replace equal-domain complete-volume evaluation
or support challenge-level claims.
