# Stage-2 streamed canonical artifacts: engineering PR 1 runbook

## Scope

This runbook covers the retrospective-only external artifact path:

```text
R source x_d
  -> frozen Variant-A N_d(x_d)                 [in memory only]
  -> hash canonical tensor and source support
  -> frozen VAE full encode_dist(...)[0]       [same streamed record]
  -> persist posterior-mean latent + packed local-valid-core support + sidecar
  -> R/train local-valid-core statistics
  -> R/train structural descriptors
```

The primary config never persists full float32 canonical volumes or full Boolean source
masks. `stage2-canonical-volume-v1` is an ephemeral, hash-sealed computation boundary. No
debug full-volume artifact mode is enabled or exposed by the primary commands.

This PR does not implement or run a translator, trainer, critic, anatomy loss, graph loss,
adversarial loss, notebook, prospective evaluation, or long job. It does not modify Variant-A
arithmetic and does not mutate, alias, or relabel `latent-bank-v1`.

All split files, source arrays, photometry/qualification artifacts, checkpoints, output
records, manifests, statistics, and descriptors are external to the repository.

## Required inputs

The operator must supply:

- the original canonical VAE split JSON with valid file, membership, and v3 recovery
  fingerprints;
- the exact frozen `stage2-photometry-factorization-v1` artifact;
- the exact passing `stage2-photometry-variant-a-qualification-v1` result with
  `canonical_latent_bank_authorized: true`;
- the exact frozen VAE architecture config and checkpoint named by that qualification; and
- a local output filesystem that passes atomic hard-link/no-clobber and capacity preflight.

The complete split is classified before source files are hashed, headers are inspected, or
arrays are loaded. Only canonical `R_` identities with matching prefix/cohort/subject metadata
and frozen `train` or `validation` roles are accepted. Every `P` identity is excluded; there is
no special-case list and no operator override. Both accepted splits must contain all 15
contrast/field domains.

## 1. Preflight the streamed build

Run preflight against the exact output location before any long build:

```powershell
fieldbridge preflight-photometry-factored-latent-bank `
  --config configs/experiment/stage2_canonical_artifacts_v2.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --vae-config <FROZEN_VAE_CONFIG> `
  --vae-checkpoint <EXTERNAL_FROZEN_VAE_CHECKPOINT> `
  --out-dir <LOCAL_SCRATCH_FACTORED_BANK_DIRECTORY> `
  --device cuda
```

The JSON report seals or reports:

- accepted/excluded record counts and proof that classification preceded array loading;
- source shapes obtained from NIfTI headers, record count, and total source voxels;
- float32 canonical-volume and full Boolean support bytes avoided;
- predicted latent, packed-support, descriptor, output, and temporary bytes;
- required free storage and peak streamed working-set estimate;
- the reviewed local-valid-core rule, graph hash, receptive field, stride, alignment, and
  separately sealed GroupNorm dependency provenance; and
- successful same-directory atomic hard-link publication and no-clobber behavior.

The estimates are conservative planning values, not memory reservations. Preflight reads
source identities, file bytes for hashes, and NIfTI headers; it does not load voxel arrays and
does not run N_d or VAE inference.

### Google Drive, FUSE, and unsupported filesystems

Publication deliberately has no rename/overwrite fallback. If the target returns `ENOTSUP`,
`EPERM`, or otherwise cannot prove hard-link no-clobber semantics, preflight stops before any
record is processed. Build on a compatible local scratch filesystem, audit the completed local
artifact, and only then perform a separate hash-verified archival copy. Do not build directly
on an unqualified Drive/FUSE mount.

## 2. Build the streamed bank

```powershell
fieldbridge build-photometry-factored-latent-bank `
  --config configs/experiment/stage2_canonical_artifacts_v2.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --vae-config <FROZEN_VAE_CONFIG> `
  --vae-checkpoint <EXTERNAL_FROZEN_VAE_CHECKPOINT> `
  --out-dir <LOCAL_SCRATCH_FACTORED_BANK_DIRECTORY> `
  --resume `
  --device cuda `
  --log-every 10
```

For each R record, the builder loads one source, computes frozen `N_d`, hashes the source,
canonical tensor, and full source support, immediately performs the frozen full
`encode_dist(...)[0]`, propagates local anatomical validity through the reviewed encoder graph,
packs that latent support, publishes one record, and releases the full tensors. The record
payload contains only:

- the stored posterior-mean latent;
- the packed latent support; and
- a hash-bound metadata/provenance sidecar.

No canonical tensor or full source mask is included in the payload.

The primary v2 config requires `strategy=full` and actual `path_used=full` with float32 encode
arithmetic. A full-encode OOM is a hard stop. There is no automatic or explicit tiled fallback;
any tiled variant requires a new artifact version and separate qualification.

## Encoder local-valid-core support

`encoder-local-valid-core-support-v1` is the operational support contract. It propagates
anatomical spatial validity through the frozen `KLVAEEncoder` graph:

- Conv3d invalidity is propagated with each layer's exact kernel, stride, padding, and
  dilation. Constant padding outside the volume is not treated as a source dependency.
- Residual outputs require both their main and skip dependencies to be supported.
- Pointwise activations and identity normalization preserve the local mask.
- GroupNorm also preserves the local mask. Its global value dependence is recorded separately
  for every module with module path, type, channels, groups, and epsilon.

The artifact seals the graph operations, graph hash, convolutional receptive-field size,
radius, output stride, source/latent alignment, GroupNorm dependency provenance and hash,
support-mask hash, shape, count, and packing rule. Local valid-core support describes
anatomical spatial validity. It does not establish independence from GroupNorm's global
normalization statistics.

The former global-normalization-collapse calculation remains available only as
`encoder-complete-dependency-support-diagnostic-v1`. It is non-operational and nonblocking;
for this VAE it may report an empty diagnostic mask. The primary v2 configuration leaves it
disabled. Enabling it records only a diagnostic summary and never replaces, weakens, or blocks
the local-valid-core mask. The build itself fails if operational support is empty, nonfinite,
shape-inconsistent, or the frozen graph contains an unsupported spatial operation.

## Supported-cell statistics and descriptors

`photometry-factored-latent-statistics-v2` uses only operational local-valid-core cells from
persisted R/train latents. It uses channel-wise streaming float64 Welford accumulation and
records per-channel supported counts, total supported value count, and supported spatial-cell
count. Empty masks, nonfinite values, fewer than two values, and channel variance at or below
the sealed `1e-12` floor fail closed.

Descriptors standardize with those masked statistics and force unsupported standardized cells
to exact zero before hashing or feature calculation. Fixed latent and valid-adjacent absolute
gradient features are support-normalized and adaptively pooled at sizes 1, 2, and 4. Arbitrary
finite values outside support cannot change either statistics or descriptors.

The descriptor artifact retains subject-group identity for later same-subject exclusion and
contains no paired endpoint or target input. It is explicitly sealed as:

```text
coupling_authorized: false
qualification_required: photometry-factored-structural-descriptor-qualification-v2
```

No trainer may consume it for nearest-neighbor coupling until a retrospective R/validation
qualification with subject-group exclusion tests subject retrieval, field predictability,
support-volume shortcuts, and descriptor stability. The artifact makes no learned-
disentanglement claim.

## 3. Complete audit

```powershell
fieldbridge audit-photometry-factored-latent-bank `
  --config configs/experiment/stage2_canonical_artifacts_v2.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --vae-config <FROZEN_VAE_CONFIG> `
  --vae-checkpoint <EXTERNAL_FROZEN_VAE_CHECKPOINT> `
  --bank-dir <LOCAL_SCRATCH_FACTORED_BANK_DIRECTORY> `
  --device cuda `
  --log-every 10
```

Audit writes nothing. After the same complete R-only preflight, it verifies current source-file
identities and recomputes every source through `source -> N_d -> full E`. It verifies canonical
and source-support hashes, posterior-mean latent bytes, propagated and packed support, masked
statistics, every descriptor, all manifests, and every input/provenance binding.

## Computational provenance

`stage2-computational-provenance-v2` contains a versioned reviewed dependency map and module
hash for every production path involved in:

- N_d and support generation;
- record contracts, source loading, and split classification/fingerprints;
- VAE construction and encoder forward arithmetic;
- `encode_latent` and full encoding;
- local-valid-core propagation, GroupNorm dependency provenance, optional diagnostic,
  support packing, Welford statistics, and descriptors; and
- config, checkpoint, qualification, and CLI loading.

It also seals the dependency-map hash, Git commit/clean-checkout state, Python implementation
and version, PyTorch and NumPy versions, CUDA build/availability, cuDNN version, and selected
device name/index/capability. A change to any required module identity, map, or runtime identity
invalidates resume and audit.

## Atomic resume and no-clobber

Every record and JSON manifest is written to a unique same-directory temporary file, flushed,
and published using a hard link that fails if the destination already exists. Interrupted
temporaries are removed. Concurrent destination creation never gets overwritten. Unsupported
hard links stop without leaving a partial destination.

Without `--resume`, any existing destination fails. With `--resume`, a complete manifest is
accepted only when the v2 resume/audit contracts, run fingerprint, local-support rule,
GroupNorm provenance, computational provenance, source identities, and every referenced
file/content hash match exactly. Compatible complete artifacts are reused byte for byte
without loading source arrays. Any v1 plan, incompatible artifact, or partial published
manifest fails; there is no overwrite, repair, relabel, or fallback path.

## Scientific boundary

These artifacts authorize no model training or coupling. Remaining trainer/objective,
coupling, checkpoint/history, anatomy scale, graph-loss, promotion, prospective, compute, and
final-fit decisions remain outside this PR. No private or prospective execution is authorized.
