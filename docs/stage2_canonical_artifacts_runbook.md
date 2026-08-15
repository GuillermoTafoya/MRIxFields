# Stage-2 canonical artifacts: engineering PR 1 runbook

## Scope

This runbook covers only the external artifact path authorized after Variant A:

```text
retrospective source x_d
  -> frozen Variant-A N_d
  -> stage2-canonical-volume-v1
  -> frozen VAE posterior mean E
  -> photometry-factored-latent-bank-v1
  -> R/train-only channel statistics
  -> R/train-only structural descriptors
```

It does not implement or run a translator, trainer, critic, anatomy loss, graph loss,
adversarial loss, notebook, prospective evaluation, or long job. It does not modify or
relabel `latent-bank-v1`.

All split files, source arrays, photometry/qualification artifacts, checkpoints, output
records, manifests, statistics, and descriptors are external to the repository. The
checked-in config and a reviewed frozen-VAE architecture config may be inside the checkout.

## Preconditions

The operator must supply:

- the original canonical VAE split JSON with valid membership and v3 recovery fingerprints;
- the exact frozen `stage2-photometry-factorization-v1` artifact;
- the exact passing `stage2-photometry-variant-a-qualification-v1` result with
  `canonical_latent_bank_authorized: true`;
- the exact frozen VAE config and checkpoint named by that qualification result; and
- external output directories that do not already contain incompatible artifacts.

The split is classified completely before a source array is loaded. Only canonical `R_`
identities with matching metadata prefix, cohort, subject identity, and `train` or
`validation` role are accepted. Every correctly labelled `P_` identity is excluded and
sealed before array loading; conflicting or missing identities fail closed. This applies to
all prospective identities, not only travellers 0006, 0007, and 0009.

Both accepted splits must contain all 15 contrast/field domains. Statistics and structural
descriptors use only `R/train`; `R/validation` latents remain non-training records and are
never added to either derived artifact.

## 1. Build canonical volumes

```powershell
fieldbridge build-stage2-canonical-volumes `
  --config configs/experiment/stage2_canonical_artifacts_v1.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --out-dir <EXTERNAL_CANONICAL_DIRECTORY> `
  --resume `
  --device cpu `
  --log-every 10
```

Each `stage2-canonical-volume-v1` record stores the float32 tensor returned by
`FrozenPhotometryArtifact.normalize_source` and its exact boolean source support. Its
manifest seals:

- original split-file, membership, and recovery identities;
- photometry artifact, artifact-file, and resolved-config hashes;
- Variant-A qualification artifact/file identity and frozen-VAE provenance;
- Git commit, clean-checkout evidence, and source-module hashes;
- record, canonical subject-group, cohort, split, domain, and source-path identities;
- source-file, loaded-source-array, canonical-tensor, support, record-payload, and record-file
  hashes;
- source/canonical/support shape and dtype, support count, normalization path, and support
  policy; and
- deterministic run and per-record resume keys.

No target tensor, paired endpoint, target statistic, or runtime prediction CDF is accepted.

## 2. Audit canonical volumes

```powershell
fieldbridge audit-stage2-canonical-volumes `
  --config configs/experiment/stage2_canonical_artifacts_v1.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --canonical-dir <EXTERNAL_CANONICAL_DIRECTORY> `
  --device cpu `
  --log-every 10
```

The audit revalidates current source-file bytes, reloads each accepted source only after the
complete cohort/role preflight, reruns `N_d`, and requires exact saved tensor and support
identities. A source-content change, manifest edit, record edit, config change, split change,
or authorization change fails closed.

## 3. Build the photometry-factored bank

```powershell
fieldbridge build-photometry-factored-latent-bank `
  --config configs/experiment/stage2_canonical_artifacts_v1.yaml `
  --canonical-dir <EXTERNAL_CANONICAL_DIRECTORY> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --vae-config <FROZEN_VAE_CONFIG> `
  --vae-checkpoint <EXTERNAL_FROZEN_VAE_CHECKPOINT> `
  --out-dir <EXTERNAL_FACTORED_BANK_DIRECTORY> `
  --resume `
  --device cuda `
  --log-every 10
```

The encoder is loaded strictly and frozen. Each record uses only
`encode_dist(canonical, domain)[0]`: the posterior mean, never a posterior sample. The
requested full/tiled encoding path, actual path, precision, downsample factor, latent shape,
stored dtype, VAE hashes, photometry hashes, source/canonical identities, and code provenance
are sealed.

Each latent also carries a packed conservative source-support mask. A latent cell is marked
supported only when every source voxel in its exact VAE downsample block was supported. The
boolean mask is flattened in C order and packed with little-bit-order `numpy.packbits`; shape,
count, byte tensor identity, downsample rule, and packing rule are sealed.

`photometry-factored-latent-statistics-v1` computes per-channel mean and standard deviation
from stored `R/train` latents only. Its record/content list and artifact hash are inputs to
the descriptor contract.

`photometry-factored-structural-descriptor-v1` is also `R/train` only. For each standardized
canonical latent it computes, in fixed order:

1. support-normalized latent pooling;
2. support-valid absolute first differences along x, y, and z; and
3. support-normalized adaptive 3-D pooling at output sizes 1, 2, and 4.

The descriptor retains canonical subject identity for later same-subject exclusion. It seals
the complete record/source/canonical/latent/support content fingerprint, latent-statistics
artifact hash, descriptor-config hash, standardized-latent identity, descriptor tensor
identity, shape, and dtype. It contains no paired endpoint or target input.

## 4. Audit the bank

```powershell
fieldbridge audit-photometry-factored-latent-bank `
  --config configs/experiment/stage2_canonical_artifacts_v1.yaml `
  --canonical-dir <EXTERNAL_CANONICAL_DIRECTORY> `
  --photometry-artifact <EXTERNAL_PHOTOMETRY_JSON> `
  --qualification <EXTERNAL_VARIANT_A_QUALIFICATION_JSON> `
  --vae-config <FROZEN_VAE_CONFIG> `
  --vae-checkpoint <EXTERNAL_FROZEN_VAE_CHECKPOINT> `
  --bank-dir <EXTERNAL_FACTORED_BANK_DIRECTORY> `
  --device cuda `
  --log-every 10
```

The audit re-encodes every canonical tensor with the exact frozen encoder, verifies every
stored latent and packed support mask, recomputes train-only statistics, and recomputes all
train-only descriptors. It writes nothing.

## Atomic resume and no-clobber rules

Every record and JSON manifest is published atomically from a same-directory temporary file.
Publication uses an atomic no-clobber link; an existing destination is never replaced.
Interrupted temporary files are removed.

Without `--resume`, any existing destination fails. With `--resume`, every existing record,
derived artifact, and final manifest must match its deterministic input/resume identity and
all internal tensor/content hashes. Compatible files are reused byte-for-byte. Incompatible
files fail; there is no overwrite, repair, relabel, or partial acceptance path.

## Scientific and operator boundary

These artifacts authorize no model training by themselves. The missing later decisions remain
outside this PR: trainer/objective contract, coupling policy, checkpoint/history contract,
anatomy scales, graph-loss order, promotion thresholds, prospective roles, compute budget, and
competition-permitted final-fit manifest. No `P` identity may be added to this v1 bank by an
operator override; any future prospective ablation requires a new forward-versioned contract.
