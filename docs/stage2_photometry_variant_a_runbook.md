# Stage-2 Variant A: frozen photometry runbook

## Scope

Variant A is a fixed, non-learned baseline. For source domain
`s = (contrast, field)` and target domain `t` with the same contrast, it computes

```text
T_A(s -> t, x_s) = M_s * P_t(N_s(x_s))
```

`N_s`, `P_t`, and the contrast-specific canonical quantile template are sealed in a
training-only artifact. `M_s` is exactly `x_s != 0`. No target or prediction is
accepted by the normalization API, and no runtime CDF is fitted.

This implementation does not train a model, build a latent bank, modify the Stage-2
translator, or implement Variants B-E.

## Contracts

The independently versioned contracts are:

- frozen artifact: `stage2-photometry-factorization-v1`;
- resolved config: `stage2-photometry-variant-a-config-v1`;
- qualification result: `stage2-photometry-variant-a-qualification-v1`;
- Gate 0.1 continuity reference: `stage2-photometry-continuity-reference-v2`;
- genuinely paired evaluation manifest: `stage2-photometry-paired-evaluation-manifest-v1`;
- resumable case shard: `stage2-photometry-paired-case-shard-v1`;
- method-neutral result: `stage2-photometry-dual-baseline-result-v2`.

The checked-in proposed defaults are in
`configs/experiment/stage2_photometry_factorization_a_v1.yaml`. Their numerical
thresholds are versioned research decisions awaiting external review, not physical
constants and not values to tune on prospective evidence.

All data-bearing inputs and outputs in the commands below must be external to the
repository. The reviewed checked-in Variant-A config, and a checked-in frozen-VAE
architecture config when applicable, are the only non-data inputs that may come from
the checkout. Examples use placeholders intentionally; do not put machine-specific or
private paths in a checked-in script or config.

## 1. Fit the frozen artifact

Run fitting from a clean checkout at the reviewed commit. The canonical split loader
must validate the membership and recovery fingerprints before fitting. The command
accepts only retrospective `R/train` records and requires all 15 contrast/field
domains. One shared classifier derives `R` or `P` independently from the `case_id`
prefix, reconciles it with the metadata prefix and supplied cohort, requires a subject
identity, and rejects missing or conflicting identities. Prospective records,
validation/test records, and travellers 0006, 0007, and 0009 are rejected by the fit
contract; the three named travellers are not the only rejected `P` identities.

```powershell
fieldbridge fit-stage2-photometry `
  --config configs/experiment/stage2_photometry_factorization_a_v1.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --out <EXTERNAL_ARTIFACT_JSON> `
  --device cpu `
  --log-every 10
```

The artifact records the source split SHA-256 and fingerprints, accepted record and
content identities, domain counts, equal-volume and equal-field weights, quantile
grids, fixed interpolation rules, resolved config hash, repository commit, clean-tree
evidence, source-file hashes, and its own content SHA-256. Existing output is never
silently overwritten.

For each contrast, the canonical grid is the exact arithmetic mean of the five field
grids, giving every field weight `0.2`. Each domain grid is the arithmetic mean of its
eligible volume grids, giving every eligible volume equal weight. The default grid has
256 evenly spaced quantile probabilities.

Duplicate source knots are deterministically collapsed to one knot whose target value
is the arithmetic mean of the corresponding target knots. Piecewise-linear mapping is
sealed at the endpoint values. Values outside source support are bitwise zero after
both normalization and target rendering; Variant A applies no erosion or morphology.

## 2. Qualify fixed photometry and canonical VAE input

Qualification accepts only held-out retrospective `R/validation` records from the
same fingerprinted split. It rejects `R/train`, all prospective records, and travellers
0006, 0007, and 0009. The supplied VAE is frozen, loaded strictly, and used with full
posterior-mean encode and full decode; there is no tiled or automatic fallback.

```powershell
fieldbridge audit-stage2-photometry `
  --config configs/experiment/stage2_photometry_factorization_a_v1.yaml `
  --split-json <EXTERNAL_SPLIT_JSON> `
  --artifact <EXTERNAL_ARTIFACT_JSON> `
  --vae-config <FROZEN_VAE_CONFIG> `
  --vae-checkpoint <EXTERNAL_FROZEN_VAE_CHECKPOINT> `
  --continuity-reference <EXTERNAL_CONTINUITY_REFERENCE_JSON> `
  --gate01-result <EXTERNAL_GATE01_RESULT_JSON> `
  --out <EXTERNAL_QUALIFICATION_JSON> `
  --device cuda `
  --precision float32 `
  --log-every 1
```

The report includes per-record, per-domain, macro, and worst-domain measurements for:

- stored-grid finiteness and tolerance-scaled monotonicity;
- exact-zero support;
- direct `P_d(N_d(x))` round-trip error;
- canonical histogram alignment across fields;
- a deterministic leave-one-subject-group-out contrast-identity control. Intensities
  are average-rank normalized to `[-1,1]` inside support; fixed 1-, 3-, and 5-voxel
  support-normalized box filters and three first-difference maps are resampled with
  trilinear interpolation (`align_corners=false`) to `4 x 4 x 4`. Every fold reports
  its held-out group, training groups, and train/validation domain counts;
- multiplicative scaling at the sealed factors, reported independently as histogram
  distance and a true uniform-window spatial 3-D SSIM. Both volumes are exactly masked
  by the same source support and use the unperturbed canonical volume's sealed data
  range;
- fitted and qualification domain counts and the sealed equal-weight policy;
- raw `D(E(x))` and factorized `P_d(D(E(N_d(x))))` reconstruction on each same
  validation volume. Both reconstruction paths are masked with the one source-derived
  support before official metrics; raw decoder leakage before masking is reported
  separately. Support shape, boolean dtype, voxel count, and canonical byte SHA-256 are
  sealed per record.

Monotonicity is not inferred from constructor acceptance. The audit evaluates both
realized `N_d` and `P_d` after duplicate-knot collapse on 4,097 fixed points spanning
the sealed 1%-99% robust range plus endpoints, and independently reports finiteness and
the minimum output step under the preregistered tolerance.

The report also compares the two held-out-retrospective macro reconstructions with the
hash-verified original Stage-1 ceiling. That comparison is labelled external
continuity evidence: it is not reinterpreted as a held-out-retrospective observation
and is not an extra pass/fail threshold.

Scaling sensitivity is evidence, not an authorization to normalize using a target or
prediction. The command never changes the fixed map in response to scaling results.

Failure is classified explicitly:

- `photometry_factorization_failure` means a fixed-map, support, balance, alignment,
  contrast, scaling, or round-trip check failed;
- `canonical_vae_distribution_shift_failure` means canonical-input VAE compatibility
  failed relative to the original-input frozen-VAE reconstruction.

`canonical_latent_bank_authorized` is true only when both classes are absent. If the
fixed map cannot remove field photometry while preserving VAE reconstruction, stop:
do not build a canonical latent bank, and do not substitute Gate 0.1 posthoc
per-prediction CDF calibration and call it factorization.

## 3. Prepare the continuity reference

The evaluator does not rerun or reinterpret Gate 0.1. Build the reference from the
explicit reviewed result; do not transcribe aggregate values manually:

```powershell
fieldbridge build-stage2-photometry-continuity-reference `
  --gate01-result <EXTERNAL_GATE01_RESULT_JSON> `
  --evaluation-id <FROZEN_PAIRED_SELECTION_IDENTITY> `
  --out <EXTERNAL_CONTINUITY_REFERENCE_JSON>
```

The builder requires the reviewed `stage2-gate01-equal-photometry-v2` contract and
exactly `nrmse`, `ssim`, and `lpips` for `calibrated_identity`, `raw_identity`, and
`stage1_reconstruction_ceiling`. It seals the source file SHA-256, extraction paths,
semantics, provenance, and its own artifact hash.

The following synthetic example is byte-for-byte loadable. Save this one-line Gate 0.1
source with exactly one final LF; its SHA-256 is
`53078368ff1fdbbdd0712f2fe5ad5d8ec7ec4b69318f3558ee5f4c7f9d773b7f`:

```json
{"contract_version":"stage2-gate01-equal-photometry-v2","overall":{"methods":{"calibrated_identity":{"lpips":0.091186,"nrmse":0.323585,"ssim":0.899374},"raw_identity":{"lpips":0.096859,"nrmse":0.595189,"ssim":0.875809},"stage1_reconstruction_ceiling":{"lpips":0.03606,"nrmse":0.131321,"ssim":0.965984}}}}
```

The deterministic builder produces this loadable reference:

```json
{
  "artifact_sha256": "0aa44819b9977728e433d86cb0adf42356e33397596977e955759afa47a478c2",
  "calibration_semantics": {
    "gate01_posthoc_calibrated_identity": "unchanged Gate 0.1 prediction-CDF diagnostic",
    "raw_identity": "no calibration",
    "stage1_reconstruction_ceiling": "frozen Stage 1 reconstruction reference"
  },
  "contract_version": "stage2-photometry-continuity-reference-v2",
  "evaluation_identity": "documented-synthetic-selection",
  "methods": {
    "gate01_posthoc_calibrated_identity": {
      "lpips": 0.091186,
      "nrmse": 0.323585,
      "ssim": 0.899374
    },
    "raw_identity": {
      "lpips": 0.096859,
      "nrmse": 0.595189,
      "ssim": 0.875809
    },
    "stage1_reconstruction_ceiling": {
      "lpips": 0.03606,
      "nrmse": 0.131321,
      "ssim": 0.965984
    }
  },
  "provenance": {
    "extraction": {
      "gate01_posthoc_calibrated_identity": "overall.methods.calibrated_identity",
      "raw_identity": "overall.methods.raw_identity",
      "stage1_reconstruction_ceiling": "overall.methods.stage1_reconstruction_ceiling"
    },
    "semantics": "external continuity only; not recomputed on Variant-A paired cases",
    "source_contract_version": "stage2-gate01-equal-photometry-v2",
    "source_result_sha256": "53078368ff1fdbbdd0712f2fe5ad5d8ec7ec4b69318f3558ee5f4c7f9d773b7f"
  },
  "source_result_sha256": "53078368ff1fdbbdd0712f2fe5ad5d8ec7ec4b69318f3558ee5f4c7f9d773b7f"
}
```

The loader rejects method, metric, provenance, artifact-hash, and source-file mutation.
These values are external continuity evidence, not new same-case observations.

## 4. Seal and evaluate a paired selection

Variant A cannot create a scientific reference pair by matching unrelated retrospective
subjects. Evaluation therefore consumes a separately sealed, genuinely paired manifest.
Every case must have the same canonical subject-group identity at both endpoints, the
same contrast, and different fields. The manifest loader applies the same cohort
classifier used by fitting and qualification and rejects contradictory case prefixes,
metadata prefixes, cohort labels, or subjects. Its authorization provenance must explain
why the pairing is permitted. Creating pseudo-pairs by domain alone is forbidden.

The `stage2-photometry-paired-evaluation-manifest-v1` manifest seals:

- the canonical split file, membership/recovery fingerprint, and their hashes;
- the fit-artifact and resolved-config hashes;
- an opaque selection identity and pairing-authorization provenance;
- exact source and target identities, source-content identities, cohort/subject/domain
  labels, shapes, and both file-byte and loaded-array SHA-256 values;
- optional same-case Stage-1 reconstruction identities and hashes;
- exactly the official `nrmse`, `ssim`, and `lpips` metrics and the preregistered raw
  nRMSE `> 1.0` catastrophic boundary.

`P_t(N_s(x_s))` and the support are computed without consulting target values. The
target enters only the subsequent metric and continuity-control calls. The single
source-derived support mask is applied bitwise to source, target, raw identity,
fixed-map identity, and optional Stage-1 reconstruction before any same-case comparison.
No target statistic changes the mask, map, interpolation, clamping, or scaling.

```powershell
fieldbridge eval-stage2-photometry-baseline `
  --artifact <EXTERNAL_ARTIFACT_JSON> `
  --manifest <EXTERNAL_PAIRED_EVALUATION_MANIFEST_JSON> `
  --continuity-reference <EXTERNAL_CONTINUITY_REFERENCE_JSON> `
  --gate01-result <EXTERNAL_GATE01_RESULT_JSON> `
  --out-dir <EXTERNAL_VARIANT_A_EVALUATION_DIRECTORY> `
  --resume `
  --device cuda `
  --log-every 10
```

The evaluator writes one atomically sealed case shard per completed case and a final
atomically sealed result. Existing incompatible files are never overwritten. `--resume`
hash-verifies the manifest, run contract, every shard, and a completed final result
before reuse. Interrupted temporary files are cleaned up by the atomic writer.

Same-case reductions contain only:

1. `fixed_map_factorized_identity` - fixed `P_t(N_s(x_s))`;
2. `raw_identity` - the untransformed source on the same source support;
3. optional `stage1_reconstruction_ceiling` - only when a same-case reconstruction is
   supplied and hash-verified in the paired manifest.

They report exact official metrics per case, equal-case per-domain summaries,
equal-domain per-contrast summaries, an equal-domain macro summary, and catastrophic
versus ordinary strata. Unthresholded intensity-to-source, edge-to-source/target, and
gradient-to-source/target continuity controls are reported at case and reduction levels.
All source, target, mask, loaded-array, split, membership, artifact, config, code,
metric-runtime, and pairing-authorization provenance is machine-readable.

`gate01_posthoc_calibrated_identity` and its historical raw identity and Stage-1 ceiling
remain in the top-level external continuity track only. They are never inserted into
same-case reductions and are never relabelled as newly observed Variant-A measurements.
The two calibration semantics remain distinct: Variant A uses fixed training-only maps;
Gate 0.1 uses its unchanged posthoc prediction-CDF diagnostic. Scientific promotion is
an external decision against both baselines, not an evaluator pass/fail result.

## 5. Deterministic boundary for a later Variant A.5 audit

This PR deliberately does not write canonical volumes, call a VAE, or build a latent
bank. Its in-memory `SourceCanonicalizedVolume` exposes the exact canonical tensor and
frozen boolean source support deterministically, which is sufficient to define the
smallest later exporter without changing `N_d` or its mask semantics.

A separately reviewed Variant A.5 may add `stage2-canonical-volume-v1`. Each external,
atomic, resumable record would seal the fit-artifact, resolved-config, split-file, and
membership hashes; record and subject-group identities; source-content identity; source
domain; tensor shape and dtype; canonical tensor byte SHA-256; support shape, dtype,
nonzero count, and canonical byte SHA-256; output file hash; code provenance; and a
deterministic resume key. It must contain no target data or target-derived statistic.
The exporter must prove that its saved tensor and mask are byte-identical to the
in-memory `normalize_source` result. VAE inference and any latent-bank contract remain
outside both Variant A and this interface definition.

## External execution checklist

Before any prospective evaluation:

- record the reviewed Git commit and resolved config hash;
- fit only on canonical-split `R/train` and qualify only on `R/validation`;
- verify all 15 domains and sealed equal weights;
- independently review scaling sensitivity and every worst-domain result;
- require `canonical_latent_bank_authorized: true` before considering later latent
  work;
- use only a sealed genuinely paired evaluation manifest; never match unrelated
  retrospective subjects to manufacture evaluation endpoints;
- retain the original Gate 0.1 result and its hash unchanged;
- keep all data, checkpoints, arrays, artifacts, reports, and output paths external;
- freeze architecture and hyperparameters before any competition-permitted final
  training or final evidence is touched.

Variant B and all later learned components require separate authorization.
