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
- Gate 0.1 continuity reference: `stage2-photometry-continuity-reference-v1`;
- method-neutral result: `stage2-photometry-dual-baseline-result-v1`.

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
domains. Prospective records, validation/test records, and travellers 0006, 0007, and
0009 are rejected by the fit contract.

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
- a contrast-identity control;
- multiplicative scaling at the sealed factors, reported independently as histogram
  distance and masked SSIM;
- fitted and qualification domain counts and the sealed equal-weight policy;
- raw `D(E(x))` and factorized `P_d(D(E(N_d(x))))` reconstruction on each same
  validation volume.

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

The new evaluator does not rerun or reinterpret Gate 0.1. An external review process
must create a small continuity-reference JSON whose `source_result_sha256` hashes the
unchanged Gate 0.1 result file and whose method values were copied from that reviewed
result. It has this shape:

```json
{
  "contract_version": "stage2-photometry-continuity-reference-v1",
  "evaluation_identity": "<FROZEN_SELECTION_IDENTITY>",
  "source_contract_version": "<REVIEWED_GATE01_RESULT_CONTRACT>",
  "source_result_sha256": "<64_LOWERCASE_HEX>",
  "methods": {
    "gate01_posthoc_calibrated_identity": {
      "nrmse": "<REVIEWED_NUMBER>",
      "ssim": "<REVIEWED_NUMBER>",
      "lpips": "<REVIEWED_NUMBER>"
    },
    "raw_identity": {
      "nrmse": "<REVIEWED_NUMBER>",
      "ssim": "<REVIEWED_NUMBER>",
      "lpips": "<REVIEWED_NUMBER>"
    },
    "stage1_reconstruction_ceiling": {
      "nrmse": "<REVIEWED_NUMBER>",
      "ssim": "<REVIEWED_NUMBER>",
      "lpips": "<REVIEWED_NUMBER>"
    }
  }
}
```

The loader validates the continuity-reference content hash and the referenced Gate 0.1
file hash. Numbers are therefore imported as continuity evidence, not hard-coded as
new observations.

## 4. Evaluate the directed fixed-map identity

The narrow CLI evaluates one same-contrast, cross-field case from external NumPy
arrays. The target is used only after prediction construction to compute metrics. The
transform and exact source mask are fully determined before the target is observed.

```powershell
fieldbridge eval-stage2-photometry-baseline `
  --artifact <EXTERNAL_ARTIFACT_JSON> `
  --source-array <EXTERNAL_SOURCE_NPY> `
  --target-array <EXTERNAL_EVALUATION_TARGET_NPY> `
  --source-field <SOURCE_FIELD_T> `
  --source-contrast <CONTRAST> `
  --target-field <TARGET_FIELD_T> `
  --target-contrast <SAME_CONTRAST> `
  --case-id <OPAQUE_CASE_ID> `
  --selection-id <FROZEN_SELECTION_IDENTITY> `
  --continuity-reference <EXTERNAL_CONTINUITY_REFERENCE_JSON> `
  --gate01-result <EXTERNAL_GATE01_RESULT_JSON> `
  --out <EXTERNAL_VARIANT_A_RESULT_JSON> `
  --device cuda
```

The result preserves four distinct method identities:

1. `fixed_map_factorized_identity` — fixed `P_t(N_s(x_s))`;
2. `gate01_posthoc_calibrated_identity` — unchanged prediction-CDF diagnostic;
3. `raw_identity` — continuity reference;
4. `stage1_reconstruction_ceiling` — continuity/upper-bound reference.

The contract forbids averaging, substituting, or implying equivalence between the two
calibration semantics. Promotion requires external application of the reviewed
dual-baseline rules; this CLI does not declare a learned-candidate promotion.

## External execution checklist

Before any prospective evaluation:

- record the reviewed Git commit and resolved config hash;
- fit only on canonical-split `R/train` and qualify only on `R/validation`;
- verify all 15 domains and sealed equal weights;
- independently review scaling sensitivity and every worst-domain result;
- require `canonical_latent_bank_authorized: true` before considering later latent
  work;
- retain the original Gate 0.1 result and its hash unchanged;
- keep all data, checkpoints, arrays, artifacts, reports, and output paths external;
- freeze architecture and hyperparameters before any competition-permitted final
  training or final evidence is touched.

Variant B and all later learned components require separate authorization.
