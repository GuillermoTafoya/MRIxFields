# Research Status

Snapshot date: 2026-07-21

Repository baseline: `origin/main` at
`dea5c27b9ad738a7f561ac2011ffec8f700b97c1`.

The shared north star is one shared-parameter conditional model that emits complete 3D
MRI volumes. The current tracks test different risks and are intentionally separate.

## Track A: Deterministic Pseudo-Pair Baseline

Status: corrected implementation available; micro-v2, duration-probe, and prospective
paired zero-shot development evidence recorded; a real-paired LOSO feasibility
experiment is preregistered; no confirmatory or promotion claim is supported.

Track A is the 2D axial `ConditionalUNetFieldTranslator` experiment for T2-FLAIR. It
uses retrospective high-field targets, synthetic 0.1T degradation, volume/subject-first
splits, `volume[:, :, :, z]` slice extraction, target-aware background loss, and
checkpoint version 2.

This track is the fastest falsifiable field-conditioning experiment. Its allowed claim
is synthetic-degradation restoration only. It does not establish translation from real
0.1T acquisitions or complete-volume performance.

### Micro-v2 Development Result

The following is user-supplied evidence from a private Colab/Drive run. Repository
maintainers and coding agents do not have access to those private artifacts and have not
independently verified the split file, checkpoint, logs, or metrics.

- Code commit: `8631962f96ea07d1dfe51bbfa486ddac266cb828`.
- Split fingerprint:
  `17f00411ab04331fa0380526b2d8f0cd0173e4ff6f8978f72c61053fa7385dbe`.
- Checkpoint metadata: pseudo-pair pipeline v2, epoch 2, global step 32.
- Runtime report: CUDA and AMP active; engineering gate passed.
- Scientific gate: failed.
- Evidence scope: sampled-slice/per-volume exploratory; `complete_volume: false`.

| Metric | Degraded input | Prediction |
| --- | ---: | ---: |
| Macro nRMSE | 0.054048 | 0.104325 |
| Macro SSIM | 0.874324 | 0.167202 |

No target field improved nRMSE (`0/4`). Correct conditioning had the best mean
selected-slice nRMSE for `1/4` volumes. The mean margin versus the best wrong target was
`-0.00113659`; relative correct-versus-wrong and correct-versus-permuted nRMSE effects
were `0.005750` and `0.002822`, respectively.

This is valid negative evidence for the predeclared micro-v2 viability gates: at the
declared endpoint, prediction was worse than the degraded-input baseline and target
conditioning was not materially discriminative. It is not evidence about complete
volumes, real 0.1T acquisitions, a confirmatory held-out split, or the eventual 3D
translation contract.

The observed split is now development evidence because its test summaries have been
examined. Reusing it for the predeclared duration probe controls the split while testing
one variable, but no result from that reuse is confirmatory evidence.

### Ten-Epoch Duration-Probe Development Result

The following is also user-supplied evidence from the private Colab/Drive run. The
repository does not have access to the duration-probe JSON, split, checkpoint, logs, or
telemetry and has not independently verified them.

- Code commit: `fe02d866deb060863f539cb30c08db608623cd69`.
- Endpoint: epoch 10, global step 160.
- Engineering gate: passed.
- Scientific gate: failed.
- Evidence scope: sampled-slice/per-volume exploratory; `complete_volume: false`.
- Split role: observed development split; not confirmatory evidence.

| Metric | Degraded input | Prediction |
| --- | ---: | ---: |
| Macro nRMSE | 0.05404807 | 0.03838583 |
| Macro SSIM | 0.87432381 | 0.53409448 |

All four fields improved nRMSE. Correct conditioning had the best mean selected-slice
nRMSE for only `1/4` volumes, with mean margin `-0.00278850` versus the best wrong
target. Relative correct-versus-wrong and correct-versus-permuted nRMSE effects were
`-0.03584306` and `-0.04334417`, respectively.

Increasing duration rescued nRMSE relative to the degraded-input baseline, but did not
rescue SSIM or target conditioning. The probe therefore remains a scientific failure
under the predeclared joint viability gates. It does not justify a scaled pilot,
confirmatory claim, complete-volume claim, or real 0.1T translation claim.

### Prospective Paired Zero-Shot Development Result

The following sanitized result is user-supplied evidence from a private Colab execution.
Repository maintainers and coding agents did not access the manifest, subjects, images,
checkpoint file, private paths, or full outputs and have not independently verified the
run or reported values.

- Audit code commit: `a7ac99f40dcaea4811452172d363347997c504e1`.
- Frozen training/checkpoint code commit:
  `e1e526ea5fa0a58f5682823f85a3957d5cc8647c`.
- Evidence scope: `prospective_paired_selected_slice_development`;
  `complete_volume: false`.
- Scope: 3 prospective cases, 4 actual paired target fields, and 8 fixed slices per
  case/field; 96 paired comparisons and 384 conditioned predictions.
- `scientific_thresholds: null`; this execution had no preregistered formal pass/fail
  gate.

| Macro selected-slice metric | Real 0.1T source | Correctly conditioned prediction |
| --- | ---: | ---: |
| nRMSE | 0.0773619951 | 0.0873034413 |
| SSIM | 0.8645790890 | 0.8571345539 |
| Masked MAE | 0.1286291652 | 0.1430917376 |
| Signed foreground bias | 0.0288569646 | 0.0956241177 |

| Actual target field | Source nRMSE | Prediction nRMSE |
| --- | ---: | ---: |
| 1.5T | 0.0549967966 | 0.0544251086 |
| 3T | 0.0619403781 | 0.0562433123 |
| 5T | 0.0502285822 | 0.0603846158 |
| 7T | 0.1422822235 | 0.1781607283 |

Only `4/12` held-out case-field units improved nRMSE. Correct conditioning was worse
than the mean wrong condition in macro nRMSE (`0.0873034413` versus `0.0867206908`),
and the correct requested field had the best aggregate nRMSE only for the true 1.5T
target. Prediction also increased positive foreground bias substantially.

This is negative observed-development evidence: the frozen synthetic residual model did
not transfer reliably to real paired 0.1T inputs, particularly at 5T and 7T, and did not
show useful target conditioning. Because thresholds were null, it is not described as a
retrospective formal gate failure. It is selected-slice evidence, not complete-volume,
held-out-confirmatory, or challenge evidence.

This and other observed Track-A development results cannot satisfy promotion or
final-volume gates.

## Track B: Volumetric Stage-1 Stack

Status: infrastructure implemented; diagnostic v1 completed with negative, non-held-out
engineering evidence; Stage 2 blocked; scientific and challenge evidence pending.

Track B contains the 3D patch-based KL-VAE, resumable patch bank, sliding-window
full-volume reconstruction, and conditional latent diffuser. It addresses volumetric
representation and reconstruction constraints, not the paired synthetic-restoration
question owned by Track A.

The Stage-1 patch bank is not a pseudo-pair dataset. Its current provenance contract does
not establish paired targets, degradation parameters, or pseudo-pair slice geometry.

### Stage-1 Reconstruction Engineering Evidence

The following is user-supplied evidence from a private run. Repository maintainers and
coding agents do not have access to its checkpoint, patch bank, manifest, volumes, logs,
or rendered reconstructions and have not independently verified the artifacts or metrics.

- Patch bank: 1,984 volumes, 63,488 patches, 32 patches per volume.
- Training endpoint: 54,000 steps; reported 3,968 steps per epoch; early stop at
  approximately epoch 13.6.
- Evaluation: five T1W full volumes from the same manifest used to build the training
  bank; therefore not held out or confirmatory.
- Manifest composition: 1,939 retrospective and 45 prospective volumes.
- Generic manifest audit: duplicate case IDs and `ok=false` under the notebook-mutated
  identity mapping.
- Evaluation command: overlap `0.25`, despite notebook prose stating `0.5`.

| Mean full-volume metric | Supplied value |
| --- | ---: |
| nRMSE | 0.48881893 |
| SSIM3D | -0.00149328 |
| LPIPS | 0.62425638 |
| MAE | 0.90241840 |
| MSE | 0.95595611 |

The supplied visual assessment reports retained coarse anatomy alongside gray background,
collapsed intensity distribution, and a regular tile grid. Taken together, this is valid
negative engineering evidence for the evaluated reconstruction path, not evidence from a
held-out subject set and not a basis for Stage 2. It does not isolate whether the dominant
failure is checkpoint reconstruction, latent sampling, normalization/background handling,
or sliding-window inference.

The predeclared diagnostic v1 addressed that isolation without training: it audited the
official manifest, used official `sample_id` as unique volume identity, validated bank and
config provenance, compared latent-mean and sampled patch reconstructions, verified the
identity tiler, and reported full-volume overlap/seam sensitivity at `0.25`, `0.5`, and
`0.75`. It reported every supplied checkpoint step chronologically and did not select a
best checkpoint post hoc.

### Stage-1 Reconstruction Diagnostic v1 Result

The following is user-supplied evidence from the completed private Colab diagnostic.
Repository maintainers and coding agents do not have access to the checkpoint, patch bank,
manifest, volumes, or full output and have not independently verified the artifacts or
reported values.

- Diagnostic code commit: `9b071cc17c545e126891ae77f7e0dd27c2815b1c`.
- Training/checkpoint code commit: `c9ee9dd738f8d9fee7acf9340dc4325c47a639cd`.
- Checkpoint: legacy/unversioned, step 54,000.
- Evidence scope: development engineering diagnostic; `held_out: false`;
  `confirmatory: false`; `complete_volume: true`.
- Manifest fingerprint:
  `a2c49959c14f5ab917425e6e42bad0381259d339e68f03619fba2429d1babe8a`.
- Patch-bank fingerprint:
  `95499028d3baab2a2aa5a53dc648454d25f4fdbce4a51b8c3f32a8a18b877a8e`.
- Resolved-config fingerprint:
  `5462b1890ab2cf55b9f89b76b3db472f8c67a337c2b4ff333094166ec53b1a65`.
- Combined provenance fingerprint:
  `b8a134fe850eca4a2de4173e9bac763384f197afb3c34bffff09850fd6259bfd`.

The identity tiler passed at overlaps `0.25`, `0.50`, and `0.75`. The fixed-patch
direct-versus-tiled check also passed, with mean absolute difference
`1.0026588448397433e-09`. These checks make a tiler implementation failure an unlikely
explanation for the large posterior-mean reconstruction error.

| Step-54,000 fixed-patch path | nRMSE | SSIM3D | LPIPS | Foreground MAE | Outside reconstruction mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Posterior mean | 0.50438815 | -0.05506255 | 0.63492680 | 0.35528192 | 0.07830130 |
| Sampled posterior, seed 13 | 0.05191850 | 0.53982496 | 0.04449676 | 0.09432714 | -0.97147036 |

The target outside mean was `-0.99989337`. For complete-volume posterior-mean
reconstruction, nRMSE ranged from `0.50107300` to `0.50455886` across the declared
overlaps, and SSIM3D ranged approximately from `-0.04117` to `-0.02156`. The overlap
nRMSE span was `0.00348586`; the seam-ratio span was `0.57896406`; and the
posterior-mean reconstruction standard-deviation ratio was `0.49903357`.

The chronological fixed-patch posterior-mean path worsened from nRMSE `0.1342` at step
2,000 to `0.6437` at step 50,000, then partially rebounded to `0.5044` at step 54,000.
No checkpoint is selected post hoc from that trajectory.

This evidence supports an engineering interpretation of a posterior mean/sample contract
mismatch and possible variance-channel information leakage. It does not establish that
sampled decoding is stable or generalizable. Although the diagnostic reconstructed a
complete volume, it used the training manifest and therefore is neither held out nor
confirmatory and does not pass the final-volume scientific gate. Stage 2 was not started
and remains blocked.

## Invalid Evidence

Commit `be8b4792344ac4f8b73112cd9ff6db4298af6b13` predates the pseudo-pair axis and
target-aware background-loss corrections. Checkpoints and reported results produced by
that implementation are pseudo-pair v1 artifacts and are invalid as scientific
evidence. They must not be resumed or compared as if they were v2 results.

## Current Blockers

- Track A's 10-epoch duration probe rescued nRMSE but failed SSIM and conditioning
  gates on the observed development split.
- The frozen synthetic residual checkpoint did not reliably improve real paired 0.1T
  inputs and did not show useful target conditioning in the zero-shot audit.
- The development split has been observed and cannot support a confirmatory claim.
- Track A does not yet reconstruct and evaluate complete held-out NIfTI volumes.
- Track B diagnostic v1 is complete. Its negative, non-held-out engineering evidence
  supports a posterior mean/sample contract mismatch and possible variance-channel
  information leakage, but not stable or generalizable sampled decoding.
- Track B has no held-out or confirmatory reconstruction evidence. A posterior experiment
  has not been implemented, and Stage 2 remains blocked.
- A shared 3D paired/degradation provenance contract does not yet exist between tracks.
- Eventual anatomical metrics require a validated toolchain and documented protocol.

## Next Decision Gate

The scaled pilot remains blocked. The next Track-A development experiment is the
preregistered three-fold, subject-level LOSO real-paired T2-FLAIR feasibility experiment
in `prospective_paired_loso_residual_v1.yaml`. Each prospective case is held out once;
the other two cases provide real 0.1T-to-target training pairs. It compares the unchanged
source, a train-fold-only target-specific affine calibration, identity-initialized paired
training, and paired fine-tuning initialized from the frozen synthetic checkpoint. The
two neural arms share exactly the same architecture, real-paired examples, optimizer,
losses, deterministic sampling, preprocessing, and fixed endpoint; synthetic examples
are not mixed into paired training.

The experiment has no validation loader, early stopping, or held-out checkpoint
selection. It trains on every predeclared brain-support slice, evaluates the frozen eight
slices for direct comparison, and separately attempts every-z-slice reconstruction on
the model and inverse-restored native grids. Complete-volume status is allowed only when
coverage and every inverse geometry are verified. Results remain observed development
evidence because all three prospective cases have already been examined.

The LOSO experiment remains unexecuted. Before private execution, its runtime handoff
recovery was corrected so a global resume validates and skips completed endpoints,
resumes partially completed arms from the last completed epoch, starts never-begun arms
fresh, and fails closed on incompatible or inconsistent artifacts. This is an
orchestration/reporting correction and does not add scientific evidence.

A diagnostic-only prospective paired Track-A audit is now implemented for a frozen
residual checkpoint. It is limited to the predeclared T2-FLAIR cases, target fields, and
eight selected slices in `prospective_paired_zero_shot_v1.yaml`. The runner fails closed
on acquisition multiplicity, physical geometry, fit-pad geometry, and checkpoint
identity; evaluates the real 0.1T source plus correct and every wrong target condition
against each actual paired target; and emits anonymous tables, fixed alignment/error
maps, hierarchical summaries, a conditioning sweep, and a sanitized JSON handoff.
The first private Colab handoff attempt stopped before inference because the validator
incorrectly required historical `training.num_workers` to appear in the checkpoint's
normalized `PseudoPairEpochConfig` payload. `num_workers` is owned by DataLoader runtime
construction, not `PseudoPairEpochConfig`, so legacy checkpoint metadata cannot
independently attest it even though both the historical YAML and run launcher predeclare
zero workers. The corrective validator normalizes the historical YAML through the
trainer's actual serialization contract, allows only explicit runtime/path handling,
and retains exact comparison of the two checkpoint config copies and every serialized
training field. This was a handoff-validation blocker, not model or audit evidence; no
private metric result was produced or recorded. Any future result from this audit is
observed selected-slice development evidence (`complete_volume: false`), not held-out or
confirmatory evidence, and cannot unblock the scaled pilot by itself. No subsequent
training experiment is implemented.

Track B remains at the volumetric reconstruction gate. No new VAE training, posterior
experiment, or Stage-2 work is authorized by the diagnostic v1 result in this snapshot.

## Artifact Location Policy

Real manifests, MRI data, checkpoints, split/run outputs, credentials, and
machine-specific absolute paths remain outside Git. This status document records
identities and decisions, not private storage locations.
