# Stage 2 — Gate 0 diagnostic

**Date:** 2026-08-03
**Config:** [`configs/experiment/stage2_gate0.yaml`](../../configs/experiment/stage2_gate0.yaml)
**Code:** `src/fieldbridge/models/translators/affine_baseline.py`,
`src/fieldbridge/evaluation/stage2_gate0.py`
**Notebook:** [`notebooks/stage2_gate0_diagnostic_A100.ipynb`](../../notebooks/stage2_gate0_diagnostic_A100.ipynb)
**Bank:** `runC_ep015_step047700_74132b9c_full_bf16` — `strategy_used: ["full"]`,
VAE sha256 `74132b9c…`, roundtrip SSIM3D 0.9735, counts 1590/189/205
**Split:** `split_v4_111` regenerated after the fingerprint fix — 0007 train / 0006 validation /
0009 test. **0009 untouched.**

## Revision, 2026-08-05 (post-review)

The question this gate answers was sharpened: not "can Stage 1 reconstruct" (run C settled
that) but **is Stage 2 learning conditional field translation from unpaired distributions, or
mostly an intensity/statistical shortcut?** Four corrections followed, and they change what the
results below may be used to claim:

1. **The latent affine alone cannot test the shortcut hypothesis.** Its closed form is exact
   only under a 1-D Gaussian marginal approximation per channel, while the channels are
   spatial, non-Gaussian and decoded nonlinearly. So `latent affine ≈ SB` is strong evidence of
   a shortcut, but `latent affine ≠ SB` rules out only a per-channel latent affine. Image-space
   baselines were added — robust-affine and full histogram matching applied to the identity
   reconstruction, fitted from training subjects only. Histogram matching is the strongest
   purely photometric map available, so anything a model gains over it is not photometric.
2. **Responsiveness is not direction.** The sweep now also assigns each output to the nearest
   of that subject's five real target latents (chance 1/5), reporting correct-target fraction,
   rank of the requested field, and the margin against the wrong domains.
3. **The cross-subject residual cosine is descriptive, not a ceiling.** Registration error and
   anatomical variation attenuate a voxelwise cosine even when a transferable effect exists. It
   is reported as a reference scale, excluded from the decision rule, and the verdict is
   explicitly one Gate-0 input rather than a standalone decision.
4. **`sb_v2_minus_affine` is a diagnostic decomposition, not a model variant** — renamed
   accordingly. It can leave the latent manifold, so its metrics are not a claim about any
   deployable system and it is never promoted on its score. `ssim_robust` likewise stays
   labelled diagnostic and is never substituted for official SSIM.

Also added: per-contrast coupling-quality measurement in the coupling's own descriptor space,
an assembled per-contrast evidence table, per-pair resumable decoding, a provenance contract,
and a `RUN_EXPENSIVE_REFERENCE` gate so the A100 step is turned on deliberately.

The step-2 and step-5 numbers below are unchanged by this revision; their **interpretation** is
narrowed as described.

## What this gate asked

v2 scored held-out 0006 at nRMSE 0.459 / SSIM 0.880 against identity 0.595 / 0.876 and a frozen
VAE ceiling of 0.131 / 0.966. A large nRMSE move with no SSIM move against that ceiling reads as
a global intensity rescaling rather than structure. Gate 0 tests that reading cheaply, before
committing A100 hours to Gate 1.

Thresholds were fixed in the config before any number was read.

## Execution status

| step | what | status |
|---|---|---|
| 1 | wrong-target sweep | **pending** — needs the SB v2 checkpoint (Colab) |
| 2 | closed-form affine baseline | **done** (CPU, 1575 retrospective train volumes) |
| 3 | four-reference gate + LPIPS | **pending** — needs decode + real NIfTI + the checkpoint (Colab) |
| 4 | robust-normalized SSIM | **pending** — a column of step 3 |
| 5 | residual energy + predictability | **done** (CPU, 120 paired traveller pairs) |

Steps 2 and 5 are latent-only, so they ran locally on CPU against the extracted bank. Steps 1,
3 and 4 need the frozen decoder, the real target volumes and the trained velocity field, none of
which are on this box.

## Step 2 — the closed-form affine barely moves the latent

Per-channel moment matching over the retrospective train pool, per (contrast, source field,
target field). Two tables were fitted in one pass: `all` (every latent voxel) and `foreground`
(above the per-volume median of the across-channel norm).

The fitted per-domain marginals are nearly identical across field strengths. Example, T1w
`all` table, per-channel std:

| domain | per-channel std |
|---|---|
| 0.1T/T1w | 0.360, 0.441, 0.424, 0.413 |
| 1.5T/T1w | 0.362, 0.446, 0.452, 0.432 |

So `a ≈ 1`, `b ≈ 0`: **in latent space there is almost no per-channel affine to apply.**

This is the first substantive finding, and it does not match the premise the gate was built on.
Whatever v2 did to image intensities, it is **not** a per-channel affine on the latent. The
affine reference in step 3 will therefore land close to identity, and it cannot be used to
explain away v2's nRMSE gain. That gain still needs a mechanism, and step 3 is what will name it.

## Step 5 — residual energy and predictability (the decisive step)

120 ordered same-contrast cross-field pairs from the two paired travellers (0007, 0006).
Energies are mean-square in raw latent units.

| baseline | E identity | E residual | explained by affine | E anatomy floor | predictable fraction | % of ceiling |
|---|---|---|---|---|---|---|
| `all` | 0.13766 | 0.13314 | **2.5 %** | 0.27695 | 0.00452 | 3.8 % |
| `foreground` | 0.13766 | 0.13637 | **0.2 %** | 0.28015 | 0.01418 | 11.9 % |

Stratified, `all` table:

| stratum | n | E identity | E residual | explained |
|---|---|---|---|---|
| far field pairs (\|log f_t/f_s\| above median) | 60 | 0.16171 | 0.15677 | 2.7 % |
| near field pairs | 60 | 0.11361 | 0.10950 | 2.3 % |
| T1w | 40 | 0.15027 | 0.14089 | 5.5 % |
| T2-FLAIR | 40 | 0.13701 | 0.13367 | 1.5 % |
| T2w | 40 | 0.12570 | 0.12485 | 0.5 % |

The `foreground` table goes **negative** on three strata (far-field −1.1 %, T2-FLAIR −2.6 %,
T2w −2.6 %): there, applying the affine is worse than doing nothing.

### The floors

- **Storage floor.** The bank holds the deterministic posterior mean, so re-encoding a volume
  returns the identical latent — there is no stochastic encode noise to measure, and float16
  quantization is the only true in-latent floor: 2.55e-08. The residual sits ~5.2e6 above it.
  The residual is nowhere near the storage floor; that check is not what closes this gate.
- **Anatomy floor.** The residual built against the *wrong subject's* target: 0.277. The real
  residual (0.133) is about half of a pure cross-subject anatomy mismatch — same order of
  magnitude, not a small perturbation on top of it.
- **Alignment ceiling.** The predictable fraction is a voxelwise cosine between two different
  brains, so it is capped by how well those brains line up. Measured, not assumed: cross-subject
  same-domain latent cosine is **0.345**, giving a ceiling of 0.119 on any squared-cosine
  statistic. Reported per baseline as "% of ceiling" above.

### The number that decides

One traveller's residual explains **0.5 % (`all`) to 1.4 % (`foreground`)** of the other's,
against a pre-declared bar of 5 %. Normalized by the alignment ceiling that is still only
**3.8 % to 11.9 %** of the transferable signal that could be measured this way.

The shared component is real but tiny: aligned cosine 0.067 / 0.119 versus a control cosine of
−0.005 / −0.007 when the field pair is deliberately mismatched. So there *is* a genuinely
field-pair-specific shared residual — it is just small enough that a model trained to predict it
would be fitting almost entirely subject-specific structure.

## Verdict

**`gate0_close_generative_branch`**, on `median_predictable_fraction` = 0.0045 (`all`) /
0.0142 (`foreground`) against the pre-declared 0.05.

The number that supports it: after the closed-form affine, what is left of the cross-field
latent displacement is **48 % as large as a whole different brain** (0.133 vs an anatomy floor
of 0.277) and only **0.5–1.4 % of it transfers between the two travellers** — 3.8–11.9 % of what
imperfect anatomical alignment would even allow. The cross-field transformation, as this frozen
VAE encodes it, is close to subject-specific.

## Evaluation protocol from here

Develop and debug on training traveller **0007**, with `--allow-training-subjects` and the
result labelled development evidence. Freeze code, thresholds and report format. Then evaluate
**once** on 0006. 0009 stays untouched.

Once a design is locked, the final read is three-fold traveller LOSO with all three
subject-level results reported. With three subjects that is descriptive evidence, not a
population estimate; the safeguard is that no model choice or threshold may change after a fold
is inspected.

## Honest limits of this verdict

- **n = 2.** The predictable fraction is a 1-vs-1 comparison between the only two paired
  travellers available. It is an indicator, not a statistic; it has no confidence interval, and
  it cannot distinguish "no shared field effect" from "two unusually dissimilar subjects".
- **The alignment ceiling was measured after the first run**, once the cross-subject voxelwise
  confound became visible. It is reported and deliberately excluded from the decision rule,
  because moving a pre-declared threshold onto a statistic chosen after seeing the numbers is
  the failure mode the pre-registration exists to prevent. Under either reading the verdict is
  the same.
- **This is a statement about the frozen run-C latent space, not about the task.** A residual
  that is unpredictable in these latents could be predictable in image space, in a different
  latent space, or after per-subject registration. Gate 0 does not test those.
- **Steps 1, 3 and 4 are not in.** If step 3 shows `affine ≈ identity` while `sb_v2` still wins
  nRMSE, then v2's mechanism is something other than the affine hypothesis this gate assumed,
  and that mechanism is still unidentified.

## Repository defect found and fixed along the way

`resplit-travellers` copied the input split's membership fingerprints into its output while
changing membership. Both fingerprints are membership-sensitive, so **every resplit file failed
`load_vae_splits` as "stale or altered"** — including `split_v4_111.json`, the file whose entire
purpose is to give the Stage-2 gate a held-out anchor. Reproduced from `split_v3.json` at HEAD.

Fixed: `promote_subjects_to_split` now drops the stale fingerprints and `resplit_file`
recomputes them through the canonical implementations. Regression tests added
(`tests/test_resplit.py`).

**This has an open implication for the v2 numbers.** `eval-stage2-transport` calls
`load_vae_splits(--split-json)`, so with the pinned code it could not have loaded
`split_v4_111.json`. How the quoted v2 held-out 0006 result was produced is therefore not
established from this repository, and whether 0006 was genuinely held out in that run needs to
be confirmed against the actual run artifact before the 0.459 / 0.880 figures are cited as
held-out evidence.
