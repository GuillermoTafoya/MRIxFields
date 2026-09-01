# Stage-2 rescue diagnostics v1

## Purpose and decision boundary

This suite investigates why target-conditioned transport did not become useful at the
sealed step-200 checkpoint. It does not authorize continuation of training and it does
not promote step-200. The authenticated training evidence remains commit
`82633d66e5ea47f96b149ea22cc192fcf4526f06`; the checkpoint SHA-256 remains
`09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021`.

All fitting and reference distributions use R/train. Predeclared diagnostic assessment
may use R/validation. R has no invented paired endpoint metrics. P:0006 remains a
separate sealed final development/model-assessment step and is not opened here. P:0009
is refused.

## One Run-all flow

1. Open `notebooks/stage2_rescue_diagnostics_colab.ipynb` in a fresh A100 80 GB Colab
   runtime.
2. Choose **Runtime → Run all** and authorize the single Google Drive mount.
3. Do not edit paths or cells. The notebook authenticates hardware, clones the pinned
   implementation commit, checks out detached and clean, mounts Drive, and delegates to
   `notebooks/stage2_rescue_diagnostics_operator.py`.
4. Read the final JSON and Markdown scorecards under
   `UnifiedStage2_1ca2b4a_01/stage2_rescue_2026_09_01/diagnostics_v1`.

The operator derives every input from the reviewed Tafoya layout. Existing bank,
photometry, VAE, checkpoint, and training namespaces are read-only. A rerun verifies
existing self-hashed results and refuses any mismatch; it never clobbers them.

## What each diagnostic answers

| Diagnostic | Falsifiable question |
|---|---|
| Conditioning plumbing | Do source/target labels remain distinct, reach the translator, alter the embedding/output, and carry nonzero gradient without an update? |
| Real-domain identifiability | Can fixed R/train probes distinguish contrast and field on independent R/validation above explicit chance baselines? |
| Same-source five-target sweep | Which differences appear in transported latent, decoded canonical output, frozen renderer-only counterfactual, and final output? |
| Off-manifold drift | Do generated latents depart from the sealed Stage-1 R/train distribution globally or by source field, target field, and contrast? |
| Per-term gradients | For each real six-term loss, are raw/weighted losses and translator/critic gradients finite, nonzero, and mutually conflicting? |
| Synthetic micro-overfit | Can the implementation learn exact identity and known target-dependent transforms in an isolated easy problem? |

The scorecard reports `PASS`, `FAIL`, or `INCONCLUSIVE` for the evidence rows. It always
leaves the architecture verdict to a human and always records that current step-200
promotion is unauthorized. Synthetic micro-overfit is implementation evidence only,
never real scientific evidence.

## Local synthetic validation

No private data, A100 workload, P:0006, P:0009, or scientific training is needed for
development validation:

```powershell
python -m pytest -q tests/test_stage2_rescue_diagnostics.py
python -m pytest -q
python -m compileall -q src tests notebooks
git diff --check
fieldbridge smoke-train --steps 2
```

The reusable conditioning entry point is:

```powershell
fieldbridge diagnose-stage2-conditioning `
  --config configs/experiment/stage2_unified_full_retrospective_v7.yaml `
  --checkpoint <external-checkpoint> `
  --bank-dir <restored-R-only-bank> `
  --expected-checkpoint-sha256 <sha256> `
  --expected-training-commit <git-sha> `
  --out <new-output.json>
```
