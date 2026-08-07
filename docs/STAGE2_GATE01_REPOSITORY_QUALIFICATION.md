# Stage 2 Gate 0.1 repository qualification

Date: 2026-08-05

## Qualified base

- Remote: `origin` (`GuillermoTafoya/MRIxFields`)
- Qualified `origin/main`: `5a0a95cc59c4135eb4c8fe3a25ee9404ae2456e4`
- Stability check: two pruned fetches returned the same `origin/main` SHA.
- Owner worktree: clean and preserved on its existing Stage-1 feature branch.
- Implementation worktree: disposable branch created directly from the qualified SHA.

## Remote findings

Main already contains the Stage-2 v1 latent transport, latent-bank and full-volume
evaluation scaffold, source-pinned official Task-3 metrics, field-to-field coupling,
and resplit utilities through merged PRs 26, 27, and 29.

The following work remains branch-only:

- SB-v2 global nearest-neighbour coupling and identity-start transport;
- leakage-safe traveller/resplit guards;
- the Gate-0 affine, intensity, target-direction, and residual diagnostics;
- the Gate-0 experiment log and notebook.

Frozen Gate-0 commit `d3476b900866019b428d52d01a6d5b26b93ca65d` is reachable from
`origin/experiment/stage2-gate0-diagnostic` and is not reachable from `origin/main`.
The only open PR during qualification was draft PR 28, unrelated Stage-1 gradient
stability work. No open PR implemented the frozen Gate 0.1 contract.

## Histogram semantic audit

The existing `ImageIntensityBaseline` histogram operation is a post-hoc
prediction-CDF-to-frozen-target-CDF projection. At application time it computes quantiles
from the supplied image and maps them onto a target-domain foreground distribution fitted
from training records. It is operation B in the scientific plan, not a source-image-only
source-to-target transform (operation A).

Gate 0 applied this operation only to identity. Its v1 API also maps exact-zero background
and does not fail closed on all Gate 0.1 provenance conditions. PR A therefore preserves
the v1 API and arithmetic for backward compatibility while introducing a separately
versioned strict post-hoc calibrator for the equal-photometry comparison.

## Conflict and data-governance check

No newer remote work conflicts with the Stage-2 scientific plan, and no duplicate Gate 0.1
implementation was found. No tracked forbidden MRI, checkpoint, tensor, archive, or model
artifact extension was present on the qualified main tree. Frozen artifacts and private run
outputs remain external and are represented only by hashes and contract metadata.
