# Stage-2 Gate 0.1 legacy-inventory recovery

This is an operator-only recovery for the completed Stage-2 v7 qualification and pilots. The
training evidence remains pinned to commit
`82633d66e5ea47f96b149ea22cc192fcf4526f06` and to the external read-only namespace
`UnifiedStage2_1ca2b4a_01/stage2_unified_v7/bank_8081ce89a0ea/implementation_82633d66e5ea`.
The operator commit is pinned separately by the sealed Colab notebook. This recovery does not
train, tune, select a checkpoint, or authorize the 100k-step run.

## Gate 0.1 archive contract

Pass `/content/drive/MyDrive/MRIxFields2026/Gate01Private_8012a3f` as the logical root. Never pass
its `archive/` child. Exactly one of these layouts is accepted:

- Modern flat: `sha256-inventory.csv` and all required artifacts are direct root children.
- Reviewed legacy: `archive/sha256-inventory.json` contains exactly 14 reviewed rows. Normal rows
  resolve only to the lexical suffix after their single `Gate01Private_8012a3f` path component.
  The sole exception is the exactly pinned historical `split_v3.json` row, which resolves only to
  `archive/split_v3.json`.

The legacy inventory must be a nonempty JSON list of exact `path`, `sha256`, and `size_bytes`
records. Digests are lowercase SHA-256 and sizes are nonnegative JSON integers, excluding
booleans. Every row is verified. Stored absolute paths are never opened or searched. The exact
validated root-relative suffix is the inventory identity, so duplicate basenames in different
directories are legitimate and cannot substitute for one another. Duplicate or case-colliding
relative paths, traversal, ambiguous separators, symlinked files or parents, special files,
missing files, size/hash changes, missing scientific contracts, ambiguous layouts, and the wrong
Gate 0.1 result identity all stop the recovery.

`colab-operational-source-split.json` and
`gate01-reviewed-module-sha256-8012a3f.json` are not inventory members. They are separately pinned
direct-root supplemental dependencies, with their exact observed sizes and SHA-256 values. The
canonical split loader verifies the operational split and its frozen bank-membership linkage; the
reviewed module document must be the exact five-field current-versus-previous comparison object.
Both maps cover the same 31 scientific modules, the declared change set must equal the computed
digest differences and the sole reviewed change is `flow_transport.py`. Only the current commit
and current map may match and authorize the Gate lock; the previous commit and map are provenance
only.
Their identities are recorded separately and never enter normalized inventory arithmetic.

The metadata-only preflight validates this graph immediately after Drive mount. It opens no
patient array payload and runs before bank restoration, checkpoint loading, any subprocess, and
any GPU work. The resulting P:0006 protocol v4 records exact relative paths, resolution rules,
normalized inventory provenance, and separate supplemental identities without changing the
existing development/model-assessment-only meaning. Valid v2 and v3 protocols retain their
published loading semantics.

## Persisted Stage-2 topology and bank restore

The exact output root is
`/content/drive/MyDrive/MRIxFields2026/UnifiedStage2_1ca2b4a_01`. The Stage-2 v7 evidence is below
its `stage2_unified_v7/` child, and the pair-feasibility receipt is the direct child
`stage2_retrospective_pair_feasibility_v2.json`. No alternate locations are searched.

The reusable bank is the direct child `photometry_factored_latent_bank_v2.tar`, not a Drive
directory. An early local-capacity preflight reserves space for the tar, the 12.8 GB extracted
tree, and a fixed safety margin. A CPU High-RAM runtime is recommended and the GPU is never used.
The Drive tar is streamed once into a unique local no-clobber attempt while its SHA-256 is
calculated from that same read stream. Copy progress is flushed at least every 30 seconds or
512 MiB. Before extraction its SHA-256 must be
`78d323c02ceccdfcb054307da3c9e14575210869d22cade6c5ecd4afa4baf8d5`. Safe extraction to a unique
local partial attempt uses only that verified local tar and rejects absolute paths, traversal,
links, special members, duplicates, overwrites, and ambiguous archive roots. Extraction and
complete-tree verification emit flushed byte/file counts without private identities or paths.
Publication to the local final directory occurs only after the complete extracted identity
matches all of:

- tree SHA-256 `f9cb09bfa177a3e389f87f087b0d756a2709e2054559a39c85e8272d5e1cfaa3`;
- bank artifact SHA-256 `8081ce89a0eac1522b4fb28cd7919de4a4ecf1d5af72552d141a0ee9b9944194`;
- 3,312 regular files; and
- 12,873,486,620 total file bytes.

The sibling `photometry_factored_latent_bank_v2/` directory is ignored only when it is empty. A
nonempty, linked, or non-directory entry at that unreceipted path is an ambiguity and fails
closed. A fresh-Colab import preflight verifies NumPy, SciPy, PyTorch, PyYAML, the pinned
FieldBridge checkout, and its CLI before Drive mount; the notebook never installs packages.
P:0006 import streams one full-volume case at a time, retains only hashes and graph counts, and
releases the case and calibrated tensor before advancing the manifest. Recovery re-verification
uses the same one-case streaming contract and returns no evaluation or baseline tensors; the
public materializing loader remains available only to later evaluation consumers. Import and
re-verification print start, per-case, and end receipts containing counts only. Before the import
subprocess starts, the parent performs ordinary Python garbage collection after completed-evidence
verification. A filtered subprocess failure prints only its operation, return code, and the
immutable archived log path and SHA-256. A negative return code is recorded as a signal number and
name; signal 9 is classified only as an external termination/resource-kill candidate, never as an
OOM without kernel evidence, and is not retried automatically.

`Runtime -> Run all` is safely repeatable in the same runtime. Each seal uses an
implementation-commit-scoped checkout directory, so a new seal can clone beside an older exact
checkout without touching it. An existing checkout for the current seal is reused
only when read-only Git probes prove that it is the exact clean detached operator commit, has
only the pinned `origin`, descends from the training-evidence commit, and retains the restricted
operator-only diff. The rerun never fetches, checks out, deletes, resets, cleans, or otherwise
mutates an existing checkout. Any mismatch fails closed. Exact local tar/bank and published
recovery artifacts are reverified; partial copy/extraction attempts remain immutable, and
transient copy/free-space observations are excluded from the exact sealed recovery receipt.
Deterministic missing-path, hash, size, schema, containment, ambiguity, and scientific-contract
failures execute once. Only genuine Drive/FUSE I/O, stale-mount, disconnected-transport, and
timeout failures receive bounded retry/remount handling.

## Completed-evidence reuse

The recovery verifier reads unknown checkpoint hashes and the run fingerprint from the sealed
selection receipt, history, and CPU-loaded checkpoints. It requires the observed step-200 receipt
file hash, final validation-plan hash, selection-rule hash, exact completed histories, matching
bank/config/run identities, a passing A100 and anatomy qualification, an unchanged decoder, and
all six qualified objective terms. P:0006 did not enter training or selection. Missing or
incompatible evidence is an error and cannot fall back to training.

Recovery outputs use a distinct namespace containing both the training-evidence and operator
commit prefixes. Existing training attempts are read-only. Publication uses immutable attempts,
Drive retries, hashes, and no-clobber writes. All long-run authorization flags remain false, and
`Runtime -> Run all` stops after P:0006 import and evaluation-readiness sealing for a separate
resource-bounded training-design review.

Scientific limits remain unchanged: R-only data trained the model; current pilot selection is
unpaired; P:0006 is development/model assessment only; population/generalization claims remain
unauthorized; P:0007 and P:0009 are not imported; P:0009 remains frozen; descriptor coupling is
disabled; and no learned-disentanglement or StarGAN-control claim is made.
