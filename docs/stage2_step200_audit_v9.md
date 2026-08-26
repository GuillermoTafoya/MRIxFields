# Stage-2 step-200 audit v9

This audit exposes the behavior of the sealed 200-step full-objective pilot before any
resource-bounded long-run decision. It does **not** declare convergence, scientific
success, or population/generalization performance. It cannot authorize or launch
training.

## Sealed evidence

- Training implementation: `82633d66e5ea47f96b149ea22cc192fcf4526f06`.
- Recovery implementation: `d3949c3591dfb5d1b5270b92af78360bf73e18aa`.
- Step-200 checkpoint SHA-256:
  `09b157d7d9b214816693a8d522d7fa9e8a75d8f08254ed2715bfb8fc13795021`.
- Run fingerprint:
  `c814c948a5b85bd3a694db7c8e074894e97c16a96a36acbfa6f370faf2dac0aa`.
- P:0006 remains development/model-assessment evidence only. It is never used for
  fitting, early stopping, checkpoint selection, or hyperparameter optimization.
- P:0007 is absent. P:0009 remains frozen and unused.

The checkpoint is a batch-1, BF16, full-volume, four-step-Heun pilot with the SB,
identity, anatomy, graph, adversarial, and domain objectives enabled. The CPU report
keeps the independent step-20 and step-200 pilots separate; it does not join them into
one trajectory.

## CPU pilot-evidence audit

Open `notebooks/stage2_step200_pilot_audit_colab.ipynb` in Colab, select **CPU
High-RAM**, then choose **Runtime → Run all** and authorize the Drive mount. The
notebook installs or downloads nothing. It hashes and validates the existing
checkpoint, JSONL history, resolved configuration, embedded pilot report, selection
receipt, validation plan, A100 qualification receipt, and recovery receipt.

It does not restore the latent bank, open patient arrays, use a GPU, or run inference.
Its CSV uses an explicit numeric/domain-label allowlist: record, subject, patient, case,
path, and filename fields cannot enter the output. The generated engineering checks
are predeclared and do not constitute a convergence assessment. Because only terminal
complete unpaired R validation exists, the report explicitly states that neither a
validation trend nor early-stopping behavior can be inferred.

Outputs are written atomically to a new namespace keyed by the training commit,
checkpoint SHA, and audit implementation commit. An existing complete manifest is
verified exactly and reused; incompatible or partial output fails closed.

## A100 inference-only visual audit

Open `notebooks/stage2_step200_inference_audit_colab.ipynb`, attach an **NVIDIA A100
80 GB** runtime, then choose **Runtime → Run all** and authorize Drive. The notebook
starts with a standard-library-only `nvidia-smi` gate. It requires one visible
`NVIDIA A100-SXM4-80GB`, at least 79 GiB total and 75 GiB free memory, and an
exact sealed implementation pin before it can clone code, invoke pip, download
weights, mount Drive, inspect the bank, or load any evidence. The gate records the
driver and prints a sanitized no-action receipt on failure.

Dependency-lock v2 accepts only the authenticated fresh-Colab profile: CPython with
the `cpython-313` ABI and Python major/minor 3.13, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, and the PyTorch CUDA 12.8 runtime. The owner-observed
Python 3.13.15 patch and NVIDIA driver 580.82.07 are recorded as provenance;
patch/driver variation is reported but does not authorize a different Python ABI or
CUDA stack. The former Python-3.12/PyTorch-2.8 profile is retained only as explicitly
unqualified provenance and is never accepted.

PyYAML 6.0.3, matplotlib 3.10.0, nibabel 5.4.2, NumPy 2.1.3, scikit-image
0.25.2, and SciPy 1.16.3 must already match exactly and are never reinstalled.
The sole allowed package installation is the missing `lpips==0.1.4` wheel from
PyPI, pinned to SHA-256
`fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e`.
Pip runs noninteractively with `--no-deps`, `--only-binary=:all:`, and
`--require-hashes`; a wrong preinstalled LPIPS version fails instead of being
replaced. Environment-provenance v2 records the complete installed environment as
a deterministic PEP-503-normalized multimap. Duplicate metadata for an unlocked,
unused package is reported under `unlocked_distribution_ambiguities` and does not
authorize or block the audit. Every metadata observation for the locked numerical
closure is still enumerated: conflicting versions fail, and same-version duplicates
are accepted only when the active imported module is owned by matching metadata.
The exact observed Python patch and driver, active imports, install decision, wheel
identity, source-root hashes, and lock-file SHA-256 are sealed without exposing
installation paths.

LPIPS bootstrap is transactional in the local implementation-scoped directory
`/content/stage2_step200_audit_v9_bootstrap/implementation_<commit-prefix>`; it
never writes a bootstrap receipt to Drive. An absent LPIPS distribution installs
the exact wheel once and atomically seals the receipt. An exact 0.1.4 installation
with a valid receipt is reverified and reused without pip or network. Exact 0.1.4
without a receipt is treated as an interrupted attempt and receives one forced,
hash-locked, no-dependency reinstall before sealing. A different version, altered
receipt, changed lock/implementation identity, shadowed import, or changed installed
package tree fails closed.

After those gates, but still before Drive access, the notebook constructs exactly one
frozen AlexNet LPIPS evaluator. The torchvision
`AlexNet_Weights.IMAGENET1K_V1` URL and full file SHA-256, the packaged LPIPS-v0.1
learned linear-weight SHA-256, and a canonical hash over every final parameter/buffer
name, shape, dtype, and raw byte are recorded. The evaluator must remain in evaluation
mode, is reused by all seven methods and all 60 cases, and is authenticated again at
the end. Network access is blocked throughout the one-case gate and 60-case loop.

Only then does the notebook mount Drive. It uses the stable
`/content/stage2_gate01_recovery_v8_scratch` cache and therefore verifies and reuses an
exact existing archive/bank when present.

Immediately after Drive mount and before any 12.9 GB bank copy or extraction, the
operator authenticates the reviewed Gate01Private direct child `stage1-run-c.yaml` as
the frozen Stage-1 Run C VAE configuration. Authorization uses its exact 4,290 raw bytes
and raw-file SHA-256
`55921ffc53bac074883b66d368051589dd3cc3f2ce5c8e2cc1d304be4245888f`;
the repository example is not an operational substitute even when its reviewed Git bytes
are identical, because the exact `stage1-run-c.yaml` role basename is also required. Only
after this role and raw-file gate passes is the YAML parsed. Its canonical hash is derived
provenance and never authorizes the configuration. After verified bank restoration, the
operator independently requires the bank manifest to name that same raw config identity
and requires both the VAE checkpoint file and manifest identity to equal
`74132b9c514bb91b86d8eb43c63542780bce11304e31e67d3bf75c90ff5d4d79`
before any checkpoint is deserialized or model is constructed. Audit protocol, run, and
artifact-manifest contracts are v3 because no v2 owner audit output was produced before
this provenance boundary was corrected.

The operator revalidates the complete step-200 evidence, the full checkpoint container,
the frozen Stage-1 VAE, the R-only photometry/bank identities, the P:0006 protocol, and
evaluation-readiness evidence. It then performs an exact one-case inference memory and
runtime gate. Any incompatible identity or a peak allocation above the sealed gate
limit aborts before the 60-case audit.

The authentic A100 qualification receipt retains its producer-defined 15-field schema;
it does not contain a selection-rule field. Its run fingerprint is independently pinned
to `502a0989591cad3d09841d7deec841d41ac000d4c9e4ff314b53a8d4067ba5d7`.
The selection rule remains independently checked against the step-200 selection receipt
and checkpoint.

The audit constructs only the unified generator and frozen VAE encoder/decoder. It
does not construct a critic, optimizer, scheduler, or scaler. Execution uses
`torch.inference_mode()`, frozen model parameters, deterministic per-case seeds, BF16,
and the sealed four-step Heun transport contract. The complete checkpoint is verified
before only the required generator state is extracted.

The P:0006 interface yields one verified tensor-bearing case at a time and refuses to
advance until the consumer explicitly releases it. The audit retains only aggregate
numeric metrics and frozen two-dimensional slices. It creates one immutable receipt per
case, verifies and skips completed receipts on exact resume, and never persists or
accumulates full-volume predictions. Corrupted receipts fail closed.

The frozen montage and inference plan is sealed before inference. Its case choice,
planes, relative slice positions, display order, graph paths, field sweeps, normalization,
and seed derivation do not inspect target tensor values or result metrics. The existing
method-agnostic Gate 0.1 calibrator is used only under its reviewed R-training-derived
contract; no P:0006 target fits or selects calibration.

Reports contain descriptive per-case, by-contrast, by-directed-field-pair, and overall
metrics plus paired descriptive differences and win/tie/loss counts. They do not report
population p-values or confidence claims because all 60 directions belong to one
prospective traveller protocol.

Visible progress is count-only. The final receipt reports LPIPS initialization time,
one-case wall time, the projected 60-case inference time, peak allocated/reserved CUDA
memory, whether a dependency or AlexNet weight download was observed, and
`training_invoked: false`. Resume reauthenticates and skips immutable case receipts;
runtime provenance observations from the first sealed attempt remain byte-identical.

The final line is always:

`STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION`

No audit output modifies the immutable training, recovery, Gate01Private, or bank
namespaces. The inference audit writes only to a new namespace keyed by training commit,
checkpoint SHA, P:0006 protocol SHA, and audit implementation commit.
