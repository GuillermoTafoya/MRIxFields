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
starts with a standard-library-only `nvidia-smi` gate. It requires one visible NVIDIA
A100, at least 79 GiB total and 75 GiB free memory, and an exact sealed implementation
pin before it can clone code, invoke pip, download weights, mount Drive, inspect the
bank, or load any evidence. A failed gate prints a no-action receipt and stops.

The CUDA stack is not reinstalled. The sealed environment requires Python 3.12.13,
PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126, and the PyTorch CUDA 12.6 runtime. This
pair follows the PyTorch 2.8/torchvision 0.23 compatibility release and the Colab
Python-3.12/PyTorch-2.8 runtime announcement. Every smaller notebook-installed package
is exact-version locked in
`notebooks/stage2_step200_inference_audit_dependency_lock.json`; installation uses
`--no-deps` so dependency solving cannot expand beyond that closed inventory. The
complete resolved distribution environment and lock-file SHA-256 are sealed.

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
