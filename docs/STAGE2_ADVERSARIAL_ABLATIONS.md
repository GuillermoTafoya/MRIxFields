# Stage 2 SB + adversarial training

## Scientific question

This experiment keeps Schrödinger-bridge matching as the generator objective and asks
whether a StarGAN-style discriminator adds target-domain fidelity. The discriminator has
a hinge real/fake head and a 15-class `(contrast, field)` head. It never creates pairs:
real targets and generated endpoints come from the same frozen latent bank and the existing
same-contrast, cross-field sampler.

The primary run discriminates standardized latents. The image-space ablation passes those
same latent endpoints through the frozen Stage-1 decoder; gradients reach the translator but
the decoder is never optimized. Do not mix original and photometry-factorized banks in one
run. If Variant A authorizes a canonical bank, repeat the complete board on that bank.

## Frozen board

Use the same bank, split, seed, step count and evaluation protocol for every row. Give each
run a distinct checkpoint directory (shown below as `<OUT>`). The proposed weights are
starting hypotheses and must not be tuned on prospective test evidence.

| Run | Purpose | Overrides |
| --- | --- | --- |
| `sb_adv_latent` | Primary: SB + latent realism + target domain + identity | none |
| `sb_only` | Establish the marginal value of all adversarial terms | space/weights off |
| `sb_adv_image` | Test whether decoded-image feedback beats latent feedback | image space |
| `sb_adv_no_domain` | Isolate the 15-domain classifier | domain weight 0 |
| `sb_adv_no_identity` | Check whether same-domain anchoring prevents drift | identity weight 0 |

Common command prefix:

```powershell
$Config = "configs/experiment/stage2_transport_sb_adversarial_v1.yaml"
$Bank = "<EXTERNAL_LATENT_BANK>"
```

Primary:

```powershell
fieldbridge train-stage2-transport --config $Config --bank-dir $Bank `
  --checkpoint-dir <OUT>/sb_adv_latent --val
```

Pure SB:

```powershell
fieldbridge train-stage2-transport --config $Config --bank-dir $Bank `
  --checkpoint-dir <OUT>/sb_only --adversarial-space none `
  --adversarial-weight 0 --domain-weight 0 --identity-weight 0 --val
```

Image discriminator (the VAE config and checkpoint must be the exact frozen producer of
the bank):

```powershell
fieldbridge train-stage2-transport --config $Config --bank-dir $Bank `
  --checkpoint-dir <OUT>/sb_adv_image --adversarial-space image `
  --vae-config <FROZEN_STAGE1_CONFIG> --vae-checkpoint <FROZEN_STAGE1_CHECKPOINT> --val
```

Component ablations:

```powershell
fieldbridge train-stage2-transport --config $Config --bank-dir $Bank `
  --checkpoint-dir <OUT>/sb_adv_no_domain --domain-weight 0 --val

fieldbridge train-stage2-transport --config $Config --bank-dir $Bank `
  --checkpoint-dir <OUT>/sb_adv_no_identity --identity-weight 0 --val
```

## Evaluation and interpretation

Select checkpoints by retrospective validation flow loss only, then run the existing
complete-volume traveller evaluation unchanged. Report official nRMSE, SSIM and LPIPS,
identity, Stage-1 ceiling, per-domain reductions, wrong-target controls and runtime/memory.
The discriminator loss is diagnostic, never a promotion metric. Reject an adversarial run
that improves appearance while worsening structural metrics, wrong-target control or
identity behavior.

Checkpoints include translator, discriminator and both optimizer states, so `--resume-from`
is exact for adversarial runs. A pure-SB checkpoint remains backward compatible and contains
only the original translator/optimizer keys.
