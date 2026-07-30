"""Etapa-2 latent transport training (OT-CFM / Schrodinger bridge) on the frozen-VAE bank.

Trains a single shared velocity field ``v_theta(z_t, t, c_source, c_target)`` on precomputed
latents (the VAE stays frozen and is not even loaded here). The ablation ladder is expressed
as config, not new networks:

- ``coupling``: how source latents are paired to target latents.
  ``independent`` (random), ``ot`` (exact minibatch optimal-transport assignment, the
  squared-L2 Hungarian plan — the unpaired coupling the v3.1 plan calls for), or ``nn``
  (global nearest-neighbour retrieval over the *whole* field pool).

  Why ``nn`` exists. ``ot`` assigns within a minibatch, but the field pools hold 37-191
  volumes, so a batch of 8 is a near-random draw and the Hungarian plan over it is much
  closer to random pairing than to optimal transport. The measured consequence on the
  held-out traveller gate: the transport learned only a per-field intensity rescaling —
  it beat the identity baseline on nRMSE solely on the 13/60 pairs where source and
  target intensity scales are wildly mismatched, and lost on the other 47 (SSIM went
  0.8730 -> 0.8755 against a 0.9573 ceiling, i.e. no structural gain at all). ``nn``
  is the same cost, taken over the full pool instead of 8 random draws: each source is
  paired with an anatomically similar target, so the residual the flow has to explain is
  the field change rather than the anatomy difference.
- ``bridge``: the conditional path + regression target.
  ``ot_cfm`` (deterministic linear interpolation, target velocity ``z1 - z0``) or
  ``schrodinger`` (Brownian bridge with volatility ``sigma``, target drift
  ``(z1 - z_t)/(1 - t)`` — simulation-free bridge matching).

Loss ladder (training-conventions): every non-flow term defaults to weight 0 and is a
one-number switch. ``flow`` is the flow/bridge-matching MSE; ``transport_cost`` penalizes
``||v||^2`` (minimal displacement); ``identity`` enforces ``v(z, t, c, c) = 0`` on real
latents. ``cycle`` needs the ODE sampler and is intentionally not yet implemented.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.latent_bank_dataset import LatentBankIndex, LatentStats
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.checkpoints import checkpoint_filename, load_checkpoint, save_checkpoint
from fieldbridge.utils.seeding import seed_everything

Precision = Literal["fp32", "bf16"]
Device = Literal["auto", "cpu", "cuda"]
Coupling = Literal["independent", "ot", "nn"]
Bridge = Literal["ot_cfm", "schrodinger"]
FieldPairing = Literal["cross", "any"]

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "flow": 1.0,
    "transport_cost": 0.0,
    "identity": 0.0,
    "cycle": 0.0,
}


@dataclass(frozen=True, slots=True)
class Stage2TransportConfig:
    steps: int = 2
    batch_size: int = 4
    seed: int = 13
    lr: float = 2e-4
    device: Device = "auto"
    precision: Precision = "bf16"
    coupling: Coupling = "ot"
    bridge: Bridge = "ot_cfm"
    # Task 3 is field-to-field: source and target share a contrast and differ (only) in field
    # strength. same_contrast constrains the coupling to a single contrast per batch; the old
    # unconstrained any-domain draw is same_contrast=False. field_pairing="cross" forces
    # field_s != field_t (a real translation); "any" allows field_s == field_t (identity coverage).
    same_contrast: bool = True
    field_pairing: FieldPairing = "cross"
    # Spatial pool applied to latents before the OT cost. Full-latent L2 is degenerate at high
    # dim (all pairwise distances concentrate), so the minibatch-OT plan is ~random; pooling to a
    # small grid makes the cost meaningful and cheap. 0 = full latent (legacy behaviour).
    ot_pool_size: int = 4
    # coupling="nn" only. Each source is paired with one of its `nn_candidates` nearest
    # targets in descriptor space (sampled uniformly), not with the single nearest: a hard
    # 1-NN would pin every source to one fixed partner for the whole run, which both
    # overfits that anatomy and removes the stochasticity the flow-matching objective needs.
    nn_candidates: int = 5
    # Fraction of steps drawn as TRUE same-subject cross-field pairs from the prospective
    # travellers in this split, instead of from the unpaired coupling. 0.0 = pure unpaired
    # (the v3.1 thesis). The official baseline recipe is "unpaired pretrain on retrospective +
    # paired fine-tune on prospective", and Task 3 is scored exclusively on travellers, so this
    # is the knob that implements that second half.
    #
    # It exists to make the paired signal EXPLICIT and measurable. With coupling="nn" a
    # traveller in the training pool already retrieves its own target-field volume as the
    # nearest neighbour (same anatomy is trivially nearest), so paired supervision was leaking
    # in by accident at an unmeasurable rate. Keep travellers out of train (1-1-1 resplit) and
    # dial this instead, so `+/- paired` is an ablation rather than a side effect.
    #
    # Caveat worth stating: with one traveller in train this is 60 ordered pairs from a single
    # anatomy. Overfitting to it is a real risk; that is what the held-out traveller gate is for.
    paired_fraction: float = 0.0
    # Where the pooled per-record descriptors are cached. Building them reads every latent
    # in the bank once (~11 GB for the train split), which is not something to repeat per
    # run. None = alongside the bank.
    descriptor_cache: Path | None = None
    sigma: float = 0.1  # Schrodinger-bridge volatility (bridge="schrodinger" only)
    time_eps: float = 1e-3  # clamp on t and (1 - t)
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    checkpoint_dir: Path | None = None
    checkpoint_every_steps: int = 0
    checkpoint_at_end: bool = True
    checkpoint_max_bytes: int = 500_000_000
    val_every_steps: int = 0
    val_batches: int = 4
    log_every_steps: int = 0
    resume_from: Path | None = None
    variant: str = "flow_matching_latent"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Stage2TransportConfig":
        defaults = cls()
        training = data.get("training", {})
        training = dict(training) if isinstance(training, Mapping) else {}

        def pick(key: str, current: Any) -> Any:
            return training.get(key, data.get(key, current))

        weights = dict(DEFAULT_LOSS_WEIGHTS)
        weights.update(training.get("loss_weights", data.get("loss_weights", {})))
        checkpoint_dir = pick("checkpoint_dir", None)
        resume_from = pick("resume_from", None)
        return cls(
            steps=int(pick("steps", defaults.steps)),
            batch_size=int(pick("batch_size", defaults.batch_size)),
            seed=int(pick("seed", defaults.seed)),
            lr=float(pick("lr", defaults.lr)),
            device=pick("device", defaults.device),
            precision=pick("precision", defaults.precision),
            coupling=pick("coupling", defaults.coupling),
            bridge=pick("bridge", defaults.bridge),
            same_contrast=bool(pick("same_contrast", defaults.same_contrast)),
            field_pairing=pick("field_pairing", defaults.field_pairing),
            ot_pool_size=int(pick("ot_pool_size", defaults.ot_pool_size)),
            nn_candidates=int(pick("nn_candidates", defaults.nn_candidates)),
            paired_fraction=float(pick("paired_fraction", defaults.paired_fraction)),
            descriptor_cache=(
                Path(descriptor_cache) if (descriptor_cache := pick("descriptor_cache", None)) else None
            ),
            sigma=float(pick("sigma", defaults.sigma)),
            time_eps=float(pick("time_eps", defaults.time_eps)),
            loss_weights=weights,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_every_steps=int(pick("checkpoint_every_steps", defaults.checkpoint_every_steps)),
            checkpoint_at_end=bool(pick("checkpoint_at_end", defaults.checkpoint_at_end)),
            checkpoint_max_bytes=int(pick("checkpoint_max_bytes", defaults.checkpoint_max_bytes)),
            val_every_steps=int(pick("val_every_steps", defaults.val_every_steps)),
            val_batches=int(pick("val_batches", defaults.val_batches)),
            log_every_steps=int(pick("log_every_steps", defaults.log_every_steps)),
            resume_from=Path(resume_from) if resume_from else None,
            variant=str(pick("variant", defaults.variant)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "lr": self.lr,
            "device": self.device,
            "precision": self.precision,
            "coupling": self.coupling,
            "bridge": self.bridge,
            "same_contrast": self.same_contrast,
            "field_pairing": self.field_pairing,
            "ot_pool_size": self.ot_pool_size,
            "nn_candidates": self.nn_candidates,
            "paired_fraction": self.paired_fraction,
            "sigma": self.sigma,
            "time_eps": self.time_eps,
            "loss_weights": dict(self.loss_weights),
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class Stage2TransportResult:
    steps: int
    losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    best_val: float | None = None

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def seconds_per_step(self) -> float:
        return self.elapsed_seconds / len(self.losses) if self.losses else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "losses": self.losses,
            "val_losses": self.val_losses,
            "final_loss": self.final_loss,
            "seconds_per_step": self.seconds_per_step,
            "elapsed_seconds": self.elapsed_seconds,
            "best_val": self.best_val,
        }


def run_stage2_transport_train(
    config: Stage2TransportConfig | Mapping[str, Any] | None,
    *,
    translator: BaseTranslator,
    train_index: LatentBankIndex,
    stats: LatentStats,
    val_index: LatentBankIndex | None = None,
) -> Stage2TransportResult:
    cfg = _coerce_config(config)
    _validate_config(cfg)
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    translator = translator.to(device)
    optimizer = torch.optim.Adam(translator.parameters(), lr=cfg.lr)
    sampler = torch.Generator().manual_seed(cfg.seed)
    train_pools = _FieldPools.from_index(train_index)
    val_pools = _FieldPools.from_index(val_index) if val_index is not None else None
    if cfg.same_contrast:
        train_pools.require_pairable(cfg.field_pairing)
    if cfg.coupling == "nn":
        train_pools = _attach_nn_tables(train_index, train_pools, stats, cfg, split="train")
        if val_index is not None and val_pools is not None:
            val_pools = _attach_nn_tables(val_index, val_pools, stats, cfg, split="validation")

    start_step = 0
    best_val: float | None = None
    if cfg.resume_from is not None:
        state = load_checkpoint(cfg.resume_from, map_location=device)
        translator.load_state_dict(state["translator"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        best_val = state.get("best_val")

    autocast_ctx = _autocast_context(device, cfg.precision)
    if cfg.log_every_steps > 0:
        _log(
            f"stage2_transport start steps={cfg.steps} batch={cfg.batch_size} device={device.type} "
            f"coupling={cfg.coupling} bridge={cfg.bridge} same_contrast={cfg.same_contrast} "
            f"field_pairing={cfg.field_pairing} ot_pool={cfg.ot_pool_size} "
            f"train_records={len(train_index)} val_records={len(val_index) if val_index else 0}"
        )

    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    last_ckpt_step: int | None = None
    train_start = time.perf_counter()
    for step in range(start_step, start_step + cfg.steps):
        step_start = time.perf_counter()
        translator.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            total, terms = _transport_loss(translator, train_index, train_pools, stats, cfg, device, sampler)
        total.backward()
        optimizer.step()
        _sync_if_cuda(device)
        losses.append(float(total.detach().cpu()))

        current_step = step + 1
        if cfg.checkpoint_dir is not None and cfg.checkpoint_every_steps and current_step % cfg.checkpoint_every_steps == 0:
            _save_checkpoint(cfg, translator, optimizer, current_step, best_val, unique=True)
            last_ckpt_step = current_step

        if val_index is not None and cfg.val_every_steps and current_step % cfg.val_every_steps == 0:
            val = _validate(translator, val_index, val_pools, stats, cfg, device)
            val_losses.append((current_step, val))
            if best_val is None or val < best_val:
                best_val = val
                if cfg.checkpoint_dir is not None:
                    _save_checkpoint(cfg, translator, optimizer, current_step, best_val, name="best")
            if cfg.log_every_steps > 0:
                _log(f"stage2_transport step={current_step} val_flow={val:.6f} best_val={best_val:.6f}")

        if cfg.log_every_steps > 0 and (len(losses) == 1 or current_step % cfg.log_every_steps == 0 or len(losses) == cfg.steps):
            elapsed = time.perf_counter() - train_start
            term_str = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
            _log(
                f"stage2_transport step={current_step}/{start_step + cfg.steps} loss={losses[-1]:.6f} "
                f"[{term_str}] step_sec={time.perf_counter() - step_start:.3f} "
                f"avg_sec_per_step={elapsed / len(losses):.3f}"
            )

    final_step = start_step + len(losses)
    if cfg.checkpoint_dir is not None and cfg.checkpoint_at_end and losses:
        _save_checkpoint(cfg, translator, optimizer, final_step, best_val, name="last")
        if last_ckpt_step != final_step and cfg.checkpoint_every_steps == 0:
            pass
    return Stage2TransportResult(
        steps=len(losses),
        losses=losses,
        val_losses=val_losses,
        elapsed_seconds=time.perf_counter() - train_start,
        best_val=best_val,
    )


def _transport_loss(
    translator: BaseTranslator,
    index: LatentBankIndex,
    pools: "_FieldPools",
    stats: LatentStats,
    cfg: Stage2TransportConfig,
    device: torch.device,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cfg.same_contrast:
        z0, dom_s, z1, dom_t, already_paired = _sample_constrained_pair(
            index, pools, stats, cfg, device, sampler
        )
    else:
        z0, dom_s = _sample_normalized_batch(index, stats, cfg.batch_size, device, sampler)
        z1, dom_t = _sample_normalized_batch(index, stats, cfg.batch_size, device, sampler)
        already_paired = False
    # "nn" and the paired draw both fix a specific target per source when the batch is drawn;
    # permuting again would throw that pairing away. This is per-batch, not per-run: with
    # 0 < paired_fraction < 1 the unpaired batches still get their OT assignment.
    if cfg.coupling == "ot" and not already_paired:
        perm = _ot_assignment(z0, z1, pool_size=cfg.ot_pool_size)
        z1 = z1[perm]
        dom_t = [dom_t[i] for i in perm.tolist()]

    t = _sample_time(cfg.batch_size, device, sampler, cfg.time_eps)
    z_t, target = _bridge_sample(z0, z1, t, cfg, sampler)
    velocity = translator(z_t, dom_s, dom_t, t)

    weights = cfg.loss_weights
    flow = F.mse_loss(velocity, target)
    total = weights.get("flow", 1.0) * flow
    terms: dict[str, float] = {"flow": float(flow.detach().cpu())}

    if weights.get("transport_cost", 0.0) > 0.0:
        transport = velocity.square().mean()
        total = total + weights["transport_cost"] * transport
        terms["transport_cost"] = float(transport.detach().cpu())

    if weights.get("identity", 0.0) > 0.0:
        z_id, dom_id = _sample_normalized_batch(index, stats, cfg.batch_size, device, sampler)
        t_id = _sample_time(cfg.batch_size, device, sampler, cfg.time_eps)
        v_id = translator(z_id, dom_id, dom_id, t_id)  # source==target ⇒ velocity should be 0
        identity = v_id.square().mean()
        total = total + weights["identity"] * identity
        terms["identity"] = float(identity.detach().cpu())

    if weights.get("cycle", 0.0) > 0.0:
        raise NotImplementedError(
            "cycle loss for latent transport needs the ODE sampler (A->B->A); "
            "keep loss_weights.cycle at 0 until it is implemented."
        )
    return total, terms


def _bridge_sample(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: torch.Tensor,
    cfg: Stage2TransportConfig,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_b = t.reshape(-1, *([1] * (z0.ndim - 1)))
    if cfg.bridge == "ot_cfm":
        z_t = (1.0 - t_b) * z0 + t_b * z1
        target = z1 - z0
        return z_t, target
    if cfg.bridge == "schrodinger":
        mean = (1.0 - t_b) * z0 + t_b * z1
        std = cfg.sigma * torch.sqrt((t_b * (1.0 - t_b)).clamp_min(0.0))
        eps = torch.randn(z0.shape, generator=sampler, device="cpu").to(z0.device, z0.dtype)
        z_t = mean + std * eps
        target = (z1 - z_t) / (1.0 - t_b).clamp_min(cfg.time_eps)
        return z_t, target
    raise ValueError(f"Unknown bridge {cfg.bridge!r}.")


def _ot_assignment(z0: torch.Tensor, z1: torch.Tensor, *, pool_size: int = 0) -> torch.Tensor:
    """Minibatch OT coupling: squared-L2 Hungarian assignment (column perm for z1).

    The cost is computed on spatially avg-pooled latents when ``pool_size > 0``. Full-latent L2
    concentrates at high dim (batch-8 vectors of ~1e6 dims are ~equidistant), so the plan
    degenerates to ~random; pooling to a small grid restores a meaningful, cheap cost. The
    returned permutation still indexes the full-resolution ``z1``.
    """

    from scipy.optimize import linear_sum_assignment

    a = _spatial_pool(z0, pool_size) if pool_size and pool_size > 0 else z0
    b = _spatial_pool(z1, pool_size) if pool_size and pool_size > 0 else z1
    flat0 = a.reshape(a.shape[0], -1).float()
    flat1 = b.reshape(b.shape[0], -1).float()
    cost = torch.cdist(flat0, flat1).square().detach().cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return torch.as_tensor(col, dtype=torch.long, device=z0.device)


def _spatial_pool(z: torch.Tensor, size: int) -> torch.Tensor:
    """Adaptive avg-pool the spatial dims of a (B, C, ...) latent to a (size,)*spatial grid."""

    tensor = z.float()
    if tensor.ndim == 5:
        return F.adaptive_avg_pool3d(tensor, size)
    if tensor.ndim == 4:
        return F.adaptive_avg_pool2d(tensor, size)
    return tensor


def _sample_time(batch_size: int, device: torch.device, sampler: torch.Generator, eps: float) -> torch.Tensor:
    t = torch.rand(batch_size, generator=sampler, device="cpu")
    return t.clamp(eps, 1.0 - eps).to(device)


def _load_normalized(
    index: LatentBankIndex,
    stats: LatentStats,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[Domain]]:
    latents, domains = index.load_batch(list(indices))
    latents = stats.normalize(latents).to(device)
    return latents, domains


def _sample_normalized_batch(
    index: LatentBankIndex,
    stats: LatentStats,
    batch_size: int,
    device: torch.device,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, list[Domain]]:
    indices = torch.randint(0, len(index), (batch_size,), generator=sampler).tolist()
    return _load_normalized(index, stats, indices, device)


DESCRIPTOR_CONTRACT_VERSION = "latent-descriptors-v1"


def build_latent_descriptors(
    index: LatentBankIndex,
    stats: LatentStats,
    *,
    pool_size: int,
    cache_path: Path | None = None,
    log: bool = False,
) -> torch.Tensor:
    """One pooled vector per bank record — the space the ``nn`` coupling measures distance in.

    Latents are standardized first (the transport trains in standardized space, so the
    coupling must live there too) and then average-pooled to a ``pool_size`` grid. Pooling is
    not an optimization: full-latent L2 concentrates at ~3.6M dimensions, where all pairwise
    distances are nearly equal and "nearest" stops meaning anything.

    Building this reads every latent in the split exactly once (~11 GB for train), so the
    result is cached. The cache is keyed by the record list and the pool size; a bank rebuild
    or a pool-size change invalidates it rather than silently returning stale descriptors.
    """

    fingerprint = {
        "contract": DESCRIPTOR_CONTRACT_VERSION,
        "pool_size": int(pool_size),
        "case_ids": [record.case_id for record in index.records],
    }
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("fingerprint") == fingerprint:
            if log:
                _log(f"stage2_transport descriptors: cache hit {cache_path}")
            return cached["vectors"]
        if log:
            _log(f"stage2_transport descriptors: cache STALE, rebuilding {cache_path}")

    vectors: list[torch.Tensor] = []
    for position in range(len(index)):
        latent = stats.normalize(index.load_latent(position).unsqueeze(0))
        vectors.append(_spatial_pool(latent, pool_size).reshape(-1))
        if log and (position == 0 or (position + 1) % 100 == 0 or position + 1 == len(index)):
            _log(f"stage2_transport descriptors {position + 1}/{len(index)}")
    stacked = torch.stack(vectors, dim=0).float()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        torch.save({"fingerprint": fingerprint, "vectors": stacked}, temporary)
        temporary.replace(cache_path)
        if log:
            _log(f"stage2_transport descriptors: wrote {cache_path} {tuple(stacked.shape)}")
    return stacked


def _nearest_neighbour_table(
    descriptors: torch.Tensor,
    source_positions: Sequence[int],
    target_positions: Sequence[int],
    *,
    candidates: int,
) -> torch.Tensor:
    """(len(source_positions), k) table of the k closest target *positions* per source.

    Distances are computed over the whole pool, which is the entire point: this is the
    global version of the assignment that ``ot`` could only approximate inside a minibatch.
    """

    source = descriptors[list(source_positions)]
    target = descriptors[list(target_positions)]
    k = max(1, min(int(candidates), len(target_positions)))
    distances = torch.cdist(source, target)
    nearest = distances.topk(k, dim=1, largest=False).indices
    lookup = torch.as_tensor(list(target_positions), dtype=torch.long)
    return lookup[nearest]


@dataclass(frozen=True, slots=True)
class _FieldPools:
    """Record indices grouped by (contrast, field strength), for contrast-constrained sampling.

    Built once per split. ``pairable_contrasts`` are the contrasts present at >= 2 field
    strengths — the only ones from which a cross-field (field_s != field_t) pair can be drawn.
    """

    by_contrast_field: dict[Contrast, dict[float, tuple[int, ...]]]
    pairable_contrasts: tuple[Contrast, ...]
    # coupling="nn" only: (contrast, field_s, field_t) -> (len(pool_s), k) target positions,
    # row i holding the k globally nearest targets to pool_s[i]. Precomputed once because the
    # pools are static; the largest table is 191x k.
    nn_tables: dict[tuple[Contrast, float, float], torch.Tensor] | None = None
    # (contrast, field_s, field_t) -> ((src_position, tgt_position), ...) for the SAME
    # prospective subject. This is genuine paired supervision and the only such data that
    # exists: retrospective subject_ids are field-scoped, so the same number at two fields is
    # two different people. Empty when the split holds no traveller (which is the point of a
    # 1-1-1 resplit — see `paired_fraction`).
    paired_by_transition: dict[tuple[Contrast, float, float], tuple[tuple[int, int], ...]] = field(
        default_factory=dict
    )

    @classmethod
    def from_index(cls, index: LatentBankIndex) -> "_FieldPools":
        table: dict[Contrast, dict[float, list[int]]] = {}
        for position, record in enumerate(index.records):
            contrast = Contrast.parse(record.domain.contrast)
            field_strength = float(record.domain.field_strength_t)
            table.setdefault(contrast, {}).setdefault(field_strength, []).append(position)
        frozen = {c: {f: tuple(idx) for f, idx in fields.items()} for c, fields in table.items()}
        pairable = tuple(c for c, fields in frozen.items() if len(fields) >= 2)
        return cls(
            by_contrast_field=frozen,
            pairable_contrasts=pairable,
            paired_by_transition=_paired_transitions(index),
        )

    def with_nn_tables(self, descriptors: torch.Tensor, *, candidates: int) -> "_FieldPools":
        """Return a copy carrying the global NN table for every (contrast, f_s, f_t)."""

        tables: dict[tuple[Contrast, float, float], torch.Tensor] = {}
        for contrast, fields in self.by_contrast_field.items():
            for field_s, pool_s in fields.items():
                for field_t, pool_t in fields.items():
                    if field_s == field_t:
                        continue
                    tables[(contrast, field_s, field_t)] = _nearest_neighbour_table(
                        descriptors, pool_s, pool_t, candidates=candidates
                    )
        return _FieldPools(
            by_contrast_field=self.by_contrast_field,
            pairable_contrasts=self.pairable_contrasts,
            nn_tables=tables,
            paired_by_transition=self.paired_by_transition,
        )

    def paired_pair_count(self) -> int:
        return sum(len(v) for v in self.paired_by_transition.values())

    def require_pairable(self, field_pairing: FieldPairing) -> None:
        if field_pairing == "cross" and not self.pairable_contrasts:
            raise ValueError(
                "same_contrast with field_pairing='cross' needs a bank where at least one "
                "contrast is present at >= 2 field strengths; found none. Check the split, or "
                "set field_pairing='any' / same_contrast=false."
            )
        if not self.by_contrast_field:
            raise ValueError("Latent bank has no records to sample contrast-constrained pairs from.")


def _paired_transitions(
    index: LatentBankIndex,
) -> dict[tuple[Contrast, float, float], tuple[tuple[int, int], ...]]:
    """Same-subject cross-field position pairs, restricted to the prospective cohort.

    The cohort restriction is not cosmetic. The official data description gives retrospective
    IDs disjoint per field strength (0001-1056, each field its own range) and prospective IDs
    0001-0040 where the same ID *is* the same volunteer at every field. Keying on the bare
    subject_id would therefore fabricate "pairs" out of two unrelated retrospective people.
    """

    by_identity: dict[tuple[str, Contrast], dict[float, int]] = {}
    for position, record in enumerate(index.records):
        if not str(record.case_id).startswith("P_"):
            continue
        key = (str(record.subject_id), Contrast.parse(record.domain.contrast))
        by_identity.setdefault(key, {})[float(record.domain.field_strength_t)] = position

    paired: dict[tuple[Contrast, float, float], list[tuple[int, int]]] = {}
    for (_subject, contrast), field_map in by_identity.items():
        for field_s, source in field_map.items():
            for field_t, target in field_map.items():
                if field_s == field_t:
                    continue
                paired.setdefault((contrast, field_s, field_t), []).append((source, target))
    return {key: tuple(value) for key, value in paired.items()}


def _attach_nn_tables(
    index: LatentBankIndex,
    pools: _FieldPools,
    stats: LatentStats,
    cfg: Stage2TransportConfig,
    *,
    split: str,
) -> _FieldPools:
    """Build (or load) the descriptors for one split and hang its neighbour tables off pools."""

    if cfg.ot_pool_size <= 0:
        raise ValueError("coupling='nn' needs ot_pool_size > 0 (the descriptor grid).")
    cache = cfg.descriptor_cache
    cache_path = (
        cache / f"descriptors_{split}_pool{cfg.ot_pool_size}.pt"
        if cache is not None
        else index.bank_dir / f"descriptors_{split}_pool{cfg.ot_pool_size}.pt"
    )
    descriptors = build_latent_descriptors(
        index, stats, pool_size=cfg.ot_pool_size, cache_path=cache_path, log=cfg.log_every_steps > 0
    )
    if cfg.log_every_steps > 0:
        _log(
            f"stage2_transport nn coupling: {split} descriptors {tuple(descriptors.shape)} "
            f"candidates={cfg.nn_candidates}"
        )
    return pools.with_nn_tables(descriptors, candidates=cfg.nn_candidates)


def _draw_with_replacement(pool: tuple[int, ...], count: int, sampler: torch.Generator) -> list[int]:
    picks = torch.randint(0, len(pool), (count,), generator=sampler).tolist()
    return [pool[p] for p in picks]


def _sample_constrained_pair(
    index: LatentBankIndex,
    pools: _FieldPools,
    stats: LatentStats,
    cfg: Stage2TransportConfig,
    device: torch.device,
    sampler: torch.Generator,
) -> tuple[torch.Tensor, list[Domain], torch.Tensor, list[Domain], bool]:
    """Draw a same-contrast (field_s -> field_t) batch: source and target share the contrast.

    One (contrast, field_s, field_t) transition is chosen per batch so the minibatch-OT coupling
    pairs anatomies for a single, well-defined field translation. Over steps this covers all
    (contrast, field_s, field_t) combinations, training the one shared any-to-any field v_theta.
    """

    contrasts = pools.pairable_contrasts if cfg.field_pairing == "cross" else tuple(pools.by_contrast_field)
    contrast = contrasts[int(torch.randint(0, len(contrasts), (1,), generator=sampler))]
    fields = sorted(pools.by_contrast_field[contrast])
    if cfg.field_pairing == "cross":
        i = int(torch.randint(0, len(fields), (1,), generator=sampler))
        j = int(torch.randint(0, len(fields) - 1, (1,), generator=sampler))
        if j >= i:
            j += 1
        field_s, field_t = fields[i], fields[j]
    else:
        field_s = fields[int(torch.randint(0, len(fields), (1,), generator=sampler))]
        field_t = fields[int(torch.randint(0, len(fields), (1,), generator=sampler))]
    paired = pools.paired_by_transition.get((contrast, field_s, field_t), ())
    if paired and cfg.paired_fraction > 0.0 and (
        float(torch.rand(1, generator=sampler)) < cfg.paired_fraction
    ):
        picks = torch.randint(0, len(paired), (cfg.batch_size,), generator=sampler).tolist()
        src_idx = [paired[i][0] for i in picks]
        tgt_idx = [paired[i][1] for i in picks]
        already_paired = True
    else:
        pool_s = pools.by_contrast_field[contrast][field_s]
        src_idx = _draw_with_replacement(pool_s, cfg.batch_size, sampler)
        if cfg.coupling == "nn":
            tgt_idx = _nn_targets(pools, contrast, field_s, field_t, pool_s, src_idx, cfg, sampler)
            already_paired = True
        else:
            tgt_idx = _draw_with_replacement(
                pools.by_contrast_field[contrast][field_t], cfg.batch_size, sampler
            )
            already_paired = False
    z0, dom_s = _load_normalized(index, stats, src_idx, device)
    z1, dom_t = _load_normalized(index, stats, tgt_idx, device)
    return z0, dom_s, z1, dom_t, already_paired


def _nn_targets(
    pools: _FieldPools,
    contrast: Contrast,
    field_s: float,
    field_t: float,
    pool_s: tuple[int, ...],
    src_idx: Sequence[int],
    cfg: Stage2TransportConfig,
    sampler: torch.Generator,
) -> list[int]:
    """Pair each drawn source with one of its k globally nearest targets, chosen uniformly."""

    if pools.nn_tables is None:
        raise ValueError(
            "coupling='nn' needs the precomputed neighbour tables; build them with "
            "_FieldPools.with_nn_tables(build_latent_descriptors(...))."
        )
    table = pools.nn_tables.get((contrast, field_s, field_t))
    if table is None:
        raise ValueError(
            f"No neighbour table for ({contrast.value}, {field_s}T -> {field_t}T). The pools "
            "and the tables were built from different indices."
        )
    row_of = {position: row for row, position in enumerate(pool_s)}
    k = int(table.shape[1])
    picks = torch.randint(0, k, (len(src_idx),), generator=sampler).tolist()
    return [int(table[row_of[position], pick]) for position, pick in zip(src_idx, picks)]


@torch.no_grad()
def _validate(
    translator: BaseTranslator,
    index: LatentBankIndex,
    pools: _FieldPools,
    stats: LatentStats,
    cfg: Stage2TransportConfig,
    device: torch.device,
) -> float:
    translator.eval()
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    total = 0.0
    for _ in range(max(1, cfg.val_batches)):
        if cfg.same_contrast:
            z0, dom_s, z1, dom_t, already_paired = _sample_constrained_pair(
                index, pools, stats, cfg, device, generator
            )
        else:
            z0, dom_s = _sample_normalized_batch(index, stats, cfg.batch_size, device, generator)
            z1, dom_t = _sample_normalized_batch(index, stats, cfg.batch_size, device, generator)
            already_paired = False
        if cfg.coupling == "ot" and not already_paired:
            perm = _ot_assignment(z0, z1, pool_size=cfg.ot_pool_size)
            z1 = z1[perm]
            dom_t = [dom_t[i] for i in perm.tolist()]
        t = _sample_time(cfg.batch_size, device, generator, cfg.time_eps)
        z_t, target = _bridge_sample(z0, z1, t, cfg, generator)
        velocity = translator(z_t, dom_s, dom_t, t)
        total += float(F.mse_loss(velocity, target).cpu())
    return total / max(1, cfg.val_batches)


def _save_checkpoint(
    cfg: Stage2TransportConfig,
    translator: BaseTranslator,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float | None,
    *,
    name: str | None = None,
    unique: bool = False,
) -> None:
    assert cfg.checkpoint_dir is not None
    state: dict[str, Any] = {
        "translator": translator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
    }
    if unique:
        filename = checkpoint_filename("transport", cfg.variant, step)
        overwrite = False
    else:
        filename = f"transport_{cfg.variant}_{name}.pt"
        overwrite = True
    save_checkpoint(
        cfg.checkpoint_dir / filename,
        state,
        max_bytes=cfg.checkpoint_max_bytes,
        overwrite=overwrite,
        seed=cfg.seed,
        config=cfg.to_dict(),
    )


def _validate_config(cfg: Stage2TransportConfig) -> None:
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if cfg.coupling not in ("independent", "ot", "nn"):
        raise ValueError(f"Unknown coupling {cfg.coupling!r}.")
    if cfg.coupling == "nn":
        if not cfg.same_contrast:
            raise ValueError(
                "coupling='nn' pairs within a (contrast, field_s -> field_t) transition, so it "
                "requires same_contrast=true."
            )
        if cfg.nn_candidates < 1:
            raise ValueError("nn_candidates must be >= 1.")
    if not 0.0 <= cfg.paired_fraction <= 1.0:
        raise ValueError(f"paired_fraction must be in [0, 1]; got {cfg.paired_fraction}.")
    if cfg.paired_fraction > 0.0 and not cfg.same_contrast:
        raise ValueError("paired_fraction > 0 pairs within a contrast, so needs same_contrast=true.")
    if cfg.bridge not in ("ot_cfm", "schrodinger"):
        raise ValueError(f"Unknown bridge {cfg.bridge!r}.")
    if cfg.bridge == "schrodinger" and cfg.sigma <= 0:
        raise ValueError("schrodinger bridge requires sigma > 0.")
    if cfg.field_pairing not in ("cross", "any"):
        raise ValueError(f"Unknown field_pairing {cfg.field_pairing!r}.")
    if cfg.ot_pool_size < 0:
        raise ValueError("ot_pool_size must be >= 0.")


def _coerce_config(config: Stage2TransportConfig | Mapping[str, Any] | None) -> Stage2TransportConfig:
    if config is None:
        return Stage2TransportConfig()
    if isinstance(config, Stage2TransportConfig):
        return config
    return Stage2TransportConfig.from_mapping(config)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is 'cuda', but CUDA is not available.")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device)


def _autocast_context(device: torch.device, precision: Precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("precision must be 'fp32' or 'bf16'.")


__all__ = [
    "Stage2TransportConfig",
    "Stage2TransportResult",
    "run_stage2_transport_train",
    "build_latent_descriptors",
    "DESCRIPTOR_CONTRACT_VERSION",
    "DEFAULT_LOSS_WEIGHTS",
]
