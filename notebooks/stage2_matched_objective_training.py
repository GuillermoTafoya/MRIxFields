"""Fit and authenticate the two retrospective v7 models at the same fixed update."""
from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

import stage2_zero_transport_operator as op
from stage2_decoder_context_inputs import RESOLVED_CONFIG_SHA256
from fieldbridge.evaluation.stage2_step200_inference_audit import _module_state_sha256
from fieldbridge.evaluation.stage2_step200_pilot_audit import STEP200_VALIDATION_PLAN_SHA256
from fieldbridge.evaluation.stage2_unified_gate01_p0006 import restore_verified_stage2_bank_tar
from fieldbridge.models.discriminators import DomainProjectionDiscriminator
from fieldbridge.models.factory import build_translator
from fieldbridge.training.stage2_unified import UNIFIED_RESUME_CONTRACT, UnifiedStage2Config, run_stage2_unified_train
from fieldbridge.utils.seeding import seed_everything

OBJECTIVES = ("forward_drift", "current_velocity")
STEPS = 200
CHECKPOINT = "stage2_unified_full_step000000200.pt"
BANK_TREE_SHA256 = "f9cb09bfa177a3e389f87f087b0d756a2709e2054559a39c85e8272d5e1cfaa3"


def _stage(scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    bank_dir = scratch / "bank"
    archive = op.copy_verified_stage2_bank_tar_to_local(
        op.BANK_ARCHIVE, scratch / op.BANK_ARCHIVE.name, expected_archive_sha256=op.BANK_ARCHIVE_SHA256,
        progress_callback=op.archive_progress,
    )
    restore_verified_stage2_bank_tar(
        archive, bank_dir, expected_archive_sha256=op.BANK_ARCHIVE_SHA256, expected_tree_sha256=BANK_TREE_SHA256,
        expected_bank_artifact_sha256=op.BANK_SHA256, expected_file_count=3312, expected_total_bytes=12873486620,
    )
    pilot_config = (op.OUTPUT_ROOT / "stage2_unified_v7" / "bank_8081ce89a0ea"
                    / "implementation_82633d66e5ea" / "unified_full_objective_pilot_200" / "resolved_config.json")
    for source in (pilot_config, op.CONFIG, op.CHECKPOINT, op.PHOTOMETRY):
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, scratch / source.name)
    if op.sha256_file(scratch / pilot_config.name) != RESOLVED_CONFIG_SHA256:
        raise ValueError("Expected the archived v7 full 200-step pilot configuration.")
    recipe = json.loads((scratch / pilot_config.name).read_bytes())
    train, validation = (op.PhotometryFactoredLatentBankIndex(
        bank_dir, role, expected_artifact_sha256=op.BANK_SHA256) for role in ("train", "validation"))
    stats = op.FactoredLatentStats.from_bank(bank_dir)
    if stats.artifact_sha256 != train.manifest["latent_statistics"]["artifact_sha256"]:
        raise ValueError("Training statistics differ from the authenticated bank.")
    vae_config = op.preflight_frozen_stage1_run_c_config(scratch / op.CONFIG.name)
    vae = op.verify_frozen_stage1_vae_bank_provenance(
        vae_config, vae_checkpoint_path=scratch / op.CHECKPOINT.name, bank_dir=bank_dir,
    )
    photo = op.preflight_reviewed_photometry_namespace_artifact(scratch / op.PHOTOMETRY.name)
    op.verify_reviewed_photometry_bank_provenance(photo, bank_dir=bank_dir)
    decoder = op.build_decoder("kl_vae", **op._kl_vae_kwargs(vae_config.parsed_config["model"], "decoder"))
    decoder.load_state_dict(op.load_checkpoint(scratch / op.CHECKPOINT.name, map_location="cpu")["decoder"], strict=True)
    op._require_linear_decoder_output_activation(decoder)
    inputs = {"bank_sha256": op.BANK_SHA256, "bank_tree_sha256": BANK_TREE_SHA256,
              "archive_sha256": op.BANK_ARCHIVE_SHA256, "statistics_sha256": stats.artifact_sha256,
              "source_split": train.manifest["source_split"], "vae_checkpoint_sha256": vae.checkpoint_file_sha256,
              "v7_resolved_config_sha256": RESOLVED_CONFIG_SHA256}
    return recipe, train, validation, stats, decoder.eval().requires_grad_(False), inputs


def _arm_config(recipe, objective, attempt):
    if objective not in OBJECTIVES:
        raise ValueError("Unknown matched objective.")
    training = dict(recipe["training"])
    # The old pilot tested readiness for promotion to 100k updates. Its SB-based
    # thresholds and gradient observation hooks do not select these fixed-update fits.
    training.update(bridge_objective=objective, device="cuda", steps=STEPS,
                    pilot={**training["pilot"], "steps": 0},
                    checkpoint={**training["checkpoint"], "dir": str(attempt / "checkpoints"),
                                "every_steps": STEPS},
                    history_jsonl=str(attempt / "history.jsonl"), resume_from=None)
    return UnifiedStage2Config.from_mapping({**recipe, "training": training})


def _translator(model_spec):
    kwargs = dict(model_spec)
    return build_translator(kwargs.pop("name"), **kwargs)


def _check_checkpoint(state, expected):
    if (state["contract_version"] != UNIFIED_RESUME_CONTRACT
            or expected["step"] != STEPS or state["training_cursor"] != STEPS
            or state["bridge_objective"] != expected["bridge_objective"]
            or state["_meta"]["config"] != expected["config"]
            or state["_meta"]["git_commit"] != expected["implementation_commit"]
            or state["validation_plan_sha256"] != STEP200_VALIDATION_PLAN_SHA256):
        raise ValueError("Fixed-update checkpoint identity mismatch.")
    for optimizer in ("generator_optimizer", "critic_optimizer"):
        counters = {float(v["step"]) for v in state[optimizer]["state"].values()}
        if counters != {float(STEPS)}:
            raise ValueError(f"{optimizer} did not make exactly {STEPS} updates.")


def load_step200_translator(fit, device):
    """Read back the published arm, validating objective, update and model state."""
    path = Path(fit["checkpoint"])
    if path.name != CHECKPOINT or op.sha256_file(path) != fit["checkpoint_sha256"]:
        raise ValueError("Published step-200 checkpoint file changed.")
    state = op.load_checkpoint(path, map_location="cpu")
    _check_checkpoint(state, fit)
    if state["run_fingerprint"] != fit["run_fingerprint"]:
        raise ValueError("Unexpected step-200 run identity.")
    model = _translator(fit["model"])
    model.load_state_dict(state["translator"], strict=True)
    if _module_state_sha256(model) != fit["translator_state_sha256"]:
        raise ValueError("Step-200 model parameters differ from the training receipt.")
    return model.eval().requires_grad_(False).to(device)


def _terminal_receipt(attempt, identity):
    checkpoint = attempt / "checkpoints" / CHECKPOINT
    state = op.load_checkpoint(checkpoint, map_location="cpu")
    _check_checkpoint(state, identity)
    history = attempt / "history.jsonl"
    if history.stat().st_size != state["history_prefix_bytes"] or op.sha256_file(history) != state["history_prefix_sha256"]:
        raise ValueError("Step-200 checkpoint history changed.")
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    if any(row["run_fingerprint"] != state["run_fingerprint"] for row in rows):
        raise ValueError("Wrong run fingerprint in training history.")
    steps = [r for r in rows if "generator/frozen_step_plan_sha256" in r]
    if [r["step"] for r in steps] != list(range(1, STEPS + 1)):
        raise ValueError("Incomplete training draw history.")
    validations = [r["validation"] for r in rows if r.get("event") == "unpaired_validation" and r["step"] == STEPS]
    if (len(validations) != 1 or validations[0]["validation_plan_sha256"] != state["validation_plan_sha256"]
            or validations[0]["complete_inventory_used"] is not True):
        raise ValueError("Expected the complete terminal retrospective validation.")
    model = _translator(identity["model"])
    model.load_state_dict(state["translator"], strict=True)
    return {**identity, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": op.sha256_file(checkpoint),
            "translator_state_sha256": _module_state_sha256(model), "run_fingerprint": state["run_fingerprint"],
            "validation_plan_sha256": state["validation_plan_sha256"], "val_sb": validations[0]["means"]["sb"],
            "training_step_plans_sha256": op.sha256_json([r["generator/frozen_step_plan_sha256"] for r in steps]),
            "history_sha256": state["history_prefix_sha256"]}


def _fit_one(arm_dir, objective, recipe, train, validation, stats, decoder, common):
    cfg = _arm_config(recipe, objective, arm_dir)
    seed_everything(cfg.seed)
    np.random.seed(cfg.seed)
    translator = _translator(recipe["model"])
    # Production constructs its critic immediately after reseeding inside the trainer.
    seed_everything(cfg.seed)
    critic = DomainProjectionDiscriminator(stats.mean.numel() + 1, cfg.critic_channels)
    identity = {**common, "step": STEPS, "bridge_objective": objective, "config": cfg.to_dict(),
                "model": recipe["model"], "initial_translator_sha256": _module_state_sha256(translator),
                "initial_critic_sha256": _module_state_sha256(critic)}
    for attempt in sorted(arm_dir.glob("run-*")):
        path = attempt / "checkpoints" / CHECKPOINT
        receipt_path = attempt / "fit.json"
        if not (path.is_file() or receipt_path.is_file()):
            continue
        if json.loads((attempt / "identity.json").read_bytes()) != identity:
            raise ValueError(f"Existing fit identity differs: {attempt}")
        receipt = _terminal_receipt(attempt, identity)
        if receipt_path.exists():
            if json.loads(receipt_path.read_bytes()) != receipt:
                raise ValueError("Published fit receipt does not match the terminal checkpoint and history.")
        else:
            op.write_result(receipt_path, receipt)
        return receipt

    attempt = op.new_run_directory(arm_dir)
    cfg = _arm_config(recipe, objective, attempt)
    op.write_result(attempt / "identity.json", identity)
    result = run_stage2_unified_train(
        cfg, translator=translator, critic=critic, decoder=decoder, stats=stats,
        train_index=train, validation_index=validation,
    )
    if result.completed_steps != STEPS or Path(result.checkpoint) != attempt / "checkpoints" / CHECKPOINT:
        raise RuntimeError("Trainer did not publish the requested fixed-update checkpoint.")
    receipt = _terminal_receipt(attempt, identity)
    op.write_result(attempt / "fit.json", receipt)
    return receipt


def verify_match(fits):
    """Bind initialization, observed stochastic training schedules and all common configuration."""
    common_rows = []
    for objective in OBJECTIVES:
        row = fits[objective]
        config = dict(row["config"])
        if config.pop("bridge_objective") != objective or row["bridge_objective"] != objective:
            raise ValueError("Wrong training objective in the matched fit.")
        common_rows.append({**{key: row[key] for key in (
            "inputs", "implementation_commit", "backend", "step", "model", "initial_translator_sha256",
            "initial_critic_sha256", "training_step_plans_sha256", "validation_plan_sha256",
        )}, "config": config})
    if common_rows[0] != common_rows[1]:
        raise ValueError("The two fits did not share initialization, draw sequence, configuration and bank.")


def require_training_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Select the A100 80 GB Colab runtime.")
    device = torch.device("cuda", 0)
    gpu = torch.cuda.get_device_properties(device)
    if ("A100" not in gpu.name or gpu.total_memory < 75 * 2**30
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise RuntimeError(f"The v7 fits require an A100 80 GB with native BF16. Found {gpu.name}.")
    return device


def run_matched_fits(run_dir, scratch):
    """Restore the retrospective-only bank and obtain both immutable final checkpoints on Drive."""
    commit = op.verified_commit(Path(__file__).resolve().parents[1])
    torch.cuda.set_device(require_training_gpu())
    recipe, train, validation, stats, decoder, inputs = _stage(Path(scratch))
    strict = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    try:
        # CUDA avg_pool3d backward in the v7 anatomy term can lack a deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        backend = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                   "deterministic_algorithms": True, "warn_only": True, "cudnn_benchmark": False,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "allow_tf32_convolution": torch.backends.cudnn.allow_tf32,
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "cpu_threads": torch.get_num_threads()}
        common = {"implementation_commit": commit, "inputs": inputs, "backend": backend}
        fits = {}
        for objective in OBJECTIVES:
            fits[objective] = _fit_one(Path(run_dir) / "fits" / objective, objective, recipe,
                                      train, validation, stats, decoder, common)
            gc.collect()
            torch.cuda.empty_cache()
        verify_match(fits)
        return fits
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.use_deterministic_algorithms(strict, warn_only=warn)
