"""Run the two retrospective fits and their fixed-update acquired-target comparison in Colab."""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

import stage2_zero_transport_operator as op
from stage2_matched_objective_evaluation import evaluate_step200
from stage2_matched_objective_training import require_training_gpu, run_matched_fits

EXPERIMENT = "stage2_matched_objectives_step200"


def main():
    repo = Path(__file__).resolve().parents[1]
    commit = op.verified_commit(repo)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the worker.")
    device = require_training_gpu()
    if not op.OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Mount the project Drive at {op.DRIVE}.")

    run_dir = op.OUTPUT_ROOT / EXPERIMENT / commit / "run-0001"
    stage_dir = Path("/content") / EXPERIMENT / commit
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"Matched checkpoints and scores: {run_dir}", flush=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    fits = run_matched_fits(run_dir, stage_dir)
    for objective, fit in fits.items():
        print(f"{objective}: step {fit['step']}, objective-specific val_sb={fit['val_sb']:.8f}\n"
              f"  {fit['checkpoint']}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(op.SEED)
    summary = evaluate_step200(run_dir=run_dir, stage_dir=stage_dir, commit=commit, fits=fits, device=device)
    print("\nCalibrated whole-volume target nRMSE (current velocity minus forward drift)")
    for label in ("development_59", "original_pilot"):
        group = summary[label]
        delta = group["mean_paired_current_minus_forward"]
        print(f"{label} (n={group['count']}): {delta:+.9f}")
        for arm, scores in group["mean_calibrated_full_nrmse"].items():
            table = ", ".join(f"{method}={score:.7f}" for method, score in scores.items())
            print(f"  {arm}: {table}")
    print(f"Per-direction scores and paired summary: {run_dir / 'evaluation' / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
