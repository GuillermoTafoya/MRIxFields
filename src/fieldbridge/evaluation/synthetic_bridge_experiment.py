"""Run the synthetic 2 x 2 Brownian objective and endpoint-pairing experiment on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from tempfile import mkdtemp

import scipy
import torch

from fieldbridge.data.domains import Domain
from fieldbridge.evaluation.synthetic_bridge_phantoms import (
    DEPTH, SOURCE_FIELD, TARGET_FIELDS, TISSUE_INTENSITY, SubjectBatch, make_cohort, measure,
)
from fieldbridge.evaluation.synthetic_bridge_training import train_arm
from fieldbridge.training.stage2_unified import integrate_transport


@dataclass(frozen=True)
class Protocol:
    cohort_seed: int = 426
    seeds: tuple[int, ...] = (13, 37, 73)
    n_train: int = 32
    n_test: int = 12
    side: int = 16
    per_field: int = 4
    updates: int = 600
    sigma: float = 0.15
    time_eps: float = 0.01
    lr: float = 0.002
    heun_steps: tuple[int, int] = (4, 32)


PAIRINGS = ("paired", "shuffled")
OBJECTIVES = ("forward_drift", "current_velocity")
METRICS = ("mean_subject_rmse", "mean_changed_tissue_rmse", "population_intensity_w1", "population_image_w2")


def split_fingerprint(train: SubjectBatch, test: SubjectBatch) -> str:
    digest = hashlib.sha256()
    for role, group in (("train", train), ("test", test)):
        digest.update(json.dumps([role, group.ids], separators=(",", ":")).encode())
        for volume in (group.source, *(group.targets[f] for f in TARGET_FIELDS), group.tissue):
            digest.update(str((volume.dtype, tuple(volume.shape))).encode())
            digest.update(volume.contiguous().numpy().tobytes())
    return digest.hexdigest()


def score(prediction, target, subjects):
    metrics = measure(prediction, target)
    metrics["subject_rmse"] = dict(zip(subjects.ids, metrics["subject_rmse"], strict=True))
    tissue = subjects.tissue.flatten(1)
    error = (prediction.double() - target.double()).flatten(1).square()
    tissue_errors = ((error * tissue).sum(1) / tissue.sum(1)).sqrt()
    metrics["mean_changed_tissue_rmse"] = tissue_errors.mean().item()
    return metrics


def summarize(rows):
    """Average fields and seeds equally, after scoring each held-out population separately."""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("pairing", "objective", "control", "steps"))
        groups.setdefault(key, []).append(row["metrics"])
    return [
        {"pairing": pairing, "objective": objective, "control": control, "steps": steps,
         **{metric: fmean(score[metric] for score in group) for metric in METRICS},
         "subject_rmse": {key: fmean(score["subject_rmse"][key] for score in group)
                          for key in group[0]["subject_rmse"]}}
        for (pairing, objective, control, steps), group in groups.items()
    ]


@torch.inference_mode()
def evaluate(model, subjects, steps):
    requests = {}
    source_labels = [Domain(SOURCE_FIELD, "T1w")] * len(subjects.ids)
    for requested in TARGET_FIELDS:
        target_labels = [Domain(requested, "T1w")] * len(subjects.ids)
        requests[requested] = integrate_transport(
            model, subjects.source, source_labels, target_labels, steps=steps, solver="heun",
        )
    return requests


def run(protocol: Protocol, commit: str) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train, test = make_cohort(n_train=protocol.n_train, n_test=protocol.n_test, side=protocol.side,
                              seed=protocol.cohort_seed)
    if set(train.ids) & set(test.ids):
        raise ValueError("Subject leakage between training and evaluation.")
    rows = [
        {"pairing": "baseline", "objective": "zero", "seed": None, "target_field": field,
         "requested_field": None, "control": "zero", "steps": 0,
         "metrics": score(test.source, test.targets[field], test)}
        for field in TARGET_FIELDS
    ]
    fits = []
    refinement = []
    for seed in protocol.seeds:
        common = None
        paired_endpoints = {}
        for pairing in PAIRINGS:
            for objective in OBJECTIVES:
                arm = {"seed": seed, "pairing": pairing, "objective": objective}
                print(f"CPU fit  {seed}  {pairing}  {objective}", flush=True)
                model, fit = train_arm(
                    train, pairing=pairing, objective=objective, seed=seed, updates=protocol.updates,
                    per_field=protocol.per_field, sigma=protocol.sigma, time_eps=protocol.time_eps, lr=protocol.lr,
                )
                identity = (fit["initial_state_sha256"], fit["draws_sha256"], fit["parameters"])
                if common is not None and identity != common:
                    raise RuntimeError("Initialization, capacity or common random draws changed between arms.")
                common = identity
                if paired_endpoints.setdefault(pairing, fit["pairing_indices_sha256"]) != fit["pairing_indices_sha256"]:
                    raise RuntimeError("Target endpoint schedule changed between objectives.")
                model.eval()
                fits.append({**arm, **fit})
                predictions = {n: evaluate(model, test, n) for n in protocol.heun_steps}
                coarse, fine = protocol.heun_steps
                for actual in TARGET_FIELDS:
                    delta = predictions[coarse][actual].double() - predictions[fine][actual].double()
                    per_subject = delta.flatten(1).square().mean(1).sqrt()
                    refinement.append({**arm, "requested_field": actual,
                                       "mean_endpoint_rmse": per_subject.mean().item(),
                                       "subject_endpoint_rmse": dict(zip(test.ids, per_subject.tolist(), strict=True))})
                    for steps, requests in predictions.items():
                        for requested, prediction in requests.items():
                            rows.append({**arm, "steps": steps, "target_field": actual,
                                         "requested_field": requested,
                                         "control": "correct" if requested == actual else "wrong",
                                         "metrics": score(prediction, test.targets[actual], test)})
        if len(set(paired_endpoints.values())) != len(PAIRINGS):
            raise RuntimeError("Paired and shuffled fits accidentally used identical endpoint schedules.")
    return {
        "experiment": "synthetic-tissue-brownian-pairing-v1", "git_commit": commit,
        "config": {**asdict(protocol), "source_field": SOURCE_FIELD, "target_fields": TARGET_FIELDS,
                   "tissue_intensity": TISSUE_INTENSITY, "depth": DEPTH, "dtype": "float32", "device": "cpu"},
        "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__), "scipy": scipy.__version__,
                     "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "split_sha256": split_fingerprint(train, test), "train_subjects": train.ids, "heldout_subjects": test.ids,
        "model_state": "final update of each independent fit (in memory)",
        "fits": fits, "measurements": rows, "heun_refinement": refinement, "summary": summarize(rows),
    }


def write_report(report, path):
    summary = report["summary"]
    lines = [
        "# Synthetic endpoint pairing and Brownian objectives", "",
        f"Implementation `{report['git_commit']}`. Split `{report['split_sha256']}`.", "",
        "Each cell averages the two requested fields and the matched training seeds. "
        "All scores are distances in synthetic intensity units. Lower is better. "
        "Subject RMSE averages per-subject full-volume errors. Intensity W1 compares pooled voxel "
        "intensities. Image W2 optimally matches the two unordered sets of whole volumes by squared voxel RMSE.",
        "", "| Pairing | Objective | Request | Heun steps | Subject RMSE | Changed tissue RMSE | Intensity W1 | Image W2 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        cells = [r["pairing"], r["objective"], r["control"], str(r["steps"]), *(f"{r[k]:.6f}" for k in METRICS)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Per-subject RMSE at four Heun steps", "",
                  "Correct requests, averaged across both target fields and all three training seeds.", "",
                  "| Held-out subject | Zero | Paired drift | Paired current | Shuffled drift | Shuffled current |",
                  "|---|---:|---:|---:|---:|---:|"])
    columns = [r for r in summary if r["control"] == "zero" or (r["steps"] == 4 and r["control"] == "correct")]
    for subject in report["heldout_subjects"]:
        lines.append("| " + " | ".join([subject, *(f"{r['subject_rmse'][subject]:.6f}" for r in columns)]) + " |")
    lines.extend(["", "## Heun refinement", "",
                  "Mean per-subject RMSE between 4- and 32-step predictions from the identical weights.", "",
                  "| Pairing | Objective | Endpoint RMSE |", "|---|---|---:|"])
    for pairing in PAIRINGS:
        for objective in OBJECTIVES:
            items = [r["mean_endpoint_rmse"] for r in report["heun_refinement"]
                     if r["pairing"] == pairing and r["objective"] == objective]
            lines.append(f"| {pairing} | {objective} | {fmean(items):.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Parent for a fresh run directory containing results.json and report.md.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    report = run(Protocol(), commit)
    args.results_root.mkdir(parents=True, exist_ok=True)
    out = Path(mkdtemp(prefix="run-", dir=args.results_root))
    (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                      encoding="utf-8")
    write_report(report, out / "report.md")
    print(out.resolve())


if __name__ == "__main__":
    main()
