#!/usr/bin/env bash
# Colab/CPU dry run: exercises install -> tests -> smoke-train -> train
# with minimal steps/batch/data so it finishes on CPU in well under a minute.
# Does not download or touch real MRI data (repo is synthetic-only, per AGENTS.md).
set -euo pipefail

STEPS="${1:-1}"
BATCH_SIZE="${2:-2}"

echo "== [1/4] install (editable, dev extra) =="
python -m pip install -q -e ".[dev]"

echo "== [2/4] pytest =="
python -m pytest -q

echo "== [3/4] fieldbridge smoke-train =="
python -m fieldbridge.cli smoke-train --steps "$STEPS" --batch-size "$BATCH_SIZE" --json

echo "== [4/4] fieldbridge train (translator stage, identity variant) =="
python -m fieldbridge.cli train --config configs/experiment/smoke.yaml --steps "$STEPS" --batch-size "$BATCH_SIZE" --json

echo "== dry run OK =="
