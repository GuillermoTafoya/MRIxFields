#!/usr/bin/env bash
# Colab dry run against a *small* slice of real MRIxFields data (Drive-mounted or
# otherwise locally available). Requires the optional 'nifti' extra (nibabel).
# Only paths are read from DATA_ROOT -- the manifest built from it is written to a
# scratch location and never committed to the repo (AGENTS.md: no real data/paths in git).
#
# Usage: DATA_ROOT=/content/drive/MyDrive/.../Data bash scripts/dry_run_real.sh [max_records] [steps] [batch_size]
set -euo pipefail

if [ -z "${DATA_ROOT:-}" ]; then
    echo "Set DATA_ROOT to the real data directory (e.g. a Drive-mounted MRIxFields2026/Data path)." >&2
    exit 1
fi

MAX_RECORDS="${1:-8}"
STEPS="${2:-1}"
BATCH_SIZE="${3:-1}"
MANIFEST_PATH="$(mktemp -t clbfield-real-manifest-XXXX.json)"

echo "== [1/3] install (editable, dev + nifti extras) =="
python -m pip install -q -e ".[dev,nifti]"

echo "== [2/3] build manifest from $DATA_ROOT (max $MAX_RECORDS records) =="
python scripts/build_real_manifest.py --data-root "$DATA_ROOT" --out "$MANIFEST_PATH" --max-records "$MAX_RECORDS"

echo "== [3/3] clbfield train (translator stage, identity variant, real data) =="
python -m clbfield.cli train \
    --config configs/experiment/smoke.yaml \
    --manifest "$MANIFEST_PATH" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --json

rm -f "$MANIFEST_PATH"
echo "== real-data dry run OK =="
