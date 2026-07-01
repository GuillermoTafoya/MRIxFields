# Local Windows counterpart to scripts/dry_run.sh: install -> tests -> smoke-train -> train
# with minimal steps/batch so it finishes on CPU in well under a minute.
param(
    [int]$Steps = 1,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"

Write-Host "== [1/4] install (editable, dev extra) =="
python -m pip install -q -e ".[dev]"

Write-Host "== [2/4] pytest =="
python -m pytest -q

Write-Host "== [3/4] fieldbridge smoke-train =="
python -m fieldbridge.cli smoke-train --steps $Steps --batch-size $BatchSize --json

Write-Host "== [4/4] fieldbridge train (translator stage, identity variant) =="
python -m fieldbridge.cli train --config configs/experiment/smoke.yaml --steps $Steps --batch-size $BatchSize --json

Write-Host "== dry run OK =="
