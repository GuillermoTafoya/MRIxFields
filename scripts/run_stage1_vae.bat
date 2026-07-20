@echo off
setlocal enabledelayedexpansion
:: ---------------------------------------------------------------------------
:: Full Etapa-1 KL-VAE pipeline on a local NVIDIA box, end to end:
::   manifest  ->  subject-level split  ->  train (+ per-epoch validation)  ->  held-out test eval
::
:: Only paths are read from DATA_ROOT; the manifest and split contain real file paths and
:: are written under WORK_DIR (outside the repo) -- never commit them (AGENTS.md).
::
:: Usage (from anywhere):
::   set DATA_ROOT=D:\MRI_Field_2026\Data
::   scripts\run_stage1_vae.bat
::
:: Optional overrides (set before calling):
::   set WORK_DIR=D:\MRI_Field_2026     (default: parent of DATA_ROOT)
::   set EPOCHS=30                       (default: 30)
::   set CONFIG=configs\experiment\stage1_vae.yaml
:: ---------------------------------------------------------------------------

if "%DATA_ROOT%"=="" (
  echo [error] Set DATA_ROOT to the extracted MRIxFields Data directory, e.g.:
  echo         set DATA_ROOT=D:\MRI_Field_2026\Data
  exit /b 1
)
:: Default WORK_DIR to the parent of DATA_ROOT (…\Data -> …\), no machine path hardcoded.
if "%WORK_DIR%"=="" ( for %%I in ("%DATA_ROOT%") do set "WORK_DIR=%%~dpI" )
if "%EPOCHS%"==""  set "EPOCHS=100"
if "%CONFIG%"==""  set "CONFIG=configs\experiment\stage1_vae.yaml"

:: Run from the repo root so the config's relative checkpoint_dir resolves there.
cd /d "%~dp0.."

set "MANIFEST=%WORK_DIR%manifest.json"
set "SPLIT=%WORK_DIR%split.json"
set "CKPT=outputs\stage1_vae\checkpoints\vae_kl_vae_best.pt"
set "OUTDIR=%WORK_DIR%runs\stage1_%RANDOM%"
mkdir "%OUTDIR%" 2>nul

echo == [1/4] manifest (%MANIFEST%) ==
if not exist "%MANIFEST%" (
  python scripts\build_real_manifest.py --data-root "%DATA_ROOT%" --out "%MANIFEST%" --max-records 100000 || exit /b 1
) else ( echo     exists, skipping )

echo == [2/4] subject-level train/val/test split (%SPLIT%) ==
if not exist "%SPLIT%" (
  python -m fieldbridge.cli build-vae-splits --manifest "%MANIFEST%" --out "%SPLIT%" --seed 13 || exit /b 1
) else ( echo     exists, skipping )

echo == [3/4] train (%EPOCHS% epochs, per-epoch validation -> history.jsonl + best checkpoint) ==
python -m fieldbridge.cli train-stage1-vae --config "%CONFIG%" --split-json "%SPLIT%" --epochs %EPOCHS% || exit /b 1

echo == [4/4] held-out test eval (diagnostics.png + metrics.json) ==
if not exist "%CKPT%" ( echo [error] best checkpoint not found: %CKPT% & exit /b 1 )
python -m fieldbridge.cli eval-stage1-vae --config "%CONFIG%" --split-json "%SPLIT%" --split test ^
  --checkpoint "%CKPT%" --out "%OUTDIR%" --per-domain || exit /b 1

echo.
echo == DONE ==
echo   history:     outputs\stage1_vae\checkpoints\history.jsonl
echo   best model:  %CKPT%
echo   test report: %OUTDIR%\metrics.json  +  %OUTDIR%\diagnostics.png
endlocal
