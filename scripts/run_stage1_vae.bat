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
::   set EPOCHS=30                       (default: 100)
::   set CONFIG=configs\experiment\stage1_vae.yaml
::   set CKPT=outputs\...\vae_kl_vae_best.pt   (must match CONFIG's checkpoint_dir; the
::                                              final eval reads the trained best from here)
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
:: CKPT must point at the best checkpoint under CONFIG's checkpoint_dir. Overridable from the
:: environment so a run with a non-default checkpoint_dir (e.g. the 75ep cosine config) still
:: finds its trained model for the eval step.
if not defined CKPT set "CKPT=outputs\stage1_vae\checkpoints\vae_kl_vae_best.pt"
for %%D in ("%CKPT%") do set "CKPT_DIR=%%~dpD"
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

echo == [4/4] held-out test eval (diagnostics.png + metrics.json + metrics.csv) ==
if not exist "%CKPT%" ( echo [error] best checkpoint not found: %CKPT% & exit /b 1 )
:: --num-samples 60 (not the default 4) so every field/contrast pair is actually covered,
:: with 0.1T over-represented (x3) — the ultra-low-field failure mode we most need to see.
python -m fieldbridge.cli eval-stage1-vae --config "%CONFIG%" --split-json "%SPLIT%" --split test ^
  --checkpoint "%CKPT%" --out "%OUTDIR%" --num-samples 60 ^
  --oversample-field 0.1 --oversample-factor 3 || exit /b 1

echo.
echo == DONE ==
echo   history:     %CKPT_DIR%history.jsonl
echo   best model:  %CKPT%
echo   test report: %OUTDIR%\metrics.json  +  %OUTDIR%\metrics.csv  +  %OUTDIR%\diagnostics.png
endlocal
