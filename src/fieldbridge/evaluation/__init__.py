"""Evaluation metrics."""

from fieldbridge.evaluation.stage1_diagnostics import (
    Stage1DiagnosticSpec,
    identity_tiler_contract,
    minus_one_one_foreground_mask,
    run_stage1_reconstruction_diagnostics,
    seam_gradient_metric,
)
from fieldbridge.evaluation.metrics import (
    gradient_mae,
    lpips_metric,
    mae,
    masked_mae,
    mse,
    normalized_cross_correlation,
    nrmse,
    outside_mask_mean_abs,
    psnr,
    ssim,
)
from fieldbridge.evaluation.mrixfields2026_official import (
    OFFICIAL_TASK3_METRIC_CONTRACT,
    evaluate_official_task3_directory,
    evaluate_official_task3_pair,
    load_official_nifti,
    match_official_task3_pairs,
    official_task3_lpips,
    official_task3_nrmse,
    official_task3_ssim,
)

__all__ = [
    "Stage1DiagnosticSpec",
    "gradient_mae",
    "lpips_metric",
    "mae",
    "masked_mae",
    "mse",
    "normalized_cross_correlation",
    "nrmse",
    "outside_mask_mean_abs",
    "psnr",
    "ssim",
    "OFFICIAL_TASK3_METRIC_CONTRACT",
    "evaluate_official_task3_directory",
    "evaluate_official_task3_pair",
    "load_official_nifti",
    "match_official_task3_pairs",
    "official_task3_lpips",
    "official_task3_nrmse",
    "official_task3_ssim",
    "identity_tiler_contract",
    "minus_one_one_foreground_mask",
    "run_stage1_reconstruction_diagnostics",
    "seam_gradient_metric",
]

