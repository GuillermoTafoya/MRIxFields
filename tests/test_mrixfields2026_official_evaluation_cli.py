import json

import fieldbridge.cli as cli
from fieldbridge.evaluation.mrixfields2026_official import (
    OFFICIAL_TASK3_METRIC_CONTRACT,
)


def test_official_task3_directory_cli_prints_and_writes_same_payload(
    tmp_path, capsys, monkeypatch
) -> None:
    prediction_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    prediction_dir.mkdir()
    target_dir.mkdir()
    out_path = tmp_path / "results" / "official.json"
    expected = {
        "metric_contract": OFFICIAL_TASK3_METRIC_CONTRACT,
        "case_count": 2,
        "summary": {
            "nrmse_mean": 0.1,
            "nrmse_std": 0.01,
            "ssim_mean": 0.9,
            "ssim_std": 0.02,
            "lpips_mean": 0.2,
            "lpips_std": 0.03,
        },
    }
    observed: dict[str, object] = {}

    def fake_evaluate(pred_dir, tgt_dir, *, metrics, device):
        observed.update(
            {
                "prediction_dir": pred_dir,
                "target_dir": tgt_dir,
                "metrics": tuple(metrics),
                "device": device,
            }
        )
        return expected

    monkeypatch.setattr(
        cli, "evaluate_official_task3_directory", fake_evaluate
    )

    exit_code = cli.main(
        [
            "mrixfields2026-evaluate-task3",
            "--pred-dir",
            str(prediction_dir),
            "--target-dir",
            str(target_dir),
            "--device",
            "cpu",
            "--out",
            str(out_path),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed == expected
    assert written == expected
    assert observed == {
        "prediction_dir": prediction_dir,
        "target_dir": target_dir,
        "metrics": ("nrmse", "ssim", "lpips"),
        "device": "cpu",
    }
