import json

from fieldbridge.cli import main
from fieldbridge.official.data_manifest import read_manifest_jsonl, write_manifest_jsonl


def test_build_manifest_cli_writes_jsonl_and_prints_json(tmp_path, capsys) -> None:
    _touch_data_file(tmp_path, "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz")
    _touch_data_file(tmp_path, "Training_prospective/T2W/7T/P_T2W_7T_0006.nii.gz")
    out_path = tmp_path / "manifest.jsonl"

    exit_code = main(
        [
            "mrixfields2026-build-manifest",
            "--data-root",
            str(tmp_path),
            "--out",
            str(out_path),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert out_path.exists()
    assert payload["out"] == str(out_path)
    assert payload["audit"]["ok"] is True
    assert payload["audit"]["total_records"] == 2
    assert len(read_manifest_jsonl(out_path)) == 2


def test_audit_data_cli_reads_manifest(tmp_path, capsys) -> None:
    _touch_data_file(tmp_path, "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz")
    manifest = tmp_path / "manifest.jsonl"

    assert main(
        [
            "mrixfields2026-build-manifest",
            "--data-root",
            str(tmp_path),
            "--out",
            str(manifest),
            "--json",
        ]
    ) == 0
    capsys.readouterr()

    exit_code = main(["mrixfields2026-audit-data", "--manifest", str(manifest), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["total_records"] == 1


def test_audit_data_cli_returns_nonzero_for_invalid_manifest(tmp_path, capsys) -> None:
    _touch_data_file(tmp_path, "Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz")
    manifest = tmp_path / "manifest.jsonl"
    assert main(
        [
            "mrixfields2026-build-manifest",
            "--data-root",
            str(tmp_path),
            "--out",
            str(manifest),
            "--json",
        ]
    ) == 0
    capsys.readouterr()

    records = read_manifest_jsonl(manifest)
    bad = [records[0], records[0]]
    write_manifest_jsonl(bad, manifest)

    exit_code = main(["mrixfields2026-audit-data", "--manifest", str(manifest), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["duplicate_sample_ids"]


def test_audit_data_cli_can_scan_data_root(tmp_path, capsys) -> None:
    _touch_data_file(tmp_path, "Validating_prospective/T2FLAIR/3T/P_T2FLAIR_3T_0010.nii.gz")

    exit_code = main(["mrixfields2026-audit-data", "--data-root", str(tmp_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["counts_by_split"] == {"Validating_prospective": 1}


def _touch_data_file(root, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
