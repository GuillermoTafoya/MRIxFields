import json

from clbfield.cli import main


def test_mrixfields2026_print_spec_cli_outputs_json(capsys) -> None:
    exit_code = main(["mrixfields2026-print-spec"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["fields"] == ["0.1T", "1.5T", "3T", "5T", "7T"]
    assert payload["modalities"] == ["T1W", "T2W", "T2FLAIR"]
    assert payload["full_shape"] == [364, 436, 364]
    assert payload["submission_z_clip"] == [150, 180]
    assert payload["submission_shape"] == [364, 436, 30]
    assert payload["task1"]["pair_count"] == 4
    assert payload["task1"]["prediction_file_count"] == 36
    assert payload["task1"]["segmentation_file_count"] == 36
    assert payload["task1"]["requires_segmentation"] is True
    assert payload["task2"]["pair_count"] == 4
    assert payload["task2"]["prediction_file_count"] == 36
    assert payload["task2"]["segmentation_file_count"] == 36
    assert payload["task2"]["requires_segmentation"] is True
    assert payload["task3"]["pair_count"] == 20
    assert payload["task3"]["prediction_file_count"] == 180
    assert payload["task3"]["segmentation_file_count"] == 0
    assert payload["task3"]["requires_segmentation"] is False
    assert payload["validation_released_ids"]["0.1T"] == ["0001", "0002", "0003"]
