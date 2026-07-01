import json

from clbfield.cli import main


def test_train_cli_runs_with_default_smoke_config(capsys) -> None:
    exit_code = main(["train", "--steps", "2", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["steps"] == 2
    assert len(payload["losses"]) == 2
