import pytest

from fieldbridge.cli import build_parser


def test_eval_stage1_vae_help_names_required_extras(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["eval-stage1-vae", "--help"])

    assert exc_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert 'pip install -e ".[nifti,evaluation]"' in output


@pytest.mark.parametrize(
    "command",
    ("select-stage1-vae-audit", "audit-stage1-vae", "smoke-stage1-audit"),
)
def test_stage1_full_volume_audit_commands_are_exposed(command: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args([command, "--help"])
    assert exc_info.value.code == 0
