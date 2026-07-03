import torch
from torch import nn

from fieldbridge.training.warm_start import load_state_dict_tolerant


class _Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matching = nn.Linear(4, 4)
        self.wrong_shape = nn.Linear(4, 4)
        self.missing_in_checkpoint = nn.Linear(4, 4)


def test_load_state_dict_tolerant_loads_compatible_subset_and_reports_the_rest() -> None:
    target = _Target()
    source_state = target.state_dict()

    # Shape-mismatched key: replace with a differently-shaped tensor under the same name.
    source_state["wrong_shape.weight"] = torch.randn(2, 2)

    # Unexpected key: not present in the target module at all.
    source_state["extra_unexpected_param"] = torch.randn(1)

    # Missing key: simulate the checkpoint not having this module's weights.
    del source_state["missing_in_checkpoint.weight"]
    del source_state["missing_in_checkpoint.bias"]

    logged: list[str] = []
    result = load_state_dict_tolerant(target, source_state, log=logged.append)

    # The compatible "matching" layer's weights should have loaded successfully.
    assert torch.equal(target.matching.weight, source_state["matching.weight"])

    assert "missing_in_checkpoint.weight" in result.missing_keys
    assert "missing_in_checkpoint.bias" in result.missing_keys
    assert "extra_unexpected_param" in result.unexpected_keys
    # The shape-mismatched key must NOT raise, and must be reported somewhere in the logs.
    assert any("wrong_shape.weight" in message for message in logged)
    assert any("missing_in_checkpoint" in message for message in logged)
    assert any("extra_unexpected_param" in message for message in logged)
