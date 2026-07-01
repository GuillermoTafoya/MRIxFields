import pytest
import torch

from fieldbridge.training.checkpoints import (
    checkpoint_filename,
    load_checkpoint,
    resolve_git_commit,
    save_checkpoint,
)


def test_save_and_load_roundtrip_with_metadata(tmp_path) -> None:
    path = tmp_path / "translator_identity_20260701_step10.pt"
    state = {"model": torch.zeros(2)}

    saved_path = save_checkpoint(path, state, seed=13, config={"lr": 0.01})
    loaded = load_checkpoint(saved_path)

    assert saved_path == path
    assert torch.equal(loaded["model"], state["model"])
    assert loaded["_meta"]["seed"] == 13
    assert loaded["_meta"]["config"] == {"lr": 0.01}
    assert isinstance(loaded["_meta"]["git_commit"], str)


def test_save_checkpoint_rejects_silent_overwrite(tmp_path) -> None:
    path = tmp_path / "translator_identity_20260701_step10.pt"
    save_checkpoint(path, {"model": torch.zeros(2)})

    with pytest.raises(FileExistsError):
        save_checkpoint(path, {"model": torch.ones(2)})

    save_checkpoint(path, {"model": torch.ones(2)}, overwrite=True)
    assert torch.equal(load_checkpoint(path)["model"], torch.ones(2))


def test_save_checkpoint_rejects_oversized_output(tmp_path) -> None:
    path = tmp_path / "oversized.pt"
    huge_state = {"model": torch.zeros(10_000_000)}

    with pytest.raises(ValueError):
        save_checkpoint(path, huge_state, max_bytes=1_000)
    assert not path.exists()


def test_checkpoint_filename_matches_naming_convention() -> None:
    name = checkpoint_filename("transport", "otcfm", 500, timestamp="20260701")

    assert name == "transport_otcfm_20260701_step500.pt"


def test_resolve_git_commit_returns_string() -> None:
    assert isinstance(resolve_git_commit(), str)
