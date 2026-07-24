import pytest
import torch

import fieldbridge.training.checkpoints as checkpoint_module
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


def test_atomic_overwrite_failure_preserves_previous_checkpoint_and_cleans_temp(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "recoverable.pt"
    save_checkpoint(path, {"model": torch.zeros(2)})
    original_replace = checkpoint_module.os.replace

    def _fail_replace(source, destination):
        if destination == path:
            raise OSError("injected replacement crash")
        return original_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="injected replacement crash"):
        save_checkpoint(path, {"model": torch.ones(2)}, overwrite=True)

    assert torch.equal(load_checkpoint(path)["model"], torch.zeros(2))
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_oversized_atomic_overwrite_preserves_previous_checkpoint(tmp_path) -> None:
    path = tmp_path / "recoverable.pt"
    save_checkpoint(path, {"model": torch.zeros(2)})

    with pytest.raises(ValueError, match="size guardrail"):
        save_checkpoint(
            path,
            {"model": torch.zeros(10_000_000)},
            max_bytes=1_000,
            overwrite=True,
        )

    assert torch.equal(load_checkpoint(path)["model"], torch.zeros(2))
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_checkpoint_filename_matches_naming_convention() -> None:
    name = checkpoint_filename("transport", "otcfm", 500, timestamp="20260701")

    assert name == "transport_otcfm_20260701_step500.pt"


def test_resolve_git_commit_returns_string() -> None:
    assert isinstance(resolve_git_commit(), str)
