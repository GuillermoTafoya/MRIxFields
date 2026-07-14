import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import SliceGeometry
from fieldbridge.data.pseudo_pairs import (
    PseudoPairSliceSample,
    collate_pseudo_pair_slices,
)
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.pseudo_pair_epochs import (
    PseudoPairEpochConfig,
    train_pseudo_pair_epochs,
)


class _FixedPseudoPairDataset(Dataset[PseudoPairSliceSample]):
    def __init__(self, count: int = 4) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> PseudoPairSliceSample:
        field = 1.5 if index % 2 == 0 else 3.0
        return PseudoPairSliceSample(
            x_low=torch.zeros(1, 4, 4),
            x_high=torch.ones(1, 4, 4),
            mask=torch.ones(1, 4, 4),
            source_domain=Domain(0.1, "T2-FLAIR"),
            target_domain=Domain(field, "T2-FLAIR"),
            record_id=f"case-{index}",
            volume_path=f"case-{index}.nii.gz",
            slice_index=index,
            degradation_seed=index,
            degradation_strength=0.5,
            geometry=SliceGeometry(
                slice_index=index,
                original_height=4,
                original_width=4,
                resized_height=4,
                resized_width=4,
                output_height=4,
                output_width=4,
            ),
        )


class _AdditiveTranslator(BaseTranslator):
    def __init__(self) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.0))
        self.grad_modes: list[bool] = []

    def forward(self, z, source_domain, target_domain, t=None):  # type: ignore[no-untyped-def]
        del source_domain, target_domain, t
        self.grad_modes.append(torch.is_grad_enabled())
        return z + self.delta.reshape(1, 1, 1, 1)


def _loader() -> DataLoader[PseudoPairSliceSample]:
    return DataLoader(
        _FixedPseudoPairDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_pseudo_pair_slices,
    )


def _config(tmp_path, *, epochs: int = 3, resume_from=None) -> PseudoPairEpochConfig:  # type: ignore[no-untyped-def]
    return PseudoPairEpochConfig(
        epochs=epochs,
        batch_size=2,
        seed=1,
        lr=0.1,
        weight_decay=0.0,
        checkpoint_dir=tmp_path,
        resume_from=resume_from,
        loss_weights={"masked_l1": 1.0, "gradient": 0.0, "background": 0.0},
        scheduler={"name": "none"},
        log_every_steps=0,
    )


def test_steps_per_epoch_and_small_epoch_complete(tmp_path) -> None:
    train_loader = _loader()
    model = _AdditiveTranslator()

    result = train_pseudo_pair_epochs(
        _config(tmp_path, epochs=1),
        model=model,
        train_loader=train_loader,
        val_loader=_loader(),
        run_metadata={"split_sha256": "abc123"},
    )
    state = load_checkpoint(result.last_checkpoint)

    assert len(train_loader) == 2
    assert result.global_step == 2
    assert result.epochs_completed == 1
    assert result.history[0]["train"]["loss"] == result.history[0]["train"]["total"]
    assert set(result.history[0]["validation"]) >= {"loss", "total", "masked_l1", "gradient", "background"}
    assert state["run_metadata"]["split_sha256"] == "abc123"
    assert state["pseudo_pair_pipeline_version"] == 2


def test_validation_runs_without_gradients_and_checkpoints_are_written(tmp_path) -> None:
    model = _AdditiveTranslator()

    result = train_pseudo_pair_epochs(
        _config(tmp_path, epochs=1),
        model=model,
        train_loader=_loader(),
        val_loader=_loader(),
    )

    assert False in model.grad_modes
    assert result.best_checkpoint is not None and result.best_checkpoint.exists()
    assert result.last_checkpoint is not None and result.last_checkpoint.exists()
    assert (tmp_path / "history.jsonl").exists()


def test_resume_continues_epoch_and_global_step(tmp_path) -> None:
    first_model = _AdditiveTranslator()
    first = train_pseudo_pair_epochs(
        _config(tmp_path, epochs=1),
        model=first_model,
        train_loader=_loader(),
        val_loader=_loader(),
    )
    assert first.last_checkpoint is not None
    first_state = load_checkpoint(first.last_checkpoint)

    resumed_model = _AdditiveTranslator()
    resumed = train_pseudo_pair_epochs(
        _config(tmp_path, epochs=2, resume_from=first.last_checkpoint),
        model=resumed_model,
        train_loader=_loader(),
        val_loader=_loader(),
    )
    resumed_state = load_checkpoint(resumed.last_checkpoint)

    assert first_state["global_step"] == 2
    assert resumed.global_step == 4
    assert resumed_state["global_step"] == 4
    assert resumed.epochs_completed == 1


def test_loss_decreases_in_tiny_synthetic_setup(tmp_path) -> None:
    model = _AdditiveTranslator()

    result = train_pseudo_pair_epochs(
        _config(tmp_path, epochs=4),
        model=model,
        train_loader=_loader(),
        val_loader=_loader(),
    )

    assert result.history[-1]["train"]["loss"] < result.history[0]["train"]["loss"]
