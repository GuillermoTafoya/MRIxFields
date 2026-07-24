from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.cli import (
    _preflight_stage1_v3_records,
    _resolve_stage1_epoch_dependent_schedules,
    _resolve_stage1_endpoint,
    _steps_per_epoch_for_volumes,
    main,
)
from fieldbridge.config import load_yaml_config
from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.datasets import (
    ALL_DOMAINS,
    StreamingPatchDataset,
    _scheduled_item_seed,
    collate_raw_batches,
)
from fieldbridge.data.domains import Domain
from fieldbridge.data.vae_splits import VaeSplits, save_vae_splits
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.latent_stats import DomainBalancedLatentStatsAccumulator
from fieldbridge.training.stage1_vae import (
    Stage1VAEConfig,
    Stage1VAETrainResult,
    _CandidateTracker,
    _capture_fixed_batch,
    _candidate_id,
    _checkpoint_payload,
    _save_pareto_checkpoint,
    preflight_stage1_resume,
    run_stage1_vae_train,
)


def _all_domain_records(*, suffix: str) -> tuple[VolumeRecord, ...]:
    return tuple(
        VolumeRecord(
            case_id=f"{suffix}-{index}",
            image_path=f"{suffix}-{index}.nii.gz",
            domain=domain,
            subject_id=f"subject-{index}",
            metadata={"prefix": suffix},
        )
        for index, domain in enumerate(ALL_DOMAINS)
    )


def test_config_only_v3_endpoint_resolves_40_complete_epochs() -> None:
    for name in (
        "stage1_ae_v3_joint_domain.yaml",
        "stage1_vae_v3_joint_domain_freebits.yaml",
        "stage1_vae_v3_target_decoder_film.yaml",
    ):
        config = load_yaml_config(Path("configs/experiment") / name)
        _resolve_stage1_endpoint(
            config,
            steps_per_epoch=7,
            cli_epochs=None,
            cli_steps=None,
        )
        assert config["training"]["endpoint_total_steps"] == 280
        assert config["training"]["resolved_epochs"] == 40
        assert config["training"]["endpoint_source"] == "config_epochs"


def test_config_only_v3_cli_invocation_uses_40_epochs(
    tmp_path, monkeypatch, capsys
) -> None:
    splits = VaeSplits(
        train=_all_domain_records(suffix="train"),
        validation=_all_domain_records(suffix="val"),
        test=(),
        seed=13,
        fractions=(0.8, 0.1, 0.1),
    )
    split_path = tmp_path / "split.json"
    save_vae_splits(splits, split_path)
    captured: dict[str, Stage1VAEConfig] = {}

    def _capture(config, **kwargs):
        del kwargs
        captured["config"] = config
        return Stage1VAETrainResult(steps=0)

    monkeypatch.setattr("fieldbridge.cli.run_stage1_vae_train", _capture)
    assert (
        main(
            [
                "train-stage1-vae",
                "--config",
                "configs/experiment/stage1_ae_v3_joint_domain.yaml",
                "--split-json",
                str(split_path),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    resolved = captured["config"]
    assert resolved.resolved_epochs == 40
    assert resolved.endpoint_total_steps == 40 * resolved.steps_per_epoch
    assert resolved.endpoint_source == "config_epochs"


def test_cli_epochs_override_config_epochs_and_config_epochs_override_steps() -> None:
    config = {"training": {"epochs": 40, "steps": 1, "require_all_validation_domains": True}}
    _resolve_stage1_endpoint(
        config, steps_per_epoch=9, cli_epochs=3, cli_steps=13
    )
    assert config["training"]["endpoint_total_steps"] == 27
    assert config["training"]["endpoint_source"] == "cli_epochs"

    config = {"training": {"epochs": 40, "steps": 1, "require_all_validation_domains": True}}
    _resolve_stage1_endpoint(
        config, steps_per_epoch=9, cli_epochs=None, cli_steps=13
    )
    assert config["training"]["endpoint_total_steps"] == 360
    assert config["training"]["endpoint_source"] == "config_epochs"


def test_free_bits_warmup_resolves_to_ten_complete_epochs() -> None:
    config = load_yaml_config(
        "configs/experiment/stage1_vae_v3_joint_domain_freebits.yaml"
    )
    _resolve_stage1_endpoint(
        config, steps_per_epoch=7, cli_epochs=None, cli_steps=None
    )
    _resolve_stage1_epoch_dependent_schedules(config, steps_per_epoch=7)
    assert config["training"]["kl_warmup_steps"] == 70
    assert config["training"]["lr_cosine_total_steps"] == 280


def test_validation_endpoint_must_reach_a_complete_epoch() -> None:
    config = {
        "data": {"joint_domain_balance": {"enabled": True}},
        "training": {"steps": 8},
    }
    with pytest.raises(ValueError, match="complete epochs"):
        _resolve_stage1_endpoint(
            config, steps_per_epoch=9, cli_epochs=None, cli_steps=None
        )


def test_iterable_worker_partial_batches_are_part_of_epoch_length() -> None:
    config = {
        "data": {"patches_per_volume": 1},
        "training": {"batch_size": 4, "num_workers": 3},
    }
    # Worker shards contain 4, 3 and 3 items, producing 1 + 1 + 1 batches.
    assert _steps_per_epoch_for_volumes(config, 10) == 3


def test_missing_validation_domain_fails_v3_preflight() -> None:
    splits = VaeSplits(
        train=_all_domain_records(suffix="train"),
        validation=_all_domain_records(suffix="val")[:-1],
        test=(),
        seed=13,
        fractions=(0.8, 0.1, 0.1),
    )
    config = {
        "data": {"joint_domain_balance": {"enabled": True}},
        "training": {"require_all_validation_domains": True},
    }
    with pytest.raises(ValueError, match="validation split must contain all 15"):
        _preflight_stage1_v3_records(config, splits)


def test_missing_domain_cli_fails_before_model_initialization(
    tmp_path, monkeypatch
) -> None:
    splits = VaeSplits(
        train=_all_domain_records(suffix="train"),
        validation=_all_domain_records(suffix="val")[:-1],
        test=(),
        seed=13,
        fractions=(0.8, 0.1, 0.1),
    )
    split_path = tmp_path / "split.json"
    save_vae_splits(splits, split_path)
    config = {
        "seed": 13,
        "data": {
            "patches_per_volume": 1,
            "joint_domain_balance": {"enabled": True},
        },
        "model": {
            "name": "kl_vae",
            "base_channels": 4,
            "latent_channels": 1,
            "spatial_dims": 3,
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "require_all_validation_domains": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def _model_initialization_is_forbidden(*args, **kwargs):
        raise AssertionError("model initialization occurred before v3 preflight")

    monkeypatch.setattr(
        "fieldbridge.cli.build_encoder", _model_initialization_is_forbidden
    )
    with pytest.raises(ValueError, match="validation split must contain all 15"):
        main(
            [
                "train-stage1-vae",
                "--config",
                str(config_path),
                "--split-json",
                str(split_path),
            ]
        )


def _latent_stats(
    *,
    repeat_first_domain_patches: int = 0,
) -> dict[str, object]:
    accumulator = DomainBalancedLatentStatsAccumulator(latent_channels=1)
    template = torch.tensor(
        [[[[[0.0, 2.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]]]]
    )
    for domain in ("0.1T/T1w", "7T/T2-FLAIR"):
        accumulator.update(
            domain=domain,
            volume_id=f"{domain}-a",
            mean=template,
            logvar=torch.zeros_like(template),
        )
        accumulator.update(
            domain=domain,
            volume_id=f"{domain}-b",
            mean=template + 1.0,
            logvar=torch.zeros_like(template),
        )
    for _ in range(repeat_first_domain_patches):
        accumulator.update(
            domain="0.1T/T1w",
            volume_id="0.1T/T1w-a",
            mean=template,
            logvar=torch.zeros_like(template),
        )
    return accumulator.compute(
        active_kl_threshold=0.01,
        active_std_threshold=0.05,
        input_dependence_threshold=0.01,
        require_raw_kl=True,
    )


def test_fixed_spatial_template_fails_input_dependence_gate() -> None:
    accumulator = DomainBalancedLatentStatsAccumulator(latent_channels=1)
    template = torch.tensor(
        [[[[[0.0, 2.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]]]]
    )
    for volume in ("a", "b", "c"):
        accumulator.update(
            domain="3T/T1w",
            volume_id=volume,
            mean=template,
            logvar=torch.zeros_like(template),
        )
    stats = accumulator.compute(
        active_kl_threshold=0.01,
        active_std_threshold=0.05,
        input_dependence_threshold=0.01,
        require_raw_kl=True,
    )
    assert stats["per_dim_std"][0] > 0.05
    assert stats["per_dim_raw_kl"][0] > 0.01
    assert stats["per_dim_input_dependence"] == [0.0]
    assert stats["active_units"] == 0


def test_patch_and_domain_imbalance_do_not_change_macro_latent_eligibility() -> None:
    balanced = _latent_stats()
    imbalanced = _latent_stats(repeat_first_domain_patches=100)
    assert balanced["active_units"] == imbalanced["active_units"] == 1
    assert balanced["active_mask"] == imbalanced["active_mask"]
    assert balanced["num_domains"] == imbalanced["num_domains"] == 2


def test_resume_with_different_split_fingerprint_fails_preflight(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=1, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=1, spatial_dims=3)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=1e-3
    )
    saved_cfg = Stage1VAEConfig(
        steps=2,
        endpoint_total_steps=2,
        steps_per_epoch=1,
        split_fingerprint="split-a",
        lr=1e-3,
    )
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        _checkpoint_payload(saved_cfg, encoder, decoder, optimizer, 1),
        checkpoint,
    )
    resume_cfg = Stage1VAEConfig(
        steps=2,
        endpoint_total_steps=2,
        steps_per_epoch=1,
        split_fingerprint="split-b",
        lr=1e-3,
        resume_from=checkpoint,
    )
    with pytest.raises(ValueError, match="split fingerprint mismatch"):
        preflight_stage1_resume(resume_cfg)


def test_resume_rejects_incompatible_scheduler_and_warmup(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=1, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=1, spatial_dims=3)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=1e-3
    )
    saved_cfg = Stage1VAEConfig(
        steps=20,
        endpoint_total_steps=20,
        steps_per_epoch=2,
        split_fingerprint="same-split",
        lr=1e-3,
        lr_schedule="cosine",
        lr_cosine_total_steps=20,
        lr_warmup_steps=4,
        kl_warmup_steps=10,
    )
    checkpoint = tmp_path / "scheduler.pt"
    torch.save(
        _checkpoint_payload(saved_cfg, encoder, decoder, optimizer, 4),
        checkpoint,
    )
    incompatible = Stage1VAEConfig(
        steps=20,
        endpoint_total_steps=20,
        steps_per_epoch=2,
        split_fingerprint="same-split",
        lr=1e-3,
        lr_schedule="cosine",
        lr_cosine_total_steps=20,
        lr_warmup_steps=5,
        kl_warmup_steps=10,
        resume_from=checkpoint,
    )
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        preflight_stage1_resume(incompatible)


class _TwoInputDataset(Dataset[RawBatch]):
    def __init__(self) -> None:
        self.domain = Domain(3.0, "T1w")

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> RawBatch:
        image = torch.full((1, 8, 8, 8), 0.25 + 0.5 * index)
        return RawBatch(
            image=image,
            source_domain=self.domain,
            target_domain=self.domain,
            metadata={"case_id": f"case-{index}", "subject_id": f"s{index}"},
        )


def test_ineligible_latent_cannot_write_promoted_checkpoint(tmp_path) -> None:
    loader = DataLoader(
        _TwoInputDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_raw_batches,
    )
    encoder = KLVAEEncoder(base_channels=4, latent_channels=1, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=1, spatial_dims=3)
    cfg = Stage1VAEConfig(
        steps=1,
        endpoint_total_steps=1,
        resolved_epochs=1,
        steps_per_epoch=1,
        latent_mode="deterministic",
        promotion_min_active_channels=2,
        ssim_window_size=3,
        loss_weights={"masked_l1": 1.0, "kl": 0.0},
        checkpoint_dir=tmp_path,
        checkpoint_max_bytes=20_000_000,
        recon_dump_every_epochs=0,
    )
    run_stage1_vae_train(
        cfg,
        encoder=encoder,
        decoder=decoder,
        loader=loader,
        val_loader=loader,
    )
    history = [
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history[0]["selection"]["candidate_class"] == "diagnostic_ineligible"
    assert not list(tmp_path.glob("vae_stage1_pareto_*.pt"))
    assert not list(tmp_path.glob("vae_stage1_candidate_*.pt"))
    assert not (tmp_path / "vae_kl_vae_best.pt").exists()
    latest = load_checkpoint(tmp_path / "vae_stage1_latest_recoverable.pt")
    assert latest["candidate_state"]["frontier"] == []
    assert latest["candidate_state"]["best"] == {}


def test_two_nondominated_points_keep_two_immutable_checkpoints(tmp_path) -> None:
    encoder = KLVAEEncoder(base_channels=4, latent_channels=1, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=1, spatial_dims=3)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=1e-3
    )
    cfg = Stage1VAEConfig(
        steps=2,
        endpoint_total_steps=2,
        steps_per_epoch=1,
        checkpoint_dir=tmp_path,
        checkpoint_max_bytes=20_000_000,
        split_fingerprint="synthetic-split",
    )
    tracker = _CandidateTracker()
    evidence = {"active_units": 1, "per_domain": {"3T/T1w": {}}}
    metric_vectors = (
        {"masked_nrmse": 0.1, "masked_mae": 0.2},
        {"masked_nrmse": 0.2, "masked_mae": 0.1},
    )
    for epoch, metrics in enumerate(metric_vectors, start=1):
        complete = {**metrics, "latent_active_channels": 1.0}
        expected_id = _candidate_id(epoch, epoch, complete)
        path = tmp_path / f"vae_stage1_pareto_{expected_id}.pt"
        _, added, candidate_id = tracker.update(
            metrics,
            active_channels=1,
            epoch=epoch,
            step=epoch,
            checkpoint_path=path,
            latent_evidence=evidence,
        )
        assert added and candidate_id == expected_id
        _save_pareto_checkpoint(
            cfg,
            encoder,
            decoder,
            optimizer,
            epoch,
            candidate_id=candidate_id,
            candidate_path=path,
            candidate_state=tracker.state_dict(),
            latent_health=evidence,
            val_epochs_no_improve=0,
            validation_sampler_state={
                "pass": epoch,
                "batch_offset": 0,
                "recoverable": True,
            },
        )

    assert len(tracker.frontier) == 2
    tracker.validate_checkpoint_mapping()
    checkpoints = sorted(tmp_path.glob("vae_stage1_pareto_*.pt"))
    assert len(checkpoints) == 2
    restored = _CandidateTracker()
    restored.load_state_dict(load_checkpoint(checkpoints[-1])["candidate_state"])
    restored.validate_checkpoint_mapping()


def test_item_crop_seed_is_deterministic_decorrelated_and_worker_invariant() -> None:
    record = VolumeRecord(
        case_id="case",
        image_path="case.nii.gz",
        domain=Domain(3.0, "T1w"),
        subject_id="subject",
        metadata={"prefix": "P"},
    )
    first = _scheduled_item_seed(
        seed=13, pass_index=2, schedule_position=7, record=record
    )
    assert first == _scheduled_item_seed(
        seed=13, pass_index=2, schedule_position=7, record=record
    )
    assert first != _scheduled_item_seed(
        seed=13, pass_index=2, schedule_position=8, record=record
    )
    assert first != _scheduled_item_seed(
        seed=13, pass_index=3, schedule_position=7, record=record
    )
    # Worker ID is deliberately absent: assignment cannot change the seed.


def test_fixed_reconstruction_capture_does_not_advance_validation_pass() -> None:
    dataset = StreamingPatchDataset(
        _resume_records("fixed"),
        image_loader=_resume_image_loader,
        patch_size=(8, 8, 8),
        patches_per_volume=1,
        seed=10_077,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_raw_batches,
    )
    assert dataset.state_dict() == {"pass": 0}
    assert _capture_fixed_batch(loader) is not None
    assert dataset.state_dict() == {"pass": 0}
    list(loader)
    assert dataset.state_dict() == {"pass": 1}


def test_checkpoint_contains_complete_resolved_experiment_identity() -> None:
    resolved = {
        "seed": 13,
        "data": {
            "patch_size": [64, 64, 64],
            "patches_per_volume": 16,
            "stratified_crop": {"foreground": 0.7, "border": 0.2, "air": 0.1},
        },
        "model": {"latent_channels": 4, "base_channels": 32},
        "training": {
            "epochs": 40,
            "endpoint_total_steps": 400,
            "steps_per_epoch": 10,
            "lr": 1e-4,
        },
    }
    cfg = Stage1VAEConfig.from_mapping(resolved)
    encoder = KLVAEEncoder(base_channels=4, latent_channels=1, spatial_dims=3)
    decoder = KLVAEDecoder(base_channels=4, latent_channels=1, spatial_dims=3)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=1e-4
    )
    payload = _checkpoint_payload(cfg, encoder, decoder, optimizer, 10)
    identity = payload["experiment_identity"]
    assert identity["complete_resolved_config"] is True
    assert identity["resolved_config"] == resolved
    assert len(identity["config_fingerprint"]) == 64
    assert identity["git_commit"]
    assert payload["split_fingerprint"] is None


class _InterruptAfterFirstPass(StreamingPatchDataset):
    def __iter__(self):
        if self._pass >= 1:
            raise RuntimeError("synthetic interruption")
        yield from super().__iter__()


def _resume_records(prefix: str) -> tuple[VolumeRecord, ...]:
    domain = Domain(3.0, "T1w")
    return tuple(
        VolumeRecord(
            case_id=f"{prefix}-{index}",
            image_path=f"{prefix}-{index}.nii.gz",
            domain=domain,
            subject_id=f"s{index}",
            metadata={"prefix": prefix},
        )
        for index in range(2)
    )


def _resume_image_loader(path: Path, record: VolumeRecord) -> torch.Tensor:
    del path
    offset = 0.05 if record.case_id.endswith("-1") else 0.0
    return torch.arange(1000, dtype=torch.float32).reshape(1, 10, 10, 10) / 1200.0 + offset


def _resume_loader(
    *,
    records: tuple[VolumeRecord, ...],
    seed: int,
    interrupt: bool = False,
    crop_log: list[str] | None = None,
) -> DataLoader[RawBatch]:
    dataset_type = _InterruptAfterFirstPass if interrupt else StreamingPatchDataset
    dataset = dataset_type(
        records,
        image_loader=_resume_image_loader,
        patch_size=(8, 8, 8),
        patches_per_volume=1,
        seed=seed,
    )

    def collate(items):
        if crop_log is not None:
            crop_log.extend(
                hashlib.sha256(
                    item.image.detach().cpu().numpy().tobytes()
                ).hexdigest()
                for item in items
            )
        return collate_raw_batches(items)

    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate,
    )


def _resume_models() -> tuple[KLVAEEncoder, KLVAEDecoder]:
    torch.manual_seed(1234)
    return (
        KLVAEEncoder(base_channels=4, latent_channels=1, spatial_dims=3),
        KLVAEDecoder(base_channels=4, latent_channels=1, spatial_dims=3),
    )


def _resume_config(
    checkpoint_dir: Path, *, resume_from: Path | None = None
) -> Stage1VAEConfig:
    return Stage1VAEConfig(
        steps=4,
        endpoint_total_steps=4,
        endpoint_source="config_epochs",
        resolved_epochs=2,
        steps_per_epoch=2,
        seed=77,
        lr=1e-3,
        lr_schedule="cosine",
        lr_warmup_steps=1,
        lr_min_factor=0.1,
        lr_cosine_total_steps=4,
        latent_mode="deterministic",
        latent_active_std_threshold=0.0,
        latent_input_dependence_threshold=0.0,
        promotion_min_active_channels=1,
        ssim_window_size=3,
        loss_weights={"masked_l1": 1.0, "background": 0.1, "kl": 0.0},
        val_early_stopping=True,
        val_early_stopping_patience=5,
        checkpoint_dir=checkpoint_dir,
        checkpoint_max_bytes=20_000_000,
        recon_dump_every_epochs=0,
        resume_from=resume_from,
    )


def _history_without_runtime(path: Path) -> list[dict[str, object]]:
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    for entry in entries:
        entry.pop("seconds", None)
    return entries


def _normalized_candidate_state(state: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(state))
    for item in normalized.get("frontier", []):
        item["checkpoint_path"] = Path(item["checkpoint_path"]).name
    for item in normalized.get("best", {}).values():
        if "checkpoint_path" in item:
            item["checkpoint_path"] = Path(item["checkpoint_path"]).name
    return normalized


def test_interrupted_resume_matches_uninterrupted_validation_and_selection(
    tmp_path,
) -> None:
    train_records = _resume_records("train")
    val_records = _resume_records("val")

    full_dir = tmp_path / "full"
    full_val_crops: list[str] = []
    full_encoder, full_decoder = _resume_models()
    run_stage1_vae_train(
        _resume_config(full_dir),
        encoder=full_encoder,
        decoder=full_decoder,
        loader=_resume_loader(records=train_records, seed=77),
        val_loader=_resume_loader(
            records=val_records, seed=10_077, crop_log=full_val_crops
        ),
    )

    resumed_dir = tmp_path / "resumed"
    first_val_crops: list[str] = []
    interrupted_encoder, interrupted_decoder = _resume_models()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_stage1_vae_train(
            _resume_config(resumed_dir),
            encoder=interrupted_encoder,
            decoder=interrupted_decoder,
            loader=_resume_loader(
                records=train_records, seed=77, interrupt=True
            ),
            val_loader=_resume_loader(
                records=val_records, seed=10_077, crop_log=first_val_crops
            ),
        )

    recovery_path = resumed_dir / "vae_stage1_latest_recoverable.pt"
    assert recovery_path.exists()
    second_val_crops: list[str] = []
    resumed_encoder, resumed_decoder = _resume_models()
    run_stage1_vae_train(
        _resume_config(resumed_dir, resume_from=recovery_path),
        encoder=resumed_encoder,
        decoder=resumed_decoder,
        loader=_resume_loader(records=train_records, seed=77),
        val_loader=_resume_loader(
            records=val_records, seed=10_077, crop_log=second_val_crops
        ),
    )

    assert first_val_crops + second_val_crops == full_val_crops
    assert _history_without_runtime(resumed_dir / "history.jsonl") == (
        _history_without_runtime(full_dir / "history.jsonl")
    )
    for name, value in full_encoder.state_dict().items():
        torch.testing.assert_close(value, resumed_encoder.state_dict()[name])
    for name, value in full_decoder.state_dict().items():
        torch.testing.assert_close(value, resumed_decoder.state_dict()[name])

    full_state = load_checkpoint(full_dir / "vae_stage1_latest_recoverable.pt")
    resumed_state = load_checkpoint(
        resumed_dir / "vae_stage1_latest_recoverable.pt"
    )
    assert full_state["scheduler"] == resumed_state["scheduler"]
    assert full_state["validation_sampler_state"] == {
        "pass": 2,
        "batch_offset": 0,
        "recoverable": True,
    }
    assert full_state["validation_sampler_state"] == resumed_state[
        "validation_sampler_state"
    ]
    assert full_state["val_epochs_no_improve"] == resumed_state[
        "val_epochs_no_improve"
    ]
    assert _normalized_candidate_state(full_state["candidate_state"]) == (
        _normalized_candidate_state(resumed_state["candidate_state"])
    )
