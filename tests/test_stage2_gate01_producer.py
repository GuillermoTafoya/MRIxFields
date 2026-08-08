from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import fieldbridge.evaluation.stage2_gate01_producer as producer
import fieldbridge.evaluation.stage2_gate01_protocol as protocol
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.vae_splits import (
    VaeSplits,
    load_vae_splits,
    save_vae_splits,
    vae_splits_fingerprint,
)
from fieldbridge.evaluation.stage2_gate01 import fixed_montage_specifications
from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    SB_V2_CONFIG_SHA256,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CONFIG_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
)
from fieldbridge.evaluation.stage2_gate01_protocol import (
    GATE01_SCIENTIFIC_MODULES,
    Gate01ProtocolLock,
    frozen_protocol_artifact_provenance,
)
from fieldbridge.evaluation.stage2_transport_eval import DecodeSpec, TransportSamplerConfig
from fieldbridge.models.factory import build_decoder, build_translator

EVALUATION_COMMIT = "e" * 40


class _SyntheticBackend:
    def __init__(self) -> None:
        self.stage1_calls: list[str] = []
        self.sb_calls: list[tuple[str, str]] = []
        self._decode_paths_used: set[str] = set()

    @property
    def decode_paths_used(self) -> tuple[str, ...]:
        return tuple(sorted(self._decode_paths_used))

    def source(self, record: VolumeRecord) -> torch.Tensor:
        return torch.from_numpy(np.load(record.image_path, allow_pickle=False))

    def reconstruct(self, record: VolumeRecord, latent_path: Path) -> torch.Tensor:
        assert latent_path.is_file()
        self.stage1_calls.append(record.domain.label)
        self._decode_paths_used.add("full")
        return self.source(record) + 0.05

    def translate(
        self, record: VolumeRecord, target_domain: Domain, latent_path: Path
    ) -> torch.Tensor:
        assert latent_path.is_file()
        self.sb_calls.append((record.domain.label, target_domain.label))
        self._decode_paths_used.add("full")
        return self.source(record) + 0.01 * target_domain.field_strength_t


def _file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    if path.name == "stage1.pt" and raw == b"stage1-frozen":
        return STAGE1_RUN_C_CHECKPOINT_SHA256
    if path.name == "sb-v2.pt" and raw == b"sb-v2-frozen":
        return SB_V2_CHECKPOINT_SHA256
    if path.name == "stage1.yaml" and normalized == b"model: {name: kl_vae}\n":
        return STAGE1_RUN_C_CONFIG_SHA256
    if path.name == "sb-v2.yaml" and normalized == b"model: {name: flow_matching_latent}\n":
        return SB_V2_CONFIG_SHA256
    return hashlib.sha256(raw).hexdigest()


def _bundle(tmp_path: Path, monkeypatch) -> dict[str, object]:
    input_root = tmp_path / "private-inputs"
    input_root.mkdir(parents=True)
    records = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for field_index, field in enumerate(FIELD_STRENGTHS_T):
            domain = Domain(field, contrast)
            image_path = input_root / "images" / f"{contrast.value}-{field:g}T.npy"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            array = np.full(
                (1, 1, 4, 4, 4),
                0.1 + 0.1 * contrast_index + 0.01 * field_index,
                dtype=np.float32,
            )
            array[..., 0, 0, 0] = 0.0
            np.save(image_path, array, allow_pickle=False)
            records.append(
                VolumeRecord(
                    case_id=f"P_SYNTHETIC_0007_{contrast.value}_{field:g}T",
                    image_path=image_path,
                    domain=domain,
                    subject_id="SYNTHETIC_0007",
                    split="validation",
                    metadata={"prefix": "P"},
                )
            )
    splits = VaeSplits(
        train=(),
        validation=tuple(records),
        test=(),
        seed=13,
        fractions=(0.0, 1.0, 0.0),
        metadata={"synthetic": True},
    )
    synthetic_fingerprint = vae_splits_fingerprint(splits)
    monkeypatch.setattr(producer, "RESPLIT_FINGERPRINT", synthetic_fingerprint)
    monkeypatch.setattr(protocol, "RESPLIT_FINGERPRINT", synthetic_fingerprint)
    monkeypatch.setattr(
        producer,
        "load_volume",
        lambda record: torch.from_numpy(np.load(record.image_path, allow_pickle=False)),
    )
    split_path = save_vae_splits(splits, input_root / "split.json")
    retrospective_path = input_root / "images" / "retrospective-unused.npy"
    np.save(
        retrospective_path,
        np.ones((1, 1, 4, 4, 4), dtype=np.float32),
        allow_pickle=False,
    )
    retrospective_record = VolumeRecord(
        case_id="R_SYNTHETIC_UNUSED_T1w_0.1T",
        image_path=retrospective_path,
        domain=Domain(0.1, "T1w"),
        subject_id="SYNTHETIC_UNUSED",
        split="train",
        metadata={"prefix": "R"},
    )
    bank_records_source = [*records, retrospective_record]
    bank_splits = VaeSplits(
        train=tuple(bank_records_source),
        validation=(),
        test=(),
        seed=13,
        fractions=(1.0, 0.0, 0.0),
        metadata={"synthetic_bank_source": True},
    )
    bank_source_split_path = save_vae_splits(
        bank_splits, input_root / "bank-source-split.json"
    )
    bank_fingerprint = vae_splits_fingerprint(bank_splits)
    bank_file_sha256 = _file_sha256(bank_source_split_path)
    for module in (producer, protocol):
        monkeypatch.setattr(
            module, "FULL_LATENT_BANK_SOURCE_SPLIT_FINGERPRINT", bank_fingerprint
        )
        monkeypatch.setattr(
            module, "FULL_LATENT_BANK_SOURCE_SPLIT_FILE_SHA256", bank_file_sha256
        )
    selection_path = input_root / "selection.json"
    selection = producer.prepare_gate01_prospective_selection(
        split_path, "SYNTHETIC_0007", selection_path
    )

    lock = Gate01ProtocolLock(
        traveller_identity_sha256=selection["traveller_identity_sha256"],
        selection_fingerprint_sha256=selection["selection_fingerprint_sha256"],
        split_fingerprint=synthetic_fingerprint,
        evaluation_split_file_sha256=_file_sha256(split_path),
        bank_source_split_file_sha256=bank_file_sha256,
        bank_source_split_fingerprint=bank_fingerprint,
        support_threshold=0.0,
        calibrator_artifact_sha256="a" * 64,
        calibrator_template_sha256="b" * 64,
        evaluation_git_commit=EVALUATION_COMMIT,
        evaluation_module_sha256={
            name: f"{index + 1:064x}"
            for index, name in enumerate(GATE01_SCIENTIFIC_MODULES)
        },
        artifact_provenance=frozen_protocol_artifact_provenance(),
        official_metrics=("nrmse", "ssim", "lpips"),
        montage_specification=fixed_montage_specifications(),
    )
    lock_path = lock.save(input_root / "protocol-lock.json")
    bank = input_root / "latent-bank"
    (bank / "latents").mkdir(parents=True)
    bank_records = []
    for index, record in enumerate(bank_records_source):
        latent_path = bank / "latents" / f"latent-{index:02d}.pt"
        latent = torch.full((1, 1, 1, 1), float(index))
        latent_payload = {
            "contract_version": "latent-bank-v1",
            "case_id": record.case_id,
            "subject_id": record.subject_id,
            "split": "train",
            "domain": record.domain.to_dict(),
            "latent": latent,
            "latent_shape": [1, 1, 1, 1],
            "source_shape": [1, 4, 4, 4],
            "downsample_factor": 4,
            "encode_strategy": "full",
            "vae_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
            "git_commit": FULL_LATENT_BANK_BUILD_COMMIT,
        }
        torch.save(latent_payload, latent_path)
        bank_records.append(
            {
                "case_id": record.case_id,
                "subject_id": record.subject_id,
                "split": "train",
                "domain": record.domain.to_dict(),
                "latent_shape": [1, 1, 1, 1],
                "source_shape": [1, 4, 4, 4],
                "path": latent_path.relative_to(bank).as_posix(),
            }
        )
    (bank / "latent_stats.json").write_text(
        json.dumps(
            {
                "per_channel_mean": [0.0],
                "per_channel_std": [1.0],
                "vae_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
                "git_commit": FULL_LATENT_BANK_BUILD_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    (bank / "latent_bank_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": "latent-bank-v1",
                "config": {
                    "strategy": "full",
                    "block_size": [4, 4, 4],
                    "halo": [0, 0, 0],
                },
                "vae_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
                "git_commit": FULL_LATENT_BANK_BUILD_COMMIT,
                "strategy_used": ["full"],
                "records": bank_records,
            }
        ),
        encoding="utf-8",
    )
    stage1_config = input_root / "stage1.yaml"
    stage1_config.write_text("model: {name: kl_vae}\n", encoding="utf-8")
    sb_config = input_root / "sb-v2.yaml"
    sb_config.write_text("model: {name: flow_matching_latent}\n", encoding="utf-8")
    stage1_checkpoint = input_root / "stage1.pt"
    stage1_checkpoint.write_bytes(b"stage1-frozen")
    sb_checkpoint = input_root / "sb-v2.pt"
    sb_checkpoint.write_bytes(b"sb-v2-frozen")
    spec_path = input_root / "producer-spec.json"
    spec = producer.write_gate01_producer_spec(
        selection_path=selection_path,
        split_path=split_path,
        bank_source_split_path=bank_source_split_path,
        bank_dir=bank,
        stage1_config_path=stage1_config,
        stage1_checkpoint_path=stage1_checkpoint,
        sb_v2_config_path=sb_config,
        sb_v2_checkpoint_path=sb_checkpoint,
        protocol_lock_path=lock_path,
        sampler=TransportSamplerConfig(solver="heun", n_steps=20),
        decode=DecodeSpec(
            block_size=(4, 4, 4), halo=(0, 0, 0), precision="float32"
        ),
        out_path=spec_path,
        file_sha256=_file_sha256,
    )
    return {
        "input_root": input_root,
        "selection": selection,
        "selection_path": selection_path,
        "split_path": split_path,
        "bank_source_split_path": bank_source_split_path,
        "lock": lock,
        "lock_path": lock_path,
        "bank": bank,
        "stage1_config": stage1_config,
        "stage1_checkpoint": stage1_checkpoint,
        "sb_config": sb_config,
        "sb_checkpoint": sb_checkpoint,
        "spec": spec,
        "spec_path": spec_path,
    }


def _produce(
    bundle: dict[str, object],
    output: Path,
    state: Path,
    backend: _SyntheticBackend,
    *,
    resume: bool,
    observer=None,
):
    lock = bundle["lock"]
    return producer.produce_gate01_private_artifacts(
        spec_path=bundle["spec_path"],
        selection_path=bundle["selection_path"],
        split_path=bundle["split_path"],
        bank_source_split_path=bundle["bank_source_split_path"],
        bank_dir=bundle["bank"],
        stage1_config_path=bundle["stage1_config"],
        stage1_checkpoint_path=bundle["stage1_checkpoint"],
        sb_v2_config_path=bundle["sb_config"],
        sb_v2_checkpoint_path=bundle["sb_checkpoint"],
        protocol_lock_path=bundle["lock_path"],
        output_dir=output,
        state_dir=state,
        device="cpu",
        resume=resume,
        backend=backend,
        file_sha256=_file_sha256,
        code_commit=EVALUATION_COMMIT,
        code_provenance={
            "git_head": EVALUATION_COMMIT,
            "checkout_clean": True,
            "module_sha256": dict(lock.evaluation_module_sha256),
        },
        progress_observer=observer,
    )


def test_private_producer_clean_build_has_exact_graph_and_no_manual_cases(
    tmp_path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    backend = _SyntheticBackend()
    result = _produce(
        bundle, tmp_path / "outputs", tmp_path / "state", backend, resume=False
    )
    assert result["acquisition_count"] == 15
    assert result["stage1_inference_count"] == 15
    assert result["direction_count"] == 60
    assert result["sb_v2_inference_count"] == 60
    assert result["wrong_target_reference_count"] == 180
    assert result["producer_provenance"] == {
        "decode_strategy": "full",
        "path_used": ["full"],
    }
    assert bundle["spec"]["decode"]["strategy"] == "full"
    assert len(bundle["spec"]["selected_source_acquisitions"]) == 15
    assert bundle["spec"]["selected_payload_count"] == 15
    assert len(bundle["spec"]["selected_payload_identity_set_sha256"]) == 64
    assert bundle["spec"]["stage1_config_sha256"] == STAGE1_RUN_C_CONFIG_SHA256
    assert bundle["spec"]["sb_v2_config_sha256"] == SB_V2_CONFIG_SHA256
    assert bundle["spec"]["split_provenance"]["evaluation"]["membership_fingerprint"] != (
        bundle["spec"]["split_provenance"]["bank_storage"]["membership_fingerprint"]
    )
    assert len(backend.stage1_calls) == len(set(backend.stage1_calls)) == 15
    assert len(backend.sb_calls) == len(set(backend.sb_calls)) == 60
    plan = json.loads(Path(result["build_plan"]).read_text(encoding="utf-8"))
    assert len(plan["cases"]) == 60
    assert sum(len(case["wrong_target_sb_v2"]) for case in plan["cases"]) == 180
    state = json.loads(
        (tmp_path / "state" / "producer-state.json").read_text(encoding="utf-8")
    )
    assert plan["producer_receipt"] == state["producer_receipt"]
    assert state["producer_receipt"]["decode_strategy"] == "full"
    assert state["producer_receipt"]["path_used"] == ["full"]
    selection_text = Path(bundle["selection_path"]).read_text(encoding="utf-8")
    assert "SYNTHETIC_0007" not in selection_text
    assert str(bundle["input_root"]) not in json.dumps(plan)
    for case in plan["cases"]:
        source = np.load(
            tmp_path / "outputs" / case["source_image"]["path"], allow_pickle=False
        )
        mask = np.load(
            tmp_path / "outputs" / case["source_support_mask"]["path"],
            allow_pickle=False,
        )
        assert np.array_equal(mask, np.abs(source) > 0.0)


@pytest.mark.parametrize("selected_split", ["train", "test"])
def test_scientific_selection_rejects_prospective_nonvalidation_travellers(
    tmp_path, monkeypatch, selected_split
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    records = load_vae_splits(bundle["split_path"]).validation
    nonvalidation = VaeSplits(
        train=records if selected_split == "train" else (),
        validation=(),
        test=records if selected_split == "test" else (),
        seed=13,
        fractions=(1.0, 0.0, 0.0),
        metadata={"synthetic": True},
    )
    path = save_vae_splits(nonvalidation, tmp_path / f"{selected_split}-split.json")
    monkeypatch.setattr(
        producer, "RESPLIT_FINGERPRINT", vae_splits_fingerprint(nonvalidation)
    )
    with pytest.raises(ValueError, match="prospective validation traveller"):
        producer.prepare_gate01_prospective_selection(
            path, "SYNTHETIC_0007", tmp_path / f"{selected_split}-selection.json"
        )


def test_latent_manifest_payload_bank_storage_disagreement_fails_closed(
    tmp_path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    manifest_path = Path(bundle["bank"]) / "latent_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The last record is retrospective and not part of the selected 15. Complete-bank
    # validation must still reject its disagreement before any selected inference.
    manifest["records"][-1]["split"] = "validation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    backend = _SyntheticBackend()
    with pytest.raises(ValueError, match="manifest/payload/source-split membership"):
        _produce(
            bundle,
            tmp_path / "outputs",
            tmp_path / "state",
            backend,
            resume=False,
        )
    assert backend.stage1_calls == []
    assert backend.sb_calls == []


def test_private_producer_interrupted_resume_is_equivalent_without_duplicate_inference(
    tmp_path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    backend = _SyntheticBackend()

    def interrupt(_key: str, operation: int) -> None:
        if operation == 25:
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _produce(
            bundle,
            tmp_path / "resumed-output",
            tmp_path / "resumed-state",
            backend,
            resume=False,
            observer=interrupt,
        )
    resumed = _produce(
        bundle,
        tmp_path / "resumed-output",
        tmp_path / "resumed-state",
        backend,
        resume=True,
    )
    assert len(backend.stage1_calls) == 15
    assert len(backend.sb_calls) == 60
    clean_backend = _SyntheticBackend()
    clean = _produce(
        bundle,
        tmp_path / "clean-output",
        tmp_path / "clean-state",
        clean_backend,
        resume=False,
    )
    assert resumed["build_plan_sha256"] == clean["build_plan_sha256"]
    resumed_state = (tmp_path / "resumed-state" / "producer-state.json").read_bytes()
    clean_state = (tmp_path / "clean-state" / "producer-state.json").read_bytes()
    assert resumed_state == clean_state


def test_private_producer_adopts_atomically_published_pending_inference(
    tmp_path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    backend = _SyntheticBackend()
    original_write = producer._write_json_atomic
    interrupted = False

    def fail_between_array_and_state(path, payload):
        nonlocal interrupted
        completed = payload.get("completed", {})
        if (
            not interrupted
            and path.name == "producer-state.json"
            and any(key.startswith("stage1:") for key in completed)
            and not payload.get("pending")
        ):
            interrupted = True
            raise RuntimeError("state-write interruption")
        return original_write(path, payload)

    monkeypatch.setattr(producer, "_write_json_atomic", fail_between_array_and_state)
    with pytest.raises(RuntimeError, match="state-write interruption"):
        _produce(
            bundle,
            tmp_path / "outputs",
            tmp_path / "state",
            backend,
            resume=False,
        )
    assert len(backend.stage1_calls) == 1
    monkeypatch.setattr(producer, "_write_json_atomic", original_write)
    _produce(
        bundle,
        tmp_path / "outputs",
        tmp_path / "state",
        backend,
        resume=True,
    )
    assert len(backend.stage1_calls) == 15
    assert len(backend.sb_calls) == 60


def test_private_producer_rejects_mutation_stale_inputs_and_unexpected_paths(
    tmp_path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    output = tmp_path / "outputs"
    state = tmp_path / "state"
    _produce(bundle, output, state, _SyntheticBackend(), resume=False)
    first_array = next((output / "arrays").glob("*.npy"))
    changed = np.load(first_array, allow_pickle=False)
    changed.flat[0] += 1.0
    np.save(first_array, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="mutated completed array"):
        _produce(bundle, output, state, _SyntheticBackend(), resume=True)

    fresh = _bundle(tmp_path / "stale-config", monkeypatch)
    Path(fresh["stage1_config"]).write_text("model: {name: changed}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage1_config_sha256"):
        _produce(
            fresh,
            tmp_path / "stale-config-output",
            tmp_path / "stale-config-state",
            _SyntheticBackend(),
            resume=False,
        )

    checkpoint = _bundle(tmp_path / "stale-checkpoint", monkeypatch)
    Path(checkpoint["sb_checkpoint"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="sb_v2_checkpoint_sha256"):
        _produce(
            checkpoint,
            tmp_path / "stale-checkpoint-output",
            tmp_path / "stale-checkpoint-state",
            _SyntheticBackend(),
            resume=False,
        )

    unexpected = _bundle(tmp_path / "unexpected", monkeypatch)
    unexpected_output = tmp_path / "unexpected-output"
    unexpected_output.mkdir()
    (unexpected_output / "untracked.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected path"):
        _produce(
            unexpected,
            unexpected_output,
            tmp_path / "unexpected-state",
            _SyntheticBackend(),
            resume=False,
        )


def test_private_producer_rejects_selected_source_mutation_before_inference(
    tmp_path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    image_path = next(
        path
        for path in (Path(bundle["input_root"]) / "images").glob("*.npy")
        if "retrospective-unused" not in path.name
    )
    changed = np.load(image_path, allow_pickle=False)
    changed[..., -1, -1, -1] += 0.125
    np.save(image_path, changed, allow_pickle=False)
    backend = _SyntheticBackend()

    with pytest.raises(ValueError, match="selected_source_acquisitions"):
        _produce(
            bundle,
            tmp_path / "outputs",
            tmp_path / "state",
            backend,
            resume=False,
        )
    assert backend.stage1_calls == []
    assert backend.sb_calls == []

    resumed_bundle = _bundle(tmp_path / "resume", monkeypatch)
    resumed_output = tmp_path / "resume-output"
    resumed_state = tmp_path / "resume-state"
    _produce(
        resumed_bundle,
        resumed_output,
        resumed_state,
        _SyntheticBackend(),
        resume=False,
    )
    resumed_image = next(
        path
        for path in (Path(resumed_bundle["input_root"]) / "images").glob("*.npy")
        if "retrospective-unused" not in path.name
    )
    resumed_changed = np.load(resumed_image, allow_pickle=False)
    resumed_changed[..., 1, 1, 1] += 0.25
    np.save(resumed_image, resumed_changed, allow_pickle=False)
    resumed_backend = _SyntheticBackend()
    with pytest.raises(ValueError, match="selected_source_acquisitions"):
        _produce(
            resumed_bundle,
            resumed_output,
            resumed_state,
            resumed_backend,
            resume=True,
        )
    assert resumed_backend.stage1_calls == []
    assert resumed_backend.sb_calls == []


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("case_id", "wrong-case", "case identity"),
        ("split", "validation", "split identity"),
        ("domain", Domain(7.0, "T2w").to_dict(), "domain identity"),
        ("downsample_factor", 2, "downsample factor"),
        ("encode_strategy", "tiled", "not encoded as a full volume"),
        ("source_shape", [1, 8, 4, 4], "source shape"),
        ("vae_checkpoint_sha256", "0" * 64, "Stage-1 checkpoint"),
        ("git_commit", "0" * 40, "bank-build commit"),
    ],
)
def test_selected_latent_payload_fails_closed_on_stale_provenance(
    tmp_path, monkeypatch, field, replacement, message
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    manifest = json.loads(
        (Path(bundle["bank"]) / "latent_bank_manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["records"][0]
    latent_path = Path(bundle["bank"]) / entry["path"]
    payload = torch.load(latent_path, map_location="cpu")
    payload[field] = replacement
    resolved = producer.load_gate01_prospective_selection(
        bundle["selection_path"], bundle["split_path"]
    )
    record = resolved.records[Domain.from_dict(entry["domain"]).label]

    with pytest.raises(ValueError, match=message):
        producer._validate_selected_latent_payload(
            payload,
            entry=entry,
            record=record,
            bank_record=record,
            evaluation_split="validation",
            bank_storage_split="train",
            source_identity=bundle["spec"]["selected_source_acquisitions"][
                record.domain.label
            ],
        )


def test_real_gate01_backend_accepts_frozen_sb_config_shape_and_decoder_factor(
    tmp_path,
) -> None:
    stage_model = {
        "name": "kl_vae",
        "base_channels": 2,
        "latent_channels": 1,
        "spatial_dims": 3,
        "use_norm": False,
        "num_res_blocks": 1,
        "out_channels": 1,
    }
    sb_model = {
        "name": "flow_matching_latent",
        "latent_channels": 1,
        "hidden_channels": [2],
        "bottleneck_channels": 4,
        "cond_dim": 4,
        "time_embed_dim": 4,
        "time_scale": 1000.0,
        "spatial_dims": 3,
        "activation": "silu",
        "use_norm": False,
        "skip_mode": "concat",
        "zero_init_output": True,
    }
    stage_config = tmp_path / "stage1.yaml"
    sb_config = tmp_path / "sb.yaml"
    stage_config.write_text(json.dumps({"model": stage_model}), encoding="utf-8")
    sb_config.write_text(json.dumps({"model": sb_model}), encoding="utf-8")
    decoder = build_decoder("kl_vae", **producer._kl_decoder_kwargs(stage_model))
    translator_parameters = dict(sb_model)
    translator_name = translator_parameters.pop("name")
    translator = build_translator(translator_name, **translator_parameters)
    stage_checkpoint = tmp_path / "stage1.pt"
    sb_checkpoint = tmp_path / "sb.pt"
    torch.save({"decoder": decoder.state_dict()}, stage_checkpoint)
    torch.save({"translator": translator.state_dict()}, sb_checkpoint)
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "latent_stats.json").write_text(
        json.dumps({"per_channel_mean": [0.0], "per_channel_std": [1.0]}),
        encoding="utf-8",
    )

    backend = producer._RealGate01Backend.create(
        bank_dir=bank,
        stage1_config=stage_config,
        stage1_checkpoint=stage_checkpoint,
        sb_config=sb_config,
        sb_checkpoint=sb_checkpoint,
        sampler=TransportSamplerConfig(solver="heun", n_steps=1),
        decode=DecodeSpec(
            block_size=(4, 4, 4), halo=(0, 0, 0), precision="float32"
        ),
        device="cpu",
    )

    assert backend.factor == 4
    assert backend.decoder.downsample_factor == 4
    assert backend.translator.zero_init_output is True
    assert backend.decode_paths_used == ()


def test_producer_decode_contract_rejects_non_full_strategy() -> None:
    with pytest.raises(ValueError, match="full-volume decode specification"):
        producer._validate_sampler_decode(
            {"solver": "heun", "n_steps": 20},
            {
                "strategy": "auto",
                "block_size": [128, 128, 128],
                "halo": [16, 16, 16],
                "precision": "bfloat16",
            },
        )


def test_real_backend_seals_full_decode_path(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_decode(_decoder, latent, _domain, **kwargs):
        observed.update(kwargs)
        return torch.zeros(1, 1, 4, 4, 4), "full"

    monkeypatch.setattr(producer, "decode_latent", fake_decode)
    backend = producer._RealGate01Backend(
        decoder=object(),
        translator=object(),
        stats=object(),  # type: ignore[arg-type]
        sampler=TransportSamplerConfig(solver="heun", n_steps=1),
        decode=DecodeSpec(
            block_size=(4, 4, 4), halo=(0, 0, 0), precision="float32"
        ),
        device=torch.device("cpu"),
        factor=4,
    )

    result = backend._decode(
        torch.zeros(1, 1, 1, 1, 1), Domain(3.0, "T1w")
    )

    assert tuple(result.shape) == (1, 1, 4, 4, 4)
    assert observed["strategy"] == "full"
    assert backend.decode_paths_used == ("full",)


def test_producer_rejects_non_full_backend_path_before_publishing_inference(
    tmp_path, monkeypatch
) -> None:
    class _TiledBackend(_SyntheticBackend):
        def reconstruct(self, record: VolumeRecord, latent_path: Path) -> torch.Tensor:
            assert latent_path.is_file()
            self.stage1_calls.append(record.domain.label)
            self._decode_paths_used.add("tiled")
            return self.source(record)

    bundle = _bundle(tmp_path, monkeypatch)
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="non-full decode path"):
        _produce(
            bundle,
            output,
            tmp_path / "state",
            _TiledBackend(),
            resume=False,
        )

    assert list((output / "arrays").glob("stage1-*.npy")) == []
