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
from fieldbridge.data.vae_splits import VaeSplits, save_vae_splits, vae_splits_fingerprint
from fieldbridge.evaluation.stage2_gate01 import fixed_montage_specifications
from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
)
from fieldbridge.evaluation.stage2_gate01_protocol import (
    GATE01_SCIENTIFIC_MODULES,
    Gate01ProtocolLock,
    frozen_protocol_artifact_provenance,
)
from fieldbridge.evaluation.stage2_transport_eval import DecodeSpec, TransportSamplerConfig

EVALUATION_COMMIT = "e" * 40


class _SyntheticBackend:
    def __init__(self) -> None:
        self.stage1_calls: list[str] = []
        self.sb_calls: list[tuple[str, str]] = []

    def source(self, record: VolumeRecord) -> torch.Tensor:
        return torch.from_numpy(np.load(record.image_path, allow_pickle=False))

    def reconstruct(self, record: VolumeRecord, latent_path: Path) -> torch.Tensor:
        assert latent_path.is_file()
        self.stage1_calls.append(record.domain.label)
        return self.source(record) + 0.05

    def translate(
        self, record: VolumeRecord, target_domain: Domain, latent_path: Path
    ) -> torch.Tensor:
        assert latent_path.is_file()
        self.sb_calls.append((record.domain.label, target_domain.label))
        return self.source(record) + 0.01 * target_domain.field_strength_t


def _file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.name == "stage1.pt" and raw == b"stage1-frozen":
        return STAGE1_RUN_C_CHECKPOINT_SHA256
    if path.name == "sb-v2.pt" and raw == b"sb-v2-frozen":
        return SB_V2_CHECKPOINT_SHA256
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
                (1, 1, 2, 2, 2),
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
    split_path = save_vae_splits(splits, input_root / "split.json")
    selection_path = input_root / "selection.json"
    selection = producer.prepare_gate01_prospective_selection(
        split_path, "SYNTHETIC_0007", selection_path
    )

    lock = Gate01ProtocolLock(
        traveller_identity_sha256=selection["traveller_identity_sha256"],
        selection_fingerprint_sha256=selection["selection_fingerprint_sha256"],
        split_fingerprint=synthetic_fingerprint,
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
    for index, record in enumerate(records):
        latent_path = bank / "latents" / f"latent-{index:02d}.pt"
        torch.save({"latent": torch.full((1, 1, 1, 1), float(index))}, latent_path)
        bank_records.append(
            {
                "case_id": record.case_id,
                "subject_id": record.subject_id,
                "split": "validation",
                "domain": record.domain.to_dict(),
                "latent_shape": [1, 1, 1, 1],
                "source_shape": [1, 1, 2, 2, 2],
                "path": latent_path.relative_to(bank).as_posix(),
            }
        )
    (bank / "latent_stats.json").write_text(
        json.dumps({"per_channel_mean": [0.0], "per_channel_std": [1.0]}),
        encoding="utf-8",
    )
    (bank / "latent_bank_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": "latent-bank-v1",
                "config": {"block_size": [2, 2, 2], "halo": [0, 0, 0]},
                "vae_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
                "git_commit": FULL_LATENT_BANK_BUILD_COMMIT,
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
        bank_dir=bank,
        stage1_config_path=stage1_config,
        stage1_checkpoint_path=stage1_checkpoint,
        sb_v2_config_path=sb_config,
        sb_v2_checkpoint_path=sb_checkpoint,
        protocol_lock_path=lock_path,
        sampler=TransportSamplerConfig(solver="heun", n_steps=20),
        decode=DecodeSpec(
            block_size=(2, 2, 2), halo=(0, 0, 0), precision="float32"
        ),
        out_path=spec_path,
        file_sha256=_file_sha256,
    )
    return {
        "input_root": input_root,
        "selection": selection,
        "selection_path": selection_path,
        "split_path": split_path,
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
    assert len(backend.stage1_calls) == len(set(backend.stage1_calls)) == 15
    assert len(backend.sb_calls) == len(set(backend.sb_calls)) == 60
    plan = json.loads(Path(result["build_plan"]).read_text(encoding="utf-8"))
    assert len(plan["cases"]) == 60
    assert sum(len(case["wrong_target_sb_v2"]) for case in plan["cases"]) == 180
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
