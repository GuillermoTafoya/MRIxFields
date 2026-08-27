from __future__ import annotations

import copy
import gc
import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import sha256_file, sha256_json
from fieldbridge.evaluation import stage2_step200_inference_audit as audit
from fieldbridge.evaluation import stage2_unified_gate01_p0006 as p0006
from fieldbridge.evaluation.stage2_gate01 import Gate01Case
from fieldbridge.evaluation.stage2_unified_preflight import (
    PAIRED_FEASIBILITY_CONTRACT,
    seal_long_run_evaluation_readiness,
)


AUDIT_COMMIT = "a" * 40


def _node_sha(contrast: str, field: float) -> str:
    return hashlib.sha256(f"{contrast}|{field:g}".encode()).hexdigest()


def _case_receipt(contrast: str, source: float, target: float) -> dict[str, object]:
    wrong_fields = [
        float(field)
        for field in FIELD_STRENGTHS_T
        if float(field) not in {float(source), float(target)}
    ]
    return {
        "case_identity": f"synthetic-{contrast}-{source:g}-{target:g}",
        "source_domain": Domain(source, contrast).to_dict(),
        "target_domain": Domain(target, contrast).to_dict(),
        "source_image_sha256": _node_sha(contrast, source),
        "source_support_sha256": hashlib.sha256(
            f"support|{contrast}|{source:g}".encode()
        ).hexdigest(),
        "target_sha256": _node_sha(contrast, target),
        "raw_identity_sha256": hashlib.sha256(
            f"identity|{contrast}|{source:g}|{target:g}".encode()
        ).hexdigest(),
        "calibrated_identity_sha256": hashlib.sha256(
            f"calibrated|{contrast}|{source:g}|{target:g}".encode()
        ).hexdigest(),
        "original_sb_v2_sha256": hashlib.sha256(
            f"sb|{contrast}|{source:g}|{target:g}".encode()
        ).hexdigest(),
        "stage1_reconstruction_ceiling_sha256": hashlib.sha256(
            f"ceiling|{contrast}|{source:g}|{target:g}".encode()
        ).hexdigest(),
        "wrong_target_sb_v2_sha256": {
            f"{field:g}T": hashlib.sha256(
                f"wrong|{field:g}|{contrast}|{source:g}|{target:g}".encode()
            ).hexdigest()
            for field in wrong_fields
        },
    }


def _protocol() -> dict[str, object]:
    receipts = [
        _case_receipt(contrast.value, float(source), float(target))
        for contrast in CONTRASTS
        for source in FIELD_STRENGTHS_T
        for target in FIELD_STRENGTHS_T
        if source != target
    ]
    body: dict[str, object] = {
        "contract_version": p0006.GATE01_P0006_EVALUATION_PROTOCOL,
        "data_role": p0006.P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
        "evidence_interpretation": p0006.P0006_EVIDENCE_LIMITATION,
        "training_or_model_selection_use": False,
        "population_or_generalization_claims_authorized": False,
        "subject_group_identity": p0006.P0006_SUBJECT_GROUP,
        "traveller_identity_sha256": p0006.P0006_IDENTITY_SHA256,
        "acquisition_count": 15,
        "directed_pair_count": 60,
        "wrong_target_reference_count": 180,
        "factored_bank": {"P_record_count": 0},
        "frozen_unpaired_validation": {"P_endpoint_count": 0},
        "gate01_result": {"file_sha256": "d" * 64},
        "private_arrays_validated": True,
        "forbidden_travellers": ["P:0007", "P:0009"],
        "P0009_confirmation_status": p0006.P0009_CONFIRMATION_STATUS,
        "P0009_executed": False,
        "case_receipts": receipts,
    }
    body["protocol_sha256"] = sha256_json(body)
    return body


def _case_from_receipt(receipt: dict[str, object], ordinal: int) -> Gate01Case:
    source = Domain.from_dict(dict(receipt["source_domain"]))
    target = Domain.from_dict(dict(receipt["target_domain"]))
    base = float(ordinal) / 100.0
    source_tensor = torch.full((1, 8, 8, 8), base)
    target_tensor = torch.full((1, 8, 8, 8), base + 0.1)
    arrays = {
        "source_image": str(receipt["source_image_sha256"]),
        "source_support_mask": str(receipt["source_support_sha256"]),
        "target": str(receipt["target_sha256"]),
        "raw_identity": str(receipt["raw_identity_sha256"]),
        "raw_sb_v2": str(receipt["original_sb_v2_sha256"]),
        "stage1_reconstruction_ceiling": str(
            receipt["stage1_reconstruction_ceiling_sha256"]
        ),
        **{
            f"wrong_target_sb_v2[{label}]": str(value)
            for label, value in dict(receipt["wrong_target_sb_v2_sha256"]).items()
        },
    }
    return Gate01Case(
        case_id=str(receipt["case_identity"]),
        source_domain=source,
        target_domain=target,
        source_image=source_tensor,
        target=target_tensor,
        raw_identity=source_tensor.clone(),
        raw_sb_v2=source_tensor + 0.02,
        stage1_reconstruction_ceiling=target_tensor - 0.01,
        support_mask=torch.ones((1, 8, 8, 8), dtype=torch.bool),
        traveller_identity_sha256=p0006.P0006_IDENTITY_SHA256,
        array_sha256=arrays,
        wrong_target_sb_v2={
            label: source_tensor + 0.03 + index * 0.01
            for index, label in enumerate(
                dict(receipt["wrong_target_sb_v2_sha256"])
            )
        },
    )


class _SentinelManifest:
    def __init__(self, receipts: list[dict[str, object]], live_counts: list[int]):
        self.receipts = receipts
        self.live_counts = live_counts

    def __iter__(self):
        previous: weakref.ReferenceType[torch.Tensor] | None = None
        for ordinal, receipt in enumerate(self.receipts, 1):
            if previous is not None:
                gc.collect()
                if previous() is not None:
                    raise AssertionError("the previous tensor-bearing case is still live")
            case = _case_from_receipt(receipt, ordinal)
            assert case.source_image is not None
            previous = weakref.ref(case.source_image)
            self.live_counts.append(1)
            yield case
            del case


class _DummyRuntime:
    device = torch.device("cpu")
    gpu_identity = {
        "name": "synthetic CPU (A100 requirement disabled by test)",
        "total_memory_bytes": 0,
        "cuda_runtime": "none",
        "torch": torch.__version__,
    }

    def __init__(self):
        self.calls = 0
        self.max_live_outputs = 0
        self._live_outputs = 0

    def state_identity(self) -> dict[str, str]:
        return {"translator": "1" * 64, "encoder": "2" * 64, "decoder": "3" * 64}

    @torch.inference_mode()
    def infer_case(self, case, calibrator, *, plan):
        del calibrator
        assert not torch.is_grad_enabled()
        self.calls += 1
        self._live_outputs += 1
        self.max_live_outputs = max(self.max_live_outputs, self._live_outputs)
        unified = case.source_image + 0.08
        methods = {
            "raw_identity": case.raw_identity,
            "calibrated_identity": case.raw_identity + 0.01,
            "raw_original_sb_v2": case.raw_sb_v2,
            "calibrated_original_sb_v2": case.raw_sb_v2 + 0.005,
            "raw_unified_step200": unified,
            "calibrated_unified_step200": unified + 0.002,
            "stage1_reconstruction_ceiling": case.stage1_reconstruction_ceiling,
        }
        contrast_index = [item.value for item in CONTRASTS].index(
            case.source_domain.contrast.value
        )
        sweeps = {}
        if float(case.source_domain.field_strength_t) == 3.0:
            sweeps = {
                f"{field:g}T": np.full((8, 8), contrast_index + field, dtype=np.float32)
                for field in FIELD_STRENGTHS_T
            }
        selected = (
            float(case.source_domain.field_strength_t),
            float(case.target_domain.field_strength_t),
        ) in {(0.1, 7.0), (7.0, 0.1)}
        graph = {"selected": False}
        if selected:
            graph = {
                "selected": True,
                "intermediate_field_t": 3.0,
                "direct_vs_composed_l1": 0.01,
                "direct_vs_composed_mse": 0.001,
                "direct_slice": np.full((8, 8), 0.2, dtype=np.float32),
                "composed_slice": np.full((8, 8), 0.21, dtype=np.float32),
                "absolute_difference_slice": np.full((8, 8), 0.01, dtype=np.float32),
            }
        outputs = audit.InferenceCaseOutputs(
            methods=methods,
            anatomy={"gradient": 0.01},
            graph=graph,
            sweep_slices=sweeps,
            decoded_canonical_sha256="4" * 64,
        )
        weakref.finalize(unified, self._output_released)
        return outputs

    def _output_released(self):
        self._live_outputs -= 1


def _metric(prediction, target, metrics, device):
    del device
    error = float(torch.mean(torch.abs(prediction - target)))
    values = {"nrmse": error, "ssim": 1.0 - error, "lpips": 2.0 * error}
    return {name: values[name] for name in metrics}


def _install_synthetic_protocol(monkeypatch, tmp_path: Path):
    protocol = _protocol()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    feasibility_body: dict[str, object] = {
        "contract_version": PAIRED_FEASIBILITY_CONTRACT,
        "complete_inventory_no_selection": True,
        "paired_evaluation_possible": False,
    }
    feasibility = {
        **feasibility_body,
        "result_sha256": sha256_json(feasibility_body),
    }
    feasibility_path = tmp_path / "feasibility.json"
    feasibility_path.write_text(
        json.dumps(feasibility, sort_keys=True), encoding="utf-8"
    )
    readiness_path = tmp_path / "readiness.json"
    readiness = seal_long_run_evaluation_readiness(
        feasibility_path,
        p0006_evaluation_protocol_path=protocol_path,
        output_path=readiness_path,
    )
    monkeypatch.setattr(audit, "P0006_PROTOCOL_FILE_SHA256", sha256_file(protocol_path))
    monkeypatch.setattr(audit, "P0006_PROTOCOL_SHA256", protocol["protocol_sha256"])
    monkeypatch.setattr(
        audit, "EVALUATION_READINESS_FILE_SHA256", sha256_file(readiness_path)
    )
    monkeypatch.setattr(
        audit, "EVALUATION_READINESS_SHA256", readiness["readiness_sha256"]
    )
    live_counts: list[int] = []

    def context(_path):
        return SimpleNamespace(
            protocol=protocol,
            archive_root=None,
            gate_manifest=_SentinelManifest(protocol["case_receipts"], live_counts),
            calibrator=object(),
        )

    monkeypatch.setattr(p0006, "_load_verified_p0006_protocol_context", context)
    monkeypatch.setattr(p0006, "_case_receipt", lambda case, _cal: _case_receipt(
        case.source_domain.contrast.value,
        float(case.source_domain.field_strength_t),
        float(case.target_domain.field_strength_t),
    ))
    return protocol, protocol_path, readiness_path, live_counts


def test_authentic_production_shaped_readiness_v3_preflight_passes(
    monkeypatch, tmp_path
):
    protocol, protocol_path, readiness_path, _ = _install_synthetic_protocol(
        monkeypatch, tmp_path
    )
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["contract_version"] == "stage2-long-run-evaluation-readiness-v3"
    assert readiness["long_run_authorized_by_evaluation_path"] is True
    assert "long_run_training_authorized" not in readiness
    preflight = audit.preflight_frozen_p0006_scientific_role(
        protocol_path, readiness_path
    )
    provenance = preflight.sanitized_provenance()
    assert preflight.protocol == protocol
    assert provenance["long_run_authorized_by_evaluation_path"] is True
    assert provenance["long_run_training_authorized"] is False
    assert provenance["acquisition_count"] == 15
    assert provenance["directed_pair_count"] == 60
    assert provenance["wrong_target_reference_count"] == 180


@pytest.mark.parametrize("value", [False, None])
def test_readiness_requires_authentic_evaluation_path_authorization(
    monkeypatch, tmp_path, value
):
    _, _, readiness_path, _ = _install_synthetic_protocol(monkeypatch, tmp_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if value is None:
        readiness.pop("long_run_authorized_by_evaluation_path")
    else:
        readiness["long_run_authorized_by_evaluation_path"] = value
    with pytest.raises(ValueError, match="key inventory|Evaluation path"):
        audit._validate_scientific_role(_protocol(), readiness)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluation_role", "training", "role"),
        ("evidence_interpretation", "population evidence", "interpretation"),
        ("prospective_protocol_used", False, "prospective inventory"),
        ("prospective_training_or_model_selection_use", True, "safety boundary"),
        ("population_or_generalization_claims_authorized", True, "safety boundary"),
        ("reviewed_prospective_protocol_available", False, "prospective inventory"),
        ("complete_inventory_no_selection", False, "prospective inventory"),
        ("retrospective_pair_feasibility", True, "safety boundary"),
        ("factored_bank_P_record_count", 1, "P-record"),
        ("unpaired_validation_P_endpoint_count", 1, "P-endpoint"),
        ("P0009_confirmation_status", "executed", "P:0009 status"),
        ("P0009_executed", True, "safety boundary"),
    ],
)
def test_readiness_v3_semantic_mutations_fail_closed(
    monkeypatch, tmp_path, field, value, message
):
    protocol, _, readiness_path, _ = _install_synthetic_protocol(monkeypatch, tmp_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness[field] = value
    with pytest.raises(ValueError, match=message):
        audit._validate_scientific_role(protocol, readiness)


def test_readiness_protocol_link_and_closed_schema_fail_closed(monkeypatch, tmp_path):
    protocol, _, readiness_path, _ = _install_synthetic_protocol(monkeypatch, tmp_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["p0006_evaluation_protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another P:0006 protocol"):
        audit._validate_scientific_role(protocol, readiness)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["long_run_training_authorized"] = False
    with pytest.raises(ValueError, match="key inventory"):
        audit._validate_scientific_role(protocol, readiness)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_role", "training", "data role"),
        ("evidence_interpretation", "generalization", "interpretation"),
        ("subject_group_identity", "P:0009", "subject-group"),
        ("traveller_identity_sha256", "0" * 64, "traveller identity"),
        ("acquisition_count", 14, "acquisition count"),
        ("directed_pair_count", 59, "directed-pair count"),
        ("wrong_target_reference_count", 179, "wrong-target reference count"),
        ("training_or_model_selection_use", True, "training or model selection"),
        ("population_or_generalization_claims_authorized", True, "population"),
        ("P0009_confirmation_status", "executed", "P:0009 status"),
        ("P0009_executed", True, "P:0009 execution"),
    ],
)
def test_protocol_v4_scientific_role_mutations_fail_closed(
    monkeypatch, tmp_path, field, value, message
):
    protocol, _, readiness_path, _ = _install_synthetic_protocol(monkeypatch, tmp_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    protocol[field] = value
    with pytest.raises(ValueError, match=message):
        audit._validate_scientific_role(protocol, readiness)


def test_scientific_role_preflight_rehash_detects_change_before_use(
    monkeypatch, tmp_path
):
    _, protocol_path, readiness_path, _ = _install_synthetic_protocol(
        monkeypatch, tmp_path
    )
    preflight = audit.preflight_frozen_p0006_scientific_role(
        protocol_path, readiness_path
    )
    readiness_path.write_bytes(readiness_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        audit.verify_frozen_p0006_scientific_role_preflight(preflight)


def test_streaming_iterator_requires_release_and_keeps_one_case_live(monkeypatch, tmp_path):
    _, protocol_path, _, live_counts = _install_synthetic_protocol(monkeypatch, tmp_path)
    iterator = p0006.iter_gate01_p0006_evaluation_cases(protocol_path)
    first = next(iterator)
    with pytest.raises(RuntimeError, match="before releasing"):
        next(iterator)
    iterator.close()

    iterator = p0006.iter_gate01_p0006_evaluation_cases(protocol_path)
    count = 0
    for item in iterator:
        case = item.case
        assert case.source_image is not None
        item.release()
        del case
        count += 1
    assert count == 60
    assert max(live_counts) == 1


def test_streaming_evaluation_progress_is_strictly_count_only():
    final = {
        "stage": "p0006_streaming_evaluation",
        "status": "end",
        "case_count": 60,
        "expected_case_count": 60,
        "acquisition_node_count": 15,
    }
    assert p0006.validate_p0006_count_progress(final) == final
    for field in ("case_count", "expected_case_count", "acquisition_node_count"):
        malformed = dict(final)
        malformed[field] = True
        with pytest.raises(ValueError, match="nonnegative integer"):
            p0006.validate_p0006_count_progress(malformed)
    with pytest.raises(ValueError, match="forbidden"):
        p0006.validate_p0006_count_progress({**final, "case_id": "private"})


def test_plan_is_frozen_before_inference_and_target_arrays_cannot_select(monkeypatch, tmp_path):
    protocol, _, _, _ = _install_synthetic_protocol(monkeypatch, tmp_path)
    plan = audit.build_frozen_step200_inference_plan(protocol)
    mutated = json.loads(json.dumps(protocol))
    mutated["case_receipts"][0]["target_sha256"] = "f" * 64
    mutated_plan = audit.build_frozen_step200_inference_plan(mutated)
    assert plan.payload["case_seeds"] == mutated_plan.payload["case_seeds"]
    assert plan.payload["montage_specification"] == mutated_plan.payload[
        "montage_specification"
    ]
    assert plan.payload["P0006_target_used_for_seed_preprocessing_or_selection"] is False
    assert plan.payload["P0007_access"] is False
    assert plan.payload["P0009_access"] is False
    assert plan.montage_specification["display_order"] == [
        "source",
        "target",
        "raw_identity",
        "calibrated_identity",
        "raw_original_sb_v2",
        "calibrated_original_sb_v2",
        "raw_unified_step200",
        "calibrated_unified_step200",
        "stage1_reconstruction_ceiling",
        "absolute_difference",
        "edge_difference",
    ]
    assert plan.montage_specification["relative_slice_positions"] == [0.35, 0.5, 0.65]
    assert plan.payload["memory_gate_case"]["selection_uses_target_tensor"] is False


def test_complete_streaming_audit_resume_and_corruption_fail_closed(
    monkeypatch, tmp_path
):
    _, protocol_path, readiness_path, live_counts = _install_synthetic_protocol(
        monkeypatch, tmp_path
    )
    runtime = _DummyRuntime()
    output = tmp_path / "audit"
    progress: list[dict[str, object]] = []
    result = audit.run_step200_p0006_inference_audit(
        protocol_path=protocol_path,
        evaluation_readiness_path=readiness_path,
        runtime=runtime,
        output_dir=output,
        audit_implementation_commit=AUDIT_COMMIT,
        require_a100=False,
        progress_callback=progress.append,
        metric_fn=_metric,
    )
    assert result["case_count"] == 60
    assert result["acquisition_count"] == 15
    assert result["directed_pair_count"] == 60
    assert result["graph_path_count"] == 6
    assert result["training_invoked"] is False
    assert result["gradients_enabled"] is False
    assert result["optimizer_loaded"] is False
    assert result["P0006_training_or_model_selection_use"] is False
    assert result["population_or_generalization_claims_authorized"] is False
    assert result["long_run_training_authorized"] is False
    assert result["frozen_p0006_scientific_role_provenance"][
        "long_run_authorized_by_evaluation_path"
    ] is True
    assert result["frozen_p0006_scientific_role_provenance"][
        "long_run_training_authorized"
    ] is False
    assert result["final_stop"] == "STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION"
    assert max(live_counts) == 1
    assert runtime.max_live_outputs == 1
    assert any(item.get("case_count") == 60 for item in progress)
    assert len(list((output / "stage2_step200_p0006_case_receipts").glob("*.json"))) == 60
    assert len(result["by_contrast"]) == 3
    assert len(result["by_directed_field_pair"]) == 20
    assert result["overall"]["raw_unified_step200"]["nrmse"] == pytest.approx(0.02)
    assert result["paired_descriptive_differences"]["raw_identity"]["nrmse"][
        "wins"
    ] == 60
    assert (output / "stage2_step200_p0006_metrics.csv").is_file()
    assert (output / "stage2_step200_p0006_montages.pdf").is_file()
    assert (output / "stage2_step200_p0006_audit.html").is_file()
    assert not list(output.rglob("*.pt"))
    for sealed_name in (
        "run_contract.json",
        "stage2_step200_p0006_summary.json",
        "artifact_manifest.json",
    ):
        sealed = json.loads((output / sealed_name).read_text(encoding="utf-8"))
        assert sealed["long_run_training_authorized"] is False
        assert sealed["frozen_p0006_scientific_role_provenance"][
            "long_run_authorized_by_evaluation_path"
        ] is True
    for path in output.rglob("*.npy"):
        assert np.load(path, allow_pickle=False).ndim < 4

    first_calls = runtime.calls
    first_artifact_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    resumed = audit.run_step200_p0006_inference_audit(
        protocol_path=protocol_path,
        evaluation_readiness_path=readiness_path,
        runtime=runtime,
        output_dir=output,
        audit_implementation_commit=AUDIT_COMMIT,
        require_a100=False,
        metric_fn=_metric,
    )
    assert resumed["summary_sha256"] == result["summary_sha256"]
    assert runtime.calls == first_calls
    assert {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    } == first_artifact_hashes

    receipt = next((output / "stage2_step200_p0006_case_receipts").glob("*.json"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["metrics"]["raw_identity"]["nrmse"] += 1.0
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Self-hash"):
        audit.run_step200_p0006_inference_audit(
            protocol_path=protocol_path,
            evaluation_readiness_path=readiness_path,
            runtime=runtime,
            output_dir=output,
            audit_implementation_commit=AUDIT_COMMIT,
            require_a100=False,
            metric_fn=_metric,
        )


def test_source_enforces_inference_only_generator_state_and_no_materializing_loader():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "load_gate01_p0006_evaluation_protocol" not in source
    assert "iter_gate01_p0006_evaluation_cases" in source
    assert "torch.inference_mode" in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "build_critic" not in source
    assert "generator_optimizer" in source  # verified as checkpoint container metadata only
    assert "load_state_dict(state[\"translator\"]" not in source
    assert "translator.load_state_dict(translator_state, strict=True)" in source
    assert "sha256_json(vae_config) !=" not in source
    assert "STOP_FOR_HUMAN_RESOURCE_BOUNDED_TRAINING_DECISION" in source


def _synthetic_frozen_vae_config(monkeypatch, tmp_path, payload: bytes):
    path = tmp_path / "stage1-run-c.yaml"
    path.write_bytes(payload)
    monkeypatch.setattr(audit, "STAGE1_RUN_C_CONFIG_SHA256", sha256_file(path))
    monkeypatch.setattr(audit, "STAGE1_RUN_C_CONFIG_SIZE_BYTES", len(payload))
    return path, audit.preflight_frozen_stage1_run_c_config(path)


def test_frozen_vae_config_authorizes_raw_bytes_not_parsed_equivalence(monkeypatch, tmp_path):
    exact = b"model:\n  latent_channels: 2\n"
    equivalent = b"model: {latent_channels: 2}\n"
    exact_path, preflight = _synthetic_frozen_vae_config(monkeypatch, tmp_path, exact)
    equivalent_dir = tmp_path / "parsed-equivalent"
    equivalent_dir.mkdir()
    equivalent_path = equivalent_dir / "stage1-run-c.yaml"
    equivalent_path.write_bytes(equivalent)

    parsed_equivalent = audit.load_yaml_config(equivalent_path)
    assert sha256_json(preflight.parsed_config) == sha256_json(parsed_equivalent)
    assert sha256_file(exact_path) != sha256_file(equivalent_path)
    with pytest.raises(ValueError, match="config raw-file SHA-256 mismatch"):
        audit.preflight_frozen_stage1_run_c_config(equivalent_path)


def test_repository_vae_example_cannot_substitute_for_owner_stage1_run_c():
    repository_example = (
        Path(__file__).resolve().parents[1]
        / "configs/experiment/stage1_vae_v2_fgw_freebits.yaml"
    )
    # Even when reviewed Git bytes match, the repository example is not the exact
    # owner Drive role and cannot substitute operationally.
    with pytest.raises(ValueError, match="config path role is incorrect"):
        audit.preflight_frozen_stage1_run_c_config(repository_example)


def test_frozen_vae_config_preflight_has_field_specific_file_and_parse_failures(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing-stage1-run-c.yaml"
    with pytest.raises(FileNotFoundError, match="config is missing"):
        audit.preflight_frozen_stage1_run_c_config(missing)

    directory = tmp_path / "not-a-file"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        audit.preflight_frozen_stage1_run_c_config(directory)

    malformed = tmp_path / "stage1-run-c.yaml"
    malformed.write_bytes(b"model: [\n")
    monkeypatch.setattr(audit, "STAGE1_RUN_C_CONFIG_SHA256", sha256_file(malformed))
    monkeypatch.setattr(audit, "STAGE1_RUN_C_CONFIG_SIZE_BYTES", malformed.stat().st_size)
    with pytest.raises(ValueError, match="model configuration is malformed"):
        audit.preflight_frozen_stage1_run_c_config(malformed)


def test_frozen_vae_config_failure_precedes_bank_checkpoint_model_and_tensor_work(
    monkeypatch, tmp_path
):
    bad_config = tmp_path / "stage1-run-c.yaml"
    bad_config.write_bytes(b"model: {latent_channels: 2}\n")
    touched: list[str] = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            touched.append(label)
            raise AssertionError(label)

        return fail

    monkeypatch.setattr(audit, "PhotometryFactoredLatentBankIndex", forbidden("bank"))
    monkeypatch.setattr(audit, "load_checkpoint", forbidden("checkpoint"))
    monkeypatch.setattr(audit, "build_translator", forbidden("translator"))
    monkeypatch.setattr(audit, "build_encoder", forbidden("encoder"))
    monkeypatch.setattr(audit, "build_decoder", forbidden("decoder"))
    with pytest.raises(ValueError, match="config raw-file SHA-256 mismatch"):
        audit.load_unified_step200_inference_runtime(
            checkpoint_path=tmp_path / "step200.pt",
            resolved_config_path=tmp_path / "resolved.json",
            vae_config_path=bad_config,
            vae_checkpoint_path=tmp_path / "vae.pt",
            photometry_artifact_path=tmp_path / "photometry.json",
            bank_dir=tmp_path / "bank",
        )
    assert touched == []


def test_frozen_vae_bank_manifest_and_checkpoint_raw_identities_are_field_specific(
    monkeypatch, tmp_path
):
    config_path, config_preflight = _synthetic_frozen_vae_config(
        monkeypatch, tmp_path, b"model:\n  latent_channels: 2\n"
    )
    checkpoint_path = tmp_path / "vae.pt"
    checkpoint_path.write_bytes(b"synthetic frozen VAE checkpoint")
    checkpoint_sha = sha256_file(checkpoint_path)
    monkeypatch.setattr(audit, "STAGE1_RUN_C_CHECKPOINT_SHA256", checkpoint_sha)
    manifest_vae = {
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_sha,
    }
    monkeypatch.setattr(
        audit,
        "PhotometryFactoredLatentBankIndex",
        lambda *_args: SimpleNamespace(manifest={"vae": dict(manifest_vae)}),
    )

    provenance = audit.verify_frozen_stage1_vae_bank_provenance(
        config_preflight,
        vae_checkpoint_path=checkpoint_path,
        bank_dir=tmp_path / "bank",
    )
    receipt = provenance.sanitized_provenance()
    assert receipt["config_role"] == "frozen_stage1_run_c"
    assert receipt["raw_config_file_sha256"] == sha256_file(config_path)
    assert receipt["bank_manifest_config_sha256"] == sha256_file(config_path)
    assert receipt["parsed_canonical_config_sha256"] == sha256_json(
        config_preflight.parsed_config
    )
    assert receipt["checkpoint_file_sha256"] == checkpoint_sha
    assert receipt["bank_manifest_checkpoint_sha256"] == checkpoint_sha

    manifest_vae["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bank manifest VAE config SHA-256 mismatch"):
        audit.verify_frozen_stage1_vae_bank_provenance(
            config_preflight,
            vae_checkpoint_path=checkpoint_path,
            bank_dir=tmp_path / "bank",
        )
    manifest_vae["config_sha256"] = sha256_file(config_path)
    manifest_vae["checkpoint_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="bank manifest VAE checkpoint SHA-256 mismatch"):
        audit.verify_frozen_stage1_vae_bank_provenance(
            config_preflight,
            vae_checkpoint_path=checkpoint_path,
            bank_dir=tmp_path / "bank",
        )
    manifest_vae["checkpoint_sha256"] = checkpoint_sha
    checkpoint_path.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint raw-file SHA-256 mismatch"):
        audit.verify_frozen_stage1_vae_bank_provenance(
            config_preflight,
            vae_checkpoint_path=checkpoint_path,
            bank_dir=tmp_path / "bank",
        )


def test_runtime_loader_verifies_complete_checkpoint_then_builds_only_generator_and_vae(
    monkeypatch, tmp_path
):
    config = {
        "training": {
            "device": "auto",
            "batch_size": 1,
            "precision": "bf16",
            "integration_steps": 4,
            "integration_solver": "heun",
        },
        "model": {"name": "synthetic-translator"},
    }
    effective = copy.deepcopy(config)
    effective["training"]["device"] = "cuda"
    effective_normalized = audit.UnifiedStage2Config.from_mapping(effective).to_dict()
    expected_keys = {
        "contract_version", "run_fingerprint", "validation_plan_sha256",
        "selection_rule_sha256", "training_cursor", "translator", "critic",
        "generator_optimizer", "critic_optimizer", "generator_scheduler",
        "critic_scheduler", "scaler", "sampler_rng", "torch_rng", "cuda_rng",
        "python_rng", "numpy_rng", "history_prefix_bytes", "history_prefix_sha256",
        "validation_selection", "pilot_report", "in_progress_pilot_rows", "_meta",
    }
    checkpoint_state = {key: None for key in expected_keys}
    checkpoint_state.update(
        {
            "contract_version": audit.UNIFIED_RESUME_CONTRACT,
            "run_fingerprint": audit.RUN_FINGERPRINT,
            "training_cursor": 200,
            "translator": {},
            "_meta": {
                "git_commit": audit.TRAINING_EVIDENCE_COMMIT,
                "config": effective_normalized,
            },
        }
    )
    vae_config = {"model": {"latent_channels": 2}}
    vae_state = {"encoder": {}, "decoder": {}}
    photo_sha = "5" * 64
    config_raw_sha = "4" * 64
    bank = SimpleNamespace(
        manifest={
            "vae": {
                "checkpoint_sha256": audit.STAGE1_RUN_C_CHECKPOINT_SHA256,
                "config_sha256": config_raw_sha,
            },
            "photometry": {
                "artifact_file_sha256": photo_sha,
                "artifact_sha256": "6" * 64,
            },
            "operational_support_rule": {"rule_sha256": "7" * 64},
        }
    )
    artifact = SimpleNamespace(artifact_sha256="6" * 64)
    built: list[str] = []

    class Empty(torch.nn.Module):
        pass

    checkpoint_path = tmp_path / "checkpoint.pt"
    vae_path = tmp_path / "vae.pt"
    config_path = tmp_path / "resolved.json"
    vae_config_path = tmp_path / "stage1-run-c.yaml"
    photo_path = tmp_path / audit.REVIEWED_PHOTOMETRY_BASENAME
    for path in (checkpoint_path, vae_path, config_path, vae_config_path, photo_path):
        path.write_bytes(b"synthetic")

    monkeypatch.setattr(audit, "STAGE1_RUN_C_CONFIG_SHA256", config_raw_sha)
    monkeypatch.setattr(audit, "STAGE1_RUN_C_CONFIG_SIZE_BYTES", len(b"synthetic"))
    monkeypatch.setattr(audit, "REVIEWED_PHOTOMETRY_FILE_SHA256", photo_sha)
    monkeypatch.setattr(audit, "REVIEWED_PHOTOMETRY_ARTIFACT_SHA256", "6" * 64)
    monkeypatch.setattr(audit, "_load_json", lambda _path: config)
    monkeypatch.setattr(audit, "load_yaml_config", lambda _path: vae_config)
    monkeypatch.setattr(
        audit,
        "sha256_file",
        lambda path: (
            audit.CHECKPOINT_SHA256
            if Path(path) == checkpoint_path
            else audit.STAGE1_RUN_C_CHECKPOINT_SHA256
            if Path(path) == vae_path
            else config_raw_sha
            if Path(path) == vae_config_path
            else photo_sha
        ),
    )
    monkeypatch.setattr(
        audit,
        "load_checkpoint",
        lambda path, map_location="cpu": (
            checkpoint_state if Path(path) == checkpoint_path else vae_state
        ),
    )
    monkeypatch.setattr(
        audit, "build_translator", lambda *args, **kwargs: built.append("translator") or Empty()
    )
    monkeypatch.setattr(
        audit, "build_encoder", lambda *args, **kwargs: built.append("encoder") or Empty()
    )
    monkeypatch.setattr(
        audit, "build_decoder", lambda *args, **kwargs: built.append("decoder") or Empty()
    )
    monkeypatch.setattr(audit, "PhotometryFactoredLatentBankIndex", lambda *_: bank)
    monkeypatch.setattr(
        audit.FactoredLatentStats, "from_bank", lambda _path: SimpleNamespace()
    )
    photo_preflight = audit.ReviewedPhotometryPreflight(
        path=photo_path,
        artifact=artifact,
        artifact_file_sha256=photo_sha,
        artifact_internal_sha256="6" * 64,
        accepted_records_sha256="8" * 64,
        excluded_records_sha256="9" * 64,
        accepted_record_count=1_560,
        prospective_accepted_count=0,
        prospective_excluded_count=30,
        retrospective_numeric_collision_count=6,
        collision_group_counts=audit.REVIEWED_PHOTOMETRY_COLLISION_GROUP_COUNTS,
    )
    monkeypatch.setattr(
        audit,
        "preflight_reviewed_photometry_namespace_artifact",
        lambda _path: photo_preflight,
    )
    monkeypatch.setattr(
        audit,
        "verify_reviewed_photometry_bank_provenance",
        lambda preflight, *, bank_dir: audit.ReviewedPhotometryBankProvenance(
            preflight=preflight,
            bank_dir=Path(bank_dir),
            bank_manifest_artifact_file_sha256=photo_sha,
            bank_manifest_artifact_sha256="6" * 64,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(name="NVIDIA A100-SXM4-80GB", total_memory=80 * 1024**3),
    )

    runtime = audit.load_unified_step200_inference_runtime(
        checkpoint_path=checkpoint_path,
        resolved_config_path=config_path,
        vae_config_path=vae_config_path,
        vae_checkpoint_path=vae_path,
        photometry_artifact_path=photo_path,
        bank_dir=tmp_path / "bank",
    )
    assert built == ["translator", "encoder", "decoder"]
    assert runtime.translator.training is False
    assert runtime.encoder.training is False
    assert runtime.decoder.training is False
    assert not any(parameter.requires_grad for parameter in runtime.translator.parameters())
    assert runtime.frozen_stage1_vae_provenance["raw_config_file_sha256"] == config_raw_sha
    assert runtime.frozen_stage1_vae_provenance["parsed_canonical_config_sha256"] == sha256_json(
        vae_config
    )
    assert (
        runtime.reviewed_photometry_provenance["artifact_file_sha256"]
        == photo_sha
    )
    assert (
        runtime.reviewed_photometry_provenance["artifact_internal_sha256"]
        == "6" * 64
    )
    assert (
        runtime.reviewed_photometry_provenance[
            "bank_manifest_artifact_file_sha256"
        ]
        == photo_sha
    )
    run_contract = audit._run_contract(
        audit.FrozenStep200InferencePlan({"inference_plan_sha256": "9" * 64}),
        runtime,
        audit_implementation_commit=AUDIT_COMMIT,
        dependency_receipt={"synthetic": True},
        lpips_receipt={"synthetic": True},
        scientific_role_provenance={"synthetic_test_role": True},
    )
    sealed_vae = run_contract["frozen_stage1_vae_provenance"]
    assert sealed_vae["raw_config_file_sha256"] == config_raw_sha
    assert sealed_vae["parsed_canonical_config_sha256"] == sha256_json(vae_config)
    assert "config_path" not in sealed_vae
    sealed_photometry = run_contract["reviewed_photometry_provenance"]
    assert sealed_photometry["contract_version"] == (
        audit.REVIEWED_PHOTOMETRY_NAMESPACE_PROVENANCE_CONTRACT
    )
    assert sealed_photometry["accepted_record_count"] == 1_560
    assert sealed_photometry["prospective_accepted_count"] == 0
    assert "artifact_path" not in sealed_photometry

    checkpoint_state["_meta"] = {
        "git_commit": audit.TRAINING_EVIDENCE_COMMIT,
        "config": audit.UnifiedStage2Config.from_mapping(config).to_dict(),
    }
    with pytest.raises(ValueError, match="metadata identity"):
        audit.load_unified_step200_inference_runtime(
            checkpoint_path=checkpoint_path,
            resolved_config_path=config_path,
            vae_config_path=vae_config_path,
            vae_checkpoint_path=vae_path,
            photometry_artifact_path=photo_path,
            bank_dir=tmp_path / "bank",
        )

    monkeypatch.setattr(
        audit,
        "sha256_file",
        lambda path: (
            "0" * 64
            if Path(path) == checkpoint_path
            else audit.STAGE1_RUN_C_CHECKPOINT_SHA256
            if Path(path) == vae_path
            else config_raw_sha
            if Path(path) == vae_config_path
            else photo_sha
        ),
    )
    with pytest.raises(ValueError, match="checkpoint file SHA-256"):
        audit.load_unified_step200_inference_runtime(
            checkpoint_path=checkpoint_path,
            resolved_config_path=config_path,
            vae_config_path=vae_config_path,
            vae_checkpoint_path=vae_path,
            photometry_artifact_path=photo_path,
            bank_dir=tmp_path / "bank",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_one_case_inference_equivalence_and_memory_is_measured_not_fabricated():
    cpu_model = torch.nn.Conv3d(1, 1, 1, bias=False).eval().requires_grad_(False)
    with torch.no_grad():
        cpu_model.weight.fill_(0.5)
    cuda_model = torch.nn.Conv3d(1, 1, 1, bias=False).cuda().eval().requires_grad_(False)
    cuda_model.load_state_dict(cpu_model.state_dict())
    case = torch.linspace(0, 1, 8**3).reshape(1, 1, 8, 8, 8)
    with torch.inference_mode():
        expected = cpu_model(case)
        torch.cuda.reset_peak_memory_stats()
        observed = cuda_model(case.cuda()).cpu()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    assert torch.allclose(expected, observed, atol=1e-6, rtol=1e-6)
    assert peak > 0
    assert not torch.is_grad_enabled() or observed.grad_fn is None
