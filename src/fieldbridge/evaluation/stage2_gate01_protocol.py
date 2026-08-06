"""Independent external protocol lock for scientific Gate 0.1 execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    GATE0_DIAGNOSTIC_COMMIT,
    GATE01_SUPPORT_THRESHOLD,
    RESPLIT_FINGERPRINT,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
    PosthocTargetCalibrator,
)

GATE01_PROTOCOL_LOCK_CONTRACT_VERSION = "stage2-gate01-protocol-lock-v1"
GATE01_PROTOCOL_OFFICIAL_METRICS = ("nrmse", "ssim", "lpips")

_LOCK_FIELDS = {
    "contract_version",
    "traveller_identity_sha256",
    "selection_fingerprint_sha256",
    "split_fingerprint",
    "support_threshold",
    "calibrator_artifact_sha256",
    "calibrator_template_sha256",
    "artifact_provenance",
    "official_metrics",
    "montage_specification",
    "montage_specification_sha256",
    "artifact_sha256",
}
_SPEC_FIELDS = _LOCK_FIELDS - {
    "contract_version",
    "montage_specification_sha256",
    "artifact_sha256",
}


def frozen_protocol_artifact_provenance() -> dict[str, str]:
    return {
        "stage1_run_c_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
        "full_latent_bank_build_commit": FULL_LATENT_BANK_BUILD_COMMIT,
        "gate0_diagnostic_commit": GATE0_DIAGNOSTIC_COMMIT,
        "sb_v2_checkpoint_sha256": SB_V2_CHECKPOINT_SHA256,
        "resplit_fingerprint": RESPLIT_FINGERPRINT,
    }


@dataclass(frozen=True, slots=True)
class Gate01ProtocolLock:
    """Values frozen independently of the prediction manifest under validation."""

    traveller_identity_sha256: str
    selection_fingerprint_sha256: str
    split_fingerprint: str
    support_threshold: float
    calibrator_artifact_sha256: str
    calibrator_template_sha256: str
    artifact_provenance: Mapping[str, str]
    official_metrics: tuple[str, ...]
    montage_specification: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "traveller_identity_sha256",
            "selection_fingerprint_sha256",
            "calibrator_artifact_sha256",
            "calibrator_template_sha256",
        ):
            if not _is_sha256(str(getattr(self, name))):
                raise ValueError(f"Gate 0.1 protocol lock {name} must be SHA-256.")
        if self.split_fingerprint != RESPLIT_FINGERPRINT:
            raise ValueError("Gate 0.1 protocol lock has a stale split fingerprint.")
        if not math.isfinite(self.support_threshold) or self.support_threshold != 0.0:
            raise ValueError("Gate 0.1 protocol support threshold is frozen at 0.0.")
        if self.support_threshold != GATE01_SUPPORT_THRESHOLD:
            raise ValueError("Gate 0.1 protocol/calibrator support thresholds disagree.")
        expected_artifacts = frozen_protocol_artifact_provenance()
        if dict(self.artifact_provenance) != expected_artifacts:
            raise ValueError("Gate 0.1 protocol lock has incompatible frozen artifacts.")
        if tuple(self.official_metrics) != GATE01_PROTOCOL_OFFICIAL_METRICS:
            raise ValueError(
                "Gate 0.1 protocol lock must pin nRMSE, SSIM, and LPIPS in order."
            )
        if not isinstance(self.montage_specification, Mapping):
            raise ValueError("Gate 0.1 protocol montage specification must be a mapping.")

    @property
    def montage_specification_sha256(self) -> str:
        return _sha256_json(self.montage_specification)

    @property
    def artifact_sha256(self) -> str:
        return _sha256_json(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": GATE01_PROTOCOL_LOCK_CONTRACT_VERSION,
            "traveller_identity_sha256": self.traveller_identity_sha256,
            "selection_fingerprint_sha256": self.selection_fingerprint_sha256,
            "split_fingerprint": self.split_fingerprint,
            "support_threshold": float(self.support_threshold),
            "calibrator_artifact_sha256": self.calibrator_artifact_sha256,
            "calibrator_template_sha256": self.calibrator_template_sha256,
            "artifact_provenance": dict(self.artifact_provenance),
            "official_metrics": list(self.official_metrics),
            "montage_specification": dict(self.montage_specification),
            "montage_specification_sha256": self.montage_specification_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            "contract_version": GATE01_PROTOCOL_LOCK_CONTRACT_VERSION,
            "artifact_sha256": self.artifact_sha256,
            "traveller_identity_sha256": self.traveller_identity_sha256,
            "selection_fingerprint_sha256": self.selection_fingerprint_sha256,
            "split_fingerprint": self.split_fingerprint,
            "support_threshold": self.support_threshold,
            "calibrator_artifact_sha256": self.calibrator_artifact_sha256,
            "calibrator_template_sha256": self.calibrator_template_sha256,
            "official_metrics": list(self.official_metrics),
            "montage_specification_sha256": self.montage_specification_sha256,
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "Gate01ProtocolLock":
        _assert_exact_keys(spec, _SPEC_FIELDS, "Gate 0.1 protocol-lock specification")
        return cls._from_values(spec)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Gate01ProtocolLock":
        _assert_exact_keys(payload, _LOCK_FIELDS, "Gate 0.1 protocol lock")
        if payload["contract_version"] != GATE01_PROTOCOL_LOCK_CONTRACT_VERSION:
            raise ValueError("Gate 0.1 protocol-lock contract is incompatible.")
        lock = cls._from_values(payload)
        if payload["montage_specification_sha256"] != lock.montage_specification_sha256:
            raise ValueError("Gate 0.1 protocol montage specification hash mismatch.")
        if payload["artifact_sha256"] != lock.artifact_sha256:
            raise ValueError("Gate 0.1 protocol-lock artifact hash mismatch.")
        return lock

    @classmethod
    def _from_values(cls, values: Mapping[str, Any]) -> "Gate01ProtocolLock":
        metrics = values["official_metrics"]
        if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
            raise ValueError("Gate 0.1 protocol official_metrics must be a sequence.")
        return cls(
            traveller_identity_sha256=str(values["traveller_identity_sha256"]),
            selection_fingerprint_sha256=str(values["selection_fingerprint_sha256"]),
            split_fingerprint=str(values["split_fingerprint"]),
            support_threshold=float(values["support_threshold"]),
            calibrator_artifact_sha256=str(values["calibrator_artifact_sha256"]),
            calibrator_template_sha256=str(values["calibrator_template_sha256"]),
            artifact_provenance={
                str(key): str(value)
                for key, value in dict(values["artifact_provenance"]).items()
            },
            official_metrics=tuple(str(value) for value in metrics),
            montage_specification=dict(values["montage_specification"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Gate01ProtocolLock":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Gate 0.1 protocol lock root must be a mapping.")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def assert_calibrator(self, calibrator: PosthocTargetCalibrator) -> None:
        if calibrator.artifact_sha256 != self.calibrator_artifact_sha256:
            raise ValueError("Gate 0.1 protocol/calibrator artifact SHA-256 mismatch.")
        if calibrator.template_sha256 != self.calibrator_template_sha256:
            raise ValueError("Gate 0.1 protocol/calibrator template SHA-256 mismatch.")
        if calibrator.split_fingerprint != self.split_fingerprint:
            raise ValueError("Gate 0.1 protocol/calibrator split mismatch.")
        if calibrator.support_threshold != self.support_threshold:
            raise ValueError("Gate 0.1 protocol/calibrator support threshold mismatch.")

    def assert_manifest_contract(
        self,
        *,
        traveller_identity_sha256: str,
        selection_fingerprint_sha256: str,
        split_fingerprint: str,
        support_threshold: float,
        artifact_provenance: Mapping[str, Any],
    ) -> None:
        observed = {
            "traveller_identity_sha256": traveller_identity_sha256,
            "selection_fingerprint_sha256": selection_fingerprint_sha256,
            "split_fingerprint": split_fingerprint,
            "support_threshold": support_threshold,
        }
        expected = {
            "traveller_identity_sha256": self.traveller_identity_sha256,
            "selection_fingerprint_sha256": self.selection_fingerprint_sha256,
            "split_fingerprint": self.split_fingerprint,
            "support_threshold": self.support_threshold,
        }
        if observed != expected:
            raise ValueError(
                "Gate 0.1 prediction manifest does not match the independent protocol lock."
            )
        if dict(artifact_provenance) != dict(self.artifact_provenance):
            raise ValueError(
                "Gate 0.1 prediction artifacts do not match the independent protocol lock."
            )

    def assert_runtime_contract(
        self,
        *,
        metrics: Sequence[str],
        montage_specification: Mapping[str, Any],
    ) -> None:
        if tuple(metrics) != self.official_metrics:
            raise ValueError("Gate 0.1 runtime metrics do not match the protocol lock.")
        if _sha256_json(montage_specification) != self.montage_specification_sha256:
            raise ValueError("Gate 0.1 runtime montage specification does not match the lock.")


def _assert_exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{name} schema mismatch: missing={missing}, unexpected={unexpected}."
        )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "GATE01_PROTOCOL_LOCK_CONTRACT_VERSION",
    "GATE01_PROTOCOL_OFFICIAL_METRICS",
    "Gate01ProtocolLock",
    "frozen_protocol_artifact_provenance",
]
