"""Sealed, single-instance LPIPS support for the step-200 inference audit."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import torch

from fieldbridge.evaluation.mrixfields2026_official import (
    OfficialTask3LPIPSEvaluator,
    official_task3_nrmse,
    official_task3_ssim,
)


LPIPS_AUDIT_PROVENANCE_CONTRACT = "stage2-step200-lpips-alexnet-provenance-v1"
ALEXNET_WEIGHT_ENUM = "AlexNet_Weights.IMAGENET1K_V1"
ALEXNET_WEIGHT_URL = (
    "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
)
ALEXNET_WEIGHT_SHA256 = (
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
)
LPIPS_LINEAR_WEIGHT_RESOURCE = "weights/v0.1/alex.pth"
LPIPS_LINEAR_WEIGHT_URL = (
    "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
    "master/lpips/weights/v0.1/alex.pth"
)
LPIPS_LINEAR_WEIGHT_SHA256 = (
    "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
)
_SEALED_LPIPS_BY_DEVICE: dict[str, SealedOfficialLPIPS] = {}


@dataclass(frozen=True, slots=True)
class SealedOfficialLPIPS:
    evaluator: OfficialTask3LPIPSEvaluator
    provenance: dict[str, Any]

    def verify_unchanged(self) -> dict[str, Any]:
        current_state = canonical_tensor_state_sha256(self.evaluator.network)
        if current_state != self.provenance["canonical_tensor_state_sha256"]:
            raise RuntimeError("LPIPS evaluator tensor state changed during the audit.")
        alex_path = _alexnet_cache_path()
        linear_path = _lpips_linear_weight_path()
        if _sha256_file(alex_path) != self.provenance["alexnet_weight_file_sha256"]:
            raise RuntimeError("Cached AlexNet weight identity changed during the audit.")
        if _sha256_file(linear_path) != self.provenance["lpips_linear_weight_file_sha256"]:
            raise RuntimeError("LPIPS learned linear-weight identity changed during the audit.")
        if self.evaluator.network.training or any(
            parameter.requires_grad for parameter in self.evaluator.network.parameters()
        ):
            raise RuntimeError("LPIPS evaluator left frozen evaluation mode.")
        return {
            "canonical_tensor_state_sha256": current_state,
            "alexnet_weight_file_sha256": self.provenance["alexnet_weight_file_sha256"],
            "lpips_linear_weight_file_sha256": self.provenance[
                "lpips_linear_weight_file_sha256"
            ],
            "unchanged": True,
        }


def initialize_sealed_official_lpips(*, device: str = "cuda") -> SealedOfficialLPIPS:
    """Resolve weights once, authenticate them, and construct exactly one evaluator."""

    device_key = str(torch.device(device))
    if device_key in _SEALED_LPIPS_BY_DEVICE:
        return _SEALED_LPIPS_BY_DEVICE[device_key]
    import lpips
    from torchvision.models import AlexNet_Weights

    selected = AlexNet_Weights.IMAGENET1K_V1
    if selected.url != ALEXNET_WEIGHT_URL or AlexNet_Weights.DEFAULT is not selected:
        raise RuntimeError("torchvision AlexNet upstream weight identity changed.")
    alex_path = _alexnet_cache_path()
    alex_downloaded = _ensure_alexnet_weight(alex_path)
    started = time.perf_counter()
    network = lpips.LPIPS(
        pretrained=True,
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        eval_mode=True,
        verbose=False,
    )
    evaluator = OfficialTask3LPIPSEvaluator(network, device=device)
    initialization_seconds = time.perf_counter() - started
    if not alex_path.is_file() or _sha256_file(alex_path) != ALEXNET_WEIGHT_SHA256:
        raise RuntimeError("Pinned torchvision AlexNet weight file is missing or changed.")
    linear_path = _lpips_linear_weight_path()
    if not linear_path.is_file() or _sha256_file(linear_path) != LPIPS_LINEAR_WEIGHT_SHA256:
        raise RuntimeError("Pinned LPIPS learned linear-weight file is missing or changed.")
    state_sha = canonical_tensor_state_sha256(evaluator.network)
    provenance = {
        "contract_version": LPIPS_AUDIT_PROVENANCE_CONTRACT,
        "lpips_network": "alex",
        "lpips_version": "0.1",
        "lpips_construction_count": 1,
        "lpips_eval_mode": evaluator.network.training is False,
        "lpips_parameters_frozen": not any(
            parameter.requires_grad for parameter in evaluator.network.parameters()
        ),
        "torchvision_alexnet_weight_enum": ALEXNET_WEIGHT_ENUM,
        "torchvision_alexnet_weight_url": ALEXNET_WEIGHT_URL,
        "alexnet_weight_filename": alex_path.name,
        "alexnet_weight_file_sha256": ALEXNET_WEIGHT_SHA256,
        "alexnet_weight_downloaded": alex_downloaded,
        "lpips_linear_weight_resource": LPIPS_LINEAR_WEIGHT_RESOURCE,
        "lpips_linear_weight_source_url": LPIPS_LINEAR_WEIGHT_URL,
        "lpips_linear_weight_file_sha256": LPIPS_LINEAR_WEIGHT_SHA256,
        "lpips_linear_weight_downloaded": False,
        "canonical_tensor_state_sha256": state_sha,
        "initialization_seconds": initialization_seconds,
    }
    sealed = SealedOfficialLPIPS(evaluator=evaluator, provenance=provenance)
    _SEALED_LPIPS_BY_DEVICE[device_key] = sealed
    return sealed


def cached_official_gate01_metric_fn(
    sealed: SealedOfficialLPIPS,
):
    """Return the unchanged official metric adapter using one injected LPIPS network."""

    def metric(
        prediction: torch.Tensor,
        target: torch.Tensor,
        metrics: Sequence[str],
        device: str,
    ) -> Mapping[str, float]:
        if device != sealed.evaluator.device:
            raise ValueError("LPIPS evaluator device and audit metric device differ.")
        pred = _official_array(prediction, "prediction")
        tgt = _official_array(target, "target")
        result: dict[str, float] = {}
        if "nrmse" in metrics:
            result["nrmse"] = official_task3_nrmse(pred, tgt)
        if "ssim" in metrics:
            result["ssim"] = official_task3_ssim(pred, tgt)
        if "lpips" in metrics:
            result["lpips"] = sealed.evaluator(pred, tgt)
        return result

    return metric


@contextmanager
def forbid_network_access() -> Iterator[None]:
    """Fail closed if any case-loop code attempts network access."""

    def denied(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("Network access is forbidden after LPIPS initialization.")

    original_socket_connect = socket.socket.connect
    original_create_connection = socket.create_connection
    original_urlopen = urllib.request.urlopen
    original_http_connect = http.client.HTTPConnection.connect
    original_https_connect = http.client.HTTPSConnection.connect
    try:
        socket.socket.connect = denied
        socket.create_connection = denied
        urllib.request.urlopen = denied
        http.client.HTTPConnection.connect = denied
        http.client.HTTPSConnection.connect = denied
        yield
    finally:
        socket.socket.connect = original_socket_connect
        socket.create_connection = original_create_connection
        urllib.request.urlopen = original_urlopen
        http.client.HTTPConnection.connect = original_http_connect
        http.client.HTTPSConnection.connect = original_https_connect


def canonical_tensor_state_sha256(module: torch.nn.Module) -> str:
    """Hash sorted names, shapes, dtypes, and raw bytes for parameters and buffers."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _official_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    array = tensor.detach().cpu().to(torch.float32).numpy().squeeze()
    if array.ndim != 3:
        raise ValueError(
            f"Gate 0.1 {name} must resolve to one 3-D full volume; got {array.shape}."
        )
    return array.astype(np.float64)


def _alexnet_cache_path() -> Path:
    filename = Path(urlparse(ALEXNET_WEIGHT_URL).path).name
    if filename != "alexnet-owt-7be5be79.pth":
        raise RuntimeError("Pinned AlexNet weight URL became ambiguous.")
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _lpips_linear_weight_path() -> Path:
    resource = resources.files("lpips").joinpath(LPIPS_LINEAR_WEIGHT_RESOURCE)
    path = Path(str(resource))
    if path.name != "alex.pth":
        raise RuntimeError("Pinned LPIPS linear-weight resource became ambiguous.")
    return path


def _ensure_alexnet_weight(path: Path) -> bool:
    if path.exists():
        if not path.is_file() or _sha256_file(path) != ALEXNET_WEIGHT_SHA256:
            raise RuntimeError("Existing cached AlexNet weight file is substituted or changed.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.bounded-download")
    if temporary.exists():
        raise RuntimeError("Ambiguous unfinished AlexNet weight download exists.")
    digest = hashlib.sha256()
    total = 0
    started = time.monotonic()
    try:
        with urllib.request.urlopen(ALEXNET_WEIGHT_URL, timeout=120) as response:
            with temporary.open("xb") as stream:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > 300 * 1024**2 or time.monotonic() - started > 600:
                        raise RuntimeError("Bounded AlexNet weight download exceeded its limit.")
                    digest.update(block)
                    stream.write(block)
        if digest.hexdigest() != ALEXNET_WEIGHT_SHA256:
            raise RuntimeError("Downloaded AlexNet weight SHA-256 is incompatible.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ALEXNET_WEIGHT_ENUM",
    "ALEXNET_WEIGHT_SHA256",
    "ALEXNET_WEIGHT_URL",
    "LPIPS_AUDIT_PROVENANCE_CONTRACT",
    "LPIPS_LINEAR_WEIGHT_RESOURCE",
    "LPIPS_LINEAR_WEIGHT_SHA256",
    "LPIPS_LINEAR_WEIGHT_URL",
    "SealedOfficialLPIPS",
    "cached_official_gate01_metric_fn",
    "canonical_tensor_state_sha256",
    "forbid_network_access",
    "initialize_sealed_official_lpips",
]
