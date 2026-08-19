from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fieldbridge.data.domains import Contrast, Domain
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.evaluation.stage2_photometry_baseline import PairedEvaluationCase
from fieldbridge.evaluation.stage2_unified import (
    _all_intermediates,
    _reductions,
    _wrong_target_controls,
)


class FieldTranslator(nn.Module):
    def forward(self, z, source_domains, target_domains, t):
        del source_domains, t
        values = torch.tensor(
            [domain.field_strength_t for domain in target_domains],
            dtype=z.dtype,
            device=z.device,
        ).reshape(-1, 1, 1, 1, 1)
        return values.expand_as(z) * 0.01


class Decoder(nn.Module):
    def decode(self, z, domains):
        del domains
        return z[:, :1]


@dataclass
class Context:
    values: torch.Tensor
    support_mask: torch.Tensor

    def with_values(self, values: torch.Tensor):
        return Context(values, self.support_mask)


class ArtifactSpy:
    def __init__(self) -> None:
        self.rendered_domains: list[str] = []

    def render_target(self, context, domain):
        self.rendered_domains.append(domain.label)
        return context.values


def _stats() -> FactoredLatentStats:
    return FactoredLatentStats(
        mean=torch.tensor([2.0, -1.0, 0.5, 3.0]),
        std=torch.tensor([0.5, 2.0, 1.5, 4.0]),
        supported_count=torch.ones(4, dtype=torch.int64),
        artifact_sha256="a" * 64,
    )


def test_wrong_target_mechanistic_control_holds_requested_render_map_fixed() -> None:
    requested = Domain(3.0, Contrast.T1W)
    case = PairedEvaluationCase(
        case_identity="R_pair",
        source=torch.zeros(1, 3, 3, 3),
        target=torch.zeros(1, 3, 3, 3),
        source_domain=Domain(0.1, Contrast.T1W),
        target_domain=requested,
        subject_group_identity="R:s",
        source_provenance={"case_id": "R_source"},
        target_provenance={"case_id": "R_target"},
    )
    artifact = ArtifactSpy()
    rows = _wrong_target_controls(
        FieldTranslator(),
        Decoder(),
        artifact,  # type: ignore[arg-type]
        _stats(),
        torch.zeros(1, 4, 3, 3, 3),
        Context(torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3, dtype=torch.bool)),
        case,
        1,
        "euler",
        torch.device("cpu"),
    )
    assert len(rows) == 4
    assert all(row["requested_render_domain"] == requested.label for row in rows)
    assert artifact.rendered_domains[::2] == [requested.label] * 4
    assert artifact.rendered_domains[1::2] == [row["conditioned_domain"] for row in rows]
    assert all("condition_native_render_nrmse" in row for row in rows)


def test_graph_evaluation_enumerates_every_valid_intermediate_field() -> None:
    values = _all_intermediates(
        Domain(0.1, Contrast.T2W), Domain(7.0, Contrast.T2W)
    )
    assert [item.field_strength_t for item in values] == [1.5, 3.0, 5.0]


def test_reductions_include_source_target_domains_and_raw_decoder_leakage() -> None:
    methods = {
        "raw_identity": {"nrmse": 1.0, "ssim": 0.1, "lpips": 0.9},
        "full_unified_model": {"nrmse": 0.5, "ssim": 0.5, "lpips": 0.4},
    }
    row = {
        "source_domain": "0.1T/T1w",
        "target_domain": "3T/T1w",
        "contrast": "T1w",
        "directed_field_pair": "0.1T->3T",
        "raw_identity_stratum": "ordinary",
        "methods": methods,
        "identical_support_methods": methods,
        "requested_vs_wrong_target": {"requested_better_than_every_wrong_target": True},
        "graph_consistency": {"mean_direct_vs_composed_l1": 0.25},
        "anatomy_preservation": {
            "low_mid": 0.1,
            "edge": 0.2,
            "gradient": 0.3,
            "total": 0.2,
        },
        "raw_pre_mask_decoder_background_leakage_mae": 0.75,
    }
    result = _reductions([row])
    assert "per_source_domain" in result and "per_target_domain" in result
    assert (
        result["overall"]["all"]["raw_pre_mask_decoder_background_leakage_mae"]
        == 0.75
    )
