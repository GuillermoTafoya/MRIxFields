import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.domains import Domain
from fieldbridge.data.preprocessing import SliceGeometry
from fieldbridge.data.pseudo_pairs import (
    PseudoPairSliceSample,
    collate_pseudo_pair_slices,
)
from fieldbridge.evaluation.pseudo_pairs import PseudoPairEvalConfig, evaluate_pseudo_pairs
from fieldbridge.models.translators.base import BaseTranslator


class _EvalDataset(Dataset[PseudoPairSliceSample]):
    def __init__(self, fields: tuple[float, ...] = (1.5, 3.0, 1.5, 3.0)) -> None:
        self.fields = fields

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index: int) -> PseudoPairSliceSample:
        field = self.fields[index]
        target_value = field / 7.0
        return PseudoPairSliceSample(
            x_low=torch.zeros(1, 8, 8),
            x_high=torch.full((1, 8, 8), target_value),
            mask=torch.ones(1, 8, 8),
            source_domain=Domain(0.1, "T2-FLAIR"),
            target_domain=Domain(field, "T2-FLAIR"),
            record_id=f"case-{index}",
            volume_path=f"case-{index}.nii.gz",
            slice_index=index,
            degradation_seed=index,
            degradation_strength=0.5,
            geometry=SliceGeometry(
                slice_index=index,
                original_height=8,
                original_width=8,
                resized_height=8,
                resized_width=8,
                output_height=8,
                output_width=8,
            ),
        )


class _TargetValueTranslator(BaseTranslator):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, z, source_domain, target_domain, t=None):  # type: ignore[no-untyped-def]
        del source_domain, t
        domains = target_domain if isinstance(target_domain, list) else [target_domain] * int(z.shape[0])
        values = [domain.field_strength_t / 7.0 for domain in domains]
        stacked = torch.stack(
            [torch.full_like(z[index], float(value)) for index, value in enumerate(values)],
            dim=0,
        )
        return stacked + self.anchor * 0.0


def _loader(
    fields: tuple[float, ...] = (1.5, 3.0, 1.5, 3.0),
) -> DataLoader[PseudoPairSliceSample]:
    return DataLoader(
        _EvalDataset(fields),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_pseudo_pair_slices,
    )


def test_evaluation_reports_degraded_predicted_and_per_field_metrics() -> None:
    payload = evaluate_pseudo_pairs(
        _TargetValueTranslator(),
        _loader(),
        PseudoPairEvalConfig(model_range="zero_one", lpips="off", target_fields=(1.5, 3.0, 5.0)),
    )

    assert payload["num_samples"] == 4
    assert "degraded" in payload["aggregate"]
    assert "predicted" in payload["aggregate"]
    assert "psnr" in payload["aggregate"]["predicted"]
    assert "macro_average" in payload
    assert "nrmse" in payload["macro_average"]["predicted"]
    assert payload["per_target_field"]["1.5T"]["samples"] == 2
    assert payload["per_target_field"]["3T"]["samples"] == 2
    assert payload["improvement_over_degraded"]["nrmse"] > 0.0


def test_target_label_permutation_audit_executes_and_detects_improvement() -> None:
    payload = evaluate_pseudo_pairs(
        _TargetValueTranslator(),
        _loader(),
        PseudoPairEvalConfig(model_range="zero_one", lpips="off", target_fields=(1.5, 3.0, 5.0)),
    )

    audit = payload["target_conditioning_audit"]

    assert audit["correct_vs_wrong_improvement"]["absolute"]["nrmse"] > 0.0
    assert audit["correct_vs_wrong_improvement"]["relative"]["nrmse"] > 0.0
    assert audit["correct_vs_permuted_improvement"]["absolute"]["nrmse"] > 0.0
    assert audit["sample_level"]["fraction_correct_best_nrmse"] == 1.0
    assert audit["sample_level"]["mean_margin_vs_best_wrong_nrmse"] > 0.0
    assert audit["by_true_target_field"]["1.5T"]["fraction_correct_best_nrmse"] == 1.0
    assert audit["by_wrong_target_field"]["5T"]["samples"] == 4
    assert "correct_conditioning_improves_nrmse" not in audit
    assert "correct_conditioning_beats_permuted_nrmse" not in audit


def test_lpips_gracefully_skips_when_optional_dependency_is_unavailable(monkeypatch) -> None:
    import fieldbridge.training.losses as losses

    def _raise_import_error(device):  # type: ignore[no-untyped-def]
        del device
        raise ImportError("lpips unavailable")

    monkeypatch.setattr(losses, "build_lpips_net", _raise_import_error)

    payload = evaluate_pseudo_pairs(
        _TargetValueTranslator(),
        _loader(),
        PseudoPairEvalConfig(model_range="zero_one", lpips="auto", target_fields=(1.5, 3.0)),
    )

    assert payload["lpips"]["skipped"] is True
    assert "lpips unavailable" in payload["lpips"]["reason"]


def test_conditioning_audit_covers_all_four_target_fields() -> None:
    fields = (1.5, 3.0, 5.0, 7.0)
    payload = evaluate_pseudo_pairs(
        _TargetValueTranslator(),
        _loader(fields),
        PseudoPairEvalConfig(model_range="zero_one", lpips="off", target_fields=fields),
    )

    audit = payload["target_conditioning_audit"]
    labels = {"1.5T", "3T", "5T", "7T"}

    assert set(payload["per_target_field"]) == labels
    assert set(audit["by_true_target_field"]) == labels
    assert set(audit["by_wrong_target_field"]) == labels
    assert audit["sample_level"]["samples_with_wrong_targets"] == 4
    assert audit["sample_level"]["fraction_correct_best_nrmse"] == 1.0
    for label in labels:
        assert audit["by_true_target_field"][label]["samples_with_wrong_targets"] == 1
        assert audit["by_wrong_target_field"][label]["samples"] == 3
