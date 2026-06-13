import pytest
import torch

from clbfield.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain


def test_domain_encodings_are_continuous_and_categorical() -> None:
    domain = Domain(3.0, "T2-FLAIR")

    field = domain.field_encoding()
    contrast = domain.contrast_encoding()
    conditioning = domain.conditioning_vector()

    assert field.shape == (1,)
    assert 0.0 <= float(field.item()) <= 1.0
    assert contrast.shape == (len(CONTRASTS),)
    assert torch.sum(contrast).item() == 1.0
    assert conditioning.shape == (1 + len(CONTRASTS),)
    assert domain.to_dict() == {"field_strength_t": 3.0, "contrast": "T2-FLAIR"}


def test_domain_rejects_unsupported_values() -> None:
    assert FIELD_STRENGTHS_T == (0.1, 1.5, 3.0, 5.0, 7.0)
    with pytest.raises(ValueError):
        Domain(9.4, "T1w")
    with pytest.raises(ValueError):
        Domain(3.0, "PDw")

